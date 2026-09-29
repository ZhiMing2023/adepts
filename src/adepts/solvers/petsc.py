from typing import Hashable, Optional

import numpy as np
import torch
from petsc4py import PETSc


INT32_MAX = np.iinfo(np.int32).max

NUMERIC_FIXED = "numeric_fixed"
STRUCTURE_FIXED = "structure_fixed"
_VALID_REUSE_MODES = {NUMERIC_FIXED, STRUCTURE_FIXED}


def ensure_cpu_csr(A: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(A):
        raise TypeError("A must be a torch tensor")
    if A.is_cuda:
        raise TypeError("A must be on the CPU")
    if A.layout == torch.sparse_csr:
        return A
    return A.to_sparse().coalesce().to_sparse_csr()


def torch_csr_to_petsc_fixed_structure(A: torch.Tensor):
    A = ensure_cpu_csr(A)
    m, n = map(int, A.shape)

    crow_t = A.crow_indices().cpu()
    col_t = A.col_indices().cpu()
    val_t = A.values().cpu()

    if crow_t.dtype != torch.int64 or col_t.dtype != torch.int64:
        raise TypeError("Torch CSR indices must be int64")
    if crow_t[-1].item() != col_t.numel():
        raise ValueError("Invalid CSR structure: crow[-1] != len(col)")
    if crow_t.max().item() > INT32_MAX or col_t.max().item() > INT32_MAX:
        raise OverflowError("CSR indices exceed the PETSc int32 limit")

    crow_np = np.asarray(crow_t.numpy(), dtype=np.int32, order="C", copy=True)
    col_np = np.asarray(col_t.numpy(), dtype=np.int32, order="C", copy=True)
    val_np = np.asarray(val_t.numpy(), dtype=np.float64, order="C", copy=True)

    M = PETSc.Mat().createAIJWithArrays(
        (m, n), (crow_np, col_np, val_np), comm=PETSc.COMM_SELF
    )
    M.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
    M.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)
    M.assemble()
    return M, crow_np, col_np, val_np


class MumpsSolverPool:
    """Cached PETSc/MUMPS solver for fixed or changing CSR structures."""

    NUMERIC_FIXED = NUMERIC_FIXED
    STRUCTURE_FIXED = STRUCTURE_FIXED

    def __init__(
        self,
        rtol=1e-8,
        mumps_pivoting=7,
        mumps_scaled_piv=1,
        reuse_ordering=True,
        reuse_fill=True,
        strict_structure: bool = True,
        debug: bool = False,
    ):
        self._store = {}
        self._rtol = rtol
        self._strict = strict_structure
        self._debug = debug

        PETSc.Options().setValue("mat_mumps_icntl_6", str(mumps_pivoting))
        PETSc.Options().setValue("mat_mumps_icntl_13", str(mumps_scaled_piv))
        if reuse_ordering:
            PETSc.Options().setValue("pc_factor_reuse_ordering", "true")
        if reuse_fill:
            PETSc.Options().setValue("pc_factor_reuse_fill", "true")

    @staticmethod
    def _normalize_reuse_mode(reuse_mode: str) -> str:
        if reuse_mode not in _VALID_REUSE_MODES:
            raise ValueError(
                f"Unknown reuse_mode {reuse_mode!r}; expected {sorted(_VALID_REUSE_MODES)}"
            )
        return reuse_mode

    @staticmethod
    def _normalize_legacy_args(
        matrix_key: Hashable = "default",
        matrix_kind: Optional[Hashable] = None,
    ) -> Hashable:
        if matrix_kind is not None:
            matrix_key = matrix_kind
        return matrix_key

    @staticmethod
    def _ensure_same_structure(A: torch.Tensor, rec: dict, tag: str = ""):
        A = ensure_cpu_csr(A)
        m, n = map(int, A.shape)

        if (m, n) != rec["shape"]:
            raise ValueError(
                f"[{tag}] shape changed: new {(m, n)} vs cached {rec['shape']}"
            )

        crow_new = np.asarray(A.crow_indices().cpu().numpy(), dtype=np.int32, order="C", copy=True)
        col_new = np.asarray(A.col_indices().cpu().numpy(), dtype=np.int32, order="C", copy=True)

        if crow_new.size != rec["crow_np"].size or col_new.size != rec["col_np"].size:
            raise ValueError(
                f"[{tag}] CSR lengths changed: "
                f"new nnz={col_new.size}, cached nnz={rec['col_np'].size}; "
                f"new crow_len={crow_new.size}, cached crow_len={rec['crow_np'].size}"
            )

        if not np.array_equal(crow_new, rec["crow_np"]) or not np.array_equal(col_new, rec["col_np"]):
            diffs = []
            if not np.array_equal(crow_new, rec["crow_np"]):
                idx = int(np.flatnonzero(crow_new != rec["crow_np"])[0])
                diffs.append(
                    f"crow differs at idx {idx}: new={crow_new[idx]} cached={rec['crow_np'][idx]}"
                )
            if not np.array_equal(col_new, rec["col_np"]):
                idx = int(np.flatnonzero(col_new != rec["col_np"])[0])
                diffs.append(
                    f"col differs at idx {idx}: new={col_new[idx]} cached={rec['col_np'][idx]}"
                )
            hint = "; ".join(diffs[:2])
            raise ValueError(f"[{tag}] cached CSR structure changed: {hint}")

    @staticmethod
    def _ensure_same_values(A: torch.Tensor, rec: dict, tag: str = ""):
        A = ensure_cpu_csr(A)
        val_new = np.asarray(A.values().cpu().numpy(), dtype=np.float64, order="C", copy=True)
        if val_new.size != rec["val_np"].size:
            raise ValueError(
                f"[{tag}] numeric_fixed requires unchanged nnz: "
                f"new={val_new.size}, cached={rec['val_np'].size}"
            )
        if not np.array_equal(val_new, rec["values_snapshot"]):
            idx = int(np.flatnonzero(val_new != rec["values_snapshot"])[0])
            raise ValueError(
                f"[{tag}] numeric_fixed values changed at {idx}: "
                f"new={val_new[idx]}, cached={rec['values_snapshot'][idx]}"
            )

    def _build_ksp(self, A_petsc: PETSc.Mat) -> PETSc.KSP:
        ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        ksp.setOperators(A_petsc)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        ksp.setTolerances(rtol=self._rtol, max_it=1)
        ksp.setFromOptions()
        ksp.setUp()
        return ksp

    def _make_cache_key(
        self,
        A: torch.Tensor,
        *,
        matrix_key: Hashable,
        reuse_mode: str,
    ):
        return (matrix_key, reuse_mode)

    def _create_entry(
        self,
        A: torch.Tensor,
        *,
        matrix_key: Hashable,
        reuse_mode: str,
        cache_key,
    ) -> dict:
        A = ensure_cpu_csr(A)
        A_p, crow_np, col_np, val_np = torch_csr_to_petsc_fixed_structure(A)
        ksp = self._build_ksp(A_p)

        entry = {
            "A": A_p,
            "ksp": ksp,
            "crow_np": crow_np,
            "col_np": col_np,
            "val_np": val_np,
            "values_snapshot": val_np.copy(),
            "shape": (int(A.shape[0]), int(A.shape[1])),
            "matrix_key": matrix_key,
            "reuse_mode": reuse_mode,
            "cache_key": cache_key,
            "factorized_once": False,
        }
        self._store[cache_key] = entry
        return entry

    def clear(self, matrix_key: Optional[Hashable] = None):
        if matrix_key is None:
            self._store.clear()
            return
        keys_to_del = [k for k in self._store if len(k) >= 1 and k[0] == matrix_key]
        for k in keys_to_del:
            del self._store[k]

    def factorize(
        self,
        A: torch.Tensor,
        matrix_key: Hashable = "default",
        *,
        reuse_mode: str = STRUCTURE_FIXED,
        matrix_kind: Optional[Hashable] = None,
    ):
        A = ensure_cpu_csr(A)
        matrix_key = self._normalize_legacy_args(matrix_key, matrix_kind)
        reuse_mode = self._normalize_reuse_mode(reuse_mode)

        cache_key = self._make_cache_key(A, matrix_key=matrix_key, reuse_mode=reuse_mode)
        tag = f"{matrix_key}, mode={reuse_mode}"
        rec = self._store.get(cache_key)

        if rec is None:
            rec = self._create_entry(
                A,
                matrix_key=matrix_key,
                reuse_mode=reuse_mode,
                cache_key=cache_key,
            )

            if reuse_mode == NUMERIC_FIXED:
                rec["factorized_once"] = True
            return

        if self._strict or reuse_mode in (NUMERIC_FIXED, STRUCTURE_FIXED):
            self._ensure_same_structure(A, rec, tag=tag)

        val_src = np.asarray(A.values().cpu().numpy(), dtype=np.float64, order="C", copy=True)
        if val_src.size != rec["val_np"].size:
            raise ValueError(
                f"[{tag}] nnz-mismatch: new nnz={val_src.size}, cached nnz={rec['val_np'].size}"
            )

        if reuse_mode == NUMERIC_FIXED:
            self._ensure_same_values(A, rec, tag=tag)
            rec["factorized_once"] = True
            return

        # Update values with a fixed CSR structure.
        rec["A"].zeroEntries()
        rec["A"].setValuesCSR(rec["crow_np"], rec["col_np"], val_src)

        rec["A"].assemblyBegin()
        rec["A"].assemblyEnd()

        rec["ksp"].setOperators(rec["A"])
        rec["ksp"].setUp()
        rec["factorized_once"] = True

    def solve(
        self,
        b: torch.Tensor,
        matrix_key: Hashable = "default",
        *,
        transpose: bool = False,
        reuse_mode: str = STRUCTURE_FIXED,
        matrix_kind: Optional[Hashable] = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(b):
            raise TypeError("b must be a torch tensor")
        if b.is_cuda:
            raise TypeError("b must be on the CPU")

        matrix_key = self._normalize_legacy_args(matrix_key, matrix_kind)
        reuse_mode = self._normalize_reuse_mode(reuse_mode)

        key = (matrix_key, reuse_mode)
        if key not in self._store:
            raise ValueError(
                f"matrix_key={matrix_key!r}, reuse_mode={reuse_mode!r} is not factorized"
            )
        rec = self._store[key]

        m, _ = rec["shape"]
        if b.numel() != m:
            raise ValueError(f"RHS size mismatch: len(b)={b.numel()} vs rows={m}")

        b_np = b.cpu().numpy().astype(np.float64, copy=False)
        b_p = PETSc.Vec().createWithArray(b_np, comm=PETSc.COMM_SELF)
        x_p = b_p.duplicate()

        if transpose:
            rec["ksp"].solveTranspose(b_p, x_p)
        else:
            rec["ksp"].solve(b_p, x_p)

        if rec["ksp"].getConvergedReason() <= 0:
            raise RuntimeError(f"MUMPS solve failed (reason={rec['ksp'].getConvergedReason()})")

        if self._debug:
            r_p = b_p.duplicate()
            if transpose:
                rec["A"].multTranspose(x_p, r_p)
            else:
                rec["A"].mult(x_p, r_p)
            r_p.axpy(-1.0, b_p)
            nr = r_p.norm()
            nb = b_p.norm()
            print(f"[debug] PETSc residual ({'A^T' if transpose else 'A'}) = {nr/(nb+1e-30):.3e}")

        x_np = x_p.getArray(readonly=True).copy()
        return torch.from_numpy(x_np).to(b.device, dtype=b.dtype)
