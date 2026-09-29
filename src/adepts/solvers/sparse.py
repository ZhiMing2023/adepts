import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from typing import Optional, Hashable

import torch
import warnings
from dataclasses import dataclass


NUMERIC_FIXED = "numeric_fixed"
STRUCTURE_FIXED = "structure_fixed"
_VALID_REUSE_MODES = {NUMERIC_FIXED, STRUCTURE_FIXED}


def ensure_torch_csr(A: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(A):
        raise TypeError("A must be a torch tensor")
    if A.layout == torch.sparse_csr:
        return A
    return A.to_sparse().coalesce().to_sparse_csr()


def _to_cpu_csr_int64(A: torch.Tensor) -> torch.Tensor:
    A = ensure_torch_csr(A)
    return torch.sparse_csr_tensor(
        A.crow_indices().detach().cpu().to(torch.int64),
        A.col_indices().detach().cpu().to(torch.int64),
        A.values().detach().cpu(),
        size=A.shape,
        device="cpu",
        dtype=A.dtype,
    )


def _normalize_matrix_key_and_mode(
    matrix_key: Optional[Hashable] = None,
    reuse_mode: str = STRUCTURE_FIXED,
    matrix_kind: Optional[Hashable] = None,
):
    if matrix_kind is not None:
        matrix_key = matrix_kind
    if matrix_key is None:
        matrix_key = "default"

    if reuse_mode not in _VALID_REUSE_MODES:
        raise ValueError(
            f"Unknown reuse_mode {reuse_mode!r}; expected {sorted(_VALID_REUSE_MODES)}"
        )

    return matrix_key, reuse_mode


def _effective_cpu_matrix_key(
    A: torch.Tensor,
    matrix_key: Hashable,
    reuse_mode: str,
) -> Hashable:
    return matrix_key, tuple(A.shape)


def scipy_spsolve_cpu(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    transpose: bool = False,
) -> torch.Tensor:
    A = _to_cpu_csr_int64(A)

    assert A.layout == torch.sparse_csr, "A must use CSR layout"
    assert not A.is_cuda and not b.is_cuda, "A and b must be on the CPU"
    assert b.dim() == 1, "b must be one-dimensional"

    if A.dtype != torch.float64 or b.dtype != torch.float64:
        A = torch.sparse_csr_tensor(
            A.crow_indices(),
            A.col_indices(),
            A.values().to(torch.float64),
            size=A.shape,
            device="cpu",
        )
        b = b.to(torch.float64)

    crow = A.crow_indices().cpu().numpy().astype(np.int32, copy=False)
    col = A.col_indices().cpu().numpy().astype(np.int32, copy=False)
    val = A.values().detach().cpu().numpy()

    A_csr = sp.csr_matrix((val, col, crow), shape=A.shape)
    A_csr.sort_indices()

    if transpose:
        A_csr = A_csr.transpose().tocsr()
        A_csr.sort_indices()

    b_np = np.asarray(b.detach().cpu().numpy(), dtype=np.float64, order="C")
    x_np = spsolve(A_csr, b_np)
    x_np = np.asarray(x_np, dtype=np.float64, order="C").reshape(-1)

    return torch.from_numpy(x_np)

def _detect_device(arrays) -> str:
    if not all(torch.is_tensor(array) for array in arrays):
        raise TypeError("Matrix and right-hand side must be torch tensors")
    devs = {"cuda" if array.is_cuda else "cpu" for array in arrays}
    if len(devs) != 1:
        raise ValueError("Matrix and right-hand side must use the same device")
    return devs.pop()


try:
    from .petsc import MumpsSolverPool

    _PETSC_SOLVER_POOL = None
    _PETSC_AVAILABLE = True
    _PETSC_IMPORT_ERROR = None
except Exception as e:
    MumpsSolverPool = None
    _PETSC_SOLVER_POOL = None
    _PETSC_AVAILABLE = False
    _PETSC_IMPORT_ERROR = e



def petsc_solve_cpu_stokes_mumps(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    matrix_key: Hashable = "default",
    reuse_mode: str = STRUCTURE_FIXED,
    transpose: bool = False,
    rtol: float = 1e-8,
    matrix_kind: Optional[Hashable] = None,
) -> torch.Tensor:
    """Solve a CPU CSR system with PETSc/MUMPS."""
    global _PETSC_SOLVER_POOL

    if not _PETSC_AVAILABLE:
        raise RuntimeError(
            f"PETSc/MUMPS is unavailable: {_PETSC_IMPORT_ERROR}"
        )
    if _PETSC_SOLVER_POOL is None:
        _PETSC_SOLVER_POOL = MumpsSolverPool(
            rtol=rtol,
            strict_structure=True,
            debug=False,
        )

    A = _to_cpu_csr_int64(A)
    assert A.layout == torch.sparse_csr and not A.is_cuda
    assert not b.is_cuda

    matrix_key, reuse_mode = _normalize_matrix_key_and_mode(
        matrix_key=matrix_key,
        reuse_mode=reuse_mode,
        matrix_kind=matrix_kind,
    )

    eff_key = _effective_cpu_matrix_key(A, matrix_key, reuse_mode)

    if abs(_PETSC_SOLVER_POOL._rtol - rtol) > 0:
        pass

    _PETSC_SOLVER_POOL.factorize(
        A,
        matrix_key=eff_key,
        reuse_mode=reuse_mode,
    )
    x = _PETSC_SOLVER_POOL.solve(
        b,
        matrix_key=eff_key,
        reuse_mode=reuse_mode,
        transpose=transpose,
    )

    return x

def solve_sparse_auto(
    A: torch.Tensor,
    b: torch.Tensor,
    rtol: float = 1e-8,
    matrix_key: Hashable = "default",
    reuse_mode: str = STRUCTURE_FIXED,
    transpose: bool = False,
    matrix_kind: Optional[Hashable] = None,
) -> torch.Tensor:

    matrix_key, reuse_mode = _normalize_matrix_key_and_mode(
        matrix_key=matrix_key,
        reuse_mode=reuse_mode,
        matrix_kind=matrix_kind,
    )

    device = _detect_device([A, b])

    if device == "cpu":
        A = _to_cpu_csr_int64(A)
        b = b.detach().cpu()

        petsc_err = None
        if _PETSC_AVAILABLE:
            try:
                return petsc_solve_cpu_stokes_mumps(
                    A,
                    b,
                    matrix_key=matrix_key,
                    reuse_mode=reuse_mode,
                    transpose=transpose,
                    rtol=rtol,
                )
            except Exception as e:
                petsc_err = e
                warnings.warn(
                    f"[solve_sparse_auto | {matrix_key}] PETSc failed; using SciPy: {e!r}",
                    RuntimeWarning,
                )

        try:
            return scipy_spsolve_cpu(A, b, transpose=transpose)
        except Exception as scipy_err:
            if petsc_err is not None:
                raise RuntimeError(
                    f"PETSc and SciPy failed.\nPETSc: {petsc_err!r}\nSciPy: {scipy_err!r}"
                ) from scipy_err
            raise

    if device == "cuda":
        raise RuntimeError("CUDA is not currently supported.")

    raise RuntimeError("Unrecognized device type.")

class CSRSparseSolveAutoFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        b: torch.Tensor,
        matrix_key: Hashable = "default",
        reuse_mode: str = STRUCTURE_FIXED,
    ):
        if torch.isnan(A.values()).any() or torch.isinf(A.values()).any():
            raise ValueError(f"matrix_key={matrix_key}: Matrix A contains NaN or Inf values.")

        if torch.isnan(b).any() or torch.isinf(b).any():
            raise ValueError(f"matrix_key={matrix_key}: Vector b contains NaN or Inf values.")

        assert A.layout == torch.sparse_csr, "A must be CSR"
        device = b.device

        x = solve_sparse_auto(
            A,
            b,
            matrix_key=matrix_key,
            reuse_mode=reuse_mode,
            transpose=False,
        )

        if not torch.is_tensor(x):
            x = torch.from_numpy(x).to(device)

        ctx.save_for_backward(x, b, A)
        ctx.matrix_key = matrix_key
        ctx.reuse_mode = reuse_mode

        tol = 1e-7
        res = A @ x - b
        rel_res = (res.norm() / (b.norm() + 1e-30)).item()
        if not rel_res < tol:
            print(f"[{matrix_key}] forward residual: {rel_res}")
            warnings.warn(
                f"Forward solve residual {rel_res:.3e} exceeds tolerance {tol:.1e}. "
                "Possible inconsistency in solver accuracy.",
                RuntimeWarning
            )

        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, b, A = ctx.saved_tensors
        device = grad_output.device
        matrix_key = ctx.matrix_key
        reuse_mode = ctx.reuse_mode

        build_graph = torch.is_grad_enabled()

        A_for_adj = A
        x_for_grad = x
        rhs_for_adj = grad_output

        lambda_vec = CSRSparseSolveTwiceBackAutoFunction.apply(
            A_for_adj,
            rhs_for_adj,
            matrix_key,
            reuse_mode,
        )

        grad_b = lambda_vec

        At = A.detach().transpose(0, 1)
        res = At @ lambda_vec.detach() - grad_output.detach()

        rnorm = res.norm()
        gnorm = grad_output.detach().norm()
        lnorm = lambda_vec.detach().norm()

        rel_res = (rnorm / (gnorm + 1e-30)).item()

        rtol = 1e-7
        atol = 1e-14
        threshold = atol + rtol * gnorm
        ok = rnorm <= threshold

        if not ok:
            print(f"\n[{matrix_key}] Adjoint residual check failed")
            print(f"   ||g||       = {gnorm.item():.6e}")
            print(f"   ||lambda||  = {lnorm.item():.6e}")
            print(f"   ||r||       = {rnorm.item():.6e}")
            print(f"   rel_res     = {rel_res:.6e}")
            print(f"   threshold   = {threshold.item():.6e}")
            print(f"   r/atol      = {(rnorm / atol).item():.3e}")
            print(f"   r/(rtol*g)  = {(rnorm / (rtol * gnorm + 1e-30)).item():.3e}")

            warnings.warn(
                f"Adjoint residual too large: "
                f"||r||={rnorm:.3e}, ||g||={gnorm:.3e}, rel={rel_res:.3e}",
                RuntimeWarning,
            )

        crow = A.crow_indices()
        col = A.col_indices()

        row_counts = (crow[1:] - crow[:-1]).to(torch.long)

        row_idx = torch.repeat_interleave(
            torch.arange(A.shape[0], device=device, dtype=torch.long),
            row_counts,
        )

        col_idx = col.to(torch.long)

        if lambda_vec.ndim == 1:
            grad_vals = -lambda_vec[row_idx] * x_for_grad[col_idx]

        elif lambda_vec.ndim == 2:
            grad_vals = -(lambda_vec[row_idx, :] * x_for_grad[col_idx, :]).sum(dim=1)

        else:
            raise RuntimeError(
                f"[CSRSparseSolveAutoFunction.backward | {matrix_key}] "
                f"lambda_vec.ndim={lambda_vec.ndim} not supported"
            )

        grad_A = torch.sparse_csr_tensor(
            crow,
            col,
            grad_vals,
            size=A.shape,
            device=device,
            dtype=A.dtype,
        )

        return grad_A, grad_b, None, None


class CSRSparseSolveTwiceBackAutoFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        b: torch.Tensor,
        matrix_key: Hashable = "default",
        reuse_mode: str = STRUCTURE_FIXED,
    ):
        if torch.isnan(A.values()).any() or torch.isinf(A.values()).any():
            raise ValueError(f"matrix_key={matrix_key}: Matrix A contains NaN or Inf values.")

        if torch.isnan(b).any() or torch.isinf(b).any():
            raise ValueError(f"matrix_key={matrix_key}: Vector b contains NaN or Inf values.")

        assert A.layout == torch.sparse_csr, "A must use CSR layout"
        device = b.device

        x = solve_sparse_auto(
            A,
            b,
            matrix_key=matrix_key,
            reuse_mode=reuse_mode,
            transpose=True,
        )

        if not torch.is_tensor(x):
            x = torch.from_numpy(x).to(device)

        ctx.save_for_backward(x, b, A)
        ctx.matrix_key = matrix_key
        ctx.reuse_mode = reuse_mode

        tol = 1e-7
        res = A.transpose(0, 1) @ x - b
        rel_res = (res.norm() / (b.norm() + 1e-30)).item()
        if not rel_res < tol:
            print(f"[{matrix_key}] forward residual: {rel_res}")
            warnings.warn(
                f"Forward solve residual {rel_res:.3e} exceeds tolerance {tol:.1e}. "
                "Possible inconsistency in solver accuracy.",
                RuntimeWarning
            )

        return x

    @staticmethod
    def backward(ctx, grad_output):
        with torch.autograd.set_detect_anomaly(False):
            x, b, A = ctx.saved_tensors
            device = grad_output.device
            matrix_key = ctx.matrix_key
            reuse_mode = ctx.reuse_mode

            grad_output_det = grad_output.detach()

            numuda_2 = solve_sparse_auto(
                A,
                grad_output_det,
                matrix_key=matrix_key,
                reuse_mode=reuse_mode,
                transpose=False,
            )

            grad_b = numuda_2 if torch.is_tensor(numuda_2) else \
                torch.from_numpy(numuda_2).to(device)

            res = A @ numuda_2 - grad_output_det

            rnorm = res.norm()
            gnorm = grad_output_det.norm()
            xnorm = grad_b.norm()

            rel_res = (rnorm / (gnorm + 1e-30)).item()

            rtol = 1e-7
            atol = 1e-14
            threshold = atol + rtol * gnorm
            ok = (rnorm <= threshold)

            if not ok:
                print(f"\n[twice back {matrix_key}] Adjoint residual check failed")
                print(f"   ||g||     = {gnorm.item():.6e}")
                print(f"   ||x||     = {xnorm.item():.6e}")
                print(f"   ||r||     = {rnorm.item():.6e}")
                print(f"   rel_res   = {rel_res:.6e}")
                print(f"   threshold = {threshold.item():.6e}")
                print(f"   r/atol    = {(rnorm / atol).item():.3e}")
                print(f"   r/(rtol*g)= {(rnorm / (rtol * gnorm + 1e-30)).item():.3e}")

                warnings.warn(
                    f"Adjoint residual too large: "
                    f"||r||={rnorm:.3e}, ||g||={gnorm:.3e}, rel={rel_res:.3e}",
                    RuntimeWarning
                )

            crow = A.crow_indices()
            col = A.col_indices()

            row_counts = (crow[1:] - crow[:-1]).to(torch.long)

            row_idx = torch.repeat_interleave(
                torch.arange(A.shape[0], device=device, dtype=torch.long),
                row_counts
            )

            col_idx = col.to(torch.long)

            grad_vals = -x[row_idx] * grad_b[col_idx]

            grad_A = torch.sparse_csr_tensor(
                crow,
                col,
                grad_vals,
                size=A.shape,
                device=device,
                dtype=A.dtype,
                )

            return grad_A, grad_b, None, None


def sparse_solve(
    A: torch.Tensor,
    b: torch.Tensor,
    matrix_kind: Hashable = "A1",
    reuse_mode: str = STRUCTURE_FIXED,
):

    return CSRSparseSolveAutoFunction.apply(A, b, matrix_kind, reuse_mode)




@dataclass
class CSRPattern:
    crow: torch.Tensor
    col: torch.Tensor
    inverse: torch.Tensor
    row_of_val: torch.Tensor
    shape: tuple
    nnz: int


def make_csr_pattern_from_coo(row: torch.Tensor, col: torch.Tensor, N: int) -> CSRPattern:
    with torch.no_grad():
        row = row.detach().to(torch.long)
        col = col.detach().to(torch.long)

        key = row * N + col

        unique_key, inverse = torch.unique(
            key,
            sorted=True,
            return_inverse=True,
        )

        unique_row = torch.div(unique_key, N, rounding_mode="floor")
        unique_col = unique_key % N

        counts = torch.bincount(unique_row, minlength=N)

        crow = torch.empty(N + 1, dtype=torch.long, device=row.device)
        crow[0] = 0
        crow[1:] = torch.cumsum(counts, dim=0)

        row_of_val = torch.repeat_interleave(
            torch.arange(N, device=row.device, dtype=torch.long),
            counts.to(torch.long),
        )

        return CSRPattern(
            crow=crow,
            col=unique_col.to(torch.long),
            inverse=inverse.to(torch.long),
            row_of_val=row_of_val.to(torch.long),
            shape=(N, N),
            nnz=int(unique_key.numel()),
        )


def coalesce_values_to_csr(val: torch.Tensor, pattern: CSRPattern) -> torch.Tensor:
    inverse = pattern.inverse.to(device=val.device)

    val_csr = val.new_zeros(pattern.nnz)
    val_csr = val_csr.index_add(0, inverse, val)

    return val_csr


class CSRSparseSolveValuesFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        val_csr: torch.Tensor,
        b: torch.Tensor,
        crow: torch.Tensor,
        col: torch.Tensor,
        row_of_val: torch.Tensor,
        shape,
        matrix_key: Hashable = "default",
        reuse_mode: str = "structure_fixed",
        transpose: bool = False,
    ):
        device = val_csr.device
        dtype = val_csr.dtype

        crow_local = crow.to(device=device, dtype=torch.long)
        col_local = col.to(device=device, dtype=torch.long)

        A = torch.sparse_csr_tensor(
            crow_local,
            col_local,
            val_csr.clone().detach(),
            size=shape,
            device=device,
            dtype=dtype,
        )

        x = solve_sparse_auto(
            A,
            b.detach(),
            matrix_key=matrix_key,
            reuse_mode=reuse_mode,
            transpose=transpose,
        )

        ctx.save_for_backward(
            val_csr,
            b,
            x,
            crow_local,
            col_local,
            row_of_val.to(device=device, dtype=torch.long),
        )
        ctx.shape = shape
        ctx.matrix_key = matrix_key
        ctx.reuse_mode = reuse_mode
        ctx.transpose = transpose

        return x

    @staticmethod
    def backward(ctx, grad_x):
        val_csr, b, x_saved, crow, col, row_of_val = ctx.saved_tensors

        shape = ctx.shape
        matrix_key = ctx.matrix_key
        reuse_mode = ctx.reuse_mode
        transpose = ctx.transpose

        build_graph = torch.is_grad_enabled()

        if build_graph:
            x_for_grad = CSRSparseSolveValuesFunction.apply(
                val_csr,
                b,
                crow,
                col,
                row_of_val,
                shape,
                f"{matrix_key}__primal_replay_{'T' if transpose else 'N'}",
                reuse_mode,
                transpose,
            )
            rhs_adj = grad_x
            val_for_adj = val_csr
        else:
            x_for_grad = x_saved.detach()
            rhs_adj = grad_x.detach()
            val_for_adj = val_csr.detach()

        if not transpose:
            lam = CSRSparseSolveValuesFunction.apply(
                val_for_adj,
                rhs_adj,
                crow,
                col,
                row_of_val,
                shape,
                f"{matrix_key}__adj_T",
                reuse_mode,
                True,
            )

            grad_b = lam

            r = row_of_val
            c = col

            if lam.ndim == 1:
                grad_val = -lam[r] * x_for_grad[c]
            elif lam.ndim == 2:
                grad_val = -(lam[r, :] * x_for_grad[c, :]).sum(dim=1)
            else:
                raise RuntimeError(f"Unsupported lam.ndim={lam.ndim}")

        else:
            mu = CSRSparseSolveValuesFunction.apply(
                val_for_adj,
                rhs_adj,
                crow,
                col,
                row_of_val,
                shape,
                f"{matrix_key}__adj_N",
                reuse_mode,
                False,
            )

            grad_b = mu

            r = row_of_val
            c = col

            if mu.ndim == 1:
                grad_val = -x_for_grad[r] * mu[c]
            elif mu.ndim == 2:
                grad_val = -(x_for_grad[r, :] * mu[c, :]).sum(dim=1)
            else:
                raise RuntimeError(f"Unsupported mu.ndim={mu.ndim}")

        return (
            grad_val,
            grad_b,
            None,  # crow
            None,  # col
            None,  # row_of_val
            None,  # shape
            None,  # matrix_key
            None,  # reuse_mode
            None,  # transpose
        )


def csr_solve_from_values(
    crow: torch.Tensor,
    col: torch.Tensor,
    row_of_val: torch.Tensor,
    val_csr: torch.Tensor,
    b: torch.Tensor,
    shape,
    matrix_key: Hashable = "default",
    reuse_mode: str = "structure_fixed",
    transpose: bool = False,
):
    return CSRSparseSolveValuesFunction.apply(
        val_csr,
        b,
        crow,
        col,
        row_of_val,
        shape,
        matrix_key,
        reuse_mode,
        transpose,
    )
