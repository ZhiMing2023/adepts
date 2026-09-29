from __future__ import annotations
from typing import Literal, Dict, Optional
import torch

GridType = Literal["node", "p", "vx", "vy"]
BCMode = Literal["identity", "neumann0", "skip"]


def build_Kfd_from_mesh_state(
    mesh_state: Dict,
    grid_type: GridType = "p",
    *,
    bc_mode: BCMode = "identity",
    coeff: Optional[torch.Tensor] = None,
):

    key_map = {
        "node": ("xnode", "ynode"),
        "p":    ("xp", "yp"),
        "vx":   ("xvx", "yvx"),
        "vy":   ("xvy", "yvy"),
    }
    if grid_type not in key_map:
        raise ValueError(f"Unsupported grid_type: {grid_type}")

    xkey, ykey = key_map[grid_type]
    if xkey not in mesh_state or ykey not in mesh_state:
        raise KeyError(f"mesh_state is missing {xkey} or {ykey}")

    x = mesh_state[xkey]
    y = mesh_state[ykey]

    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError(f"{xkey}/{ykey} must be torch tensors")

    x = x.detach()
    y = y.detach()
    device = x.device
    dtype = x.dtype

    Nx = int(x.shape[0])
    Ny = int(y.shape[0])

    if Nx < 2 or Ny < 2:
        raise ValueError(f"grid is too small: Nx={Nx}, Ny={Ny}")

    shape2d = (Ny, Nx)
    numel = Nx * Ny

    if coeff is None:
        coeff = torch.ones((Ny, Nx), dtype=dtype, device=device)
    else:
        coeff = coeff.to(device=device, dtype=dtype)
        if tuple(coeff.shape) != (Ny, Nx):
            raise ValueError(
                f"coeff.shape must be {(Ny, Nx)}, got {tuple(coeff.shape)}"
            )

    i = torch.arange(Ny, device=device)
    j = torch.arange(Nx, device=device)
    i_idx, j_idx = torch.meshgrid(i, j, indexing="ij")
    gk = i_idx * Nx + j_idx

    mask = {
    "top":    (i_idx == 0)      & (j_idx > 0) & (j_idx < Nx - 1),
    "bottom": (i_idx == Ny - 1) & (j_idx > 0) & (j_idx < Nx - 1),
    "left":   (j_idx == 0),
    "right":  (j_idx == Nx - 1),
    }
    mask_boundary = mask["top"] | mask["bottom"] | mask["left"] | mask["right"]
    mask_internal = ~mask_boundary

    j_int = j_idx[mask_internal].to(torch.long)
    i_int = i_idx[mask_internal].to(torch.long)
    g_int = gk[mask_internal].to(torch.long)

    dx1 = x[j_int]     - x[j_int - 1]
    dx2 = x[j_int + 1] - x[j_int]
    dx12 = 0.5 * (dx1 + dx2)

    dy1 = y[i_int]     - y[i_int - 1]
    dy2 = y[i_int + 1] - y[i_int]
    dy12 = 0.5 * (dy1 + dy2)

    c_left   = 0.5 * (coeff[i_int, j_int] + coeff[i_int, j_int - 1])
    c_right  = 0.5 * (coeff[i_int, j_int] + coeff[i_int, j_int + 1])
    c_down   = 0.5 * (coeff[i_int, j_int] + coeff[i_int - 1, j_int])
    c_up     = 0.5 * (coeff[i_int, j_int] + coeff[i_int + 1, j_int])

    # Kfd approximates -div(c grad).
    row_left = g_int
    col_left = g_int - 1
    val_left = -c_left / dx1 / dx12

    row_down = g_int
    col_down = g_int - Nx
    val_down = -c_down / dy1 / dy12

    row_self = g_int
    col_self = g_int
    val_self = (
        (c_left / dx1 + c_right / dx2) / dx12
        + (c_down / dy1 + c_up / dy2) / dy12
    )

    row_up = g_int
    col_up = g_int + Nx
    val_up = -c_up / dy2 / dy12

    row_right = g_int
    col_right = g_int + 1
    val_right = -c_right / dx2 / dx12

    rows_list = [row_left, row_down, row_self, row_up, row_right]
    cols_list = [col_left, col_down, col_self, col_up, col_right]
    vals_list = [val_left, val_down, val_self, val_up, val_right]

    if bc_mode != "skip":
        def add_identity_boundary(g: torch.Tensor):
            rows_list.append(g)
            cols_list.append(g)
            vals_list.append(torch.ones_like(g, dtype=dtype))

        def add_neumann_boundary(side: str, g: torch.Tensor):
            if g.numel() == 0:
                return
            if side == "top":
                nbr = g + Nx
            elif side == "bottom":
                nbr = g - Nx
            elif side == "left":
                nbr = g + 1
            elif side == "right":
                nbr = g - 1
            else:
                raise ValueError(f"Unknown side: {side}")

            row = torch.stack([g, g], dim=1).flatten()
            col = torch.stack([g, nbr], dim=1).flatten()
            val = torch.tensor([1.0, -1.0], dtype=dtype, device=device).repeat(g.numel())

            rows_list.append(row)
            cols_list.append(col)
            vals_list.append(val)

        for side in ["top", "bottom", "left", "right"]:
            g = gk[mask[side]].reshape(-1).to(torch.long)
            if bc_mode == "identity":
                add_identity_boundary(g)
            elif bc_mode == "neumann0":
                add_neumann_boundary(side, g)
            else:
                raise ValueError(f"Unsupported bc_mode: {bc_mode}")

    rows = torch.cat(rows_list) if rows_list else torch.empty(0, dtype=torch.long, device=device)
    cols = torch.cat(cols_list) if cols_list else torch.empty(0, dtype=torch.long, device=device)
    vals = torch.cat(vals_list) if vals_list else torch.empty(0, dtype=dtype, device=device)

    return rows, cols, vals, shape2d, numel


def build_Kfd_sparse_csr_from_mesh_state(
    mesh_state: Dict,
    grid_type: GridType = "p",
    *,
    bc_mode: BCMode = "identity",
    coeff: Optional[torch.Tensor] = None,
):

    rows, cols, vals, shape2d, numel = build_Kfd_from_mesh_state(
        mesh_state,
        grid_type=grid_type,
        bc_mode=bc_mode,
        coeff=coeff,
    )

    idx = torch.stack([rows, cols], dim=0)
    Kfd_coo = torch.sparse_coo_tensor(
        idx,
        vals,
        size=(numel, numel),
        device=vals.device,
        dtype=vals.dtype,
    ).coalesce()

    Kfd_csr = Kfd_coo.to_sparse_csr()
    return Kfd_csr, shape2d
