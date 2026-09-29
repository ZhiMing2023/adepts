import torch
from typing import Mapping, Any


def _to_bc_ints(boundary_constant):
    if isinstance(boundary_constant, (tuple, list)):
        bc = boundary_constant
    elif isinstance(boundary_constant, torch.Tensor):
        bc = boundary_constant.detach().cpu().to(torch.int64).tolist()
    else:
        raise TypeError(f"Unsupported boundary_constant type: {type(boundary_constant)}")

    return int(bc[0]), int(bc[1]), int(bc[2]), int(bc[3])

def _empty_long(device):
    return torch.empty((0,), dtype=torch.long, device=device)


def _empty_val(device, dtype):
    return torch.empty((0,), dtype=dtype, device=device)


def stokes_assembly(
    viscosity: torch.Tensor,
    viscosity_p: torch.Tensor,
    density_x: torch.Tensor,
    density_y: torch.Tensor,
    dt: torch.Tensor,
    pscale: torch.Tensor,
    mesh_state: Mapping[str, Any],
    gx,
    gy,
    boundary_constant,
    *,
    is_stick_air: bool = False,
    p_fix_value: torch.Tensor = None,
):
    """Assemble the Stokes COO entries and right-hand side."""

    if p_fix_value is None:
        p_fix_value = torch.tensor(
            0.0,
            dtype=pscale.dtype,
            device=pscale.device,
        )

    rows, cols, vals, R = stokes_assembly_forward(
        viscosity=viscosity,
        viscosity_p=viscosity_p,
        density_x=density_x,
        density_y=density_y,
        dt=dt,
        pscale=pscale,
        p_fix_value=p_fix_value,
        gx=gx,
        gy=gy,
        boundary_constant=boundary_constant,
        mesh_state=mesh_state,
        stick_air=is_stick_air,
    )

    return rows, cols, vals, R


def stokes_assembly_forward(
    viscosity,
    viscosity_p,
    density_x,
    density_y,
    dt,
    pscale,
    p_fix_value,
    gx,
    gy,
    boundary_constant,
    mesh_state,
    stick_air,
):
    """Assemble the vectorized Stokes system."""

    def _check_min_shape(name, a, min_ny, min_nx):
        if a.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape={tuple(a.shape)}.")
        if a.shape[0] < min_ny or a.shape[1] < min_nx:
            raise ValueError(
                f"{name} has shape={tuple(a.shape)}, but this assembly requires "
                f"at least ({min_ny}, {min_nx}) for Ny1={Ny1}, Nx1={Nx1}."
            )



    device = viscosity.device
    dtype = viscosity.dtype

    Nx1 = int(mesh_state["Nx1"])
    Ny1 = int(mesh_state["Ny1"])

    _check_min_shape("viscosity",   viscosity,   Ny1 - 1, Nx1 - 1)
    _check_min_shape("viscosity_p", viscosity_p, Ny1 - 1, Nx1 - 1)
    _check_min_shape("density_x",   density_x,   Ny1,     Nx1 - 1)
    _check_min_shape("density_y",   density_y,   Ny1 - 1, Nx1)

    xvx = mesh_state["xvx"].to(device=device, dtype=dtype)
    yvx = mesh_state["yvx"].to(device=device, dtype=dtype)
    xvy = mesh_state["xvy"].to(device=device, dtype=dtype)
    yvy = mesh_state["yvy"].to(device=device, dtype=dtype)

    bcleft, bcright, bctop, bcbottom = _to_bc_ints(boundary_constant)

    if p_fix_value is None:
        p_fix_value = torch.tensor(0.0, device=device, dtype=dtype)
    elif not torch.is_tensor(p_fix_value):
        p_fix_value = torch.tensor(p_fix_value, device=device, dtype=dtype)
    else:
        p_fix_value = p_fix_value.to(device=device, dtype=dtype)

    if not torch.is_tensor(pscale):
        pscale = torch.tensor(pscale, device=device, dtype=dtype)
    else:
        pscale = pscale.to(device=device, dtype=dtype)

    if not torch.is_tensor(dt):
        dt = torch.tensor(dt, device=device, dtype=dtype)
    else:
        dt = dt.to(device=device, dtype=dtype)

    dt_eff = dt if stick_air else dt * 0.0

    if not torch.is_tensor(gx):
        gx = torch.tensor(gx, device=device, dtype=dtype)
    else:
        gx = gx.to(device=device, dtype=dtype)

    if not torch.is_tensor(gy):
        gy = torch.tensor(gy, device=device, dtype=dtype)
    else:
        gy = gy.to(device=device, dtype=dtype)

    j_range = torch.arange(Nx1, device=device)
    i_range = torch.arange(Ny1, device=device)
    j_idx, i_idx = torch.meshgrid(j_range, i_range, indexing="ij")

    base = (j_idx * Ny1 + i_idx) * 3
    kvx = base
    kvy = base + 1
    kpm = base + 2

    total_size = Nx1 * Ny1 * 3
    R = torch.zeros(total_size, dtype=dtype, device=device)

    empty_l = _empty_long(device)
    empty_v = _empty_val(device, dtype)


    mask_vx_boundary = (
        (i_idx == 0)
        | (i_idx == (Ny1 - 1))
        | (j_idx == 0)
        | (j_idx == (Nx1 - 2))
        | (j_idx == (Nx1 - 1))
    )

    kvx_boundary = kvx[mask_vx_boundary].reshape(-1).to(torch.long)

    row_vx_boundary = kvx_boundary
    col_vx_boundary = kvx_boundary
    val_vx_boundary = torch.ones(kvx_boundary.numel(), dtype=dtype, device=device)

    row_vx_top = empty_l
    col_vx_top = empty_l
    val_vx_top = empty_v

    row_vx_bottom = empty_l
    col_vx_bottom = empty_l
    val_vx_bottom = empty_v

    mask_vx_top = (
        mask_vx_boundary
        & (i_idx == 0)
        & (j_idx > 0)
        & (j_idx < (Nx1 - 2))
    )
    kvx_top = kvx[mask_vx_top].reshape(-1).to(torch.long)

    if bctop == -1:
        row_vx_top = kvx_top
        col_vx_top = kvx_top + 3
        val_vx_top = torch.full(
            (kvx_top.numel(),),
            float(bctop),
            dtype=dtype,
            device=device,
        )

    mask_vx_bottom = (
        mask_vx_boundary
        & (i_idx == (Ny1 - 1))
        & (j_idx > 0)
        & (j_idx < (Nx1 - 2))
    )
    kvx_bottom = kvx[mask_vx_bottom].reshape(-1).to(torch.long)

    if bcbottom == -1:
        row_vx_bottom = kvx_bottom
        col_vx_bottom = kvx_bottom - 3
        val_vx_bottom = torch.full(
            (kvx_bottom.numel(),),
            float(bcbottom),
            dtype=dtype,
            device=device,
        )

    mask_vx_internal = ~mask_vx_boundary

    j_int = j_idx[mask_vx_internal].to(torch.long)
    i_int = i_idx[mask_vx_internal].to(torch.long)

    kvx_int = kvx[mask_vx_internal].to(torch.long)
    kvy_int = kvy[mask_vx_internal].to(torch.long)
    kpm_int = kpm[mask_vx_internal].to(torch.long)

    dx1 = xvx[j_int] - xvx[j_int - 1]
    dx2 = xvx[j_int + 1] - xvx[j_int]
    dx12 = 0.5 * (dx1 + dx2)

    dy1 = yvx[i_int] - yvx[i_int - 1]
    dy2 = yvx[i_int + 1] - yvx[i_int]
    dy12 = 0.5 * (dy1 + dy2)

    ETA1 = viscosity[i_int - 1, j_int]
    ETA2 = viscosity[i_int, j_int]

    ETAP1 = viscosity_p[i_int, j_int]
    ETAP2 = viscosity_p[i_int, j_int + 1]

    dRHOdx = (
        density_x[i_int, j_int + 1]
        - density_x[i_int, j_int - 1]
    ) / (dx1 + dx2)

    dRHOdy = (
        density_x[i_int + 1, j_int]
        - density_x[i_int - 1, j_int]
    ) / (dy1 + dy2)

    row_vx_int_1 = kvx_int
    col_vx_int_1 = kvx_int - Ny1 * 3
    val_vx_int_1 = 2.0 * ETAP1 / dx1 / dx12

    row_vx_int_2 = kvx_int
    col_vx_int_2 = kvx_int - 3
    val_vx_int_2 = ETA1 / dy1 / dy12

    row_vx_int_3 = kvx_int
    col_vx_int_3 = kvx_int
    val_vx_int_3 = (
        -2.0 * (ETAP1 / dx1 + ETAP2 / dx2) / dx12
        - (ETA1 / dy1 + ETA2 / dy2) / dy12
        - dRHOdx * gx * dt_eff
    )

    row_vx_int_4 = kvx_int
    col_vx_int_4 = kvx_int + 3
    val_vx_int_4 = ETA2 / dy2 / dy12

    row_vx_int_5 = kvx_int
    col_vx_int_5 = kvx_int + Ny1 * 3
    val_vx_int_5 = 2.0 * ETAP2 / dx2 / dx12

    row_vx_int_6 = kvx_int
    col_vx_int_6 = kvy_int - 3
    val_vx_int_6 = ETA1 / dx12 / dy12 - dRHOdy * gx * dt_eff / 4.0

    row_vx_int_7 = kvx_int
    col_vx_int_7 = kvy_int
    val_vx_int_7 = -ETA2 / dx12 / dy12 - dRHOdy * gx * dt_eff / 4.0

    row_vx_int_8 = kvx_int
    col_vx_int_8 = kvy_int + Ny1 * 3 - 3
    val_vx_int_8 = -ETA1 / dx12 / dy12 - dRHOdy * gx * dt_eff / 4.0

    row_vx_int_9 = kvx_int
    col_vx_int_9 = kvy_int + Ny1 * 3
    val_vx_int_9 = ETA2 / dx12 / dy12 - dRHOdy * gx * dt_eff / 4.0

    row_vx_int_10 = kvx_int
    col_vx_int_10 = kpm_int
    val_vx_int_10 = pscale / dx12

    row_vx_int_11 = kvx_int
    col_vx_int_11 = kpm_int + Ny1 * 3
    val_vx_int_11 = -pscale / dx12

    R = R.index_put(
        (kvx_int,),
        -density_x[i_int, j_int] * gx,
        accumulate=False,
    )

    val_vx_blocks = [
        val_vx_boundary,
        val_vx_top,
        val_vx_bottom,
        val_vx_int_1,
        val_vx_int_2,
        val_vx_int_3,
        val_vx_int_4,
        val_vx_int_5,
        val_vx_int_6,
        val_vx_int_7,
        val_vx_int_8,
        val_vx_int_9,
        val_vx_int_10,
        val_vx_int_11,
    ]

    row_vx = torch.cat([
        row_vx_boundary,
        row_vx_top,
        row_vx_bottom,
        row_vx_int_1,
        row_vx_int_2,
        row_vx_int_3,
        row_vx_int_4,
        row_vx_int_5,
        row_vx_int_6,
        row_vx_int_7,
        row_vx_int_8,
        row_vx_int_9,
        row_vx_int_10,
        row_vx_int_11,
    ])

    col_vx = torch.cat([
        col_vx_boundary,
        col_vx_top,
        col_vx_bottom,
        col_vx_int_1,
        col_vx_int_2,
        col_vx_int_3,
        col_vx_int_4,
        col_vx_int_5,
        col_vx_int_6,
        col_vx_int_7,
        col_vx_int_8,
        col_vx_int_9,
        col_vx_int_10,
        col_vx_int_11,
    ])

    val_vx = torch.cat(val_vx_blocks)


    mask_vy_boundary = (
        (j_idx == 0)
        | (j_idx == (Nx1 - 1))
        | (i_idx == 0)
        | (i_idx == (Ny1 - 1))
        | (i_idx == (Ny1 - 2))
    )

    kvy_boundary = kvy[mask_vy_boundary].reshape(-1).to(torch.long)

    row_vy_boundary = kvy_boundary
    col_vy_boundary = kvy_boundary
    val_vy_boundary = torch.ones(kvy_boundary.numel(), dtype=dtype, device=device)

    row_vy_left = empty_l
    col_vy_left = empty_l
    val_vy_left = empty_v

    row_vy_right = empty_l
    col_vy_right = empty_l
    val_vy_right = empty_v

    mask_vy_left = (
        mask_vy_boundary
        & (j_idx == 0)
        & (i_idx > 0)
        & (i_idx < (Ny1 - 2))
    )
    kvy_left = kvy[mask_vy_left].reshape(-1).to(torch.long)

    if bcleft == -1:
        row_vy_left = kvy_left
        col_vy_left = kvy_left + 3 * Ny1
        val_vy_left = torch.full(
            (kvy_left.numel(),),
            float(bcleft),
            dtype=dtype,
            device=device,
        )

    mask_vy_right = (
        mask_vy_boundary
        & (j_idx == (Nx1 - 1))
        & (i_idx > 0)
        & (i_idx < (Ny1 - 2))
    )
    kvy_right = kvy[mask_vy_right].reshape(-1).to(torch.long)

    if bcright == -1:
        row_vy_right = kvy_right
        col_vy_right = kvy_right - 3 * Ny1
        val_vy_right = torch.full(
            (kvy_right.numel(),),
            float(bcright),
            dtype=dtype,
            device=device,
        )

    mask_vy_internal = ~mask_vy_boundary

    j_int_vy = j_idx[mask_vy_internal].to(torch.long)
    i_int_vy = i_idx[mask_vy_internal].to(torch.long)

    kvy_int_vy = kvy[mask_vy_internal].to(torch.long)
    kvx_int_vy = kvx[mask_vy_internal].to(torch.long)
    kpm_int_vy = kpm[mask_vy_internal].to(torch.long)

    dxvy1 = xvy[j_int_vy] - xvy[j_int_vy - 1]
    dxvy2 = xvy[j_int_vy + 1] - xvy[j_int_vy]
    dxvy = 0.5 * (dxvy1 + dxvy2)

    dyvy1 = yvy[i_int_vy] - yvy[i_int_vy - 1]
    dyvy2 = yvy[i_int_vy + 1] - yvy[i_int_vy]
    dyvy = 0.5 * (dyvy1 + dyvy2)

    ETA1 = viscosity[i_int_vy, j_int_vy - 1]
    ETA2 = viscosity[i_int_vy, j_int_vy]

    ETAP1 = viscosity_p[i_int_vy, j_int_vy]
    ETAP2 = viscosity_p[i_int_vy + 1, j_int_vy]

    dRHOdx = (
        density_y[i_int_vy, j_int_vy + 1]
        - density_y[i_int_vy, j_int_vy - 1]
    ) / (dxvy1 + dxvy2)

    dRHOdy = (
        density_y[i_int_vy + 1, j_int_vy]
        - density_y[i_int_vy - 1, j_int_vy]
    ) / (dyvy1 + dyvy2)

    row_vy_int_1 = kvy_int_vy
    col_vy_int_1 = kvy_int_vy - Ny1 * 3
    val_vy_int_1 = ETA1 / dxvy1 / dxvy

    row_vy_int_2 = kvy_int_vy
    col_vy_int_2 = kvy_int_vy - 3
    val_vy_int_2 = 2.0 * ETAP1 / dyvy1 / dyvy

    row_vy_int_3 = kvy_int_vy
    col_vy_int_3 = kvy_int_vy
    val_vy_int_3 = (
        -2.0 * (ETAP1 / dyvy1 + ETAP2 / dyvy2) / dyvy
        - (ETA1 / dxvy1 + ETA2 / dxvy2) / dxvy
        - dRHOdy * gy * dt_eff
    )

    row_vy_int_4 = kvy_int_vy
    col_vy_int_4 = kvy_int_vy + 3
    val_vy_int_4 = 2.0 * ETAP2 / dyvy2 / dyvy

    row_vy_int_5 = kvy_int_vy
    col_vy_int_5 = kvy_int_vy + Ny1 * 3
    val_vy_int_5 = ETA2 / dxvy2 / dxvy

    row_vy_int_6 = kvy_int_vy
    col_vy_int_6 = kvx_int_vy - Ny1 * 3
    val_vy_int_6 = ETA1 / dxvy / dyvy - dRHOdx * gy * dt_eff / 4.0

    row_vy_int_7 = kvy_int_vy
    col_vy_int_7 = kvx_int_vy + 3 - Ny1 * 3
    val_vy_int_7 = -ETA1 / dxvy / dyvy - dRHOdx * gy * dt_eff / 4.0

    row_vy_int_8 = kvy_int_vy
    col_vy_int_8 = kvx_int_vy
    val_vy_int_8 = -ETA2 / dxvy / dyvy - dRHOdx * gy * dt_eff / 4.0

    row_vy_int_9 = kvy_int_vy
    col_vy_int_9 = kvx_int_vy + 3
    val_vy_int_9 = ETA2 / dxvy / dyvy - dRHOdx * gy * dt_eff / 4.0

    row_vy_int_10 = kvy_int_vy
    col_vy_int_10 = kpm_int_vy
    val_vy_int_10 = pscale / dyvy

    row_vy_int_11 = kvy_int_vy
    col_vy_int_11 = kpm_int_vy + 3
    val_vy_int_11 = -pscale / dyvy

    R = R.index_put(
        (kvy_int_vy,),
        -density_y[i_int_vy, j_int_vy] * gy,
        accumulate=False,
    )

    val_vy_blocks = [
        val_vy_boundary,
        val_vy_left,
        val_vy_right,
        val_vy_int_1,
        val_vy_int_2,
        val_vy_int_3,
        val_vy_int_4,
        val_vy_int_5,
        val_vy_int_6,
        val_vy_int_7,
        val_vy_int_8,
        val_vy_int_9,
        val_vy_int_10,
        val_vy_int_11,
    ]

    row_vy = torch.cat([
        row_vy_boundary,
        row_vy_left,
        row_vy_right,
        row_vy_int_1,
        row_vy_int_2,
        row_vy_int_3,
        row_vy_int_4,
        row_vy_int_5,
        row_vy_int_6,
        row_vy_int_7,
        row_vy_int_8,
        row_vy_int_9,
        row_vy_int_10,
        row_vy_int_11,
    ])

    col_vy = torch.cat([
        col_vy_boundary,
        col_vy_left,
        col_vy_right,
        col_vy_int_1,
        col_vy_int_2,
        col_vy_int_3,
        col_vy_int_4,
        col_vy_int_5,
        col_vy_int_6,
        col_vy_int_7,
        col_vy_int_8,
        col_vy_int_9,
        col_vy_int_10,
        col_vy_int_11,
    ])

    val_vy = torch.cat(val_vy_blocks)


    mask_p_boundary = (
        (i_idx == 0)
        | (j_idx == 0)
        | (i_idx == (Ny1 - 1))
        | (j_idx == (Nx1 - 1))
        | ((i_idx == 1) & (j_idx == 1))
    )

    mask_p_special = mask_p_boundary & ((i_idx == 1) & (j_idx == 1))
    kpm_special = kpm[mask_p_special].reshape(-1).to(torch.long)

    row_p_special = kpm_special
    col_p_special = kpm_special
    val_p_special = torch.full(
        (kpm_special.numel(),),
        1.0,
        dtype=dtype,
        device=device,
    ) * pscale

    if kpm_special.numel() > 0:
        R = R.index_put(
            (kpm_special,),
            p_fix_value.reshape(()).expand(kpm_special.numel()),
            accumulate=False,
        )

    mask_p_other = mask_p_boundary & (~mask_p_special)
    kpm_other = kpm[mask_p_other].reshape(-1).to(torch.long)

    row_p_other = kpm_other
    col_p_other = kpm_other
    val_p_other = torch.ones(kpm_other.numel(), dtype=dtype, device=device)

    mask_p_internal = ~mask_p_boundary

    i_int_p = i_idx[mask_p_internal].to(torch.long)
    j_int_p = j_idx[mask_p_internal].to(torch.long)

    kpm_int_p = kpm[mask_p_internal].to(torch.long)
    kvx_int_p = kvx[mask_p_internal].to(torch.long)
    kvy_int_p = kvy[mask_p_internal].to(torch.long)

    dxp = xvx[j_int_p] - xvx[j_int_p - 1]
    dyp = yvy[i_int_p] - yvy[i_int_p - 1]

    row_p_int_1 = kpm_int_p
    col_p_int_1 = kvx_int_p - Ny1 * 3
    val_p_int_1 = -1.0 / dxp

    row_p_int_2 = kpm_int_p
    col_p_int_2 = kvx_int_p
    val_p_int_2 = 1.0 / dxp

    row_p_int_3 = kpm_int_p
    col_p_int_3 = kvy_int_p - 3
    val_p_int_3 = -1.0 / dyp

    row_p_int_4 = kpm_int_p
    col_p_int_4 = kvy_int_p
    val_p_int_4 = 1.0 / dyp


    row_p_int_5 = kpm_int_p
    col_p_int_5 = kpm_int_p
    val_p_int_5 = torch.full_like(val_p_int_4, 0)

    val_p_blocks = [
        val_p_special,
        val_p_other,
        val_p_int_1,
        val_p_int_2,
        val_p_int_3,
        val_p_int_4,
        val_p_int_5,
    ]

    row_p = torch.cat([
        row_p_special,
        row_p_other,
        row_p_int_1,
        row_p_int_2,
        row_p_int_3,
        row_p_int_4,
        row_p_int_5,
    ])

    col_p = torch.cat([
        col_p_special,
        col_p_other,
        col_p_int_1,
        col_p_int_2,
        col_p_int_3,
        col_p_int_4,
        col_p_int_5,
    ])

    val_p = torch.cat(val_p_blocks)


    stokes_row_indices = torch.cat([row_vx, row_vy, row_p]).to(torch.long)
    stokes_col_indices = torch.cat([col_vx, col_vy, col_p]).to(torch.long)
    stokes_values = torch.cat([val_vx, val_vy, val_p])

    return stokes_row_indices, stokes_col_indices, stokes_values, R
