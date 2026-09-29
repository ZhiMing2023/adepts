import torch


def thermal_assembly(
    KX, KY, RHOCP, dt, tk1, HSUM,
    mesh_state,
    thermal_boundary_dict,
):
    xp  = mesh_state["xp"]
    yp  = mesh_state["yp"]
    Nx1 = mesh_state["Nx1"]
    Ny1 = mesh_state["Ny1"]

    rows, cols, vals, R = thermal_assembly_forward(
        KX, KY, RHOCP, dt, tk1, HSUM,
        xp, yp,
        Nx1, Ny1,
        thermal_boundary_dict,
    )

    return rows, cols, vals, R

def thermal_assembly_forward(
    KX: torch.Tensor,
    KY: torch.Tensor,
    RHOCP: torch.Tensor,
    dt: torch.Tensor,
    tk1: torch.Tensor,
    HSUM: torch.Tensor,
    xp: torch.Tensor,
    yp: torch.Tensor,
    Nx1: int,
    Ny1: int,
    boundary_dict: dict,
):
    """Assemble the five-point transient diffusion system."""
    device, dtype = KX.device, KX.dtype

    j = torch.arange(Nx1, device=device)
    i = torch.arange(Ny1, device=device)
    j_idx, i_idx = torch.meshgrid(j, i, indexing="ij")
    gk = j_idx * Ny1 + i_idx

    mask = {
        "top": (i_idx == 0) & (j_idx > 0) & (j_idx < Nx1 - 1),
        "bottom": (i_idx == Ny1 - 1) & (j_idx > 0) & (j_idx < Nx1 - 1),
        "left": (j_idx == 0),
        "right": (j_idx == Nx1 - 1),
    }
    mask_boundary = mask["top"] | mask["bottom"] | mask["left"] | mask["right"]
    mask_internal = ~mask_boundary

    j_int = j_idx[mask_internal].to(torch.long)
    i_int = i_idx[mask_internal].to(torch.long)
    g_int = gk[mask_internal].to(torch.long)

    dx1 = xp[j_int] - xp[j_int - 1]
    dx2 = xp[j_int + 1] - xp[j_int]
    dx12 = (dx1 + dx2) / 2
    dy1 = yp[i_int] - yp[i_int - 1]
    dy2 = yp[i_int + 1] - yp[i_int]
    dy12 = (dy1 + dy2) / 2

    Kx1 = KX[i_int, j_int - 1]
    Kx2 = KX[i_int, j_int]
    Ky1 = KY[i_int - 1, j_int]
    Ky2 = KY[i_int, j_int]
    rhocp = RHOCP[i_int, j_int]

    row_int_left = g_int
    col_int_left = g_int - Ny1
    val_int_left = -Kx1 / dx1 / dx12

    row_int_down = g_int
    col_int_down = g_int - 1
    val_int_down = -Ky1 / dy1 / dy12

    row_int_self = g_int
    col_int_self = g_int
    val_int_self = (
        rhocp / dt
        + (Kx1 / dx1 + Kx2 / dx2) / dx12
        + (Ky1 / dy1 + Ky2 / dy2) / dy12
    )

    row_int_up = g_int
    col_int_up = g_int + 1
    val_int_up = -Ky2 / dy2 / dy12

    row_int_right = g_int
    col_int_right = g_int + Ny1
    val_int_right = -Kx2 / dx2 / dx12

    R = torch.zeros(Nx1 * Ny1, dtype=dtype, device=device)
    R = R.index_put(
        (g_int,),
        rhocp / dt * tk1[i_int, j_int] + HSUM[i_int, j_int],
        accumulate=True,
    )

    row_b_list, col_b_list, val_b_blocks = [], [], []

    def add_boundary_block(side_name: str, g: torch.Tensor, nbr: torch.Tensor, cfg: dict):
        if cfg["type"] == "dirichlet":
            row = torch.stack([g, g], dim=1).flatten()
            col = torch.stack([g, nbr], dim=1).flatten()
            val = torch.ones(2 * g.numel(), dtype=dtype, device=device)
            R[g] = 2.0 * cfg["value"]
        elif cfg["type"] == "neumann":
            row = torch.stack([g, g], dim=1).flatten()
            col = torch.stack([g, nbr], dim=1).flatten()
            val = torch.tensor([1.0, -1.0], dtype=dtype, device=device).repeat(g.numel())
            R[g] = 0.0
        else:
            raise ValueError(f"Unknown boundary type for {side_name}: {cfg['type']}")

        row_b_list.append(row)
        col_b_list.append(col)
        val_b_blocks.append(val)

    for side, cfg in boundary_dict.items():
        g = gk[mask[side]].view(-1).to(torch.long)
        if g.numel() == 0:
            continue
        if side == "top":
            nbr = g + 1
        elif side == "bottom":
            nbr = g - 1
        elif side == "left":
            nbr = g + Ny1
        elif side == "right":
            nbr = g - Ny1
        else:
            raise ValueError(f"Unknown side: {side}")
        add_boundary_block(side, g, nbr, cfg)

    row_b = torch.cat(row_b_list) if row_b_list else torch.empty(0, dtype=torch.long, device=device)
    col_b = torch.cat(col_b_list) if col_b_list else torch.empty(0, dtype=torch.long, device=device)

    rows = torch.cat([
        row_int_left, row_int_down, row_int_self, row_int_up, row_int_right,
        row_b,
    ])
    cols = torch.cat([
        col_int_left, col_int_down, col_int_self, col_int_up, col_int_right,
        col_b,
    ])
    vals = torch.cat([
        val_int_left, val_int_down, val_int_self, val_int_up, val_int_right,
        *val_b_blocks,
    ])

    return rows, cols, vals, R
