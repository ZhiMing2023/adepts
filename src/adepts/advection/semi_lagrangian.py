import torch
from .. import interpolation


def _apply_scalar_bc_on_p(q, mesh_state, scalar_boundary_dict):
    """Apply p-grid ghost values; Neumann data are outward derivatives."""
    if scalar_boundary_dict is None:
        return q

    q = q.clone()

    xp = mesh_state["xp"]
    yp = mesh_state["yp"]

    dx_left = xp[1] - xp[0]
    dx_right = xp[-1] - xp[-2]
    dy_top = yp[1] - yp[0]
    dy_bottom = yp[-1] - yp[-2]

    def get_bc(side):
        bc = scalar_boundary_dict.get(side, None)
        if bc is None:
            return None, None
        return bc["type"].lower(), bc["value"]

    btype, bval = get_bc("top")
    if btype == "dirichlet":
        q[0, 1:-1] = 2.0 * bval - q[1, 1:-1]
    elif btype == "neumann":
        q[0, 1:-1] = q[1, 1:-1] + bval * dy_top

    btype, bval = get_bc("bottom")
    if btype == "dirichlet":
        q[-1, 1:-1] = 2.0 * bval - q[-2, 1:-1]
    elif btype == "neumann":
        q[-1, 1:-1] = q[-2, 1:-1] + bval * dy_bottom

    btype, bval = get_bc("left")
    if btype == "dirichlet":
        q[1:-1, 0] = 2.0 * bval - q[1:-1, 1]
    elif btype == "neumann":
        q[1:-1, 0] = q[1:-1, 1] + bval * dx_left

    btype, bval = get_bc("right")
    if btype == "dirichlet":
        q[1:-1, -1] = 2.0 * bval - q[1:-1, -2]
    elif btype == "neumann":
        q[1:-1, -1] = q[1:-1, -2] + bval * dx_right

    q[0, 0] = 0.5 * (q[0, 1] + q[1, 0])
    q[0, -1] = 0.5 * (q[0, -2] + q[1, -1])
    q[-1, 0] = 0.5 * (q[-1, 1] + q[-2, 0])
    q[-1, -1] = 0.5 * (q[-1, -2] + q[-2, -1])

    return q

def advect_temperature(
    q_old,
    vx, vy,
    dt,
    mesh_state,
    *,
    edge_mode="nearest",
    constant_value=0.0,
    backtrace="rk2",
    scalar_boundary_dict=None,
    enforce_bc_after_advect=True,
    use_dirichlet_value_for_oob=True,
    scalar_interp_method="linear",
    velocity_interp_method="linear"
):
    xp = mesh_state["xp"]
    yp = mesh_state["yp"]
    Xp, Yp = torch.meshgrid(xp, yp, indexing="xy")

    xmin, xmax = xp[0], xp[-1]
    ymin, ymax = yp[0], yp[-1]

    if scalar_boundary_dict is not None:
        q_in = _apply_scalar_bc_on_p(q_old, mesh_state, scalar_boundary_dict)
    else:
        q_in = q_old

    ux_p = interpolation.interp_between_grids(
        mesh_state, vx, src="vx", tgt="p",
        edge_mode=edge_mode,
        constant_value=constant_value,
        method=velocity_interp_method,
    )
    uy_p = interpolation.interp_between_grids(
        mesh_state, vy, src="vy", tgt="p",
        edge_mode=edge_mode,
        constant_value=constant_value,
        method=velocity_interp_method,
    )

    def sample_u(xq, yq):
        uq = interpolation.safe_bilinear_interpolate(
            src_val=ux_p,
            src_x=xp,
            src_y=yp,
            tgt_x=xq,
            tgt_y=yq,
            edge_mode=edge_mode,
            constant_value=constant_value,
        )
        vq = interpolation.safe_bilinear_interpolate(
            src_val=uy_p,
            src_x=xp,
            src_y=yp,
            tgt_x=xq,
            tgt_y=yq,
            edge_mode=edge_mode,
            constant_value=constant_value,
        )
        return uq, vq

    if backtrace == "euler":
        u1, v1 = ux_p, uy_p
        Xd = Xp - dt * u1
        Yd = Yp - dt * v1

    elif backtrace == "rk2":
        u1, v1 = ux_p, uy_p
        Xmid = Xp - 0.5 * dt * u1
        Ymid = Yp - 0.5 * dt * v1
        umid, vmid = sample_u(Xmid, Ymid)
        Xd = Xp - dt * umid
        Yd = Yp - dt * vmid

    elif backtrace == "rk4":
        u1, v1 = ux_p, uy_p

        X2 = Xp - 0.5 * dt * u1
        Y2 = Yp - 0.5 * dt * v1
        u2, v2 = sample_u(X2, Y2)

        X3 = Xp - 0.5 * dt * u2
        Y3 = Yp - 0.5 * dt * v2
        u3, v3 = sample_u(X3, Y3)

        X4 = Xp - dt * u3
        Y4 = Yp - dt * v3
        u4, v4 = sample_u(X4, Y4)

        ueff = (u1 + 2.0 * u2 + 2.0 * u3 + u4) / 6.0
        veff = (v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0

        Xd = Xp - dt * ueff
        Yd = Yp - dt * veff

    else:
        raise ValueError(f"Unknown backtrace mode: {backtrace}")

    if scalar_interp_method == "linear":
        q_star = interpolation.safe_bilinear_interpolate(
            src_val=q_in,
            src_x=xp,
            src_y=yp,
            tgt_x=Xd,
            tgt_y=Yd,
            edge_mode=edge_mode,
            constant_value=constant_value,
        )
    else:
        raise ValueError(f"Unknown scalar_interp_method={scalar_interp_method}")

    if use_dirichlet_value_for_oob and (scalar_boundary_dict is not None):
        top_oob = Yd < ymin
        bottom_oob = Yd > ymax
        left_oob = Xd < xmin
        right_oob = Xd > xmax

        if "top" in scalar_boundary_dict and scalar_boundary_dict["top"]["type"].lower() == "dirichlet":
            q_star = torch.where(
                top_oob,
                torch.as_tensor(scalar_boundary_dict["top"]["value"], device=q_star.device, dtype=q_star.dtype),
                q_star,
            )

        if "bottom" in scalar_boundary_dict and scalar_boundary_dict["bottom"]["type"].lower() == "dirichlet":
            q_star = torch.where(
                bottom_oob,
                torch.as_tensor(scalar_boundary_dict["bottom"]["value"], device=q_star.device, dtype=q_star.dtype),
                q_star,
            )

        if "left" in scalar_boundary_dict and scalar_boundary_dict["left"]["type"].lower() == "dirichlet":
            q_star = torch.where(
                left_oob,
                torch.as_tensor(scalar_boundary_dict["left"]["value"], device=q_star.device, dtype=q_star.dtype),
                q_star,
            )

        if "right" in scalar_boundary_dict and scalar_boundary_dict["right"]["type"].lower() == "dirichlet":
            q_star = torch.where(
                right_oob,
                torch.as_tensor(scalar_boundary_dict["right"]["value"], device=q_star.device, dtype=q_star.dtype),
                q_star,
            )

    if enforce_bc_after_advect:
        q_star = _apply_scalar_bc_on_p(q_star, mesh_state, scalar_boundary_dict)

    return q_star


def advect_scalar_sl_on_p_bfecc(
    q_old,
    vx, vy,
    dt,
    mesh_state,
    *,
    edge_mode="nearest",
    constant_value=0.0,
    backtrace="rk2",
    scalar_boundary_dict=None,
    enforce_bc_after_advect=True,
    use_dirichlet_value_for_oob=True,
    scalar_interp_method="linear",
    velocity_interp_method="linear",
    clip_range=None,
):
    """Advect a p-grid scalar with BFECC correction and optional clipping."""
    q1 = advect_temperature(
        q_old=q_old,
        vx=vx,
        vy=vy,
        dt=dt,
        mesh_state=mesh_state,
        edge_mode=edge_mode,
        constant_value=constant_value,
        backtrace=backtrace,
        scalar_boundary_dict=scalar_boundary_dict,
        enforce_bc_after_advect=enforce_bc_after_advect,
        use_dirichlet_value_for_oob=use_dirichlet_value_for_oob,
        scalar_interp_method=scalar_interp_method,
        velocity_interp_method=velocity_interp_method,
    )

    if clip_range is not None:
        q1 = torch.clamp(q1, clip_range[0], clip_range[1])

    q0_back = advect_temperature(
        q_old=q1,
        vx=vx,
        vy=vy,
        dt=-dt,
        mesh_state=mesh_state,
        edge_mode=edge_mode,
        constant_value=constant_value,
        backtrace=backtrace,
        scalar_boundary_dict=scalar_boundary_dict,
        enforce_bc_after_advect=enforce_bc_after_advect,
        use_dirichlet_value_for_oob=use_dirichlet_value_for_oob,
        scalar_interp_method=scalar_interp_method,
        velocity_interp_method=velocity_interp_method,
    )

    if clip_range is not None:
        q0_back = torch.clamp(q0_back, clip_range[0], clip_range[1])

    q_corr = q_old + 0.5 * (q_old - q0_back)

    if clip_range is not None:
        q_corr = torch.clamp(q_corr, clip_range[0], clip_range[1])

    if enforce_bc_after_advect:
        q_corr = _apply_scalar_bc_on_p(
            q_corr,
            mesh_state,
            scalar_boundary_dict,
        )

    q_new = advect_temperature(
        q_old=q_corr,
        vx=vx,
        vy=vy,
        dt=dt,
        mesh_state=mesh_state,
        edge_mode=edge_mode,
        constant_value=constant_value,
        backtrace=backtrace,
        scalar_boundary_dict=scalar_boundary_dict,
        enforce_bc_after_advect=enforce_bc_after_advect,
        use_dirichlet_value_for_oob=use_dirichlet_value_for_oob,
        scalar_interp_method=scalar_interp_method,
        velocity_interp_method=velocity_interp_method,
    )

    if clip_range is not None:
        q_new = torch.clamp(q_new, clip_range[0], clip_range[1])

    if enforce_bc_after_advect:
        q_new = _apply_scalar_bc_on_p(
            q_new,
            mesh_state,
            scalar_boundary_dict,
        )

    return q_new
