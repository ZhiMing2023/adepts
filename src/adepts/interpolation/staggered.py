import torch
from typing import Mapping, Any, Literal
import warnings

EdgeMode = Literal["clamp", "nearest", "constant"]
GridName  = Literal["node", "vx", "vy", "p"]

def _interp_between_grids_impl(
    src_vals,
    mesh_state,
    src,
    tgt,
    edge_mode,
    constant_value,
    method,
):
    coords = {
        "node": (mesh_state["xnode"], mesh_state["ynode"]),
        "vx"  : (mesh_state["xvx"],  mesh_state["yvx"]),
        "vy"  : (mesh_state["xvy"],  mesh_state["yvy"]),
        "p"   : (mesh_state["xp"],   mesh_state["yp"]),
    }

    if src not in coords:
        raise ValueError(f"Unknown src grid type: {src}. Expected one of {list(coords.keys())}.")
    if tgt not in coords:
        raise ValueError(f"Unknown tgt grid type: {tgt}. Expected one of {list(coords.keys())}.")

    if src == tgt:
        return src_vals.clone()

    sx, sy = coords[src]
    tx, ty = torch.meshgrid(*coords[tgt], indexing="xy")

    if method == "linear":
        if src == "vx" and tgt == "p":
            return interp_vx_to_p(src_vals, mesh_state, edge_mode=edge_mode, constant_value=constant_value)
        elif src == "vy" and tgt == "p":
            return interp_vy_to_p(src_vals, mesh_state, edge_mode=edge_mode, constant_value=constant_value)
        elif src == "p" and tgt == "vx":
            return interp_p_to_vx(src_vals, mesh_state, edge_mode=edge_mode, constant_value=constant_value)
        elif src == "p" and tgt == "vy":
            return interp_p_to_vy(src_vals, mesh_state, edge_mode=edge_mode, constant_value=constant_value)
        elif src == "node" and tgt == "p":
            return interp_node_to_p(src_vals, mesh_state, edge_mode=edge_mode, constant_value=constant_value)
        elif src == "p" and tgt == "node":
            return interp_p_to_node(src_vals, mesh_state, edge_mode=edge_mode, constant_value=constant_value)
        elif src == "vx" and tgt == "node":
            return interp_vx_to_node(src_vals, mesh_state, edge_mode=edge_mode, constant_value=constant_value)
        elif src == "node" and tgt == "vx":
            return interp_node_to_vx(src_vals, mesh_state, edge_mode=edge_mode, constant_value=constant_value)
        elif src == "vy" and tgt == "node":
            return interp_vy_to_node(src_vals, mesh_state, edge_mode=edge_mode, constant_value=constant_value)
        elif src == "node" and tgt == "vy":
            return interp_node_to_vy(src_vals, mesh_state, edge_mode=edge_mode, constant_value=constant_value)
        elif src == "vx" and tgt == "vy":
            return interp_vx_to_vy(src_vals, mesh_state, edge_mode=edge_mode, constant_value=constant_value)
        elif src == "vy" and tgt == "vx":
            return interp_vy_to_vx(src_vals, mesh_state, edge_mode=edge_mode, constant_value=constant_value)
        else:
            return safe_bilinear_interpolate(
                src_val=src_vals,
                src_x=sx,
                src_y=sy,
                tgt_x=tx,
                tgt_y=ty,
                edge_mode=edge_mode,
                constant_value=constant_value,
            )

    else:
        raise ValueError(f"Unknown interpolation method={method}, expected 'linear'")


def interp_between_grids(
    mesh_state: Mapping[str, Any],
    src_vals: torch.Tensor,
    src: str,
    tgt: str,
    edge_mode: str = "clamp",
    constant_value: float = 0.0,
    method: str = "linear",
) -> torch.Tensor:


    return _interp_between_grids_impl(
        src_vals,
        mesh_state,
        src,
        tgt,
        edge_mode,
        constant_value,
        method,
    )

def safe_bilinear_interpolate(
    src_val: torch.Tensor,
    src_x: torch.Tensor,
    src_y: torch.Tensor,
    tgt_x: torch.Tensor,
    tgt_y  : torch.Tensor,
    *,
    edge_mode: EdgeMode = "clamp",
    constant_value: float = 0.0,
    eps: float = 1e-12,
) -> torch.Tensor:

    if src_val.ndim != 2:
        raise ValueError(f"src_val must be 2D, got shape={tuple(src_val.shape)}")
    if src_x.ndim != 1 or src_y.ndim != 1:
        raise ValueError("src_x and src_y must be 1D")
    if tgt_x.shape != tgt_y.shape:
        raise ValueError("tgt_x and tgt_y must have the same shape")

    Ny, Nx = src_val.shape
    if Nx != src_x.numel() or Ny != src_y.numel():
        raise ValueError(
            f"src_val shape={tuple(src_val.shape)} inconsistent with "
            f"src_x={src_x.numel()}, src_y={src_y.numel()}"
        )
    if Nx < 2 or Ny < 2:
        raise ValueError("Need at least 2 points in each direction")

    ix = torch.bucketize(tgt_x.contiguous(), src_x.contiguous()) - 1
    iy = torch.bucketize(tgt_y.contiguous(), src_y.contiguous()) - 1
    ix = ix.clamp(0, src_x.numel() - 2)
    iy = iy.clamp(0, src_y.numel() - 2)

    x0, x1 = src_x[ix], src_x[ix + 1]
    y0, y1 = src_y[iy], src_y[iy + 1]

    dx = x1 - x0
    dy = y1 - y0
    mask_dx = dx.abs() < eps
    mask_dy = dy.abs() < eps
    dx = torch.where(mask_dx, torch.ones_like(dx), dx)
    dy = torch.where(mask_dy, torch.ones_like(dy), dy)

    n_bad = int((mask_dx | mask_dy).sum())
    if n_bad:
        msg = (f"[safe_bilinear_interpolate] Detected {n_bad} degenerate cell(s): "
               f"min|dx|={dx.abs().min():.3e}, min|dy|={dy.abs().min():.3e}. "
               "Check mesh quality.")
        warnings.warn(msg, RuntimeWarning)

    wx = (tgt_x - x0) / dx
    wy = (tgt_y - y0) / dy

    if edge_mode == "nearest":
        wx = wx.clamp(0.0, 1.0)
        wy = wy.clamp(0.0, 1.0)
    elif edge_mode == "constant":
        outside = (tgt_x < src_x[0]) | (tgt_x > src_x[-1]) | \
                  (tgt_y < src_y[0]) | (tgt_y > src_y[-1])
    elif edge_mode == "clamp":
        pass

    else:
        raise ValueError(f"Unknown edge_mode={edge_mode}")

    Ny, Nx = src_val.shape
    base = iy * Nx + ix
    f = src_val.flatten()

    f00 = f[base]
    f10 = f[base + 1]
    f01 = f[base + Nx]
    f11 = f[base + Nx + 1]

    out = torch.where(
        mask_dx,
        (1 - wy) * f00 + wy * f01,
        torch.zeros_like(wx)
    )

    out = torch.where(
        mask_dy,
        (1 - wx) * f00 + wx * f10,
        out
    )

    normal_mask = ~(mask_dx | mask_dy)
    out = torch.where(
        normal_mask,
        (1 - wx) * (1 - wy) * f00 +
        wx       * (1 - wy) * f10 +
        (1 - wx) * wy       * f01 +
        wx       * wy       * f11,
        out
    )

    if edge_mode == "constant":
        out = torch.where(outside, torch.tensor(constant_value, dtype=out.dtype, device=out.device), out)

    return out


def _assert_2d_shape(
    field: torch.Tensor,
    ny: int,
    nx: int,
    name: str,
):
    if field.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={tuple(field.shape)}")
    if field.shape != (ny, nx):
        raise ValueError(
            f"{name} shape mismatch: got {tuple(field.shape)}, expected {(ny, nx)}"
        )


def _linear_interp_1d_along_x(
    src_vals: torch.Tensor,
    src_x: torch.Tensor,
    tgt_x: torch.Tensor,
    *,
    edge_mode: EdgeMode = "clamp",
    constant_value: float = 0.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Interpolate each row along x."""
    if src_vals.ndim != 2:
        raise ValueError(f"src_vals must be 2D, got shape={tuple(src_vals.shape)}")
    if src_x.ndim != 1 or tgt_x.ndim != 1:
        raise ValueError("src_x and tgt_x must be 1D tensors")
    if src_vals.shape[1] != src_x.numel():
        raise ValueError(
            f"src_vals.shape[1]={src_vals.shape[1]} != src_x.numel()={src_x.numel()}"
        )
    if src_x.numel() < 2:
        raise ValueError("src_x must contain at least 2 points")

    ix = torch.bucketize(tgt_x.contiguous(), src_x.contiguous()) - 1
    ix = ix.clamp(0, src_x.numel() - 2)

    x0 = src_x[ix]
    x1 = src_x[ix + 1]
    dx = x1 - x0

    mask_dx = dx.abs() < eps
    dx_safe = torch.where(mask_dx, torch.ones_like(dx), dx)

    wx = (tgt_x - x0) / dx_safe

    outside = None
    if edge_mode == "nearest":
        wx = wx.clamp(0.0, 1.0)
    elif edge_mode == "constant":
        outside = (tgt_x < src_x[0]) | (tgt_x > src_x[-1])
    elif edge_mode != "clamp":
        raise ValueError(f"Unknown edge_mode={edge_mode}")

    f0 = src_vals[:, ix]
    f1 = src_vals[:, ix + 1]

    out = torch.where(
        mask_dx.unsqueeze(0),
        f0,
        (1.0 - wx).unsqueeze(0) * f0 + wx.unsqueeze(0) * f1,
    )

    if edge_mode == "constant":
        fill = torch.as_tensor(
            constant_value, dtype=out.dtype, device=out.device
        )
        out = torch.where(outside.unsqueeze(0), fill, out)

    return out


def _linear_interp_1d_along_y(
    src_vals: torch.Tensor,
    src_y: torch.Tensor,
    tgt_y: torch.Tensor,
    *,
    edge_mode: EdgeMode = "clamp",
    constant_value: float = 0.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    if src_vals.ndim != 2:
        raise ValueError(f"src_vals must be 2D, got shape={tuple(src_vals.shape)}")
    if src_y.ndim != 1 or tgt_y.ndim != 1:
        raise ValueError("src_y and tgt_y must be 1D tensors")
    if src_vals.shape[0] != src_y.numel():
        raise ValueError(
            f"src_vals.shape[0]={src_vals.shape[0]} != src_y.numel()={src_y.numel()}"
        )
    if src_y.numel() < 2:
        raise ValueError("src_y must contain at least 2 points")

    iy = torch.bucketize(tgt_y.contiguous(), src_y.contiguous()) - 1
    iy = iy.clamp(0, src_y.numel() - 2)

    y0 = src_y[iy]
    y1 = src_y[iy + 1]
    dy = y1 - y0

    mask_dy = dy.abs() < eps
    dy_safe = torch.where(mask_dy, torch.ones_like(dy), dy)

    wy = (tgt_y - y0) / dy_safe

    outside = None
    if edge_mode == "nearest":
        wy = wy.clamp(0.0, 1.0)
    elif edge_mode == "constant":
        outside = (tgt_y < src_y[0]) | (tgt_y > src_y[-1])
    elif edge_mode != "clamp":
        raise ValueError(f"Unknown edge_mode={edge_mode}")

    f0 = src_vals[iy, :]
    f1 = src_vals[iy + 1, :]

    out = torch.where(
        mask_dy.unsqueeze(1),
        f0,
        (1.0 - wy).unsqueeze(1) * f0 + wy.unsqueeze(1) * f1,
    )

    if edge_mode == "constant":
        fill = torch.as_tensor(
            constant_value, dtype=out.dtype, device=out.device
        )
        out = torch.where(outside.unsqueeze(1), fill, out)

    return out


def interp_vx_to_node(
    vx: torch.Tensor,
    mesh_state: Mapping[str, Any],
    *,
    edge_mode: EdgeMode = "clamp",
    constant_value: float = 0.0,
) -> torch.Tensor:
    xvx = mesh_state["xvx"][:-1]
    yvx = mesh_state["yvx"]
    xnode = mesh_state["xnode"]
    ynode = mesh_state["ynode"]

    vx_phys = vx[:, :-1]

    if xvx.numel() != xnode.numel():
        raise ValueError(
            f"interp_vx_to_node expects physical xvx and xnode aligned, "
            f"got len(xvx)={xvx.numel()}, len(xnode)={xnode.numel()}"
        )

    _assert_2d_shape(vx_phys, yvx.numel(), xvx.numel(), "vx_phys")

    return _linear_interp_1d_along_y(
        vx_phys,
        yvx,
        ynode,
        edge_mode=edge_mode,
        constant_value=constant_value,
    )

def interp_vy_to_node(
    vy: torch.Tensor,
    mesh_state: Mapping[str, Any],
    *,
    edge_mode: EdgeMode = "clamp",
    constant_value: float = 0.0,
) -> torch.Tensor:
    xvy = mesh_state["xvy"]
    yvy = mesh_state["yvy"][:-1]
    xnode = mesh_state["xnode"]
    ynode = mesh_state["ynode"]

    vy_phys = vy[:-1, :]

    if yvy.numel() != ynode.numel():
        raise ValueError(
            f"interp_vy_to_node expects physical yvy and ynode aligned, "
            f"got len(yvy)={yvy.numel()}, len(ynode)={ynode.numel()}"
        )

    _assert_2d_shape(vy_phys, yvy.numel(), xvy.numel(), "vy_phys")

    return _linear_interp_1d_along_x(
        vy_phys,
        xvy,
        xnode,
        edge_mode=edge_mode,
        constant_value=constant_value,
    )

def interp_p_to_node(
    phi_p: torch.Tensor,
    mesh_state: Mapping[str, Any],
    *,
    edge_mode: EdgeMode = "clamp",
    constant_value: float = 0.0,
) -> torch.Tensor:
    xp = mesh_state["xp"][1:-1]
    yp = mesh_state["yp"][1:-1]
    xnode = mesh_state["xnode"]
    ynode = mesh_state["ynode"]

    phi_p_phys = phi_p[1:-1, 1:-1]

    _assert_2d_shape(phi_p_phys, yp.numel(), xp.numel(), "phi_p_phys")

    tx, ty = torch.meshgrid(xnode, ynode, indexing="xy")
    return safe_bilinear_interpolate(
        src_val=phi_p_phys,
        src_x=xp,
        src_y=yp,
        tgt_x=tx,
        tgt_y=ty,
        edge_mode=edge_mode,
        constant_value=constant_value,
    )


def interp_node_to_p(
    phi_node: torch.Tensor,
    mesh_state: Mapping[str, Any],
    *,
    edge_mode: EdgeMode = "clamp",
    constant_value: float = 0.0,
) -> torch.Tensor:
    """Interpolate node values to the full pressure grid."""
    xnode = mesh_state["xnode"]
    ynode = mesh_state["ynode"]
    xp = mesh_state["xp"]
    yp = mesh_state["yp"]

    _assert_2d_shape(phi_node, ynode.numel(), xnode.numel(), "phi_node")

    tx, ty = torch.meshgrid(xp, yp, indexing="xy")
    return safe_bilinear_interpolate(
        src_val=phi_node,
        src_x=xnode,
        src_y=ynode,
        tgt_x=tx,
        tgt_y=ty,
        edge_mode=edge_mode,
        constant_value=constant_value,
    )


def interp_vx_to_p(
    vx: torch.Tensor,
    mesh_state: Mapping[str, Any],
    *,
    edge_mode: EdgeMode = "nearest",
    constant_value: float = 0.0,
) -> torch.Tensor:
    """Interpolate x-velocity values to the full pressure grid."""
    xvx_phys = mesh_state["xvx"][:-1]
    yvx_phys = mesh_state["yvx"]
    xp = mesh_state["xp"]
    yp = mesh_state["yp"]

    vx_phys = vx[:, :-1]

    _assert_2d_shape(vx_phys, yvx_phys.numel(), xvx_phys.numel(), "vx_phys")

    tx, ty = torch.meshgrid(xp, yp, indexing="xy")
    return safe_bilinear_interpolate(
        src_val=vx_phys,
        src_x=xvx_phys,
        src_y=yvx_phys,
        tgt_x=tx,
        tgt_y=ty,
        edge_mode=edge_mode,
        constant_value=constant_value,
    )


def interp_vy_to_p(
    vy: torch.Tensor,
    mesh_state: Mapping[str, Any],
    *,
    edge_mode: EdgeMode = "nearest",
    constant_value: float = 0.0,
) -> torch.Tensor:
    """Interpolate y-velocity values to the full pressure grid."""
    xvy_phys = mesh_state["xvy"]
    yvy_phys = mesh_state["yvy"][:-1]
    xp = mesh_state["xp"]
    yp = mesh_state["yp"]

    vy_phys = vy[:-1, :]

    _assert_2d_shape(vy_phys, yvy_phys.numel(), xvy_phys.numel(), "vy_phys")

    tx, ty = torch.meshgrid(xp, yp, indexing="xy")
    return safe_bilinear_interpolate(
        src_val=vy_phys,
        src_x=xvy_phys,
        src_y=yvy_phys,
        tgt_x=tx,
        tgt_y=ty,
        edge_mode=edge_mode,
        constant_value=constant_value,
    )


def interp_node_to_vx(
    phi_node: torch.Tensor,
    mesh_state: Mapping[str, Any],
    *,
    edge_mode: EdgeMode = "nearest",
    constant_value: float = 0.0,
) -> torch.Tensor:
    """Interpolate node values to the full x-velocity grid."""
    xnode = mesh_state["xnode"]
    ynode = mesh_state["ynode"]
    xvx = mesh_state["xvx"]
    yvx = mesh_state["yvx"]

    _assert_2d_shape(phi_node, ynode.numel(), xnode.numel(), "phi_node")

    tx, ty = torch.meshgrid(xvx, yvx, indexing="xy")
    return safe_bilinear_interpolate(
        src_val=phi_node,
        src_x=xnode,
        src_y=ynode,
        tgt_x=tx,
        tgt_y=ty,
        edge_mode=edge_mode,
        constant_value=constant_value,
    )

def interp_vy_to_vx(
    vy: torch.Tensor,
    mesh_state: Mapping[str, Any],
    *,
    edge_mode: EdgeMode = "nearest",
    constant_value: float = 0.0,
) -> torch.Tensor:
    """Interpolate y-velocity values to the full x-velocity grid."""
    xvy = mesh_state["xvy"]
    yvy = mesh_state["yvy"][:-1]
    xvx = mesh_state["xvx"]
    yvx = mesh_state["yvx"]

    vy_phys = vy[:-1, :]

    _assert_2d_shape(vy_phys, yvy.numel(), xvy.numel(), "vy_phys")

    tx, ty = torch.meshgrid(xvx, yvx, indexing="xy")
    return safe_bilinear_interpolate(
        src_val=vy_phys,
        src_x=xvy,
        src_y=yvy,
        tgt_x=tx,
        tgt_y=ty,
        edge_mode=edge_mode,
        constant_value=constant_value,
    )

def interp_p_to_vx(
    phi_p: torch.Tensor,
    mesh_state: Mapping[str, Any],
    *,
    edge_mode: EdgeMode = "nearest",
    constant_value: float = 0.0,
) -> torch.Tensor:
    """Interpolate pressure values to the full x-velocity grid."""
    xp = mesh_state["xp"][1:-1]
    yp = mesh_state["yp"][1:-1]
    xvx = mesh_state["xvx"]
    yvx = mesh_state["yvx"]

    phi_p_phys = phi_p[1:-1, 1:-1]

    _assert_2d_shape(phi_p_phys, yp.numel(), xp.numel(), "phi_p_phys")

    tx, ty = torch.meshgrid(xvx, yvx, indexing="xy")
    return safe_bilinear_interpolate(
        src_val=phi_p_phys,
        src_x=xp,
        src_y=yp,
        tgt_x=tx,
        tgt_y=ty,
        edge_mode=edge_mode,
        constant_value=constant_value,
    )


def interp_node_to_vy(
    phi_node: torch.Tensor,
    mesh_state: Mapping[str, Any],
    *,
    edge_mode: EdgeMode = "nearest",
    constant_value: float = 0.0,
) -> torch.Tensor:
    """Interpolate node values to the full y-velocity grid."""
    xnode = mesh_state["xnode"]
    ynode = mesh_state["ynode"]
    xvy = mesh_state["xvy"]
    yvy = mesh_state["yvy"]

    _assert_2d_shape(phi_node, ynode.numel(), xnode.numel(), "phi_node")

    tx, ty = torch.meshgrid(xvy, yvy, indexing="xy")
    return safe_bilinear_interpolate(
        src_val=phi_node,
        src_x=xnode,
        src_y=ynode,
        tgt_x=tx,
        tgt_y=ty,
        edge_mode=edge_mode,
        constant_value=constant_value,
    )


def interp_vx_to_vy(
    vx: torch.Tensor,
    mesh_state: Mapping[str, Any],
    *,
    edge_mode: EdgeMode = "nearest",
    constant_value: float = 0.0,
) -> torch.Tensor:
    """Interpolate x-velocity values to the full y-velocity grid."""
    xvx_phys = mesh_state["xvx"][:-1]
    yvx_phys = mesh_state["yvx"]
    xvy = mesh_state["xvy"]
    yvy = mesh_state["yvy"]

    vx_phys = vx[:, :-1]

    _assert_2d_shape(vx_phys, yvx_phys.numel(), xvx_phys.numel(), "vx_phys")

    tx, ty = torch.meshgrid(xvy, yvy, indexing="xy")
    return safe_bilinear_interpolate(
        src_val=vx_phys,
        src_x=xvx_phys,
        src_y=yvx_phys,
        tgt_x=tx,
        tgt_y=ty,
        edge_mode=edge_mode,
        constant_value=constant_value,
    )


def interp_p_to_vy(
    phi_p: torch.Tensor,
    mesh_state: Mapping[str, Any],
    *,
    edge_mode: EdgeMode = "nearest",
    constant_value: float = 0.0,
) -> torch.Tensor:
    """Interpolate pressure values to the full y-velocity grid."""
    xp_phys = mesh_state["xp"][1:-1]
    yp_phys = mesh_state["yp"][1:-1]
    xvy = mesh_state["xvy"]
    yvy = mesh_state["yvy"]

    phi_p_phys = phi_p[1:-1, 1:-1]

    _assert_2d_shape(phi_p_phys, yp_phys.numel(), xp_phys.numel(), "phi_p_phys")

    tx, ty = torch.meshgrid(xvy, yvy, indexing="xy")
    return safe_bilinear_interpolate(
        src_val=phi_p_phys,
        src_x=xp_phys,
        src_y=yp_phys,
        tgt_x=tx,
        tgt_y=ty,
        edge_mode=edge_mode,
        constant_value=constant_value,
    )
