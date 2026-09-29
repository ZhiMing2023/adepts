import os
import sys
import math
from pathlib import Path

import torch
import adepts as ad


SEED = 0
DEVICE = "cpu"
DTYPE = torch.float64

L0 = 660e3
ETA0 = 1e21
KAPPA0 = 1e-6
T_SURFACE = 273.0
T_MANTLE = 1574.0
T_BOTTOM = 1574.0
DTEMP = 1574.0 - 273.0

T0_SCALE = L0 ** 2 / KAPPA0
P0 = ETA0 * KAPPA0 / L0 ** 2
G_PHYS = 10.0
GX = 0.0
GY = 1.0

torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)
torch.manual_seed(SEED)

device = DEVICE
dtype = DTYPE
L0 = L0
eta0 = ETA0
kappa0 = KAPPA0
T_surface = T_SURFACE
Tmantle_phys = T_MANTLE
Tbot_phys = T_BOTTOM
dT0 = DTEMP
t0 = T0_SCALE
p0 = P0
g_phys = G_PHYS
gx = GX
gy = GY


xsize = 1500e3
ysize = 660e3

xbase_resolution = torch.tensor(10e3, dtype=dtype, device=device)
ybase_resolution = torch.tensor(10e3, dtype=dtype, device=device)
xfine_resolution = torch.tensor(5e3, dtype=dtype, device=device)
yfine_resolution = torch.tensor(5e3, dtype=dtype, device=device)

xfine_ranges = [(800e3, 1000e3)]
yfine_ranges = [(0.0, 102e3)]

xtransition_distance = torch.tensor(100e3, dtype=dtype, device=device)
ytransition_distance = torch.tensor(100e3, dtype=dtype, device=device)

mesh = ad.mesh.CartesianMesh(
    xsize,
    ysize,
    xbase_resolution,
    ybase_resolution,
    xfine_ranges=xfine_ranges,
    xfine_resolution=xfine_resolution,
    xtransition_distance=xtransition_distance,
    yfine_ranges=yfine_ranges,
    yfine_resolution=yfine_resolution,
    ytransition_distance=ytransition_distance,
    device=device,
)


def build_nd_mesh_state(mesh_obj):
    mesh_state = mesh_obj.to_mesh_state()
    for key in ["xnode", "ynode", "xp", "yp", "xvx", "yvx", "xvy", "yvy"]:
        mesh_state[key] = mesh_state[key] / L0
    mesh_state["xsize"] = mesh_state["xsize"] / L0
    mesh_state["ysize"] = mesh_state["ysize"] / L0
    return mesh_state


mesh_state = build_nd_mesh_state(mesh)
coloring_pack = ad.stokes.StokesSystem.build_Fu_coloring_from_mesh(
    mesh_state,
    mode="safe",
    device=device,
    debug_print=False,
)

num_steps = 30
dt = torch.tensor(30e4 * 31536000.0, device=device, dtype=dtype)
time_total = None
pscale = torch.ones_like(mesh.dx)
p_fix_value = torch.tensor(3e8, device=device, dtype=dtype) * L0 * L0 / eta0 / kappa0

A_param = torch.tensor([21.0], dtype=dtype, device=device, requires_grad=False)
E_param = torch.tensor([3.0], dtype=dtype, device=device, requires_grad=False)
V_disl = torch.tensor([0.0], dtype=dtype, device=device, requires_grad=False)
n_param = torch.tensor([3.0], dtype=dtype, device=device, requires_grad=False)
rho_param = torch.tensor([3300.0, 3300.0], dtype=dtype, device=device, requires_grad=False)
rhocp = torch.tensor([3.3e6], dtype=dtype, device=device, requires_grad=False)
alpha = torch.tensor([3e-5], dtype=dtype, device=device, requires_grad=False)
k_thermal = torch.tensor([3.0], dtype=dtype, device=device, requires_grad=False)

stokes_boundary_type = {
    "left": "free_slip",
    "right": "free_slip",
    "top": "free_slip",
    "bottom": "free_slip",
}

temperature_bc = {
    "top": {"type": "dirichlet", "value": 0.0},
    "bottom": {"type": "dirichlet", "value": (Tbot_phys - T_surface) / dT0},
    "left": {"type": "neumann", "value": 0.0},
    "right": {"type": "neumann", "value": 0.0},
}

composition_bc = {
    "top": {"type": "neumann", "value": 0.0},
    "bottom": {"type": "neumann", "value": 0.0},
    "left": {"type": "neumann", "value": 0.0},
    "right": {"type": "neumann", "value": 0.0},
}


def half_space_cooling_field_ridge_slab(mesh, ridge_right_width=200000.0, ridge_age_min=1.0, ridge_age_max=40000000.0, x_trench=1000000.0, slab_length=650000.0, slab_thickness=100000.0, smooth_width=10000.0, tanh_center=None, tanh_top=-10000.0, tanh_bottom=None, tanh_b=None, curve_npts=512, dist_chunk=20000, use_upper_half_tanh=True, target_slab_depth=400000.0, tanh_b_factor=3.0, weakzone_shift_x=10000.0, weakzone_y_cut=100000.0, weakzone_smooth=5000.0, age_background=40000000.0, age_slab=40000000.0, age_lm=500000000.0, T_surface=300.0, T_mantle=Tmantle_phys, T_cmb=Tbot_phys, kappa=1e-06, z_660=660000.0):

    device = mesh.xp.device
    dtype = mesh.xp.dtype
    xp = mesh.xp
    yp = mesh.yp
    X = xp[None, :].expand(yp.numel(), -1)
    Y = yp[:, None].expand(-1, xp.numel())
    z_cmb = yp[-1]
    if yp.numel() < 2:
        raise ValueError('mesh.yp must contain at least two points.')
    dy_p = yp[1] - yp[0]
    y_top_eff = yp[0] - 0.5 * dy_p
    if tanh_top is None:
        tanh_top = y_top_eff
    tanh_top_t_early = torch.as_tensor(tanh_top, device=device, dtype=dtype)
    hsc_top_y = tanh_top_t_early

    def hsc(depth, age_yr, T_hot, T_cold):
        depth = torch.clamp(depth, min=0.0)
        age_yr_t = torch.as_tensor(age_yr, device=device, dtype=dtype)
        age_sec = age_yr_t * 365.25 * 24.0 * 3600.0
        age_sec = torch.clamp(age_sec, min=1.0)
        kappa_t = torch.as_tensor(kappa, device=device, dtype=dtype)
        denom = 2.0 * torch.sqrt(kappa_t * age_sec)
        arg = depth / denom
        return T_cold + (T_hot - T_cold) * torch.erf(arg)

    def tanh_curve_torch(x, center, top, bottom, b):
        a = -(top - bottom) / 2.0
        d_mid = (top + bottom) / 2.0
        return a * torch.tanh(b * (x - center)) + d_mid
    depth_from_surface = torch.clamp(Y - hsc_top_y, min=0.0)
    T_um_bg = hsc(depth=depth_from_surface, age_yr=age_background, T_hot=T_mantle, T_cold=T_surface)
    depth_from_bottom = torch.clamp(z_cmb - Y, min=0.0)
    T_bottom_reverse_bg = hsc(depth=depth_from_bottom, age_yr=age_lm, T_hot=T_mantle, T_cold=T_cmb)
    bottom_reverse_thickness = torch.as_tensor(50000.0, device=device, dtype=dtype)
    bottom_reverse_smooth = torch.as_tensor(10000.0, device=device, dtype=dtype)
    bottom_reverse_mask = torch.sigmoid((bottom_reverse_thickness - depth_from_bottom) / bottom_reverse_smooth)
    T = (1.0 - bottom_reverse_mask) * T_um_bg + bottom_reverse_mask * T_bottom_reverse_bg
    age_ridge_right = ridge_age_min + (ridge_age_max - 0.0) / ridge_right_width * X
    age_ridge_right = torch.clamp(age_ridge_right, min=ridge_age_min, max=ridge_age_max)
    T_ridge_right = hsc(depth=depth_from_surface, age_yr=age_ridge_right, T_hot=T_mantle, T_cold=T_surface)
    ridge_mask = (X <= ridge_right_width) & (Y <= 200000.0)
    T = torch.where(ridge_mask, T_ridge_right, T)
    T_bg = T.clone()
    x_slab_start = x_trench
    x_slab_end = x_trench + slab_length
    if use_upper_half_tanh:
        if tanh_center is None:
            tanh_center = x_slab_end
        if tanh_bottom is None:
            tanh_bottom = 2.0 * target_slab_depth - tanh_top
        if tanh_b is None:
            tanh_b = tanh_b_factor / max(float(slab_length), 1.0)
    else:
        if tanh_center is None:
            tanh_center = x_trench + 0.5 * slab_length
        if tanh_bottom is None:
            tanh_bottom = min(560000.0, 0.85 * slab_length)
        if tanh_b is None:
            tanh_b = 5.189e-06 * (1000000.0 / max(float(slab_length), 1.0))
    tanh_center_t = torch.as_tensor(tanh_center, device=device, dtype=dtype)
    tanh_top_t = torch.as_tensor(tanh_top, device=device, dtype=dtype)
    tanh_bottom_t = torch.as_tensor(tanh_bottom, device=device, dtype=dtype)
    tanh_b_t = torch.as_tensor(tanh_b, device=device, dtype=dtype)
    Y_curve = tanh_curve_torch(X, center=tanh_center_t, top=tanh_top_t, bottom=tanh_bottom_t, b=tanh_b_t)
    x_curve = torch.linspace(x_slab_start, x_slab_end, curve_npts, device=device, dtype=dtype)
    y_curve = tanh_curve_torch(x_curve, center=tanh_center_t, top=tanh_top_t, bottom=tanh_bottom_t, b=tanh_b_t)
    X_flat = X.reshape(-1)
    Y_flat = Y.reshape(-1)
    dist_flat = torch.empty_like(X_flat)
    for i0 in range(0, X_flat.numel(), dist_chunk):
        i1 = min(i0 + dist_chunk, X_flat.numel())
        dx = X_flat[i0:i1, None] - x_curve[None, :]
        dy = Y_flat[i0:i1, None] - y_curve[None, :]
        dist2 = dx * dx + dy * dy
        dist_flat[i0:i1] = torch.sqrt(torch.min(dist2, dim=1).values)
    dist_to_curve = dist_flat.reshape_as(X)
    mask_x = torch.sigmoid((X - x_slab_start) / smooth_width) * torch.sigmoid((x_slab_end - X) / smooth_width)
    mask_below_curve = torch.sigmoid((Y - Y_curve) / smooth_width)
    mask_thickness = torch.sigmoid((slab_thickness - dist_to_curve) / smooth_width)
    slab_mask = mask_x * mask_below_curve * mask_thickness
    depth_slab_T = torch.clamp(Y - Y_curve, min=0.0, max=slab_thickness)
    T_slab = hsc(depth=depth_slab_T, age_yr=age_slab, T_hot=T_mantle, T_cold=T_surface)
    T = (1.0 - slab_mask) * T + slab_mask * T_slab

    def inverse_tanh_curve_torch(y, center, top, bottom, b):
        a = -(top - bottom) / 2.0
        d_mid = (top + bottom) / 2.0
        z = (y - d_mid) / a
        z = torch.clamp(z, min=-0.999999, max=0.999999)
        atanh_z = 0.5 * torch.log((1.0 + z) / (1.0 - z))
        return center + atanh_z / b

    def build_horizontal_shift_weakzone_mask(xcoord, ycoord):
        xcoord = torch.as_tensor(xcoord, device=device, dtype=dtype)
        ycoord = torch.as_tensor(ycoord, device=device, dtype=dtype)
        Xg = xcoord[None, :].expand(ycoord.numel(), -1)
        Yg = ycoord[:, None].expand(-1, xcoord.numel())
        x_curve_at_Y = inverse_tanh_curve_torch(Yg, center=tanh_center_t, top=tanh_top_t, bottom=tanh_bottom_t, b=tanh_b_t)
        x_slab_start_t = torch.as_tensor(x_slab_start, device=device, dtype=dtype)
        x_slab_end_t = torch.as_tensor(x_slab_end, device=device, dtype=dtype)
        y_curve_start = tanh_curve_torch(x_slab_start_t, center=tanh_center_t, top=tanh_top_t, bottom=tanh_bottom_t, b=tanh_b_t)
        y_curve_end = tanh_curve_torch(x_slab_end_t, center=tanh_center_t, top=tanh_top_t, bottom=tanh_bottom_t, b=tanh_b_t)
        y_curve_min = torch.minimum(y_curve_start, y_curve_end)
        y_curve_max = torch.maximum(y_curve_start, y_curve_end)
        y_valid_max = torch.minimum(y_curve_max, torch.as_tensor(weakzone_y_cut, device=device, dtype=dtype))
        mask_y_range = torch.sigmoid((Yg - y_curve_min) / weakzone_smooth) * torch.sigmoid((y_valid_max - Yg) / weakzone_smooth)
        mask_curve_x_domain = torch.sigmoid((x_curve_at_Y - x_slab_start) / weakzone_smooth) * torch.sigmoid((x_slab_end - x_curve_at_Y) / weakzone_smooth)
        weak_mask_x_between = torch.sigmoid((Xg - x_curve_at_Y) / weakzone_smooth) * torch.sigmoid((x_curve_at_Y + weakzone_shift_x - Xg) / weakzone_smooth)
        weakzone_mask = mask_y_range * mask_curve_x_domain * weak_mask_x_between
        weakzone_mask = torch.clamp(weakzone_mask, min=0.0, max=1.0)
        return weakzone_mask
    xnode = torch.as_tensor(mesh.xnode, device=device, dtype=dtype)
    ynode = torch.as_tensor(mesh.ynode, device=device, dtype=dtype)
    weakzone_mask_node = build_horizontal_shift_weakzone_mask(xnode, ynode)
    xp_for_mask = torch.as_tensor(mesh.xp, device=device, dtype=dtype)
    yp_for_mask = torch.as_tensor(mesh.yp, device=device, dtype=dtype)
    weakzone_mask_p = build_horizontal_shift_weakzone_mask(xp_for_mask, yp_for_mask)
    weakzone_mask_node = torch.clamp(weakzone_mask_node, min=0.0, max=1.0)
    weakzone_mask_p = torch.clamp(weakzone_mask_p, min=0.0, max=1.0)
    T[0, :] = T_surface
    T[-1, :] = T_cmb
    T_bg[0, :] = T_surface
    T_bg[-1, :] = T_cmb
    return (T, T_bg, weakzone_mask_node, weakzone_mask_p)


def smooth_clip(x, x_min, x_max, beta=10.0):
    x1 = x_min + torch.nn.functional.softplus(beta * (x - x_min)) / beta
    x2 = x_max - torch.nn.functional.softplus(beta * (x_max - x1)) / beta
    return x2


def smooth_bound_eta_logspace(eta_raw, eta_min, eta_max, beta=10.0, floor=1e-300):
    eta_min_t = torch.as_tensor(eta_min, dtype=eta_raw.dtype, device=eta_raw.device)
    eta_max_t = torch.as_tensor(eta_max, dtype=eta_raw.dtype, device=eta_raw.device)

    log_eta = torch.log(torch.clamp(eta_raw, min=floor))
    log_min = torch.log(eta_min_t)
    log_max = torch.log(eta_max_t)

    x = log_min + torch.nn.functional.softplus(beta * (log_eta - log_min)) / beta
    y = log_max - torch.nn.functional.softplus(beta * (log_max - x)) / beta
    return torch.exp(y)


def mixed_eta_nd(
    strain_II_nd,
    tk_nd,
    mesh_state,
    A_param,
    n_param,
    E_param,
    V_param,
    pressure_nd=None,
    weakzone_mask=None,
):
    dtype_loc = tk_nd.dtype
    device_loc = tk_nd.device

    T = tk_nd * dT0 + T_surface
    epsII_phys = torch.clamp(strain_II_nd / t0, min=1e-20)

    A_loc = torch.as_tensor(A_param, device=device_loc, dtype=dtype_loc).reshape(-1)[0]
    n_loc = torch.as_tensor(n_param, device=device_loc, dtype=dtype_loc).reshape(-1)[0]
    E_loc = torch.as_tensor(E_param, device=device_loc, dtype=dtype_loc).reshape(-1)[0] * 1e5
    V_loc = torch.as_tensor(V_param, device=device_loc, dtype=dtype_loc).reshape(-1)[0]

    R = torch.tensor(8.31446261815324, dtype=dtype_loc, device=device_loc)
    T_ref = torch.tensor(Tmantle_phys, dtype=dtype_loc, device=device_loc)
    eta_ref = 10.0 ** A_loc

    if pressure_nd is None:
        expo_arg = (E_loc / (n_loc * R)) * (1.0 / T - 1.0 / T_ref)
    else:
        p_phys = pressure_nd * p0
        expo_arg = ((E_loc + p_phys * V_loc) / (n_loc * R)) * (1.0 / T - 1.0 / T_ref)

    expo_arg = smooth_clip(expo_arg, -50.0, 50.0, beta=10.0)

    eps_ref = torch.tensor(1e-15, dtype=dtype_loc, device=device_loc)
    eta_viscous = eta_ref * (epsII_phys / eps_ref) ** ((1.0 - n_loc) / n_loc) * torch.exp(expo_arg)

    sigma_y = torch.tensor(100e6, dtype=dtype_loc, device=device_loc)
    eta_yield_raw = sigma_y / (2.0 * epsII_phys)
    eta_yield = smooth_bound_eta_logspace(eta_yield_raw, eta_min=1e18, eta_max=1e24, beta=10.0)

    eta_eff_raw = 1.0 / (
        1.0 / torch.clamp(eta_viscous, min=1e-300)
        + 1.0 / torch.clamp(eta_yield, min=1e-300)
    )
    eta = smooth_bound_eta_logspace(eta_eff_raw, eta_min=1e18, eta_max=1e24, beta=10.0)

    if weakzone_mask is not None:
        weakzone_mask = weakzone_mask.to(device=device_loc, dtype=dtype_loc)
        if weakzone_mask.shape != eta.shape:
            raise ValueError(
                f"weakzone_mask shape {tuple(weakzone_mask.shape)} does not match eta shape {tuple(eta.shape)}"
            )
        weakzone_mask = torch.clamp(weakzone_mask, min=0.0, max=1.0)
        log_eta = torch.log(torch.clamp(eta, min=1e-300))
        log_eta_weak = torch.log(torch.tensor(1e18, dtype=dtype_loc, device=device_loc))
        eta = torch.exp((1.0 - weakzone_mask) * log_eta + weakzone_mask * log_eta_weak)

    eta = torch.clamp(eta, min=1e12, max=1e30)
    return eta / eta0


def viscosity_update_fn_grid(mesh_state, strain_II_p, strain_II_node, pressure_ext, tkp_nd, ctx):
    tk_node_nd = ad.interpolation.interp_between_grids(mesh_state, tkp_nd, src="p", tgt="node")

    eta_node = mixed_eta_nd(
        strain_II_node,
        tk_node_nd,
        mesh_state,
        ctx["A_param"],
        ctx["n_param"],
        ctx["E_param"],
        ctx["V_param"],
        pressure_nd=None,
        weakzone_mask=ctx["weakzone_mask_node"],
    )

    eta_p = mixed_eta_nd(
        strain_II_p,
        tkp_nd,
        mesh_state,
        ctx["A_param"],
        ctx["n_param"],
        ctx["E_param"],
        ctx["V_param"],
        pressure_nd=None,
        weakzone_mask=ctx["weakzone_mask_p"],
    )

    return eta_node, eta_p


tkp0_base, T_bg, weakzone_mask_node, weakzone_mask_p = half_space_cooling_field_ridge_slab(
    mesh,
    ridge_right_width=200e3,
    ridge_age_max=40e6,
    x_trench=550e3,
    slab_length=600e3,
    slab_thickness=100e3,
    age_background=40e6,
    age_slab=40e6,
    use_upper_half_tanh=True,
    target_slab_depth=300e3,
    tanh_top=-10e3,
    tanh_b_factor=3.0,
    T_surface=T_surface,
    T_mantle=Tmantle_phys,
    T_cmb=Tbot_phys,
)


def _store_tensor(x, store_on_cpu=True):
    y = x.clone()
    return y.cpu() if store_on_cpu else y


def forward_fn(
    num_steps,
    tkp_init,
    composition_field_init=None,
    A_param=A_param,
    n_param=n_param,
    E_param=E_param,
    rho_param=rho_param,
    dt=dt,
    seed=SEED,
    store_on_cpu=True,
):
    torch.manual_seed(seed)

    if tkp_init is None:
        raise ValueError("tkp_init must be provided")

    if tkp_init.abs().max() > 1000.0:
        tkp = (tkp_init - T_surface) / dT0
    else:
        tkp = tkp_init.clone()

    if composition_field_init is None:
        composition_field = torch.zeros_like(tkp)
    else:
        composition_field = composition_field_init.clone().to(device=tkp.device, dtype=tkp.dtype)

    if composition_field.shape != tkp.shape:
        raise ValueError(
            f"composition_field_init shape {tuple(composition_field.shape)} must match tkp shape {tuple(tkp.shape)}"
        )

    composition_field = torch.clamp(composition_field, 0.0, 1.0)

    dt_nd = dt / t0
    sumt = torch.zeros((), device=device, dtype=dtype)
    u0 = None

    all_temperature = []
    all_surface_vx = []
    all_surface_sigmayy = []
    all_viscosity_p = []
    for timestep in range(num_steps):

        alpha_p = torch.ones_like(tkp) * alpha[0]
        if rho_param.numel() < 2:
            raise ValueError("rho_param must contain two values")

        rho_p = rho_param[0] + composition_field * (rho_param[1] - rho_param[0])
        rhoT_p = rho_p * (1.0 - alpha_p * (tkp * dT0  + T_surface - 273.0 ))
        buoyancy_p = rhoT_p * L0 ** 3 / (eta0 * kappa0) * g_phys

        density_x = ad.interpolation.interp_between_grids(mesh_state, buoyancy_p, src="p", tgt="vx")
        density_y = ad.interpolation.interp_between_grids(mesh_state, buoyancy_p, src="p", tgt="vy")

        rheology_context = {
            "A_param": A_param,
            "n_param": n_param,
            "E_param": E_param,
            "V_param": V_disl,
            "weakzone_mask_node": weakzone_mask_node,
            "weakzone_mask_p": weakzone_mask_p,
            "epsII_ref_nd": 1e-15 * t0,
        }

        stokes_state = ad.stokes.StokesSystem.implicit_solve_state(
            mesh_state=mesh_state,
            density_x=density_x,
            density_y=density_y,
            tkp=tkp,
            gx=gx,
            gy=gy,
            dt=dt_nd,
            pscale=pscale,
            boundary_const=stokes_boundary_type,
            timestep=timestep,
            viscosity_update_fn=viscosity_update_fn_grid,
            rheology_context=rheology_context,
            p_fix_value=p_fix_value,
            picard_steps=20,
            rel_tol=1e-8,
            newton_steps=20,
            is_stick_air=False,
            u0=u0,
            coloring_pack=coloring_pack,
        )

        u0 = stokes_state["u"]
        vx = stokes_state["vx"]
        vy = stokes_state["vy"]
        sigma_yy_surface = stokes_state["sigma_yy_surface"]
        viscosity_p = stokes_state["viscosity_p"]

        kp_p = torch.ones_like(tkp) * k_thermal[0] / kappa0
        rhocp_p = torch.ones_like(tkp) * rhocp[0]

        thermal_out = ad.thermal.ThermalSystemGrid.advance_temperature_strang(
            T_old=tkp,
            vx=vx,
            vy=vy,
            KX=kp_p,
            KY=kp_p,
            RHOCP=rhocp_p,
            dt=dt_nd,
            mesh_state=mesh_state,
            timestep=timestep,
            thermal_boundary_dict=temperature_bc,
            HSUM=None,
            edge_mode="nearest",
            constant_value=0.0,
            backtrace="rk2",
            recompute_coeff_second_half=False,
        )
        tkp_new = thermal_out["T"]

        composition_field_new = ad.advection.advect_scalar_sl_on_p_bfecc(
            q_old=composition_field,
            vx=vx,
            vy=vy,
            dt=dt_nd,
            mesh_state=mesh_state,
            edge_mode="nearest",
            constant_value=0.0,
            backtrace="rk2",
            scalar_boundary_dict=composition_bc,
            enforce_bc_after_advect=True,
            use_dirichlet_value_for_oob=False,
            scalar_interp_method="linear",
            velocity_interp_method="linear",
        )
        composition_field_new = torch.clamp(composition_field_new, 0.0, 1.0)

        vx_node = ad.interpolation.interp_between_grids(mesh_state, vx, src="vx", tgt="node")
        surface_vx = vx_node[0, :]

        all_temperature.append(_store_tensor(tkp_new, store_on_cpu))
        all_surface_vx.append(_store_tensor(surface_vx, store_on_cpu))
        all_surface_sigmayy.append(_store_tensor(sigma_yy_surface, store_on_cpu))
        all_viscosity_p.append(_store_tensor(viscosity_p, store_on_cpu))
        sumt = sumt + dt_nd
        tkp = tkp_new
        composition_field = composition_field_new

        if time_total is not None:
            target_time = time_total.to(device=device, dtype=dtype) if torch.is_tensor(time_total) else torch.tensor(time_total, device=device, dtype=dtype)
            remain = target_time - sumt
            if remain <= 0:
                break
            if dt_nd > remain:
                dt_nd = remain

    return {
        "tkp_end_out": _store_tensor(tkp, store_on_cpu),
        "mesh_state":mesh_state,
        "all_temperature": all_temperature,
        "all_surface_vx": all_surface_vx,
        "all_surface_sigmayy": all_surface_sigmayy,
        "all_surface_traction": all_surface_sigmayy,
        "all_viscosity_p": all_viscosity_p,
        "actual_num_steps": len(all_temperature),
    }


def save_forward_output(output, filename="observe_data.pt"):
    path = Path(filename)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"data": output}, path)
    print(f"saved: {path}")



def forward_main(
    tkp_init=None,
    filename="observe_data.pt",
    composition_field_init=None,
):
    print("Torch threads:", torch.get_num_threads())
    print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
    print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))

    if tkp_init is None:
        tkp_init = tkp0_base

    if composition_field_init is None:
        composition_field_init = torch.zeros_like(tkp_init)

    with torch.no_grad():
        output = forward_fn(
            num_steps=num_steps,
            tkp_init=tkp_init,
            composition_field_init=composition_field_init,
            A_param=A_param,
            n_param=n_param,
            E_param=E_param,
            rho_param=rho_param,
            dt=dt,
            store_on_cpu=True,
        )

    save_forward_output(output, filename=filename)
    return output



def get_grid_coords(mesh_obj, grid_kind: str):
    if grid_kind in ("p", "xp"):
        return mesh_obj.xp, mesh_obj.yp
    if grid_kind == "node":
        return mesh_obj.xnode, mesh_obj.ynode

    raise ValueError(f"Unsupported grid_kind: {grid_kind}")


def _reduce_loss(x: torch.Tensor, reduction: str):
    if reduction == "mean":
        return x.mean() if x.numel() > 0 else torch.zeros((), device=x.device, dtype=x.dtype)
    if reduction == "sum":
        return x.sum() if x.numel() > 0 else torch.zeros((), device=x.device, dtype=x.dtype)

    raise ValueError(f"Unsupported reduction: {reduction}")


def smoothness_regularization(
    field: torch.Tensor,
    mesh_obj,
    grid_kind: str = "p",
    order: int = 1,
    remove_mean: bool = False,
    reduction: str = "mean",
    eps: float = 1e-30,
):
    if field.ndim != 2:
        raise ValueError(f"Expected a 2D field, got shape={tuple(field.shape)}")

    x, y = get_grid_coords(mesh_obj, grid_kind)
    x = x.to(device=field.device, dtype=field.dtype) / L0
    y = y.to(device=field.device, dtype=field.dtype) / L0

    f = field - field.mean() if remove_mean else field

    if order == 1:
        dx = (x[1:] - x[:-1]).clamp_min(eps)
        dy = (y[1:] - y[:-1]).clamp_min(eps)

        dfdx = (f[:, 1:] - f[:, :-1]) / dx[None, :]
        dfdy = (f[1:, :] - f[:-1, :]) / dy[:, None]

        reg_x = _reduce_loss(dfdx.square(), reduction)
        reg_y = _reduce_loss(dfdy.square(), reduction)

    elif order == 2:
        dx_l = (x[1:-1] - x[:-2]).clamp_min(eps)
        dx_r = (x[2:] - x[1:-1]).clamp_min(eps)
        dy_l = (y[1:-1] - y[:-2]).clamp_min(eps)
        dy_r = (y[2:] - y[1:-1]).clamp_min(eps)

        d2fdx2 = 2.0 * (
            (f[:, 2:] - f[:, 1:-1]) / dx_r[None, :]
            - (f[:, 1:-1] - f[:, :-2]) / dx_l[None, :]
        ) / (dx_l + dx_r)[None, :]

        d2fdy2 = 2.0 * (
            (f[2:, :] - f[1:-1, :]) / dy_r[:, None]
            - (f[1:-1, :] - f[:-2, :]) / dy_l[:, None]
        ) / (dy_l + dy_r)[:, None]

        reg_x = _reduce_loss(d2fdx2.square(), reduction)
        reg_y = _reduce_loss(d2fdy2.square(), reduction)

    else:
        raise ValueError(f"order must be 1 or 2, got {order}")

    return 0.5 * reg_x + 0.5 * reg_y


def as_time_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if torch.is_tensor(x):
        return list(x)
    raise TypeError(f"Unsupported time series type: {type(x)}")


def get_final_temperature(out):
    if "tkp_end_out" in out:
        return out["tkp_end_out"]

    if "all_temperature" in out and len(out["all_temperature"]) > 0:
        return out["all_temperature"][-1]

    raise KeyError("Forward output must contain 'tkp_end_out' or non-empty 'all_temperature'.")


def get_final_composition(out):
    if "c2_end_out" in out:
        return out["c2_end_out"]
    if "composition_field_end_out" in out:
        return out["composition_field_end_out"]
    if "all_composition_field" in out and len(out["all_composition_field"]) > 0:
        return out["all_composition_field"][-1]
    return None


def get_surface_series(out, key_main: str, key_fallback: str = None):
    if key_main in out:
        return as_time_list(out[key_main])

    if key_fallback is not None and key_fallback in out:
        return as_time_list(out[key_fallback])

    raise KeyError(f"Missing surface series: {key_main}")


def surface_line(x: torch.Tensor):
    if isinstance(x, (tuple, list)):
        x = x[0]

    if x.ndim == 2:
        x = x[0, :]

    return x.reshape(-1)


def match_and_crop_1d(a, b, crop: int = 3):
    a = surface_line(a)
    b = surface_line(b)

    if crop > 0:
        a = a[crop:-crop]
        b = b[crop:-crop]

    n = min(a.numel(), b.numel())
    return a[:n], b[:n]


def normalized_series_mse(pred_series, obs_series, crop: int = 3, eps: float = 1e-30):
    nstep = min(len(pred_series), len(obs_series))
    if nstep == 0:
        raise ValueError("Empty time series in normalized_series_mse.")

    device = pred_series[0].device
    dtype = pred_series[0].dtype

    sq_err = torch.zeros((), device=device, dtype=dtype)
    energy = torch.zeros((), device=device, dtype=dtype)

    for k in range(nstep):
        pred = pred_series[k]
        obs = obs_series[k].to(device=pred.device, dtype=pred.dtype)

        pred, obs = match_and_crop_1d(pred, obs, crop=crop)

        sq_err = sq_err + torch.mean((pred - obs) ** 2)
        energy = energy + torch.mean(obs ** 2)

    sq_err = sq_err / nstep
    energy = energy / nstep

    return sq_err / (energy + eps)


def normalized_field_mse(pred, obs, ref=None, eps: float = 1e-30):
    obs = obs.to(device=pred.device, dtype=pred.dtype)

    if ref is not None:
        ref = ref.to(device=pred.device, dtype=pred.dtype).detach()
        pred = pred - ref
        obs = obs - ref

    return torch.mean((pred - obs) ** 2) / (torch.mean(obs ** 2) + eps)


def build_loss_terms(
    ob,
    num_steps,
    tkp0,
    composition_field_init=None,
    tkp0_prior=None,
    A_param=None,
    n_param=None,
    E_param=None,
    rho_param=None,
    dt=None,
    w_T: float = 1.0,
    w_vx: float = 0.1,
    w_sigmayy: float = 0.0,
    w_C: float = 0.0,
    reg_weight_tkp: float = 0.0,
    reg_tkp_order: int = 1,
    crop_surface: int = 3,
):
    if tkp0 is None:
        raise ValueError("tkp0 must be provided.")

    if A_param is None:
        A_param = globals()["A_param"]
    if n_param is None:
        n_param = globals()["n_param"]
    if E_param is None:
        E_param = globals()["E_param"]
    if rho_param is None:
        rho_param = globals()["rho_param"]
    if dt is None:
        dt = globals()["dt"]

    pred = forward_fn(
        num_steps=num_steps,
        tkp_init=tkp0,
        composition_field_init=composition_field_init,
        A_param=A_param,
        n_param=n_param,
        E_param=E_param,
        rho_param=rho_param,
        dt=dt,
        store_on_cpu=False,
    )

    device = tkp0.device
    dtype = tkp0.dtype
    zero = torch.zeros((), device=device, dtype=dtype)

    T_pred = get_final_temperature(pred)
    T_obs = get_final_temperature(ob)

    loss_T = normalized_field_mse(
        pred=T_pred,
        obs=T_obs,
        ref=tkp0_prior,
    )

    vx_pred = get_surface_series(pred, "all_surface_vx")
    vx_obs = get_surface_series(ob, "all_surface_vx")

    loss_vx = normalized_series_mse(
        pred_series=vx_pred,
        obs_series=vx_obs,
        crop=crop_surface,
    )

    loss_sigmayy = zero.clone()
    if w_sigmayy != 0.0:
        sig_pred = get_surface_series(
            pred,
            key_main="all_surface_sigmayy",
            key_fallback="all_surface_traction",
        )
        sig_obs = get_surface_series(
            ob,
            key_main="all_surface_sigmayy",
            key_fallback="all_surface_traction",
        )

        loss_sigmayy = normalized_series_mse(
            pred_series=sig_pred,
            obs_series=sig_obs,
            crop=crop_surface,
        )

    loss_C = zero.clone()
    if w_C != 0.0:
        C_pred = get_final_composition(pred)
        C_obs = get_final_composition(ob)

        if C_pred is None or C_obs is None:
            raise KeyError("Composition loss is enabled, but final composition is missing.")

        loss_C = normalized_field_mse(
            pred=C_pred,
            obs=C_obs,
            ref=None,
        )

    reg_tkp = zero.clone()
    if reg_weight_tkp != 0.0:
        if tkp0_prior is None:
            raise ValueError("tkp0_prior is required when reg_weight_tkp is nonzero.")

        reg_field = tkp0 - tkp0_prior.to(device=tkp0.device, dtype=tkp0.dtype)

        reg_tkp = smoothness_regularization(
            field=reg_field,
            mesh_obj=mesh,
            grid_kind="p",
            order=reg_tkp_order,
            remove_mean=False,
            reduction="mean",
        )

    data_loss = (
        w_T * loss_T
        + w_vx * loss_vx
        + w_sigmayy * loss_sigmayy
        + w_C * loss_C
    )

    total_loss = data_loss + reg_weight_tkp * reg_tkp

    return {
        "total_loss": total_loss,
        "data_loss": data_loss,
        "loss_T": loss_T,
        "loss_vx": loss_vx,
        "loss_sigmayy": loss_sigmayy,
        "loss_C": loss_C,
        "reg_tkp": reg_tkp,
        "forward_out": pred,
    }


def _as_scalar_loss(loss_out, loss_key="total_loss"):
    if torch.is_tensor(loss_out):
        return loss_out

    if isinstance(loss_out, dict):
        if loss_key not in loss_out:
            raise KeyError(f"loss_key='{loss_key}' is not found in loss_out.")
        return loss_out[loss_key]

    raise TypeError(f"Unsupported loss output type: {type(loss_out)}")


def run_taylor_test(
    *,
    param_name: str,
    param_base: torch.Tensor,
    make_loss,
    loss_key: str = "total_loss",
    eps_list=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
    direction="random",
    direction_mask=None,
    seed: int = 0,
    do_plot: bool = True,
    save_path: str | None = None,
    show_plot: bool = False,
):

    if param_base is None:
        raise ValueError(f"{param_name}: param_base is None.")

    p0 = param_base.detach().clone().requires_grad_(True)

    loss0 = _as_scalar_loss(make_loss(p0), loss_key=loss_key)

    if not torch.is_tensor(loss0):
        raise TypeError("Taylor test loss must be a torch scalar tensor.")

    if not loss0.requires_grad:
        raise RuntimeError(
            f"{param_name}: loss does not require grad. "
            "Check whether the forward output was detached."
        )

    grad0 = torch.autograd.grad(
        loss0,
        p0,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]

    if grad0 is None:
        raise RuntimeError(f"{param_name}: gradient is None.")

    with torch.no_grad():
        if isinstance(direction, str):
            if direction == "random":
                torch.manual_seed(seed)
                d = torch.randn_like(p0)
            elif direction == "grad":
                d = grad0.detach().clone()
            else:
                raise ValueError("direction must be 'random', 'grad', or a tensor.")
        elif torch.is_tensor(direction):
            d = direction.detach().clone().to(device=p0.device, dtype=p0.dtype)
        else:
            raise TypeError("direction must be 'random', 'grad', or a tensor.")

        if direction_mask is not None:
            mask = direction_mask.to(device=p0.device, dtype=p0.dtype)
            d = d * mask

        d_norm = d.norm()
        if d_norm.item() == 0.0:
            raise RuntimeError(f"{param_name}: perturbation direction has zero norm.")

        d = d / (d_norm + 1e-30)

    J0 = float(loss0.detach().cpu())
    g_dot_d = float(torch.sum(grad0.detach() * d).cpu())
    grad_norm = float(grad0.detach().norm().cpu())

    print(f"\n[TaylorTest] param = {param_name}, loss = {loss_key}")
    print(f"[TaylorTest] J0 = {J0:.8e}")
    print(f"[TaylorTest] ||grad|| = {grad_norm:.8e}")
    print(f"[TaylorTest] grad dot direction = {g_dot_d:.8e}")

    results = {
        "param_name": param_name,
        "loss_key": loss_key,
        "J0": J0,
        "grad_norm": grad_norm,
        "g_dot_d": g_dot_d,
        "eps": [],
        "J_eps": [],
        "R0": [],
        "R1": [],
        "rate_R0": [],
        "rate_R1": [],
    }

    prev_eps = None
    prev_R0 = None
    prev_R1 = None

    for eps in eps_list:
        eps = float(eps)

        with torch.no_grad():
            p_eps = p0.detach() + eps * d

        J_eps_tensor = _as_scalar_loss(
            make_loss(p_eps),
            loss_key=loss_key,
        )

        J_eps = float(J_eps_tensor.detach().cpu())

        R0 = abs(J_eps - J0)
        R1 = abs(J_eps - J0 - eps * g_dot_d)

        rate_R0 = float("nan")
        rate_R1 = float("nan")

        if prev_eps is not None:
            if R0 > 0.0 and prev_R0 > 0.0:
                rate_R0 = math.log(prev_R0 / R0) / math.log(prev_eps / eps)
            if R1 > 0.0 and prev_R1 > 0.0:
                rate_R1 = math.log(prev_R1 / R1) / math.log(prev_eps / eps)

        print(
            f"  eps={eps:.1e} | "
            f"J_eps={J_eps:.8e} | "
            f"R0={R0:.8e} | "
            f"R1={R1:.8e} | "
            f"rate_R0={rate_R0:.3f} | "
            f"rate_R1={rate_R1:.3f}"
        )

        results["eps"].append(eps)
        results["J_eps"].append(J_eps)
        results["R0"].append(R0)
        results["R1"].append(R1)
        results["rate_R0"].append(rate_R0)
        results["rate_R1"].append(rate_R1)

        prev_eps = eps
        prev_R0 = R0
        prev_R1 = R1

    if do_plot:
        import matplotlib.pyplot as plt

        eps_arr = results["eps"]
        R0_arr = results["R0"]
        R1_arr = results["R1"]

        ref_R0 = []
        ref_R1 = []

        if len(eps_arr) > 0:
            eps_ref = eps_arr[0]
            R0_ref = R0_arr[0] if R0_arr[0] > 0.0 else 1e-30
            R1_ref = R1_arr[0] if R1_arr[0] > 0.0 else 1e-30

            for eps in eps_arr:
                ref_R0.append(R0_ref * (eps / eps_ref))
                ref_R1.append(R1_ref * (eps / eps_ref) ** 2)

        plt.figure(figsize=(6, 5))
        plt.loglog(eps_arr, R0_arr, "o-", label="R0")
        plt.loglog(eps_arr, R1_arr, "s-", label="R1")
        plt.loglog(eps_arr, ref_R0, "--", label="O(h)")
        plt.loglog(eps_arr, ref_R1, "--", label="O(h²)")
        plt.xlabel("h")
        plt.ylabel("Taylor remainder")
        plt.title(f"Taylor test: {param_name}")
        plt.grid(True, which="both", linestyle=":")
        plt.legend()
        plt.tight_layout()

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[TaylorTest] saved figure: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    return results

def taylor_test_main(
    observation_path="observe_data.pt",
    save_dir="taylor_test",
    *,
    eps_list=None,
    do_plot=True,
):
    os.makedirs(save_dir, exist_ok=True)

    ob = load_observation(observation_path)

    tkp0_ref, tkp_bg, composition_field_init = build_inversion_initial_fields()
    interior_mask = build_interior_mask(tkp_bg)

    n_interior = int(interior_mask.sum().item())
    dT_base = torch.zeros(n_interior, dtype=dtype, device=device)

    num_steps_test = int(ob.get("actual_num_steps", num_steps))

    rho_base = rho_param.detach().clone().to(device=device, dtype=dtype)
    A_base = A_param.detach().clone().to(device=device, dtype=dtype)
    n_base = n_param.detach().clone().to(device=device, dtype=dtype)

    w_T_test = 1.0
    w_vx_test = 0.1
    w_sigmayy_test = 0.0
    w_C_test = 0.0
    reg_weight_tkp_test = 0.0
    bound_weight_test = 0.0

    def make_loss_dT(dT_test):
        tkp0_test = compose_tkp0_from_dT(
            dT_interior=dT_test,
            tkp_bg=tkp_bg,
            tkp0_ref=tkp0_ref,
            interior_mask=interior_mask,
        )

        terms = build_loss_terms(
            ob=ob,
            num_steps=num_steps_test,
            tkp0=tkp0_test,
            composition_field_init=composition_field_init,
            tkp0_prior=tkp_bg,
            A_param=A_base,
            n_param=n_base,
            E_param=E_param,
            rho_param=rho_base,
            dt=dt,
            w_T=w_T_test,
            w_vx=w_vx_test,
            w_sigmayy=w_sigmayy_test,
            w_C=w_C_test,
            reg_weight_tkp=reg_weight_tkp_test,
            reg_tkp_order=1,
            crop_surface=3,
        )

        loss_T_bound = torch.mean(
            torch.relu(-tkp0_test).square()
            + torch.relu(tkp0_test - 1.0).square()
        )

        total_loss = terms["total_loss"] + bound_weight_test * loss_T_bound

        return {
            **terms,
            "total_loss": total_loss,
            "loss_T_bound": loss_T_bound,
        }

    if eps_list is None:
        eps_list = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)

    results = run_taylor_test(
        param_name="dT_param",
        param_base=dT_base,
        make_loss=make_loss_dT,
        loss_key="total_loss",
        direction="random",
        direction_mask=None,
        seed=0,
        eps_list=eps_list,
        do_plot=do_plot,
        save_path=os.path.join(save_dir, "taylor_dT_param.png"),
        show_plot=False,
    )

    torch.save(results, os.path.join(save_dir, "taylor_dT_param.pt"))
    print(f"[TaylorTest] saved data: {os.path.join(save_dir, 'taylor_dT_param.pt')}")

    return results


def to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_cpu(v) for v in x]
    if isinstance(x, tuple):
        return tuple(to_cpu(v) for v in x)
    return x


def print_tensor_stats(name, x):
    x = x.detach()
    print(
        f"{name}: shape={tuple(x.shape)}, "
        f"mean={x.mean().item():.6e}, "
        f"std={x.std().item():.6e}, "
        f"min={x.min().item():.6e}, "
        f"max={x.max().item():.6e}"
    )


def print_field_error(name, pred, ref):
    pred = pred.detach()
    ref = ref.detach().to(device=pred.device, dtype=pred.dtype)

    diff = pred - ref
    mae = diff.abs().mean().item()
    rmse = torch.sqrt(torch.mean(diff.square())).item()
    bias = diff.mean().item()
    maxae = diff.abs().max().item()

    print(
        f"{name}: "
        f"MAE={mae:.6e}, RMSE={rmse:.6e}, "
        f"Bias={bias:.6e}, MaxAE={maxae:.6e}"
    )


def load_observation(filename="observe_data.pt"):
    payload = torch.load(filename, map_location=device)
    return payload["data"]


def build_inversion_initial_fields():
    tkp0_ref = ((tkp0_base - T_surface) / dT0).to(device=device, dtype=dtype)

    tkp_bg = T_bg.detach().clone().to(device=device, dtype=dtype)
    tkp_bg = (tkp_bg - T_surface) / dT0

    tkp_bg[:, 0] = tkp_bg[:, 1]
    tkp_bg[:, -1] = tkp_bg[:, -2]

    tkp_bg[0, :] = tkp0_ref[0, :]
    tkp_bg[-1, :] = tkp0_ref[-1, :]

    composition_field_init = torch.zeros_like(tkp0_ref)

    return tkp0_ref, tkp_bg, composition_field_init


def build_interior_mask(field):
    mask = torch.zeros_like(field, dtype=torch.bool, device=field.device)
    mask[1:-1, 1:-1] = True
    return mask


def compose_tkp0_from_dT(dT_interior, tkp_bg, tkp0_ref, interior_mask):
    tkp0 = tkp_bg.clone()
    tkp0[interior_mask] = tkp0[interior_mask] + dT_interior

    tkp0[0, :] = tkp0_ref[0, :]
    tkp0[-1, :] = tkp0_ref[-1, :]

    tkp0[:, 0] = tkp0[:, 1]
    tkp0[:, -1] = tkp0[:, -2]

    return tkp0


def build_temperature_preconditioner(
    dT_param,
    interior_mask,
    R_smooth_km=75.0,
):
    mesh_state_pre = mesh.to_mesh_state()
    mesh_state_pre = ad.mesh.get_nondimensional_mesh_state(mesh_state_pre, L0)

    Kpre, shape_pre = ad.optimization.build_Kfd_sparse_csr_from_mesh_state(
        mesh_state_pre,
        grid_type="p",
        bc_mode="identity",
    )

    Kpre = Kpre.to(device=device, dtype=dtype)

    Ny_p, Nx_p = shape_pre
    numel = Ny_p * Nx_p

    if Kpre.shape[0] != numel:
        raise RuntimeError("Preconditioner matrix size is inconsistent with p-grid size.")

    interior_mask_1d = interior_mask.reshape(-1)

    if dT_param.numel() != int(interior_mask_1d.sum().item()):
        raise RuntimeError("dT_param size is inconsistent with interior_mask.")

    alpha_pre = (R_smooth_km / (float(L0) / 1e3)) ** 2

    I_csr = torch.sparse_csr_tensor(
        crow_indices=torch.arange(numel + 1, device=device, dtype=torch.long),
        col_indices=torch.arange(numel, device=device, dtype=torch.long),
        values=torch.ones(numel, device=device, dtype=dtype),
        size=(numel, numel),
        device=device,
        dtype=dtype,
    )

    M_csr = I_csr + torch.sparse_csr_tensor(
        Kpre.crow_indices(),
        Kpre.col_indices(),
        Kpre.values() * alpha_pre,
        size=(numel, numel),
        device=device,
        dtype=dtype,
    )

    def apply_temperature_precond(M_matrix, g_interior):
        if g_interior.ndim != 1:
            raise ValueError(f"Expected 1D gradient, got {tuple(g_interior.shape)}")

        g_full = torch.zeros(numel, device=g_interior.device, dtype=g_interior.dtype)
        g_full[interior_mask_1d] = g_interior

        s_full = ad.solvers.sparse_solve(
            M_matrix,
            g_full,
            matrix_kind="Kpre_dT",
            reuse_mode="numeric_fixed",
        )

        return s_full[interior_mask_1d].reshape_as(g_interior)

    return M_csr, apply_temperature_precond


def pack_lbfgs_history(optimizer):
    if not hasattr(optimizer, "history"):
        return []

    packed = []
    for rec in optimizer.history:
        item = {}

        if "n_iter" in rec:
            item["n_iter"] = int(rec["n_iter"])
        if "loss" in rec:
            item["loss"] = float(rec["loss"])
        if "params" in rec:
            item["params"] = [
                p.detach().cpu().clone() if torch.is_tensor(p) else p
                for p in rec["params"]
            ]

        packed.append(item)

    return packed


def inversion_main(
    observation_path="observe_data.pt",
    save_path="tkp_inversion_summary.pt",
    *,
    invert_rho2=True,
    invert_A=True,
    invert_n=True,
    use_preconditioner=True,
    R_smooth_km=75.0,
    max_iter=200,
    max_eval=400,
):

    print("[INVERSION] Load observation")
    ob = load_observation(observation_path)
    print("[INVERSION] Observation keys:", ob.keys())

    tkp0_ref, tkp_bg, composition_field_init = build_inversion_initial_fields()
    interior_mask = build_interior_mask(tkp_bg)

    n_interior = int(interior_mask.sum().item())
    num_steps_inv = int(ob.get("actual_num_steps", num_steps))

    print(f"[INVERSION] n_interior = {n_interior}")
    print(f"[INVERSION] num_steps_inv = {num_steps_inv}")

    dT_param = torch.nn.Parameter(
        torch.zeros(n_interior, dtype=dtype, device=device)
    )

    rho2_init = rho_param[1].detach().clone() if not invert_rho2 else torch.tensor(3300.0, dtype=dtype, device=device)
    A_init = A_param[0].detach().clone() if not invert_A else torch.tensor(22.0, dtype=dtype, device=device)
    n_init = n_param[0].detach().clone() if not invert_n else torch.tensor(3.5, dtype=dtype, device=device)

    rho0_fixed = rho_param[0].detach().clone()
    rho_scale = torch.tensor(50.0, dtype=dtype, device=device)

    rho2_param_nd = torch.nn.Parameter(
        ((rho2_init - rho0_fixed) / rho_scale).reshape(1)
    )

    A_param_inv = torch.nn.Parameter(A_init.reshape(1))

    n_param_inv = torch.nn.Parameter(n_init.reshape(1))

    if not invert_rho2:
        rho2_param_nd.requires_grad_(False)
    if not invert_A:
        A_param_inv.requires_grad_(False)
    if not invert_n:
        n_param_inv.requires_grad_(False)

    def compose_rho_param():
        rho2 = rho0_fixed + rho_scale * rho2_param_nd.reshape(())
        return torch.stack([rho0_fixed, rho2])

    def compose_A_param():
        return A_param_inv.reshape(1)

    def compose_n_param():
        return n_param_inv.reshape(1)

    print_tensor_stats("dT_param(init)", dT_param)
    print(f"[INVERSION] rho2_init = {compose_rho_param()[1].detach().item():.6e}")
    print(f"[INVERSION] A_init    = {compose_A_param()[0].detach().item():.6e}")
    print(f"[INVERSION] n_init    = {compose_n_param()[0].detach().item():.6e}")

    params = [dT_param]
    block_specs = [("dT", dT_param.numel())]
    scales = {}

    if invert_rho2:
        params.append(rho2_param_nd)
        block_specs.append(("rho2", rho2_param_nd.numel()))
        scales["rho2"] = {"alpha": 1.0}

    if invert_A:
        params.append(A_param_inv)
        block_specs.append(("A", A_param_inv.numel()))
        scales["A"] = {"alpha": 1.0}

    if invert_n:
        params.append(n_param_inv)
        block_specs.append(("n", n_param_inv.numel()))
        scales["n"] = {"alpha": 1.0}

    precond_fn = None

    if use_preconditioner:
        M_csr_dT, Kapply_dT = build_temperature_preconditioner(
            dT_param=dT_param,
            interior_mask=interior_mask,
            R_smooth_km=R_smooth_km,
        )

        scales["dT"] = {
            "alpha": 1.0,
            "K": M_csr_dT,
            "apply": Kapply_dT,
        }

        precond_fn = ad.optimization.make_block_diag_precond(
            block_specs=block_specs,
            scales=scales,
        )

        with torch.no_grad():
            g_test = torch.cat([torch.randn_like(p).reshape(-1) for p in params])
            s_test = precond_fn(g_test)
            dot_g_Rg = torch.dot(g_test, s_test).item()

            print(f"[PRECOND] R_smooth_km = {R_smooth_km}")
            print(f"[PRECOND] dot(g, Rg) = {dot_g_Rg:.6e}")

            if dot_g_Rg <= 0.0:
                raise RuntimeError("Preconditioner is not positive definite.")

    optimizer = ad.optimization.MyLBFGS(
        params,
        lr=1.0,
        max_iter=max_iter,
        max_eval=max_eval,
        tolerance_grad=1e-10,
        tolerance_change=1e-11,
        history_size=200,
        line_search_fn="strong_wolfe",
        precond_fn=precond_fn,
    )

    w_T = 1.0
    w_vx = 0.1
    w_sigmayy = 0.0
    w_C = 0.0
    reg_weight_tkp = 1e-2
    bound_weight = 1e3

    print(
        "[INVERSION] weights: "
        f"w_T={w_T}, w_vx={w_vx}, w_sigmayy={w_sigmayy}, "
        f"w_C={w_C}, reg_weight_tkp={reg_weight_tkp}, "
        f"bound_weight={bound_weight}"
    )

    with torch.no_grad():
        ref_terms = build_loss_terms(
            ob=ob,
            num_steps=num_steps_inv,
            tkp0=tkp0_ref,
            composition_field_init=composition_field_init,
            tkp0_prior=tkp_bg,
            A_param=A_param,
            n_param=n_param,
            E_param=E_param,
            rho_param=rho_param,
            dt=dt,
            w_T=w_T,
            w_vx=w_vx,
            w_sigmayy=w_sigmayy,
            w_C=w_C,
            reg_weight_tkp=reg_weight_tkp,
            reg_tkp_order=1,
            crop_surface=3,
        )

        print("[INVERSION] reference loss terms:")
        print(f"  {'total_loss':14s} = {ref_terms['total_loss'].detach().item():.6e}")
        print(f"  {'data_loss':14s} = {ref_terms['data_loss'].detach().item():.6e}")
        print(f"  {'loss_T':14s} = {ref_terms['loss_T'].detach().item():.6e}")
        print(f"  {'loss_vx':14s} = {ref_terms['loss_vx'].detach().item():.6e}")
        print(f"  {'loss_sigmayy':14s} = {ref_terms['loss_sigmayy'].detach().item():.6e}")
        print(f"  {'loss_C':14s} = {ref_terms['loss_C'].detach().item():.6e}")
        print(f"  {'reg_tkp':14s} = {ref_terms['reg_tkp'].detach().item():.6e}")
        print(f"[INVERSION] true rho_param = {rho_param.detach().cpu().numpy()}")
        print(f"[INVERSION] true A_param = {A_param.detach().cpu().numpy()}")
        print(f"[INVERSION] true n_param = {n_param.detach().cpu().numpy()}")
    history = []
    state = {
        "closure_count": 0,
        "best_loss": float("inf"),
        "best_dT": None,
        "best_rho2": None,
        "best_A": None,
        "best_n": None,
    }


    def closure():
        optimizer.zero_grad(set_to_none=True)

        tkp0_cur = compose_tkp0_from_dT(
            dT_interior=dT_param,
            tkp_bg=tkp_bg,
            tkp0_ref=tkp0_ref,
            interior_mask=interior_mask,
        )

        rho_cur = compose_rho_param()
        A_cur = compose_A_param()
        n_cur = compose_n_param()

        loss_T_bound = torch.mean(
            torch.relu(-tkp0_cur).square()
            + torch.relu(tkp0_cur - 1.0).square()
        )

        try:
            terms = build_loss_terms(
                ob=ob,
                num_steps=num_steps_inv,
                tkp0=tkp0_cur,
                composition_field_init=composition_field_init,
                tkp0_prior=tkp_bg,
                A_param=A_cur,
                n_param=n_cur,
                E_param=E_param,
                rho_param=rho_cur,
                dt=dt,
                w_T=w_T,
                w_vx=w_vx,
                w_sigmayy=w_sigmayy,
                w_C=w_C,
                reg_weight_tkp=reg_weight_tkp,
                reg_tkp_order=1,
                crop_surface=3,
            )

            loss = terms["total_loss"] + bound_weight * loss_T_bound

        except (RuntimeError, ValueError) as err:
            print(
                "[LOSS FAILED] nonlinear solve failed. "
                f"Return large loss. Reason: {type(err).__name__}: {err}"
            )

            anchor = 0.0 * tkp0_cur.sum()
            for p in params:
                anchor = anchor + 0.0 * p.square().sum()

            loss = torch.as_tensor(1e20, device=device, dtype=dtype) + anchor

            zero = torch.zeros((), device=device, dtype=dtype)
            terms = {
                "total_loss": loss,
                "data_loss": zero,
                "loss_T": zero,
                "loss_vx": zero,
                "loss_sigmayy": zero,
                "loss_C": zero,
                "reg_tkp": zero,
            }

        if not torch.isfinite(loss):
            anchor = sum(0.0 * p.square().sum() for p in params)
            loss = torch.as_tensor(1e20, device=device, dtype=dtype) + anchor

        loss.backward()

        with torch.no_grad():
            state["closure_count"] += 1
            loss_val = float(loss.detach().item())

            grad_dT = (
                float(dT_param.grad.norm().item())
                if dT_param.grad is not None
                else float("nan")
            )

            grad_rho2 = (
                float(rho2_param_nd.grad.reshape(-1)[0].item())
                if rho2_param_nd.grad is not None
                else float("nan")
            )

            grad_A = (
                float(A_param_inv.grad.reshape(-1)[0].item())
                if A_param_inv.grad is not None
                else float("nan")
            )

            grad_n = (
                float(n_param_inv.grad.reshape(-1)[0].item())
                if n_param_inv.grad is not None
                else float("nan")
            )

            if loss_val < state["best_loss"]:
                state["best_loss"] = loss_val
                state["best_dT"] = dT_param.detach().clone()
                state["best_rho2"] = rho2_param_nd.detach().clone()
                state["best_A"] = A_param_inv.detach().clone()
                state["best_n"] = n_param_inv.detach().clone()

            row = {
                "closure": state["closure_count"],
                "total_loss": loss_val,
                "data_loss": float(terms["data_loss"].detach().item()),
                "loss_T": float(terms["loss_T"].detach().item()),
                "loss_vx": float(terms["loss_vx"].detach().item()),
                "loss_sigmayy": float(terms["loss_sigmayy"].detach().item()),
                "loss_C": float(terms["loss_C"].detach().item()),
                "reg_tkp": float(terms["reg_tkp"].detach().item()),
                "loss_T_bound": float(loss_T_bound.detach().item()),
                "rho2": float(rho_cur[1].detach().item()),
                "A": float(A_cur[0].detach().item()),
                "n": float(n_cur[0].detach().item()),
                "grad_dT": grad_dT,
                "grad_rho2": grad_rho2,
                "grad_A": grad_A,
                "grad_n": grad_n,
            }
            history.append(row)

            print(
                f"[LBFGS {row['closure']:04d}] "
                f"loss={row['total_loss']:.6e}, "
                f"T={row['loss_T']:.3e}, "
                f"vx={row['loss_vx']:.3e}, "
                f"sigmayy={row['loss_sigmayy']:.3e}, "
                f"reg={row['reg_tkp']:.3e}, "
                f"Tbound={row['loss_T_bound']:.3e}, "
                f"rho2={row['rho2']:.3e}, "
                f"A={row['A']:.3e}, "
                f"n={row['n']:.3e}, "
                f"gradT={row['grad_dT']:.3e}"
            )

        return loss

    print("[INVERSION] Start LBFGS")
    final_loss = optimizer.step(closure)

    final_loss_value = (
        float(final_loss.detach().item())
        if torch.is_tensor(final_loss)
        else float(final_loss)
    )

    print(f"[INVERSION] Finished. final_loss = {final_loss_value:.6e}")
    print(f"[INVERSION] best_loss  = {state['best_loss']:.6e}")

    with torch.no_grad():
        if state["best_dT"] is None:
            state["best_dT"] = dT_param.detach().clone()
            state["best_rho2"] = rho2_param_nd.detach().clone()
            state["best_A"] = A_param_inv.detach().clone()
            state["best_n"] = n_param_inv.detach().clone()

        tkp0_final = compose_tkp0_from_dT(
            dT_interior=state["best_dT"],
            tkp_bg=tkp_bg,
            tkp0_ref=tkp0_ref,
            interior_mask=interior_mask,
        )

        dT_full_final = torch.zeros_like(tkp_bg)
        dT_full_final[interior_mask] = state["best_dT"]

        rho2_param_nd.copy_(state["best_rho2"])
        A_param_inv.copy_(state["best_A"])
        n_param_inv.copy_(state["best_n"])

        rho_final = compose_rho_param()
        A_final = compose_A_param()
        n_final = compose_n_param()

        final_forward = forward_fn(
            num_steps=num_steps_inv,
            tkp_init=tkp0_final,
            composition_field_init=composition_field_init,
            A_param=A_final,
            n_param=n_final,
            E_param=E_param,
            rho_param=rho_final,
            dt=dt,
            store_on_cpu=True,
        )

    print_field_error("tkp0_final vs tkp0_ref", tkp0_final, tkp0_ref)

    tkp_end_final = final_forward["tkp_end_out"].to(device=device, dtype=dtype)
    tkp_end_obs = ob["tkp_end_out"].to(device=device, dtype=dtype)
    print_field_error("tkp_end_final vs obs", tkp_end_final, tkp_end_obs)

    print("[INVERSION] Final parameters")
    print(f"  rho_final = {rho_final.detach().cpu().numpy()}")
    print(f"  A_final   = {A_final.detach().cpu().numpy()}")
    print(f"  n_final   = {n_final.detach().cpu().numpy()}")

    summary = {
        "mode": "joint_T0_rho2_A_n",
        "history": history,
        "best_loss": state["best_loss"],
        "final_loss_value": final_loss_value,

        "weights": {
            "w_T": w_T,
            "w_vx": w_vx,
            "w_sigmayy": w_sigmayy,
            "w_C": w_C,
            "reg_weight_tkp": reg_weight_tkp,
            "bound_weight": bound_weight,
        },

        "tkp_bg": to_cpu(tkp_bg),
        "tkp0_ref": to_cpu(tkp0_ref),
        "tkp0_final": to_cpu(tkp0_final),
        "dT_full_final": to_cpu(dT_full_final),
        "dT_interior_final": to_cpu(state["best_dT"]),
        "interior_mask": to_cpu(interior_mask),

        "rho_param_ref": to_cpu(rho_param),
        "rho_param_final": to_cpu(rho_final),
        "A_param_ref": to_cpu(A_param),
        "A_param_final": to_cpu(A_final),
        "n_param_ref": to_cpu(n_param),
        "n_param_final": to_cpu(n_final),

        "tkp_end_final": to_cpu(tkp_end_final),
        "tkp_end_obs": to_cpu(tkp_end_obs),

        "lbfgs_history": pack_lbfgs_history(optimizer),
    }

    torch.save(summary, save_path)
    print(f"[INVERSION] Saved: {save_path}")

    return summary

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "forward"

    if mode == "forward":
        forward_main()
    elif mode == "invert":
        inversion_main(
            observation_path="observe_data.pt",
            save_path="tkp_inversion_summary.pt",
            invert_rho2=False,
            invert_A=False,
            invert_n=False,
            use_preconditioner=True,
            R_smooth_km=50.0,
            max_iter=200,
            max_eval=400,
        )
    elif mode in ["taylor", "taylortest", "taylor_test"]:
        taylor_test_main(
            observation_path="observe_data.pt",
            save_dir="taylor_test",
        )
    elif mode.startswith("forward_pre_"):
        iter_number = int(mode.replace("forward_pre_", ""))

        summary = torch.load("tkp_inversion_summary.pt", map_location=device)

        if "lbfgs_history" not in summary:
            raise KeyError("summary does not contain 'lbfgs_history'.")

        lbfgs_history = summary["lbfgs_history"]

        if len(lbfgs_history) == 0:
            raise RuntimeError("lbfgs_history is empty.")

        idx = iter_number - 1

        if idx < 0 or idx >= len(lbfgs_history):
            raise IndexError(
                f"Requested forward_pre_{iter_number}, "
                f"but lbfgs_history has only {len(lbfgs_history)} records."
            )

        rec = lbfgs_history[idx]

        if "params" not in rec:
            raise KeyError(
                "This lbfgs_history record does not contain 'params'. "
                "Cannot reconstruct tkp_init."
            )

        dT_iter = rec["params"][0].to(device=device, dtype=dtype)

        tkp_bg = summary["tkp_bg"].to(device=device, dtype=dtype)
        tkp0_ref = summary["tkp0_ref"].to(device=device, dtype=dtype)
        interior_mask = summary["interior_mask"].to(device=device).bool()

        tkp_init_iter = compose_tkp0_from_dT(
            dT_interior=dT_iter,
            tkp_bg=tkp_bg,
            tkp0_ref=tkp0_ref,
            interior_mask=interior_mask,
        )

        forward_main(
            tkp_init=tkp_init_iter,
            filename=f"observed_{iter_number}.pt",
            composition_field_init=torch.zeros_like(tkp_init_iter),
        )
    else:
        raise ValueError("mode must be 'forward' or 'invert'")
