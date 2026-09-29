'Slab-detachment convergence benchmark.'

import os
import csv
import json
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import adepts as ad



device = "cpu"
dtype = torch.float64

torch.set_default_device(device)
torch.set_default_dtype(dtype)
torch.manual_seed(0)

L0 = 660e3
eta0 = 1e21
kappa0 = 1e-6
t0 = L0**2 / kappa0

year = 365.25 * 24.0 * 3600.0
Myr = 1.0e6 * year

g_phys = 9.81

xsize = 1000e3
ysize = 660e3

resolution_km = float(os.environ.get("RES_KM", "5"))
resolution = resolution_km * 1e3

SAVE_DIR = f"benchmark4_slab_detachment_{resolution_km:g}km"

dt_years = 1.0e5
dt_phys = torch.tensor(dt_years * year, dtype=dtype, device=device)
dt_nd = dt_phys / t0

final_time_years = 23.0e6
num_steps = int(round(final_time_years / dt_years))

diag_interval = 1

plot_interval = 20

plate_thickness = 80e3
slab_width = 80e3
slab_depth_below_plate = 250e3
slab_overlap_with_plate = 0.0

slab_top = plate_thickness - slab_overlap_with_plate
slab_bottom = plate_thickness + slab_depth_below_plate

slab_center_x = 0.5 * xsize
slab_left = slab_center_x - 0.5 * slab_width
slab_right = slab_center_x + 0.5 * slab_width

geom_smooth = 1e3

rho_mantle = 3150.0
rho_slab = 3300.0
drho = rho_slab - rho_mantle

eta_mantle = 1e21
eta_min = 1e21
eta_max = 1e25

eta0_powerlaw = 4.75e11
n_powerlaw = 4.0

REMOVE_HYDROSTATIC = False

BUOYANCY_SIGN = 1.0

SIDE_BC = "no_slip"

COMPOSITION_LEVEL = 0.5



def compute_characteristic_time():
    H = slab_depth_below_plate
    B = (2.0 * eta0_powerlaw) ** (-n_powerlaw)

    tc_seconds = 1.0 / (
        B
        * (0.5 * drho * g_phys * H) ** n_powerlaw
    )

    return tc_seconds


tc_phys = compute_characteristic_time()
tc_myr = tc_phys / Myr



def build_mesh():
    xbase_resolution = torch.tensor(
        resolution,
        dtype=dtype,
        device=device,
    )
    ybase_resolution = torch.tensor(
        resolution,
        dtype=dtype,
        device=device,
    )

    mesh = ad.mesh.CartesianMesh(
        xsize,
        ysize,
        xbase_resolution,
        ybase_resolution,
        device=device,
    )
    return mesh


def build_nd_mesh_state(mesh):
    ms = mesh.to_mesh_state()

    for key in [
        "xnode",
        "ynode",
        "xp",
        "yp",
        "xvx",
        "yvx",
        "xvy",
        "yvy",
    ]:
        ms[key] = ms[key] / L0

    ms["xsize"] = ms["xsize"] / L0
    ms["ysize"] = ms["ysize"] / L0

    return ms



def apply_neumann_ghost_scalar(C):
    C = C.clone()
    C[0, :] = C[1, :]
    C[-1, :] = C[-2, :]
    C[:, 0] = C[:, 1]
    C[:, -1] = C[:, -2]
    return C


def build_initial_composition(mesh):
    xp = mesh.xp.to(device=device, dtype=dtype)
    yp = mesh.yp.to(device=device, dtype=dtype)

    X = xp[None, :].expand(yp.numel(), -1)
    Y = yp[:, None].expand(-1, xp.numel())

    plate_mask = torch.sigmoid(
        (plate_thickness - Y) / geom_smooth
    )

    slab_x_mask = (
        torch.sigmoid((X - slab_left) / geom_smooth)
        * torch.sigmoid((slab_right - X) / geom_smooth)
    )

    slab_bottom_mask = torch.sigmoid(
        (slab_bottom - Y) / geom_smooth
    )

    slab_mask = slab_x_mask * slab_bottom_mask

    C = torch.maximum(plate_mask, slab_mask)

    C = torch.clamp(C, 0.0, 1.0)
    C = apply_neumann_ghost_scalar(C)

    return C


def log_blend_eta(eta_a, eta_b, w):
    w = torch.clamp(w, 0.0, 1.0)

    return torch.exp(
        (1.0 - w) * torch.log(eta_a)
        + w * torch.log(eta_b)
    )

def slab_powerlaw_eta_from_strain(strain_II_nd):
    eps_phys = torch.clamp(
        strain_II_nd / t0,
        min=1e-20,
    )

    eta_raw = (
        eta0_powerlaw
        * eps_phys ** ((1.0 / n_powerlaw) - 1.0)
    )

    eta = torch.clamp(
        eta_raw,
        min=eta_min,
        max=eta_max,
    )

    return eta


def viscosity_update_fn_slab(
    mesh_state,
    strain_II_p,
    strain_II_node,
    pressure_ext,
    tkp_nd,
    ctx,
):
    C_p = ctx["C_p"].to(
        device=strain_II_p.device,
        dtype=strain_II_p.dtype,
    )

    C_node = ad.interpolation.interp_between_grids(
        mesh_state,
        C_p,
        src="p",
        tgt="node",
    )
    C_node = torch.clamp(C_node, 0.0, 1.0)

    eta_m_p = torch.ones_like(strain_II_p) * eta_mantle
    eta_m_node = torch.ones_like(strain_II_node) * eta_mantle

    eta_s_p = slab_powerlaw_eta_from_strain(strain_II_p)
    eta_s_node = slab_powerlaw_eta_from_strain(strain_II_node)

    eta_p = log_blend_eta(
        eta_m_p,
        eta_s_p,
        C_p,
    )

    eta_node = log_blend_eta(
        eta_m_node,
        eta_s_node,
        C_node,
    )

    eta_p = torch.clamp(
        eta_p,
        min=eta_min,
        max=eta_max,
    )
    eta_node = torch.clamp(
        eta_node,
        min=eta_min,
        max=eta_max,
    )

    return eta_node / eta0, eta_p / eta0



def make_buoyancy(C_p, mesh_state):
    buoyancy_phys = (
        BUOYANCY_SIGN
        * drho
        * C_p
        * g_phys
    )

    if REMOVE_HYDROSTATIC:
        buoyancy_phys = (
            buoyancy_phys
            - buoyancy_phys.mean(
                dim=1,
                keepdim=True,
            )
        )

    buoyancy_nd = (
        buoyancy_phys
        * (L0**3)
        / (eta0 * kappa0)
    )

    density_x = ad.interpolation.interp_between_grids(
        mesh_state,
        torch.zeros_like(buoyancy_nd),
        src="p",
        tgt="vx",
    )

    density_y = ad.interpolation.interp_between_grids(
        mesh_state,
        buoyancy_nd,
        src="p",
        tgt="vy",
    )

    return density_x, density_y


def stokes_boundary():
    return {
        "left": SIDE_BC,
        "right": SIDE_BC,
        "top": "free_slip",
        "bottom": "free_slip",
    }


def composition_bc():
    return {
        "top": {
            "type": "neumann",
            "value": 0.0,
        },
        "bottom": {
            "type": "neumann",
            "value": 0.0,
        },
        "left": {
            "type": "neumann",
            "value": 0.0,
        },
        "right": {
            "type": "neumann",
            "value": 0.0,
        },
    }


def solve_stokes_for_C(
    C_p,
    T_dummy,
    mesh,
    mesh_state,
    timestep,
    u0,
    coloring_pack,
):
    density_x, density_y = make_buoyancy(
        C_p,
        mesh_state,
    )

    try:
        pscale = torch.ones_like(
            mesh.dx,
            dtype=dtype,
            device=device,
        )
    except Exception:
        pscale = torch.ones(
            int(mesh_state["Nx"]),
            dtype=dtype,
            device=device,
        )

    rheology_context = {
        "C_p": C_p,
    }

    stokes_state = (
        ad.stokes.StokesSystem.implicit_solve_state(
            mesh_state=mesh_state,
            density_x=density_x,
            density_y=density_y,
            tkp=T_dummy,
            gx=0.0,
            gy=1.0,
            dt=dt_nd,
            pscale=pscale,
            boundary_const=stokes_boundary(),
            timestep=timestep,
            viscosity_update_fn=viscosity_update_fn_slab,
            rheology_context=rheology_context,
            p_fix_value=torch.tensor(
                0.0,
                dtype=dtype,
                device=device,
            ),
            is_stick_air=False,
            u0=u0,
            coloring_pack=coloring_pack,
        )
    )

    return stokes_state


def advect_composition(
    C_p,
    vx,
    vy,
    mesh_state,
):
    C_new = ad.advection.advect_scalar_sl_on_p_bfecc(
        q_old=C_p,
        vx=vx,
        vy=vy,
        dt=dt_nd,
        mesh_state=mesh_state,
        edge_mode="clamp",
        constant_value=0.0,
        backtrace="rk2",
        scalar_boundary_dict=composition_bc(),
        enforce_bc_after_advect=True,
        use_dirichlet_value_for_oob=False,
        scalar_interp_method="linear",
        velocity_interp_method="linear",
    )

    C_new = torch.clamp(
        C_new,
        0.0,
        1.0,
    )
    C_new = apply_neumann_ghost_scalar(C_new)

    return C_new



def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def interp_velocity_to_p(
    mesh_state,
    vx,
    vy,
):
    vx_p = ad.interpolation.interp_between_grids(
        mesh_state,
        vx,
        src="vx",
        tgt="p",
    )

    vy_p = ad.interpolation.interp_between_grids(
        mesh_state,
        vy,
        src="vy",
        tgt="p",
    )

    return vx_p, vy_p


def _interp_level_crossing(
    x0,
    x1,
    c0,
    c1,
    level=0.5,
):
    dc = c1 - c0

    if abs(dc) < 1e-14:
        return 0.5 * (x0 + x1)

    return (
        x0
        + (level - c0)
        * (x1 - x0)
        / dc
    )


def _central_component_width(
    x,
    c,
    center_x,
    level=0.5,
):
    mask = c >= level
    ids = np.flatnonzero(mask)

    if ids.size == 0:
        return 0.0

    split_ids = (
        np.where(np.diff(ids) > 1)[0]
        + 1
    )
    components = np.split(
        ids,
        split_ids,
    )

    component = min(
        components,
        key=lambda ind: abs(
            0.5
            * (
                x[ind[0]]
                + x[ind[-1]]
            )
            - center_x
        ),
    )

    i_left = int(component[0])
    i_right = int(component[-1])

    if i_left > 0:
        x_left = _interp_level_crossing(
            x[i_left - 1],
            x[i_left],
            c[i_left - 1],
            c[i_left],
            level,
        )
    else:
        x_left = x[i_left]

    if i_right < len(x) - 1:
        x_right = _interp_level_crossing(
            x[i_right],
            x[i_right + 1],
            c[i_right],
            c[i_right + 1],
            level,
        )
    else:
        x_right = x[i_right]

    return max(
        0.0,
        x_right - x_left,
    )


def compute_neck_width_and_tip(
    C_p,
    mesh,
):
    xp = to_np(mesh.xp)
    yp = to_np(mesh.yp)
    C = to_np(C_p)

    level = COMPOSITION_LEVEL

    Y = yp[:, None]

    slab_region = (
        (C >= level)
        & (
            Y
            > plate_thickness
            + 0.5 * resolution
        )
    )

    if np.any(slab_region):
        row_ids = np.where(slab_region)[0]
        tip_depth_km = float(
            np.max(yp[row_ids])
            / 1e3
        )
    else:
        tip_depth_km = float("nan")

    y_min = (
        plate_thickness
        + 2.0 * resolution
    )
    y_max = (
        slab_bottom
        - 2.0 * resolution
    )

    widths_km = []
    y_rows_km = []

    for j, yj in enumerate(yp):
        if yj < y_min or yj > y_max:
            continue

        width_m = _central_component_width(
            x=xp,
            c=C[j, :],
            center_x=slab_center_x,
            level=level,
        )

        widths_km.append(
            width_m / 1e3
        )
        y_rows_km.append(
            yj / 1e3
        )

    if len(widths_km) == 0:
        return (
            tip_depth_km,
            float("nan"),
            float("nan"),
        )

    widths_km = np.asarray(
        widths_km,
        dtype=float,
    )
    y_rows_km = np.asarray(
        y_rows_km,
        dtype=float,
    )

    i_min = int(
        np.argmin(widths_km)
    )

    neck_width_km = float(
        widths_km[i_min]
    )
    neck_depth_km = float(
        y_rows_km[i_min]
    )

    return (
        tip_depth_km,
        neck_width_km,
        neck_depth_km,
    )


def compute_diagnostics(
    C_p,
    vx,
    vy,
    eta_p_nd,
    epsII_p_nd,
    mesh,
    mesh_state,
    step,
):
    vx_p, vy_p = interp_velocity_to_p(
        mesh_state,
        vx,
        vy,
    )

    speed_nd = torch.sqrt(
        vx_p**2
        + vy_p**2
    )

    speed_phys = (
        speed_nd
        * kappa0
        / L0
    )

    speed_cm_yr = (
        speed_phys
        * 100.0
        * year
    )

    (
        tip_depth_km,
        neck_width_km,
        neck_depth_km,
    ) = compute_neck_width_and_tip(
        C_p,
        mesh,
    )

    eta_phys = eta_p_nd * eta0
    eps_phys = epsII_p_nd / t0

    time_myr = (
        step
        * dt_years
        / 1.0e6
    )

    D0_km = slab_width / 1e3

    if np.isfinite(neck_width_km):
        D_over_D0 = (
            neck_width_km
            / D0_km
        )
    else:
        D_over_D0 = float("nan")

    row = {
        "step": int(step),
        "time_myr": float(time_myr),
        "t_over_tc": float(
            time_myr / tc_myr
        ),
        "tc_myr": float(tc_myr),
        "dt_years": float(dt_years),
        "tip_depth_km": float(
            tip_depth_km
        ),
        "neck_width_km": float(
            neck_width_km
        ),
        "D_over_D0": float(
            D_over_D0
        ),
        "neck_depth_km": float(
            neck_depth_km
        ),
        "max_speed_cm_yr": float(
            speed_cm_yr[
                1:-1,
                1:-1,
            ]
            .max()
            .detach()
            .cpu()
            .item()
        ),
        "rms_speed_cm_yr": float(
            torch.sqrt(
                torch.mean(
                    speed_cm_yr[
                        1:-1,
                        1:-1,
                    ]
                    ** 2
                )
            )
            .detach()
            .cpu()
            .item()
        ),
        "max_strain_1_s": float(
            eps_phys[
                1:-1,
                1:-1,
            ]
            .max()
            .detach()
            .cpu()
            .item()
        ),
        "min_eta_Pa_s": float(
            eta_phys[
                1:-1,
                1:-1,
            ]
            .min()
            .detach()
            .cpu()
            .item()
        ),
        "max_eta_Pa_s": float(
            eta_phys[
                1:-1,
                1:-1,
            ]
            .max()
            .detach()
            .cpu()
            .item()
        ),
        "mean_C": float(
            C_p[
                1:-1,
                1:-1,
            ]
            .mean()
            .detach()
            .cpu()
            .item()
        ),
    }

    return (
        row,
        vx_p,
        vy_p,
        speed_cm_yr,
        eta_phys,
        eps_phys,
    )



def positive_range(
    arr,
    pmin=1,
    pmax=99,
    fallback=(1e-20, 1.0),
):
    arr = np.asarray(arr)

    vals = arr[
        np.isfinite(arr)
        & (arr > 0)
    ]

    if vals.size == 0:
        return fallback

    vmin = np.percentile(
        vals,
        pmin,
    )
    vmax = np.percentile(
        vals,
        pmax,
    )

    if vmax <= vmin:
        vmax = vmin * 10.0

    return (
        max(vmin, fallback[0]),
        vmax,
    )


def plot_state(
    C_p,
    vx_p,
    vy_p,
    eta_phys,
    eps_phys,
    speed_cm_yr,
    mesh,
    step,
):
    xp_km = to_np(
        mesh.xp / 1e3
    )
    yp_km = to_np(
        mesh.yp / 1e3
    )

    C_np = to_np(C_p)
    vx_np = to_np(vx_p)

    vy_np = -to_np(vy_p)

    eta_np = to_np(eta_phys)
    eps_np = to_np(eps_phys)
    speed_np = to_np(speed_cm_yr)

    X, Y = np.meshgrid(
        xp_km,
        yp_km,
    )

    extent = [
        xp_km.min(),
        xp_km.max(),
        yp_km.max(),
        yp_km.min(),
    ]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(9, 15),
        sharex=True,
    )

    im0 = axes[0].imshow(
        C_np,
        origin="upper",
        extent=extent,
        aspect="equal",
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
    )

    stride_y = max(
        1,
        C_np.shape[0] // 25,
    )
    stride_x = max(
        1,
        C_np.shape[1] // 35,
    )

    axes[0].quiver(
        X[
            1:-1:stride_y,
            1:-1:stride_x,
        ],
        Y[
            1:-1:stride_y,
            1:-1:stride_x,
        ],
        vx_np[
            1:-1:stride_y,
            1:-1:stride_x,
        ],
        vy_np[
            1:-1:stride_y,
            1:-1:stride_x,
        ],
        width=0.002,
    )

    axes[0].set_title(
        f"Composition + velocity | "
        f"step={step}, "
        f"t={step * dt_years / 1e6:.2f} Myr"
    )

    fig.colorbar(
        im0,
        ax=axes[0],
        label="C",
    )

    vmin_eta, vmax_eta = positive_range(
        eta_np,
        fallback=(1e21, 1e25),
    )

    im1 = axes[1].imshow(
        eta_np,
        origin="upper",
        extent=extent,
        aspect="equal",
        cmap="viridis",
        norm=LogNorm(
            vmin=vmin_eta,
            vmax=vmax_eta,
        ),
    )

    axes[1].set_title(
        "Viscosity"
    )

    fig.colorbar(
        im1,
        ax=axes[1],
        label="Pa s",
    )

    vmin_eps, vmax_eps = positive_range(
        eps_np,
        fallback=(1e-18, 1e-12),
    )

    im2 = axes[2].imshow(
        eps_np,
        origin="upper",
        extent=extent,
        aspect="equal",
        cmap="magma",
        norm=LogNorm(
            vmin=vmin_eps,
            vmax=vmax_eps,
        ),
    )

    axes[2].set_title(
        "Second invariant of strain rate"
    )

    fig.colorbar(
        im2,
        ax=axes[2],
        label="1/s",
    )

    vmin_v, vmax_v = positive_range(
        speed_np,
        fallback=(1e-6, 1.0),
    )

    im3 = axes[3].imshow(
        speed_np,
        origin="upper",
        extent=extent,
        aspect="equal",
        cmap="turbo",
        norm=LogNorm(
            vmin=vmin_v,
            vmax=vmax_v,
        ),
    )

    axes[3].set_title(
        "Speed"
    )

    fig.colorbar(
        im3,
        ax=axes[3],
        label="cm/yr",
    )

    for ax in axes:
        ax.set_ylabel(
            "Depth (km)"
        )
        ax.set_xlim(
            0,
            xsize / 1e3,
        )
        ax.set_ylim(
            ysize / 1e3,
            0,
        )

    axes[-1].set_xlabel(
        "x (km)"
    )

    fig.tight_layout()

    path = os.path.join(
        SAVE_DIR,
        f"state_step_{step:04d}.png",
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[plot] saved {path}"
    )


def save_csv(
    rows,
    path,
):
    if len(rows) == 0:
        return

    with open(
        path,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_diagnostics(rows):
    if not rows:
        return

    t = np.array(
        [
            r["time_myr"]
            for r in rows
        ]
    )

    tip = np.array(
        [
            r["tip_depth_km"]
            for r in rows
        ]
    )

    neck = np.array(
        [
            r["neck_width_km"]
            for r in rows
        ]
    )

    speed = np.array(
        [
            r["max_speed_cm_yr"]
            for r in rows
        ]
    )

    epsmax = np.array(
        [
            r["max_strain_1_s"]
            for r in rows
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        t,
        tip,
        label="slab tip depth",
    )
    ax.plot(
        t,
        neck,
        label="neck width",
    )

    ax.set_xlabel(
        "time (Myr)"
    )
    ax.set_ylabel(
        "km"
    )
    ax.grid(
        True,
        linestyle=":",
    )
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            SAVE_DIR,
            "diagnostics_depth_width.png",
        ),
        dpi=200,
    )

    plt.close(fig)

    fig, ax1 = plt.subplots(
        figsize=(7, 5)
    )

    ax1.plot(
        t,
        speed,
        label="max speed",
    )

    ax1.set_xlabel(
        "time (Myr)"
    )
    ax1.set_ylabel(
        "max speed (cm/yr)"
    )
    ax1.grid(
        True,
        linestyle=":",
    )

    ax2 = ax1.twinx()

    ax2.semilogy(
        t,
        epsmax,
        label="max strain rate",
    )

    ax2.set_ylabel(
        "max strain rate (1/s)"
    )

    lines1, labels1 = (
        ax1.get_legend_handles_labels()
    )
    lines2, labels2 = (
        ax2.get_legend_handles_labels()
    )

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best",
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            SAVE_DIR,
            "diagnostics_speed_strain.png",
        ),
        dpi=200,
    )

    plt.close(fig)


def plot_necking_benchmark(rows):
    if not rows:
        return

    t_norm = np.array(
        [
            r["t_over_tc"]
            for r in rows
        ],
        dtype=float,
    )

    D_norm = np.array(
        [
            r["D_over_D0"]
            for r in rows
        ],
        dtype=float,
    )

    valid = (
        np.isfinite(t_norm)
        & np.isfinite(D_norm)
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        t_norm[valid],
        D_norm[valid],
        label=(
            f"This study, "
            f"dx={resolution_km:g} km"
        ),
    )

    ax.set_xlabel(
        r"$t/t_c$"
    )
    ax.set_ylabel(
        r"$D/D_0$"
    )

    ax.set_xlim(
        0.0,
        max(
            1.05,
            np.nanmax(
                t_norm[valid]
            )
            if np.any(valid)
            else 1.05,
        ),
    )

    ax.set_ylim(
        0.0,
        1.05,
    )

    ax.grid(
        True,
        linestyle=":",
    )
    ax.legend()

    fig.tight_layout()

    path = os.path.join(
        SAVE_DIR,
        "benchmark_necking_D_D0_vs_t_tc.png",
    )

    fig.savefig(
        path,
        dpi=200,
    )

    plt.close(fig)

    print(
        f"[plot] saved {path}"
    )



def main():
    print("=" * 80)
    print(
        "[Benchmark4] "
        "Quantitative slab detachment benchmark"
    )
    print(
        f"domain      = "
        f"{xsize / 1e3:.0f} km x "
        f"{ysize / 1e3:.0f} km"
    )
    print(
        f"resolution  = "
        f"{resolution_km:.1f} km"
    )
    print(
        f"dt          = "
        f"{dt_years:.2e} yr"
    )
    print(
        f"num_steps   = "
        f"{num_steps}"
    )
    print(
        f"final time  = "
        f"{final_time_years / 1e6:.2f} Myr"
    )
    print(
        f"tc          = "
        f"{tc_myr:.3f} Myr"
    )
    print(
        f"final t/tc  = "
        f"{final_time_years / 1e6 / tc_myr:.3f}"
    )
    print(
        f"side BC     = "
        f"{SIDE_BC}"
    )
    print(
        f"save dir    = "
        f"{SAVE_DIR}"
    )
    print("=" * 80)

    mesh = build_mesh()
    mesh_state = build_nd_mesh_state(
        mesh
    )

    coloring_pack = (
        ad.stokes.StokesSystem
        .build_Fu_coloring_from_mesh(
            mesh_state,
            mode="safe",
            device=device,
            debug_print=False,
        )
    )

    C = build_initial_composition(
        mesh
    )

    T_dummy = torch.zeros_like(
        C
    )

    u0 = None
    rows = []

    t_wall0 = time.time()

    snapshot_times_myr = [
        0.0,
        5.8,
        11.4,
        17.1,
        19.3,
        20.5,
        21.7,
        22.8,
    ]

    snapshot_steps = {
        int(
            round(
                t_myr
                * 1.0e6
                / dt_years
            )
        )
        for t_myr in snapshot_times_myr
    }

    for step in range(
        num_steps + 1
    ):
        step_wall0 = time.time()

        stokes_state = solve_stokes_for_C(
            C_p=C,
            T_dummy=T_dummy,
            mesh=mesh,
            mesh_state=mesh_state,
            timestep=step,
            u0=u0,
            coloring_pack=coloring_pack,
        )

        u0 = stokes_state["u"]
        vx = stokes_state["vx"]
        vy = stokes_state["vy"]

        eta_p_nd = (
            stokes_state[
                "viscosity_p"
            ]
        )

        epsII_p_nd = (
            stokes_state[
                "epsII_p"
            ]
        )

        if (
            step % diag_interval == 0
            or step == num_steps
        ):
            (
                row,
                vx_p,
                vy_p,
                speed_cm_yr,
                eta_phys,
                eps_phys,
            ) = compute_diagnostics(
                C,
                vx,
                vy,
                eta_p_nd,
                epsII_p_nd,
                mesh,
                mesh_state,
                step,
            )

            row["wall_time_sec"] = float(
                time.time()
                - t_wall0
            )

            rows.append(
                row
            )

            print(
                f"step={step:04d} "
                f"t={row['time_myr']:.2f} Myr | "
                f"t/tc={row['t_over_tc']:.3f} | "
                f"tip={row['tip_depth_km']:.1f} km | "
                f"neck={row['neck_width_km']:.2f} km | "
                f"D/D0={row['D_over_D0']:.4f} | "
                f"neck_z={row['neck_depth_km']:.1f} km | "
                f"vmax={row['max_speed_cm_yr']:.3e} cm/yr | "
                f"epsmax={row['max_strain_1_s']:.3e} 1/s | "
                f"eta=["
                f"{row['min_eta_Pa_s']:.1e}, "
                f"{row['max_eta_Pa_s']:.1e}"
                f"] | "
                f"step_time="
                f"{time.time() - step_wall0:.2f}s"
            )

        if (
            step % plot_interval == 0
            or step in snapshot_steps
            or step == num_steps
        ):
            (
                row,
                vx_p,
                vy_p,
                speed_cm_yr,
                eta_phys,
                eps_phys,
            ) = compute_diagnostics(
                C,
                vx,
                vy,
                eta_p_nd,
                epsII_p_nd,
                mesh,
                mesh_state,
                step,
            )

            plot_state(
                C,
                vx_p,
                vy_p,
                eta_phys,
                eps_phys,
                speed_cm_yr,
                mesh,
                step,
            )

            torch.save(
                {
                    "step": step,
                    "time_myr": (
                        step
                        * dt_years
                        / 1e6
                    ),
                    "t_over_tc": (
                        step
                        * dt_years
                        / 1e6
                        / tc_myr
                    ),
                    "tc_myr": tc_myr,
                    "C": (
                        C
                        .detach()
                        .cpu()
                    ),
                    "vx": (
                        vx
                        .detach()
                        .cpu()
                    ),
                    "vy": (
                        vy
                        .detach()
                        .cpu()
                    ),
                    "eta_p_nd": (
                        eta_p_nd
                        .detach()
                        .cpu()
                    ),
                    "epsII_p_nd": (
                        epsII_p_nd
                        .detach()
                        .cpu()
                    ),
                    "mesh_xp_m": (
                        mesh.xp
                        .detach()
                        .cpu()
                    ),
                    "mesh_yp_m": (
                        mesh.yp
                        .detach()
                        .cpu()
                    ),
                    "row": row,
                },
                os.path.join(
                    SAVE_DIR,
                    f"state_step_{step:04d}.pt",
                ),
            )

        if step < num_steps:
            C = advect_composition(
                C,
                vx,
                vy,
                mesh_state,
            )

    csv_path = os.path.join(
        SAVE_DIR,
        "diagnostics.csv",
    )

    save_csv(
        rows,
        csv_path,
    )

    plot_diagnostics(
        rows
    )

    plot_necking_benchmark(
        rows
    )

    detachment_row = None

    for row in rows:
        if (
            np.isfinite(
                row["D_over_D0"]
            )
            and row["D_over_D0"] <= 0.0
        ):
            detachment_row = row
            break

    summary = {
        "benchmark": (
            "Quantitative "
            "Schmalholz-Hillebrand-style "
            "slab detachment"
        ),
        "domain_km": [
            xsize / 1e3,
            ysize / 1e3,
        ],
        "resolution_km": (
            resolution_km
        ),
        "dt_years": (
            dt_years
        ),
        "num_steps": (
            num_steps
        ),
        "final_time_myr": (
            final_time_years
            / 1e6
        ),
        "characteristic_time_myr": (
            tc_myr
        ),
        "final_t_over_tc": (
            final_time_years
            / 1e6
            / tc_myr
        ),
        "geometry": {
            "plate_thickness_km": (
                plate_thickness
                / 1e3
            ),
            "initial_slab_width_D0_km": (
                slab_width
                / 1e3
            ),
            "hanging_slab_length_H_km": (
                slab_depth_below_plate
                / 1e3
            ),
            "slab_center_x_km": (
                slab_center_x
                / 1e3
            ),
        },
        "materials": {
            "rho_mantle": (
                rho_mantle
            ),
            "rho_slab": (
                rho_slab
            ),
            "drho": (
                drho
            ),
            "eta_mantle": (
                eta_mantle
            ),
            "eta0_powerlaw": (
                eta0_powerlaw
            ),
            "n_powerlaw": (
                n_powerlaw
            ),
            "eta_min": (
                eta_min
            ),
            "eta_max": (
                eta_max
            ),
        },
        "diagnostic_definition": {
            "composition_level": (
                COMPOSITION_LEVEL
            ),
            "D_definition": (
                "minimum width of the central "
                "C>=0.5 slab component inside "
                "the neck-search window"
            ),
            "D0_km": (
                slab_width
                / 1e3
            ),
            "tc_definition": (
                "1 / [B * "
                "(0.5*drho*g*H)^n], "
                "B=(2*eta0_powerlaw)^(-n)"
            ),
        },
        "detachment": (
            {
                "time_myr": (
                    detachment_row[
                        "time_myr"
                    ]
                ),
                "t_over_tc": (
                    detachment_row[
                        "t_over_tc"
                    ]
                ),
            }
            if detachment_row is not None
            else None
        ),
        "last_diagnostics": (
            rows[-1]
            if rows
            else None
        ),
        "csv": csv_path,
    }

    summary_path = os.path.join(
        SAVE_DIR,
        "summary.json",
    )

    with open(
        summary_path,
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    torch.save(
        {
            "summary": summary,
            "rows": rows,
        },
        os.path.join(
            SAVE_DIR,
            "summary.pt",
        ),
    )

    print("=" * 80)
    print(
        "[Benchmark4 finished]"
    )
    print(
        f"CSV       : {csv_path}"
    )
    print(
        f"Summary   : {summary_path}"
    )

    if detachment_row is None:
        print(
            "Detachment: "
            "not detected within "
            "the simulated time."
        )
    else:
        print(
            "Detachment: "
            f"t={detachment_row['time_myr']:.3f} Myr, "
            f"t/tc={detachment_row['t_over_tc']:.3f}"
        )

    print("=" * 80)


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    main()
