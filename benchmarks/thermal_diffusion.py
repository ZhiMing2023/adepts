'Thermal diffusion convergence against an analytical solution.'

import os
import math
import csv
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

import adepts as ad


device = "cpu"
dtype = torch.float64

torch.set_default_dtype(dtype)
torch.set_default_device(device)
torch.manual_seed(0)

L0 = 660e3
kappa0 = 1e-6

k_thermal_phys = 3.0  # W m^-1 K^-1
rhocp_phys = 3.3e6  # J m^-3 K^-1

kappa_nd = (k_thermal_phys / rhocp_phys) / kappa0

def print_configuration():
    print("[thermal diffusion]")
    print(f"device={device}, dtype={dtype}, L0={L0:.6e} m")
    print(f"kappa0={kappa0:.6e} m^2/s, kappa_nd={kappa_nd:.12e}")

def build_mesh(
    *,
    xsize=1500e3,
    ysize=660e3,
    resolution=10e3,
    device="cpu",
    dtype=torch.float64,
):
    xbase_resolution = torch.tensor(resolution, dtype=dtype, device=device)
    ybase_resolution = torch.tensor(resolution, dtype=dtype, device=device)

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

    for key in ["xnode", "ynode", "xp", "yp", "xvx", "yvx", "xvy", "yvy"]:
        ms[key] = ms[key] / L0

    ms["xsize"] = ms["xsize"] / L0
    ms["ysize"] = ms["ysize"] / L0

    return ms


def exact_temperature_pgrid(mesh_state, t_nd):
    xp = mesh_state["xp"]
    yp = mesh_state["yp"]

    Lx = mesh_state["xsize"]
    Ly = mesh_state["ysize"]

    X = xp[None, :].expand(yp.numel(), -1)
    Y = yp[:, None].expand(-1, xp.numel())

    kx = math.pi / float(Lx)
    ky = math.pi / float(Ly)

    lam = kappa_nd * (kx ** 2 + ky ** 2)

    T = torch.cos(kx * X) * torch.sin(ky * Y) * torch.exp(
        torch.as_tensor(-lam * t_nd, dtype=X.dtype, device=X.device)
    )

    T[0, 1:-1] = -T[1, 1:-1]
    T[-1, 1:-1] = -T[-2, 1:-1]

    T[1:-1, 0] = T[1:-1, 1]
    T[1:-1, -1] = T[1:-1, -2]

    T[0, 0] = 0.5 * (T[0, 1] + T[1, 0])
    T[0, -1] = 0.5 * (T[0, -2] + T[1, -1])
    T[-1, 0] = 0.5 * (T[-1, 1] + T[-2, 0])
    T[-1, -1] = 0.5 * (T[-1, -2] + T[-2, -1])

    T[:, 0] = T[:, 1]       # Neumann-like copy, consistent with side zero flux
    T[:, -1] = T[:, -2]

    return T


def make_zero_velocity(mesh_state, dtype=torch.float64, device="cpu"):
    xvx = mesh_state["xvx"]
    yvx = mesh_state["yvx"]
    xvy = mesh_state["xvy"]
    yvy = mesh_state["yvy"]

    vx = torch.zeros(
        (yvx.numel(), xvx.numel()),
        dtype=dtype,
        device=device,
    )

    vy = torch.zeros(
        (yvy.numel(), xvy.numel()),
        dtype=dtype,
        device=device,
    )

    return vx, vy


def run_diffusion_case(
    *,
    resolution=10e3,
    t_end_nd=0.01,
    nstep=40,
    save_dir="benchmark1_thermal_diffusion",
    make_plots=True,
    save_tensor=True,
):
    os.makedirs(save_dir, exist_ok=True)

    mesh = build_mesh(
        xsize=1500e3,
        ysize=660e3,
        resolution=resolution,
        device=device,
        dtype=dtype,
    )

    mesh_state = build_nd_mesh_state(mesh)

    T = exact_temperature_pgrid(mesh_state, t_nd=0.0)
    T0 = T.clone()

    vx, vy = make_zero_velocity(mesh_state, dtype=dtype, device=device)

    kp_p = torch.ones_like(T) * (k_thermal_phys / kappa0)
    rhocp_p = torch.ones_like(T) * rhocp_phys

    dt_nd = torch.as_tensor(
        t_end_nd / float(nstep),
        dtype=dtype,
        device=device,
    )

    thermal_boundary_dict = {
        "top": {"type": "dirichlet", "value": 0.0},
        "bottom": {"type": "dirichlet", "value": 0.0},
        "left": {"type": "neumann", "value": 0.0},
        "right": {"type": "neumann", "value": 0.0},
    }

    t0_wall = time.time()

    with torch.no_grad():
        for timestep in range(nstep):
            thermal_out = ad.thermal.ThermalSystemGrid.advance_temperature_strang(
                T_old=T,
                vx=vx,
                vy=vy,
                KX=kp_p,
                KY=kp_p,
                RHOCP=rhocp_p,
                dt=dt_nd,
                mesh_state=mesh_state,
                timestep=timestep,
                thermal_boundary_dict=thermal_boundary_dict,
                HSUM=None,
                edge_mode="nearest",
                constant_value=0.0,
                backtrace="rk2",
                recompute_coeff_second_half=False,
            )

            T = thermal_out["T"]

    wall_time = time.time() - t0_wall

    T_exact = exact_temperature_pgrid(mesh_state, t_nd=t_end_nd)

    err = T - T_exact

    L2_full = torch.sqrt(torch.mean(err ** 2)).item()
    Linf_full = torch.max(torch.abs(err)).item()

    err_int = err[1:-1, 1:-1]
    L2_int = torch.sqrt(torch.mean(err_int ** 2)).item()
    Linf_int = torch.max(torch.abs(err_int)).item()

    row = {
        "resolution_m": float(resolution),
        "Nx_p": int(T.shape[1]),
        "Ny_p": int(T.shape[0]),
        "t_end_nd": float(t_end_nd),
        "nstep": int(nstep),
        "dt_nd": float(dt_nd.detach().cpu().item()),
        "kappa_nd": float(kappa_nd),
        "L2_full": float(L2_full),
        "Linf_full": float(Linf_full),
        "L2_int": float(L2_int),
        "Linf_int": float(Linf_int),
        "wall_time_sec": float(wall_time),
    }

    print(
        f"[case] res={resolution/1e3:.2f} km | "
        f"nstep={nstep:04d} | dt_nd={row['dt_nd']:.4e} | "
        f"L2_int={L2_int:.6e} | Linf_int={Linf_int:.6e} | "
        f"time={wall_time:.2f}s"
    )

    if save_tensor:
        torch.save(
            {
                "row": row,
                "T0": T0.detach().cpu(),
                "T_num": T.detach().cpu(),
                "T_exact": T_exact.detach().cpu(),
                "error": err.detach().cpu(),
                "xp_nd": mesh_state["xp"].detach().cpu(),
                "yp_nd": mesh_state["yp"].detach().cpu(),
            },
            os.path.join(save_dir, f"case_res_{int(resolution)}_nstep_{nstep}.pt"),
        )

    if make_plots:
        plot_fields(
            T_num=T,
            T_exact=T_exact,
            err=err,
            row=row,
            save_path=os.path.join(
                save_dir,
                f"fields_res_{int(resolution)}_nstep_{nstep}.png",
            ),
        )

    return row


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_fields(T_num, T_exact, err, row, save_path):
    T_num_np = to_np(T_num)
    T_exact_np = to_np(T_exact)
    err_np = to_np(err)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(T_exact_np, origin="lower", aspect="auto")
    axes[0].set_title("T exact")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(T_num_np, origin="lower", aspect="auto")
    axes[1].set_title("T numerical")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(err_np, origin="lower", aspect="auto")
    axes[2].set_title(
        f"error\nL2_int={row['L2_int']:.2e}, Linf_int={row['Linf_int']:.2e}"
    )
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    for ax in axes:
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_convergence(rows, save_dir):
    dt = np.array([r["dt_nd"] for r in rows], dtype=float)
    L2 = np.array([r["L2_int"] for r in rows], dtype=float)
    Linf = np.array([r["Linf_int"] for r in rows], dtype=float)

    idx = np.argsort(dt)
    dt = dt[idx]
    L2 = L2[idx]
    Linf = Linf[idx]

    mask = np.isfinite(dt) & np.isfinite(L2) & (dt > 0) & (L2 > 0)
    if mask.sum() >= 2:
        slope_L2 = np.polyfit(np.log(dt[mask]), np.log(L2[mask]), 1)[0]
    else:
        slope_L2 = np.nan

    mask_inf = np.isfinite(dt) & np.isfinite(Linf) & (dt > 0) & (Linf > 0)
    if mask_inf.sum() >= 2:
        slope_Linf = np.polyfit(np.log(dt[mask_inf]), np.log(Linf[mask_inf]), 1)[0]
    else:
        slope_Linf = np.nan

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(dt, L2, marker="o", label=f"L2 interior, slope={slope_L2:.2f}")
    ax.loglog(dt, Linf, marker="s", label=f"Linf interior, slope={slope_Linf:.2f}")

    if len(dt) > 0 and np.isfinite(L2[-1]) and L2[-1] > 0:
        ref = L2[-1] * (dt / dt[-1]) ** 1
        ax.loglog(dt, ref, linestyle="--", label="O(dt)")

    if len(dt) > 0 and np.isfinite(L2[-1]) and L2[-1] > 0:
        ref2 = L2[-1] * (dt / dt[-1]) ** 2
        ax.loglog(dt, ref2, linestyle=":", label="O(dt^2)")

    ax.set_xlabel("dt_nd")
    ax.set_ylabel("error at t_end")
    ax.set_title("Benchmark1: temporal convergence")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()

    fig.tight_layout()

    fig_path = os.path.join(save_dir, "temporal_convergence.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "slope_L2": float(slope_L2),
        "slope_Linf": float(slope_Linf),
        "figure": fig_path,
    }


def save_rows_csv(rows, path):
    if len(rows) == 0:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    print_configuration()
    save_dir = "benchmark1_thermal_diffusion"
    os.makedirs(save_dir, exist_ok=True)

    resolution = 2e3

    t_end_nd = 0.01

    nsteps_list = [10, 20, 40, 80]

    rows = []

    for nstep in nsteps_list:
        row = run_diffusion_case(
            resolution=resolution,
            t_end_nd=t_end_nd,
            nstep=nstep,
            save_dir=save_dir,
            make_plots=True,
        )
        rows.append(row)

    csv_path = os.path.join(save_dir, "benchmark1_temporal_convergence.csv")
    save_rows_csv(rows, csv_path)

    conv_info = plot_temporal_convergence(rows, save_dir)

    summary = {
        "benchmark": "thermal_diffusion_analytical_solution",
        "equation": "dT/dt = kappa_nd * Laplacian(T)",
        "exact_solution": "T = cos(pi*x/Lx) * sin(pi*y/Ly) * exp(-kappa_nd*((pi/Lx)^2 + (pi/Ly)^2)*t)",
        "bc": {
            "top": "Dirichlet T=0",
            "bottom": "Dirichlet T=0",
            "left": "Neumann dT/dx=0",
            "right": "Neumann dT/dx=0",
        },
        "rows": rows,
        "convergence": conv_info,
    }

    with open(os.path.join(save_dir, "benchmark1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    torch.save(summary, os.path.join(save_dir, "benchmark1_summary.pt"))

    print("=" * 80)
    print("[Benchmark1 finished]")
    print(f"CSV saved     : {csv_path}")
    print(f"Figure saved  : {conv_info['figure']}")
    print(f"slope L2      : {conv_info['slope_L2']:.4f}")
    print(f"slope Linf    : {conv_info['slope_Linf']:.4f}")
    print(f"Summary saved : {os.path.join(save_dir, 'benchmark1_summary.json')}")
    print("=" * 80)
