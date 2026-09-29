'Isoviscous thermal-convection benchmark.'

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

SAVE_DIR = "benchmark3_isoviscous_convection"

RA = 1.0e4

N_LIST = [48, 64, 96, 128]

L0 = 1.0
XSIZE = 1.0
YSIZE = 1.0

DT_ND = 1.0e-4

MAX_STEPS = 5000
MIN_STEPS = 300

DIAG_INTERVAL = 10
SAVE_FIG_INTERVAL = 500

STEADY_TOL = 1.0e-5
STEADY_COUNT_TARGET = 20

BUOYANCY_SIGN = -1.0

REMOVE_HYDROSTATIC = True

T_TOP = 0.0
T_BOTTOM = 1.0

ETA_ND = 1.0

REF_NU_1A = 4.884
REF_VRMS_1A = 42.86


def build_mesh_and_state(N):
    resolution = XSIZE / float(N)

    mesh = ad.mesh.CartesianMesh(
        XSIZE,
        YSIZE,
        torch.tensor(resolution, dtype=dtype, device=device),
        torch.tensor(resolution, dtype=dtype, device=device),
        device=device,
    )

    mesh_state = mesh.to_mesh_state()

    for key in ["xnode", "ynode", "xp", "yp", "xvx", "yvx", "xvy", "yvy"]:
        mesh_state[key] = mesh_state[key] / L0

    mesh_state["xsize"] = mesh_state["xsize"] / L0
    mesh_state["ysize"] = mesh_state["ysize"] / L0

    return mesh, mesh_state


def thermal_bc():
    return {
        "top": {"type": "dirichlet", "value": T_TOP},
        "bottom": {"type": "dirichlet", "value": T_BOTTOM},
        "left": {"type": "neumann", "value": 0.0},
        "right": {"type": "neumann", "value": 0.0},
    }


def stokes_bc():
    return {
        "left": "free_slip",
        "right": "free_slip",
        "top": "free_slip",
        "bottom": "free_slip",
    }


def initial_temperature(mesh_state, perturb_amp=1.0e-2):
    xp = mesh_state["xp"]
    yp = mesh_state["yp"]

    X = xp[None, :].expand(yp.numel(), -1)
    Y = yp[:, None].expand(-1, xp.numel())

    Lx = float(mesh_state["xsize"])
    Ly = float(mesh_state["ysize"])

    T = Y / Ly
    T = T + perturb_amp * torch.cos(math.pi * X / Lx) * torch.sin(math.pi * Y / Ly)

    T = ad.thermal.ThermalSystemGrid.apply_temperature_bc_on_p(
        T,
        thermal_bc(),
        mesh_state,
    )

    return T



def constant_viscosity_update_fn(
    mesh_state,
    strain_II_p,
    strain_II_node,
    pressure_ext,
    tkp_nd,
    ctx,
):
    eta_node = torch.ones_like(strain_II_node) * ETA_ND
    eta_p = torch.ones_like(strain_II_p) * ETA_ND
    return eta_node, eta_p



def make_buoyancy(T, mesh_state):
    if REMOVE_HYDROSTATIC:
        T_dyn = T - T.mean(dim=1, keepdim=True)
    else:
        T_dyn = T

    buoyancy_p = BUOYANCY_SIGN * RA * T_dyn

    density_x = ad.interpolation.interp_between_grids(
        mesh_state,
        torch.zeros_like(buoyancy_p),
        src="p",
        tgt="vx",
    )

    density_y = ad.interpolation.interp_between_grids(
        mesh_state,
        buoyancy_p,
        src="p",
        tgt="vy",
    )

    return density_x, density_y


def solve_stokes(T, mesh, mesh_state, timestep, u0, coloring_pack):
    density_x, density_y = make_buoyancy(T, mesh_state)

    try:
        pscale = torch.ones_like(mesh.dx, dtype=dtype, device=device)
    except Exception:
        pscale = torch.ones(
            int(mesh_state["Nx"]),
            dtype=dtype,
            device=device,
        )

    p_fix_value = torch.tensor(0.0, dtype=dtype, device=device)

    stokes_state = ad.stokes.StokesSystem.implicit_solve_state(
        mesh_state=mesh_state,
        density_x=density_x,
        density_y=density_y,
        tkp=T,
        gx=0.0,
        gy=1.0,
        dt=torch.tensor(DT_ND, dtype=dtype, device=device),
        pscale=pscale,
        boundary_const=stokes_bc(),
        timestep=timestep,
        viscosity_update_fn=constant_viscosity_update_fn,
        rheology_context={},
        p_fix_value=p_fix_value,
        is_stick_air=False,
        u0=u0,
        coloring_pack=coloring_pack,
    )

    return stokes_state



def advance_temperature(T, vx, vy, mesh_state, timestep):
    K = torch.ones_like(T)
    RHOCP = torch.ones_like(T)

    out = ad.thermal.ThermalSystemGrid.advance_temperature_strang(
        T_old=T,
        vx=vx,
        vy=vy,
        KX=K,
        KY=K,
        RHOCP=RHOCP,
        dt=torch.tensor(DT_ND, dtype=dtype, device=device),
        mesh_state=mesh_state,
        timestep=timestep,
        thermal_boundary_dict=thermal_bc(),
        HSUM=None,
        split_order="DAD",
        edge_mode="nearest",
        constant_value=0.0,
        backtrace="rk2",
        recompute_coeff_second_half=False,
    )

    return out["T"]



def compute_nusselt(T, mesh_state):
    yp = mesh_state["yp"]

    dy_top = 0.5 * (yp[1] - yp[0])
    dy_bottom = 0.5 * (yp[-1] - yp[-2])

    Nu_top = torch.mean((T[1, 1:-1] - T_TOP) / dy_top).item()
    Nu_bottom = torch.mean((T_BOTTOM - T[-2, 1:-1]) / dy_bottom).item()

    Nu_mean = 0.5 * (Nu_top + Nu_bottom)

    return float(Nu_top), float(Nu_bottom), float(Nu_mean)


def interp_velocity_to_p(mesh_state, vx, vy):
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


def compute_vrms(mesh_state, vx, vy):
    vx_p, vy_p = interp_velocity_to_p(mesh_state, vx, vy)

    speed2 = vx_p[1:-1, 1:-1] ** 2 + vy_p[1:-1, 1:-1] ** 2
    vrms = torch.sqrt(torch.mean(speed2)).item()

    vmax = torch.sqrt(vx_p[1:-1, 1:-1] ** 2 + vy_p[1:-1, 1:-1] ** 2).max().item()

    return float(vrms), float(vmax), vx_p, vy_p


def compute_mean_temperature(T):
    return float(torch.mean(T[1:-1, 1:-1]).item())



def save_csv(rows, path):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_timeseries(rows, save_path, title):
    if not rows:
        return

    t = np.array([r["time_nd"] for r in rows])
    Nu_top = np.array([r["Nu_top"] for r in rows])
    Nu_bottom = np.array([r["Nu_bottom"] for r in rows])
    Nu_mean = np.array([r["Nu_mean"] for r in rows])
    Vrms = np.array([r["Vrms"] for r in rows])

    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.plot(t, Nu_top, label="Nu top")
    ax1.plot(t, Nu_bottom, label="Nu bottom")
    ax1.plot(t, Nu_mean, linestyle="--", label="Nu mean")
    ax1.set_xlabel("time")
    ax1.set_ylabel("Nusselt number")
    ax1.grid(True, linestyle=":")

    ax2 = ax1.twinx()
    ax2.plot(t, Vrms, linestyle=":", label="Vrms")
    ax2.set_ylabel("Vrms")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_state(T, vx_p, vy_p, mesh_state, save_path, title):
    T_np = T.detach().cpu().numpy()
    vx_np = vx_p.detach().cpu().numpy()
    vy_np = vy_p.detach().cpu().numpy()

    xp = mesh_state["xp"].detach().cpu().numpy()
    yp = mesh_state["yp"].detach().cpu().numpy()

    X, Y = np.meshgrid(xp, yp)

    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(
        T_np,
        origin="upper",
        aspect="equal",
        extent=[xp.min(), xp.max(), yp.max(), yp.min()],
    )
    fig.colorbar(im, ax=ax, label="T")

    step_y = max(1, T_np.shape[0] // 24)
    step_x = max(1, T_np.shape[1] // 24)

    ax.quiver(
        X[1:-1:step_y, 1:-1:step_x],
        Y[1:-1:step_y, 1:-1:step_x],
        vx_np[1:-1:step_y, 1:-1:step_x],
        vy_np[1:-1:step_y, 1:-1:step_x],
        scale=None,
        width=0.002,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def run_one_resolution(N):
    print("=" * 80)
    print(f"[Benchmark3] Isoviscous thermal convection")
    print(f"N              = {N}")
    print(f"Ra             = {RA:.3e}")
    print(f"dt             = {DT_ND:.3e}")
    print(f"max_steps      = {MAX_STEPS}")
    print(f"buoyancy sign  = {BUOYANCY_SIGN:+.1f}")
    print(f"remove hydro   = {REMOVE_HYDROSTATIC}")
    print("=" * 80)

    mesh, mesh_state = build_mesh_and_state(N)

    coloring_pack = ad.stokes.StokesSystem.build_Fu_coloring_from_mesh(
        mesh_state,
        mode="safe",
        device=device,
        debug_print=False,
    )

    T = initial_temperature(mesh_state, perturb_amp=1.0e-2)

    u0 = None
    rows = []

    steady_counter = 0
    prev_diag = None

    run_dir = os.path.join(SAVE_DIR, f"N{N}_Ra{int(RA):d}")
    os.makedirs(run_dir, exist_ok=True)

    t_start_wall = time.time()

    final_state = None

    for step in range(MAX_STEPS + 1):
        step_wall0 = time.time()

        stokes_state = solve_stokes(
            T=T,
            mesh=mesh,
            mesh_state=mesh_state,
            timestep=step,
            u0=u0,
            coloring_pack=coloring_pack,
        )

        u0 = stokes_state["u"]
        vx = stokes_state["vx"]
        vy = stokes_state["vy"]

        if step % DIAG_INTERVAL == 0:
            Nu_top, Nu_bottom, Nu_mean = compute_nusselt(T, mesh_state)
            Vrms, Vmax, vx_p, vy_p = compute_vrms(mesh_state, vx, vy)
            Tmean = compute_mean_temperature(T)

            row = {
                "N": int(N),
                "Ra": float(RA),
                "step": int(step),
                "time_nd": float(step * DT_ND),
                "dt_nd": float(DT_ND),
                "Nu_top": float(Nu_top),
                "Nu_bottom": float(Nu_bottom),
                "Nu_mean": float(Nu_mean),
                "Vrms": float(Vrms),
                "Vmax": float(Vmax),
                "Tmean": float(Tmean),
                "wall_time_sec": float(time.time() - t_start_wall),
            }
            rows.append(row)

            print(
                f"step={step:05d} "
                f"t={step*DT_ND:.4e} "
                f"Nu_top={Nu_top:.6f} "
                f"Nu_bot={Nu_bottom:.6f} "
                f"Nu={Nu_mean:.6f} "
                f"Vrms={Vrms:.6f} "
                f"Tmean={Tmean:.6f} "
                f"step_time={time.time()-step_wall0:.2f}s"
            )

            if prev_diag is not None and step >= MIN_STEPS:
                rel_Nu = abs(Nu_mean - prev_diag["Nu_mean"]) / max(abs(Nu_mean), 1e-30)
                rel_V = abs(Vrms - prev_diag["Vrms"]) / max(abs(Vrms), 1e-30)

                if rel_Nu < STEADY_TOL and rel_V < STEADY_TOL:
                    steady_counter += 1
                else:
                    steady_counter = 0

                if steady_counter >= STEADY_COUNT_TARGET:
                    print(
                        f"[steady] reached at step={step}, "
                        f"time={step*DT_ND:.6e}, "
                        f"rel_Nu={rel_Nu:.3e}, rel_V={rel_V:.3e}"
                    )
                    final_state = {
                        "T": T.detach().cpu(),
                        "vx": vx.detach().cpu(),
                        "vy": vy.detach().cpu(),
                        "vx_p": vx_p.detach().cpu(),
                        "vy_p": vy_p.detach().cpu(),
                        "mesh_state": {
                            k: v.detach().cpu() if torch.is_tensor(v) else v
                            for k, v in mesh_state.items()
                        },
                        "last_row": row,
                    }
                    break

            prev_diag = {
                "Nu_mean": Nu_mean,
                "Vrms": Vrms,
            }

            if step % SAVE_FIG_INTERVAL == 0:
                fig_path = os.path.join(run_dir, f"state_step_{step:05d}.png")
                plot_state(
                    T,
                    vx_p,
                    vy_p,
                    mesh_state,
                    fig_path,
                    title=f"N={N}, Ra={RA:.0e}, step={step}, t={step*DT_ND:.3e}",
                )

        if step < MAX_STEPS:
            T = advance_temperature(
                T=T,
                vx=vx,
                vy=vy,
                mesh_state=mesh_state,
                timestep=step,
            )

    if final_state is None:
        Nu_top, Nu_bottom, Nu_mean = compute_nusselt(T, mesh_state)
        stokes_state = solve_stokes(
            T=T,
            mesh=mesh,
            mesh_state=mesh_state,
            timestep=MAX_STEPS,
            u0=u0,
            coloring_pack=coloring_pack,
        )
        vx = stokes_state["vx"]
        vy = stokes_state["vy"]
        Vrms, Vmax, vx_p, vy_p = compute_vrms(mesh_state, vx, vy)
        Tmean = compute_mean_temperature(T)

        last_row = {
            "N": int(N),
            "Ra": float(RA),
            "step": int(MAX_STEPS),
            "time_nd": float(MAX_STEPS * DT_ND),
            "dt_nd": float(DT_ND),
            "Nu_top": float(Nu_top),
            "Nu_bottom": float(Nu_bottom),
            "Nu_mean": float(Nu_mean),
            "Vrms": float(Vrms),
            "Vmax": float(Vmax),
            "Tmean": float(Tmean),
            "wall_time_sec": float(time.time() - t_start_wall),
        }

        final_state = {
            "T": T.detach().cpu(),
            "vx": vx.detach().cpu(),
            "vy": vy.detach().cpu(),
            "vx_p": vx_p.detach().cpu(),
            "vy_p": vy_p.detach().cpu(),
            "mesh_state": {
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in mesh_state.items()
            },
            "last_row": last_row,
        }

    csv_path = os.path.join(run_dir, "diagnostics.csv")
    save_csv(rows, csv_path)

    fig_ts = os.path.join(run_dir, "timeseries_Nu_Vrms.png")
    plot_timeseries(
        rows,
        fig_ts,
        title=f"Benchmark3: N={N}, Ra={RA:.0e}",
    )

    mesh_state_cpu = final_state["mesh_state"]
    fig_final = os.path.join(run_dir, "final_state.png")
    plot_state(
        final_state["T"],
        final_state["vx_p"],
        final_state["vy_p"],
        mesh_state_cpu,
        fig_final,
        title=f"Final state: N={N}, Ra={RA:.0e}",
    )

    pt_path = os.path.join(run_dir, "final_state.pt")
    torch.save(final_state, pt_path)

    summary = {
        "benchmark": "isoviscous_thermal_convection",
        "N": int(N),
        "Ra": float(RA),
        "dt_nd": float(DT_ND),
        "max_steps": int(MAX_STEPS),
        "steady_tol": float(STEADY_TOL),
        "steady_count_target": int(STEADY_COUNT_TARGET),
        "buoyancy_sign": float(BUOYANCY_SIGN),
        "remove_hydrostatic": bool(REMOVE_HYDROSTATIC),
        "reference_guidance": {
            "Blankenbach_case_1a_Nu_approx": REF_NU_1A,
            "Blankenbach_case_1a_Vrms_approx": REF_VRMS_1A,
            "note": "Compare only if nondimensional convention matches.",
        },
        "last_row": final_state["last_row"],
        "csv": csv_path,
        "timeseries_figure": fig_ts,
        "final_figure": fig_final,
        "final_state_pt": pt_path,
    }

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 80)
    print(f"[N={N}] finished")
    print(f"CSV          : {csv_path}")
    print(f"Timeseries   : {fig_ts}")
    print(f"Final figure : {fig_final}")
    print(f"Final state  : {pt_path}")
    print(f"Summary      : {summary_path}")
    print("Last diagnostics:")
    print(json.dumps(final_state["last_row"], indent=2))
    print("-" * 80)

    return summary



if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    all_summaries = []

    for N in N_LIST:
        summary = run_one_resolution(N)
        all_summaries.append(summary)

    all_summary_path = os.path.join(SAVE_DIR, "all_summary.json")
    with open(all_summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("=" * 80)
    print("[Benchmark3 all finished]")
    print(f"All summary: {all_summary_path}")
    print("=" * 80)
