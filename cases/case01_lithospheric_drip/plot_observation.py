#!/usr/bin/env python3
'Plot forward observation data for the 500 km lithospheric-drip case in a 5x1 layout.'

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


DEFAULT_L0 = 500e3
DEFAULT_ETA0 = 1e21
DEFAULT_KAPPA0 = 1e-6
DEFAULT_T_SURFACE = 273.0
DEFAULT_DTEMP = 1574.0 - 273.0
DEFAULT_T_MANTLE = 1574.0
DEFAULT_DELTA_RHO = 3300.0
DEFAULT_G = 10.0
YEAR = 365.25 * 24.0 * 3600.0


def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def scalar_from_any(x: Any) -> float:
    arr = to_numpy(x)
    return float(np.asarray(arr).reshape(-1)[0])


def load_observation(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    data = payload.get("data", payload) if isinstance(payload, dict) else payload

    if not isinstance(data, dict):
        raise TypeError("The observation file must contain a dictionary or {'data': dictionary}.")
    return data


def get_series(data: dict[str, Any], key: str, required: bool = True) -> list[Any]:
    if key not in data:
        if required:
            raise KeyError(f"Missing key in observation data: {key}")
        return []

    value = data[key]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if torch.is_tensor(value) and value.ndim >= 1:
        return list(value)

    raise TypeError(f"Expected '{key}' to be a list/tuple/tensor series, got {type(value)}.")


def infer_length_scale(data: dict[str, Any], fallback_l0: float) -> float:
    mesh_state = data.get("mesh_state", None)
    if not isinstance(mesh_state, dict):
        return float(fallback_l0)

    for key in ("xsize", "ysize"):
        if key in mesh_state:
            val = scalar_from_any(mesh_state[key])
            if np.isfinite(val) and val > 100.0:
                return float(val)

    for key in ("xp", "xnode", "xvx", "xvy"):
        if key in mesh_state:
            arr = to_numpy(mesh_state[key]).astype(float)
            span = float(np.nanmax(arr) - np.nanmin(arr))
            if np.isfinite(span) and span > 100.0:
                return span

    return float(fallback_l0)


def get_mesh_array(mesh_state: dict[str, Any], key: str, l0: float) -> np.ndarray:
    if key not in mesh_state:
        raise KeyError(f"mesh_state does not contain '{key}'.")

    arr = to_numpy(mesh_state[key]).astype(float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"mesh_state['{key}'] is empty.")

    if np.nanmax(np.abs(arr)) < 100.0:
        arr = arr * l0

    return arr


def centers_to_edges(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float).reshape(-1)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5], dtype=float)

    mid = 0.5 * (c[:-1] + c[1:])
    first = c[0] - 0.5 * (c[1] - c[0])
    last = c[-1] + 0.5 * (c[-1] - c[-2])
    return np.concatenate([[first], mid, [last]])


def get_field_coordinates(
    data: dict[str, Any],
    field: np.ndarray,
    l0: float,
    grid: str = "p",
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = field.shape
    mesh_state = data.get("mesh_state", None)

    if not isinstance(mesh_state, dict):
        return np.arange(nx, dtype=float), np.arange(ny, dtype=float)

    if grid == "p":
        x_key, y_key = "xp", "yp"
    elif grid == "node":
        x_key, y_key = "xnode", "ynode"
    else:
        raise ValueError(f"Unsupported grid='{grid}'.")

    if x_key in mesh_state and y_key in mesh_state:
        x = get_mesh_array(mesh_state, x_key, l0)
        y = get_mesh_array(mesh_state, y_key, l0)
    else:
        x = np.arange(nx, dtype=float)
        y = np.arange(ny, dtype=float)

    if x.size != nx:
        x = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), nx)
    if y.size != ny:
        y = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), ny)

    return x, y


def get_surface_coordinates(data: dict[str, Any], values: np.ndarray, l0: float) -> np.ndarray:
    n = values.reshape(-1).size
    mesh_state = data.get("mesh_state", None)

    if not isinstance(mesh_state, dict):
        return np.arange(n, dtype=float)

    candidates: list[np.ndarray] = []
    for key in ("xnode", "xp", "xvx", "xvy"):
        if key in mesh_state:
            candidates.append(get_mesh_array(mesh_state, key, l0))

    for x in candidates:
        if x.size == n:
            return x

    if candidates:
        x0 = float(np.nanmin(candidates[0]))
        x1 = float(np.nanmax(candidates[0]))
        return np.linspace(x0, x1, n)

    return np.arange(n, dtype=float)


def crop_field_and_coords(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if field.ndim != 2:
        raise ValueError(f"Expected 2D field, got shape={field.shape}.")
    if field.shape[0] <= 2 or field.shape[1] <= 2:
        return field, x, y
    return field[1:-1, 1:-1], x[1:-1], y[1:-1]


def crop_profile_and_coords(values: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values).reshape(-1)
    x = np.asarray(x).reshape(-1)
    if values.size <= 2 or x.size <= 2:
        return values, x
    return values[1:-1], x[1:-1]


def maybe_temperature_kelvin(T: np.ndarray, t_surface: float, dtemp: float) -> np.ndarray:
    finite = T[np.isfinite(T)]
    if finite.size == 0:
        return T
    if np.nanmax(np.abs(finite)) <= 5.0:
        return t_surface + T * dtemp
    return T


def maybe_viscosity_pas(eta: np.ndarray, eta0: float) -> np.ndarray:
    finite = eta[np.isfinite(eta)]
    if finite.size == 0:
        return eta
    if np.nanmax(np.abs(finite)) < 1e10:
        return eta * eta0
    return eta


def velocity_cm_per_year(v: np.ndarray, kappa0: float, l0: float) -> np.ndarray:
    return v * (kappa0 / l0) * YEAR * 100.0


def dynamic_topography_m(
    sigma: np.ndarray,
    *,
    eta0: float,
    kappa0: float,
    l0: float,
    delta_rho: float,
    g: float,
    remove_mean: bool = False,
    positive_up: bool = True,
) -> np.ndarray:
    stress_scale = eta0 * kappa0 / (l0 ** 2)
    sigma_pa = sigma * stress_scale

    if remove_mean:
        sigma_pa = sigma_pa - np.nanmean(sigma_pa)

    topo = sigma_pa / (delta_rho * g)
    return -topo if positive_up else topo


def safe_log_limits(values: np.ndarray, default: tuple[float, float]) -> tuple[float, float]:
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return default

    vmin = float(np.nanpercentile(finite, 1.0))
    vmax = float(np.nanpercentile(finite, 99.0))

    if not np.isfinite(vmin) or vmin <= 0.0:
        vmin = default[0]
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = max(vmin * 10.0, default[1])
    return vmin, vmax


def choose_steps(n_steps: int, every: int, include_last: bool, explicit_steps: Iterable[int] | None) -> list[int]:
    if n_steps <= 0:
        return []

    if explicit_steps is not None:
        steps = sorted({int(s) for s in explicit_steps if 0 <= int(s) < n_steps})
        if not steps:
            raise ValueError("No valid --steps were provided.")
        return steps

    every = max(1, int(every))
    steps = list(range(0, n_steps, every))
    if include_last and (n_steps - 1) not in steps:
        steps.append(n_steps - 1)
    return steps


def plot_field_pcolormesh(
    ax: plt.Axes,
    x_m: np.ndarray,
    y_m: np.ndarray,
    field: np.ndarray,
    *,
    title: str,
    label: str,
    cmap: str,
    norm: Any | None = None,
):
    xe_km = centers_to_edges(x_m) / 1e3
    ye_km = centers_to_edges(y_m) / 1e3

    pc = ax.pcolormesh(
        xe_km,
        ye_km,
        field,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(title)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("Depth y (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    return pc


def plot_step(
    data: dict[str, Any],
    step: int,
    output_dir: str | Path,
    *,
    l0: float,
    eta0: float,
    kappa0: float,
    t_surface: float,
    dtemp: float,
    t_mantle: float,
    delta_rho: float,
    g: float,
    use_physical_units: bool = True,
    remove_ghosts: bool = False,
    dpi: int = 200,
) -> Path:
    temperature_series = get_series(data, "all_temperature")
    viscosity_series = get_series(data, "all_viscosity_p")
    vx_series = get_series(data, "all_surface_vx")
    sigmayy_series = get_series(data, "all_surface_sigmayy")

    n_steps = min(len(temperature_series), len(viscosity_series), len(vx_series), len(sigmayy_series))
    if step < 0 or step >= n_steps:
        raise IndexError(f"step={step} is out of range for n_steps={n_steps}.")

    T_raw = to_numpy(temperature_series[step]).squeeze()
    eta_raw = to_numpy(viscosity_series[step]).squeeze()
    vx_raw = to_numpy(vx_series[step]).squeeze()
    sig_raw = to_numpy(sigmayy_series[step]).squeeze()

    if T_raw.ndim != 2:
        raise ValueError(f"Temperature must be 2D, got shape={T_raw.shape}.")
    if eta_raw.ndim != 2:
        raise ValueError(f"Viscosity must be 2D, got shape={eta_raw.shape}.")

    x_field, y_field = get_field_coordinates(data, T_raw, l0, grid="p")
    x_vx = get_surface_coordinates(data, vx_raw, l0)
    x_sig = get_surface_coordinates(data, sig_raw, l0)

    if remove_ghosts:
        T_raw, x_crop, y_crop = crop_field_and_coords(T_raw, x_field, y_field)
        eta_raw, _, _ = crop_field_and_coords(eta_raw, x_field, y_field)
        x_field, y_field = x_crop, y_crop
        vx_raw, x_vx = crop_profile_and_coords(vx_raw, x_vx)
        sig_raw, x_sig = crop_profile_and_coords(sig_raw, x_sig)

    if use_physical_units:
        T_plot = maybe_temperature_kelvin(T_raw, t_surface=t_surface, dtemp=dtemp)
        eta_plot = maybe_viscosity_pas(eta_raw, eta0=eta0)
        vx_plot = velocity_cm_per_year(vx_raw, kappa0=kappa0, l0=l0)
        topo_plot = dynamic_topography_m(
            sig_raw,
            eta0=eta0,
            kappa0=kappa0,
            l0=l0,
            delta_rho=delta_rho,
            g=g,
            remove_mean=True,
            positive_up=True,
        )
        T_anom = T_plot - t_mantle
        T_label = "Temperature (K)"
        T_anom_label = "T - T_mantle (K)"
        eta_label = "Viscosity (Pa s)"
        vx_label = "Surface vx (cm/yr)"
        topo_label = "Dynamic topography (m)"
    else:
        T_plot = T_raw
        eta_plot = eta_raw
        vx_plot = vx_raw
        topo_plot = sig_raw
        T_anom = T_raw - np.nanmedian(T_raw)
        T_label = "Temperature (nondim)"
        T_anom_label = "Temperature anomaly (nondim)"
        eta_label = "Viscosity (nondim)"
        vx_label = "Surface vx (nondim)"
        topo_label = "Surface sigma_yy (nondim)"

    eta_vmin, eta_vmax = safe_log_limits(
        eta_plot,
        default=(1e18, 1e24) if use_physical_units else (1e-3, 1e3),
    )

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(8.8, 27.0),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.15, 1.15, 1.15, 0.45, 0.45]},
    )

    ax_T, ax_Tanom, ax_eta, ax_vx, ax_topo = axes

    im0 = plot_field_pcolormesh(
        ax_T,
        x_field,
        y_field,
        T_plot,
        title=f"Step {step}: temperature",
        label=T_label,
        cmap="turbo",
    )
    fig.colorbar(im0, ax=ax_T, label=T_label, fraction=0.046, pad=0.04)

    im1 = plot_field_pcolormesh(
        ax_Tanom,
        x_field,
        y_field,
        T_anom,
        title=f"Step {step}: temperature anomaly",
        label=T_anom_label,
        cmap="coolwarm",
    )
    fig.colorbar(im1, ax=ax_Tanom, label=T_anom_label, fraction=0.046, pad=0.04)

    im2 = plot_field_pcolormesh(
        ax_eta,
        x_field,
        y_field,
        eta_plot,
        title=f"Step {step}: viscosity",
        label=eta_label,
        cmap="viridis",
        norm=LogNorm(vmin=eta_vmin, vmax=eta_vmax),
    )
    fig.colorbar(im2, ax=ax_eta, label=eta_label, fraction=0.046, pad=0.04)

    for ax in (ax_T, ax_Tanom, ax_eta):
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1.0)

    x_min_km = float(np.nanmin(x_field) / 1e3)
    x_max_km = float(np.nanmax(x_field) / 1e3)

    ax_vx.plot(x_vx / 1e3, np.asarray(vx_plot).reshape(-1), linewidth=1.5)
    ax_vx.set_title(f"Step {step}: surface vx")
    ax_vx.set_xlabel("x (km)")
    ax_vx.set_ylabel(vx_label)
    ax_vx.set_xlim(x_min_km, x_max_km)
    ax_vx.grid(True, linestyle=":", linewidth=0.7)

    ax_topo.plot(x_sig / 1e3, np.asarray(topo_plot).reshape(-1), linewidth=1.5)
    ax_topo.set_title(f"Step {step}: dynamic topography")
    ax_topo.set_xlabel("x (km)")
    ax_topo.set_ylabel(topo_label)
    ax_topo.set_xlim(x_min_km, x_max_km)
    ax_topo.grid(True, linestyle=":", linewidth=0.7)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"drip_observation_step_{step:04d}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path

def print_summary(data: dict[str, Any], l0: float, t_surface: float, dtemp: float, eta0: float) -> None:
    print("Observation keys:")
    for key in sorted(data.keys()):
        print(f"  - {key}")

    print(f"Plot length scale L0 = {l0:.6e} m")
    print(f"Temperature conversion: T(K) = {t_surface:.3f} + T_nd * {dtemp:.3f}")
    print(f"Viscosity conversion: eta(Pa s) = eta_nd * {eta0:.3e}")

    if "mesh_state" in data and isinstance(data["mesh_state"], dict):
        mesh_state = data["mesh_state"]
        if "xp" in mesh_state and "yp" in mesh_state:
            xp = get_mesh_array(mesh_state, "xp", l0)
            yp = get_mesh_array(mesh_state, "yp", l0)
            print(f"p-grid x range: {xp.min()/1e3:.3f} to {xp.max()/1e3:.3f} km, n={xp.size}")
            print(f"p-grid y range: {yp.min()/1e3:.3f} to {yp.max()/1e3:.3f} km, n={yp.size}")

    if "all_temperature" in data:
        T0 = to_numpy(data["all_temperature"][0])
        T0_k = maybe_temperature_kelvin(T0, t_surface=t_surface, dtemp=dtemp)
        print(f"Temperature steps: {len(data['all_temperature'])}")
        print(f"Temperature shape: {tuple(T0.shape)}")
        print(f"Initial T range: {np.nanmin(T0_k):.6e} to {np.nanmax(T0_k):.6e} K")

    if "all_viscosity_p" in data:
        eta0_arr = maybe_viscosity_pas(to_numpy(data["all_viscosity_p"][0]), eta0=eta0)
        print(f"Viscosity steps: {len(data['all_viscosity_p'])}")
        print(f"Viscosity shape: {tuple(to_numpy(data['all_viscosity_p'][0]).shape)}")
        print(f"Initial eta range: {np.nanmin(eta0_arr):.6e} to {np.nanmax(eta0_arr):.6e} Pa s")

    if "all_surface_vx" in data:
        print(f"Surface vx steps: {len(data['all_surface_vx'])}")
        print(f"Surface vx shape: {tuple(to_numpy(data['all_surface_vx'][0]).shape)}")

    if "all_surface_sigmayy" in data:
        print(f"Surface sigma_yy steps: {len(data['all_surface_sigmayy'])}")
        print(f"Surface sigma_yy shape: {tuple(to_numpy(data['all_surface_sigmayy'][0]).shape)}")

    if "actual_num_steps" in data:
        print(f"actual_num_steps: {data['actual_num_steps']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot observe_data.pt for the 500 km lithospheric-drip case.",
    )
    parser.add_argument("--input", type=str, default="observe_data.pt", help="Path to observe_data.pt.")
    parser.add_argument("--output-dir", type=str, default="drip_observation_figures", help="Output directory.")
    parser.add_argument("--every", type=int, default=10, help="Plot every N steps.")
    parser.add_argument("--steps", type=int, nargs="*", default=None, help="Explicit step indices to plot.")
    parser.add_argument("--no-last", action="store_true", help="Do not force plotting the last step.")
    parser.add_argument("--dimensionless", action="store_true", help="Plot saved nondimensional values.")
    parser.add_argument("--remove-ghosts", action="store_true", help="Remove one outer layer from fields/profiles.")
    parser.add_argument("--summary-only", action="store_true", help="Only print summary.")
    parser.add_argument("--dpi", type=int, default=200, help="Figure resolution.")

    parser.add_argument("--length-scale", type=float, default=DEFAULT_L0, help="Fallback L0 in meters.")
    parser.add_argument("--eta0", type=float, default=DEFAULT_ETA0, help="Reference viscosity in Pa s.")
    parser.add_argument("--kappa0", type=float, default=DEFAULT_KAPPA0, help="Reference diffusivity in m^2/s.")
    parser.add_argument("--t-surface", type=float, default=DEFAULT_T_SURFACE, help="Temperature offset for nondim T conversion.")
    parser.add_argument("--dtemp", type=float, default=DEFAULT_DTEMP, help="Temperature scale for nondim T conversion.")
    parser.add_argument("--t-mantle", type=float, default=DEFAULT_T_MANTLE, help="Mantle background temperature for anomaly plot.")
    parser.add_argument("--delta-rho", type=float, default=DEFAULT_DELTA_RHO, help="Density contrast for dynamic topography.")
    parser.add_argument("--g", type=float, default=DEFAULT_G, help="Gravity in m/s^2.")

    args = parser.parse_args()

    data = load_observation(args.input)
    l0 = infer_length_scale(data, fallback_l0=args.length_scale)

    print_summary(
        data,
        l0=l0,
        t_surface=args.t_surface,
        dtemp=args.dtemp,
        eta0=args.eta0,
    )

    if args.summary_only:
        return

    n_steps = min(
        len(get_series(data, "all_temperature")),
        len(get_series(data, "all_viscosity_p")),
        len(get_series(data, "all_surface_vx")),
        len(get_series(data, "all_surface_sigmayy")),
    )

    steps = choose_steps(
        n_steps=n_steps,
        every=args.every,
        include_last=not args.no_last,
        explicit_steps=args.steps,
    )

    print(f"Selected steps: {steps}")
    print(f"Remove ghost layer: {args.remove_ghosts}")

    for step in steps:
        output_path = plot_step(
            data,
            step,
            args.output_dir,
            l0=l0,
            eta0=args.eta0,
            kappa0=args.kappa0,
            t_surface=args.t_surface,
            dtemp=args.dtemp,
            t_mantle=args.t_mantle,
            delta_rho=args.delta_rho,
            g=args.g,
            use_physical_units=not args.dimensionless,
            remove_ghosts=args.remove_ghosts,
            dpi=args.dpi,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
