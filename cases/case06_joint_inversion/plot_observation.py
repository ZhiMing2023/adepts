#!/usr/bin/env python3
'Plot forward observation data saved by the thermo-mechanical model.'

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


L0 = 660e3
ETA0 = 1e21
KAPPA0 = 1e-6
T_SURFACE = 273.0
T_BOTTOM = 1874.0
DTEMP = T_BOTTOM - T_SURFACE

YEAR = 365.25 * 24.0 * 3600.0
VELOCITY_SCALE = KAPPA0 / L0          # m/s
STRESS_SCALE = ETA0 * KAPPA0 / L0**2  # Pa
DELTA_RHO = 3300.0                    # kg/m^3
G_PHYS = 10.0                         # m/s^2


def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_observation(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict) and "data" in payload:
        data = payload["data"]
    else:
        data = payload

    if not isinstance(data, dict):
        raise TypeError("The observation file must contain a dictionary or {'data': dictionary}.")

    return data


def get_series(data: dict[str, Any], key: str) -> list[Any]:
    if key not in data:
        raise KeyError(f"Missing key in observation data: {key}")

    value = data[key]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)

    raise TypeError(f"Expected '{key}' to be a list or tuple, got {type(value)}.")


def get_mesh_array(mesh_state: dict[str, Any], key: str) -> np.ndarray:
    if key not in mesh_state:
        raise KeyError(f"mesh_state does not contain '{key}'.")

    arr = to_numpy(mesh_state[key]).astype(float)

    if np.nanmax(np.abs(arr)) < 100.0:
        arr = arr * L0

    return arr


def get_field_coordinates(data: dict[str, Any], field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if "mesh_state" not in data:
        ny, nx = field.shape
        return np.arange(nx, dtype=float), np.arange(ny, dtype=float)

    mesh_state = data["mesh_state"]
    x = get_mesh_array(mesh_state, "xp")
    y = get_mesh_array(mesh_state, "yp")

    ny, nx = field.shape

    if x.size != nx:
        x = np.linspace(float(x.min()), float(x.max()), nx)
    if y.size != ny:
        y = np.linspace(float(y.min()), float(y.max()), ny)

    return x, y


def get_surface_coordinates(data: dict[str, Any], values: np.ndarray) -> np.ndarray:
    n = values.size

    if "mesh_state" not in data:
        return np.arange(n, dtype=float)

    mesh_state = data["mesh_state"]

    candidates = []
    for key in ("xnode", "xp", "xvx", "xvy"):
        if key in mesh_state:
            x = get_mesh_array(mesh_state, key)
            candidates.append(x)

    for x in candidates:
        if x.size == n:
            return x

    if len(candidates) > 0:
        x0 = float(candidates[0].min())
        x1 = float(candidates[0].max())
        return np.linspace(x0, x1, n)

    return np.arange(n, dtype=float)


def crop_ghost_field(field: np.ndarray) -> np.ndarray:
    if field.ndim != 2:
        raise ValueError(f"Expected a 2D field, got shape={field.shape}.")

    if field.shape[0] <= 2 or field.shape[1] <= 2:
        return field

    return field[1:-1, 1:-1]


def crop_ghost_profile(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        values = values.reshape(-1)

    if values.size <= 2:
        return values

    return values[1:-1]


def crop_coordinate(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).reshape(-1)

    if values.size <= 2:
        return values

    return values[1:-1]


def choose_steps(n_steps: int, every: int, include_last: bool) -> list[int]:
    if n_steps <= 0:
        return []

    every = max(1, int(every))
    steps = list(range(0, n_steps, every))

    if include_last and (n_steps - 1) not in steps:
        steps.append(n_steps - 1)

    return steps


def maybe_temperature_kelvin(T: np.ndarray) -> np.ndarray:
    if np.nanmax(T) <= 5.0:
        return T * DTEMP + T_SURFACE
    return T


def maybe_viscosity_pas(eta: np.ndarray) -> np.ndarray:
    if np.nanmax(eta) < 1e10:
        return eta * ETA0
    return eta


def velocity_cm_per_year(v: np.ndarray) -> np.ndarray:
    return v * VELOCITY_SCALE * YEAR * 100.0


def dynamic_topography_m(
    sigma: np.ndarray,
    *,
    delta_rho: float = DELTA_RHO,
    remove_mean: bool = False,
    positive_up: bool = True,
) -> np.ndarray:
    sigma_pa = sigma * STRESS_SCALE

    if remove_mean:
        sigma_pa = sigma_pa - np.nanmean(sigma_pa)

    topography = sigma_pa / (delta_rho * G_PHYS)

    if positive_up:
        topography = -topography

    return topography


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


def plot_step(
    data: dict[str, Any],
    step: int,
    output_dir: str | Path,
    *,
    use_physical_units: bool = True,
    remove_ghosts: bool = True,
    dpi: int = 200,
) -> Path:
    temperature_series = get_series(data, "all_temperature")
    viscosity_series = get_series(data, "all_viscosity_p")
    vx_series = get_series(data, "all_surface_vx")
    sigmayy_series = get_series(data, "all_surface_sigmayy")

    n_steps = min(
        len(temperature_series),
        len(viscosity_series),
        len(vx_series),
        len(sigmayy_series),
    )

    if step < 0 or step >= n_steps:
        raise IndexError(f"step={step} is out of range for n_steps={n_steps}.")

    T = to_numpy(temperature_series[step]).squeeze()
    eta = to_numpy(viscosity_series[step]).squeeze()
    vx = to_numpy(vx_series[step]).squeeze()
    sigmayy = to_numpy(sigmayy_series[step]).squeeze()

    if T.ndim != 2:
        raise ValueError(f"Temperature must be 2D, got shape={T.shape}.")
    if eta.ndim != 2:
        raise ValueError(f"Viscosity must be 2D, got shape={eta.shape}.")

    x_field, y_field = get_field_coordinates(data, T)
    x_surface_vx = get_surface_coordinates(data, vx)
    x_surface_sigmayy = get_surface_coordinates(data, sigmayy)

    if remove_ghosts:
        T = crop_ghost_field(T)
        eta = crop_ghost_field(eta)
        vx = crop_ghost_profile(vx)
        sigmayy = crop_ghost_profile(sigmayy)
        x_field = crop_coordinate(x_field)
        y_field = crop_coordinate(y_field)
        x_surface_vx = crop_coordinate(x_surface_vx)
        x_surface_sigmayy = crop_coordinate(x_surface_sigmayy)

    if use_physical_units:
        T_plot = maybe_temperature_kelvin(T)
        eta_plot = maybe_viscosity_pas(eta)
        vx_plot = velocity_cm_per_year(vx)
        topo_plot = dynamic_topography_m(sigmayy)

        T_label = "Temperature (K)"
        eta_label = "Viscosity (Pa s)"
        vx_label = "Surface vx (cm/yr)"
        topo_label = "Dynamic topography (m)"
    else:
        T_plot = T
        eta_plot = eta
        vx_plot = vx
        topo_plot = sigmayy

        T_label = "Temperature"
        eta_label = "Viscosity"
        vx_label = "Surface vx"
        topo_label = "Surface sigma_yy"

    x_field_km = x_field / 1e3
    y_field_km = y_field / 1e3
    x_surface_vx_km = x_surface_vx / 1e3
    x_surface_sigmayy_km = x_surface_sigmayy / 1e3

    extent = [
        float(x_field_km.min()),
        float(x_field_km.max()),
        float(y_field_km.max()),
        float(y_field_km.min()),
    ]

    eta_vmin, eta_vmax = safe_log_limits(eta_plot, default=(1e18, 1e24))

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(9.0, 13.0),
        constrained_layout=True,
    )

    im0 = axes[0].imshow(
        T_plot,
        origin="upper",
        extent=extent,
        aspect="auto",
        cmap="turbo",
    )
    axes[0].set_title(f"Step {step}: temperature")
    axes[0].set_xlabel("x (km)")
    axes[0].set_ylabel("Depth (km)")
    fig.colorbar(im0, ax=axes[0], label=T_label)

    im1 = axes[1].imshow(
        eta_plot,
        origin="upper",
        extent=extent,
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=eta_vmin, vmax=eta_vmax),
    )
    axes[1].set_title(f"Step {step}: viscosity")
    axes[1].set_xlabel("x (km)")
    axes[1].set_ylabel("Depth (km)")
    fig.colorbar(im1, ax=axes[1], label=eta_label)

    axes[2].plot(x_surface_vx_km, vx_plot, linewidth=1.5)
    axes[2].set_title(f"Step {step}: surface vx")
    axes[2].set_xlabel("x (km)")
    axes[2].set_ylabel(vx_label)
    axes[2].grid(True, linestyle=":", linewidth=0.7)

    axes[3].plot(x_surface_sigmayy_km, topo_plot, linewidth=1.5)
    axes[3].set_title(f"Step {step}: dynamic topography")
    axes[3].set_xlabel("x (km)")
    axes[3].set_ylabel(topo_label)
    axes[3].grid(True, linestyle=":", linewidth=0.7)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"observation_step_{step:04d}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def print_summary(data: dict[str, Any]) -> None:
    keys = sorted(data.keys())
    print("Observation keys:")
    for key in keys:
        print(f"  - {key}")

    if "all_temperature" in data:
        T0 = to_numpy(data["all_temperature"][0])
        print(f"Temperature steps: {len(data['all_temperature'])}")
        print(f"Temperature shape: {tuple(T0.shape)}")
        if T0.ndim == 2 and min(T0.shape) > 2:
            T0_inner = T0[1:-1, 1:-1]
            print(
                "Temperature inner range: "
                f"min={np.nanmin(T0_inner):.6e}, max={np.nanmax(T0_inner):.6e}"
            )

    if "all_viscosity_p" in data:
        print(f"Viscosity steps: {len(data['all_viscosity_p'])}")
        print(f"Viscosity shape: {tuple(to_numpy(data['all_viscosity_p'][0]).shape)}")

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
        description="Plot saved forward observation data. Surface sigma_yy is converted to dynamic topography in physical-unit mode.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="observe_data.pt",
        help="Path to the saved observation file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="observation_figures",
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=10,
        help="Plot every N steps.",
    )
    parser.add_argument(
        "--no-last",
        action="store_true",
        help="Do not force plotting the last step.",
    )
    parser.add_argument(
        "--dimensionless",
        action="store_true",
        help="Use saved nondimensional values without physical conversion.",
    )
    parser.add_argument(
        "--keep-ghosts",
        action="store_true",
        help="Keep the outer ghost layer when plotting.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure resolution.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print the observation summary and do not save figures.",
    )

    args = parser.parse_args()

    data = load_observation(args.input)
    print_summary(data)

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
    )

    print(f"Selected steps: {steps}")
    print(f"Remove ghost layer: {not args.keep_ghosts}")

    for step in steps:
        output_path = plot_step(
            data,
            step,
            args.output_dir,
            use_physical_units=not args.dimensionless,
            remove_ghosts=not args.keep_ghosts,
            dpi=args.dpi,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
