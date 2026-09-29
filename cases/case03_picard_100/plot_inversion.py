#!/usr/bin/env python3
'Composite figures for Case 2.'

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap



L0 = 660e3
XSIZE = 1500e3
YSIZE = 660e3

KAPPA0 = 1e-6
T_SURFACE = 273.0
T_MANTLE = 1574.0
T_BOTTOM = 1574.0
DTEMP = T_MANTLE - T_SURFACE

YEAR = 365.25 * 24.0 * 3600.0



def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_pt(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def try_load_pt(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    return load_pt(path)


def load_summary(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        print(f"[WARN] Missing summary: {path}")
        return None
    return torch.load(path, map_location="cpu")


def get_series(data: dict[str, Any], key: str) -> list[Any]:
    value = data[key]

    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if torch.is_tensor(value):
        return list(value)

    raise TypeError(f"Unsupported series type for {key}: {type(value)}")


def maybe_temperature_kelvin(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T)
    finite = T[np.isfinite(T)]

    if finite.size == 0:
        return T

    if np.nanmax(np.abs(finite)) <= 5.0:
        return T_SURFACE + T * DTEMP

    return T


def velocity_cm_per_year(v: np.ndarray) -> np.ndarray:
    return np.asarray(v) * (KAPPA0 / L0) * YEAR * 100.0



def _maybe_dimensionalize_coord(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)

    if x.size == 0:
        return x

    if np.nanmax(np.abs(x)) < 100.0:
        x = x * L0

    return x


def get_mesh_xy(data: dict[str, Any], field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mesh_state = data.get("mesh_state", None)

    if isinstance(mesh_state, dict) and "xp" in mesh_state and "yp" in mesh_state:
        x = _maybe_dimensionalize_coord(to_numpy(mesh_state["xp"]))
        y = _maybe_dimensionalize_coord(to_numpy(mesh_state["yp"]))

        if x.size != field.shape[1]:
            x = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), field.shape[1])

        if y.size != field.shape[0]:
            y = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), field.shape[0])

    else:
        ny, nx = field.shape
        x = np.linspace(0.0, XSIZE, nx)
        y = np.linspace(0.0, YSIZE, ny)

    return x / 1e3, y / 1e3


def centers_to_edges(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float).reshape(-1)

    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])

    mid = 0.5 * (c[:-1] + c[1:])
    first = c[0] - 0.5 * (c[1] - c[0])
    last = c[-1] + 0.5 * (c[-1] - c[-2])

    return np.concatenate([[first], mid, [last]])


def get_surface_x(data: dict[str, Any], n: int) -> np.ndarray:
    mesh_state = data.get("mesh_state", None)

    if isinstance(mesh_state, dict):
        for key in ("xnode", "xp", "xvx"):
            if key in mesh_state:
                x = _maybe_dimensionalize_coord(to_numpy(mesh_state[key]))

                if x.size == n:
                    return x / 1e3

    return np.linspace(0.0, XSIZE / 1e3, n)



def get_num_saved_steps(data: dict[str, Any]) -> int:
    if "actual_num_steps" in data:
        return int(data["actual_num_steps"])

    if "all_temperature" in data:
        return len(get_series(data, "all_temperature"))

    raise KeyError("Cannot determine number of saved steps.")


def safe_step_index(data: dict[str, Any], step_label: int, series_key: str) -> int:
    series = get_series(data, series_key)
    n = len(series)

    if n == 0:
        raise ValueError(f"Empty time series: {series_key}")

    if step_label <= 0:
        return 0

    return min(step_label - 1, n - 1)


def get_temp_at(
    data: dict[str, Any],
    step_label: int,
    initial_field: np.ndarray | None = None,
) -> np.ndarray:
    if step_label == 0:
        if initial_field is not None:
            return maybe_temperature_kelvin(initial_field)

        idx = safe_step_index(data, 1, "all_temperature")
        return maybe_temperature_kelvin(to_numpy(get_series(data, "all_temperature")[idx]))

    idx = safe_step_index(data, step_label, "all_temperature")
    T = to_numpy(get_series(data, "all_temperature")[idx])

    return maybe_temperature_kelvin(T)


def get_vx_at(data: dict[str, Any], step_label: int) -> np.ndarray:
    idx = safe_step_index(data, step_label, "all_surface_vx")
    vx = to_numpy(get_series(data, "all_surface_vx")[idx]).reshape(-1)

    return velocity_cm_per_year(vx)


def get_initial_field_for_row(
    row_name: str,
    data: dict[str, Any],
    summary: dict[str, Any] | None,
) -> np.ndarray | None:
    if summary is None:
        return None

    if row_name == "True":
        key = "tkp0_ref"
    elif row_name == "Initial":
        key = "tkp_bg"
    elif row_name == "Final":
        key = "tkp0_final"
    else:
        return None

    if key in summary:
        return maybe_temperature_kelvin(to_numpy(summary[key]))

    return None



def make_temperature_cmap(
    vmin: float,
    vmax: float,
    tref: float = T_MANTLE,
):
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return plt.get_cmap("turbo")

    p = (tref - vmin) / (vmax - vmin)
    p = float(np.clip(p, 0.05, 0.96))

    return LinearSegmentedColormap.from_list(
        "temperature_1574_red",
        [
            (0.0, "#313695"),
            (0.45 * p, "#74add1"),
            (0.75 * p, "#fee090"),
            (p, "#d73027"),
            (1.0, "#7f0000"),
        ],
    )


def compute_temperature_limits(
    rows: list[tuple[str, dict[str, Any]]],
    step_labels: list[int],
    summary: dict[str, Any] | None,
) -> tuple[float, float]:
    all_T = []

    for row_name, data in rows:
        initial_field = get_initial_field_for_row(row_name, data, summary)

        for step in step_labels:
            T = get_temp_at(data, step, initial_field=initial_field)
            all_T.append(T)

    all_T_arr = np.concatenate(
        [T[np.isfinite(T)].reshape(-1) for T in all_T if np.isfinite(T).any()]
    )

    data_min = float(np.nanpercentile(all_T_arr, 0.5))
    data_max = float(np.nanpercentile(all_T_arr, 99.5))

    vmin = min(T_SURFACE, data_min)
    vmax = max(T_MANTLE, data_max)

    if vmax <= T_MANTLE:
        vmax = T_MANTLE + 1.0

    return float(vmin), float(vmax)



def plot_temperature_with_vx(
    ax: plt.Axes,
    data: dict[str, Any],
    true_data: dict[str, Any],
    T: np.ndarray,
    vx: np.ndarray,
    vx_true: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap,
    vx_ylim: tuple[float, float] = (-5, 5),
):
    x_km, y_km = get_mesh_xy(data, T)
    xe = centers_to_edges(x_km)
    ye = centers_to_edges(y_km)

    im = ax.pcolormesh(
        xe,
        ye,
        T,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False,
    )

    inset = ax.inset_axes([0.0, 1.00, 1.0, 0.22])

    x_vx = get_surface_x(data, vx.size)
    x_true = get_surface_x(true_data, vx_true.size)

    x_min = float(xe[0])
    x_max = float(xe[-1])

    inset.plot(
        x_true,
        vx_true,
        color="black",
        linewidth=1.0,
        label="true",
    )
    inset.plot(
        x_vx,
        vx,
        color="red",
        linewidth=1.0,
        linestyle="--",
        label="pred",
    )

    inset.set_xlim(x_min, x_max)
    inset.set_ylim(vx_ylim)
    inset.margins(x=0.0)

    inset.set_xticks([])
    inset.set_yticks([vx_ylim[0], 0.0, vx_ylim[1]])
    inset.tick_params(axis="x", bottom=False, labelbottom=False)
    inset.tick_params(axis="y", labelsize=6, length=2)

    inset.grid(True, axis="y", linestyle=":", linewidth=0.4)

    for spine in inset.spines.values():
        spine.set_linewidth(0.5)

    return im


def plot_loss_panel(ax: plt.Axes, summary: dict[str, Any]):
    history = summary.get("history", [])

    if len(history) == 0:
        ax.text(0.5, 0.5, "No history found", ha="center", va="center")
        ax.axis("off")
        return

    value_max = 1e2

    closure = np.array(
        [h.get("closure", i + 1) for i, h in enumerate(history)],
        dtype=float,
    )
    total = np.array(
        [h.get("total_loss", np.nan) for h in history],
        dtype=float,
    )
    loss_T = np.array(
        [h.get("loss_T", np.nan) for h in history],
        dtype=float,
    )
    loss_vx = np.array(
        [h.get("loss_vx", np.nan) for h in history],
        dtype=float,
    )
    grad_dT = np.array(
        [h.get("grad_dT", np.nan) for h in history],
        dtype=float,
    )

    valid = (
        np.isfinite(closure)
        & np.isfinite(total)
        & np.isfinite(loss_T)
        & np.isfinite(loss_vx)
        & np.isfinite(grad_dT)
        & (total > 0.0)
        & (loss_T > 0.0)
        & (loss_vx > 0.0)
        & (grad_dT > 0.0)
        & (total < value_max)
        & (loss_T < value_max)
        & (loss_vx < value_max)
        & (grad_dT < value_max)
    )

    n_all = closure.size
    n_valid = int(np.sum(valid))

    if n_valid == 0:
        ax.text(
            0.5,
            0.5,
            "No valid values below 1e2",
            ha="center",
            va="center",
        )
        ax.axis("off")
        return

    closure = closure[valid]
    total = total[valid]
    loss_T = loss_T[valid]
    loss_vx = loss_vx[valid]
    grad_dT = grad_dT[valid]

    ax.semilogy(
        closure,
        total,
        linewidth=1.6,
        label="total",
    )
    ax.semilogy(
        closure,
        loss_T,
        linewidth=1.3,
        label="final T",
    )
    ax.semilogy(
        closure,
        loss_vx,
        linewidth=1.3,
        label="surface vx",
    )
    ax.semilogy(
        closure,
        grad_dT,
        linewidth=1.2,
        linestyle=":",
        label=r"$||g_T||$",
    )

    ax.set_xlabel("Closure / function evaluation")
    ax.set_ylabel("Value")
    ax.set_title("Inversion convergence")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)

    ax.legend(ncol=4, fontsize=8)

    ax.text(
        0.98,
        0.04,
        f"kept {n_valid}/{n_all} points",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
def plot_result_figure(
    rows: list[tuple[str, dict[str, Any]]],
    step_labels: list[int],
    true_data: dict[str, Any],
    summary: dict[str, Any] | None,
    output_dir: Path,
):
    vmin, vmax = compute_temperature_limits(rows, step_labels, summary)
    cmap = make_temperature_cmap(vmin, vmax, tref=T_MANTLE)

    fig = plt.figure(figsize=(13.5, 9.2), constrained_layout=True)

    gs = GridSpec(
        nrows=len(rows),
        ncols=len(step_labels),
        figure=fig,
        height_ratios=[1.0] * len(rows),
    )

    temp_axes = []
    last_im = None

    column_labels = []
    for step in step_labels:
        if step == 0:
            column_labels.append("Initial")
        else:
            column_labels.append(f"{step} steps")

    for i, (row_name, data) in enumerate(rows):
        initial_field = get_initial_field_for_row(row_name, data, summary)

        for j, step in enumerate(step_labels):
            ax = fig.add_subplot(gs[i, j])
            temp_axes.append(ax)

            T = get_temp_at(data, step, initial_field=initial_field)
            vx = get_vx_at(data, step)
            vx_true = get_vx_at(true_data, step)

            last_im = plot_temperature_with_vx(
                ax,
                data,
                true_data,
                T,
                vx,
                vx_true,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                vx_ylim=(-5.0, 20.0),
            )

            if i == 0:
                ax.text(
                    0.5,
                    1.28,
                    column_labels[j],
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            if j == 0:
                ax.text(
                    -0.045,
                    0.5,
                    row_name,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=10,
                )

    cbar = fig.colorbar(
        last_im,
        ax=temp_axes,
        orientation="vertical",
        fraction=0.025,
        pad=0.015,
    )
    cbar.set_label("Temperature (K)")

    tick_values = [T_SURFACE, 800.0, 1200.0, T_MANTLE]
    tick_values = [v for v in tick_values if vmin <= v <= vmax]
    if vmax > T_MANTLE:
        tick_values.append(vmax)

    tick_values = sorted(set([round(float(v), 6) for v in tick_values]))
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{v:.0f}" for v in tick_values])

    png_path = output_dir / "case3_true_initial_iter50_final_snapshots.png"
    pdf_path = output_dir / "case3_true_initial_iter50_final_snapshots.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_loss_figure(
    summary: dict[str, Any] | None,
    output_dir: Path,
):
    fig, ax = plt.subplots(figsize=(8.0, 3.2), constrained_layout=True)

    if summary is not None:
        plot_loss_panel(ax, summary)
    else:
        ax.text(
            0.5,
            0.5,
            "tkp_inversion_summary.pt not found",
            ha="center",
            va="center",
        )
        ax.axis("off")

    png_path = output_dir / "case3_loss_history.png"
    pdf_path = output_dir / "case3_loss_history.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")



def main():
    true_path = Path("observe_data.pt")
    initial_path = Path("observed_1.pt")
    iter50_path = Path("observed_50.pt")
    final_path = Path("observed_201.pt")
    summary_path = Path("tkp_inversion_summary.pt")

    output_dir = Path("case3_composite_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    true_data = try_load_pt(true_path)
    initial_data = try_load_pt(initial_path)
    iter50_data = try_load_pt(iter50_path)
    final_data = try_load_pt(final_path)

    if true_data is None:
        raise FileNotFoundError("observe_data.pt is required.")

    summary = load_summary(summary_path)

    rows: list[tuple[str, dict[str, Any]]] = [("True", true_data)]

    if initial_data is not None:
        rows.append(("Initial", initial_data))

    if iter50_data is not None:
        rows.append(("Iter. 50", iter50_data))

    if final_data is not None:
        rows.append(("Final", final_data))

    nsteps = get_num_saved_steps(true_data)
    mid_step = max(1, nsteps // 2)
    step_labels = [0, mid_step, nsteps]

    print(f"[INFO] nsteps = {nsteps}")
    print(f"[INFO] step_labels = {step_labels}")
    print(f"[INFO] rows = {[r[0] for r in rows]}")

    plot_result_figure(
        rows=rows,
        step_labels=step_labels,
        true_data=true_data,
        summary=summary,
        output_dir=output_dir,
    )

    plot_loss_figure(
        summary=summary,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
