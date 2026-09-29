#!/usr/bin/env python3
'Plot the Case 06 inversion summary.'

from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


LANGUAGE = (
    "chinese"
    if len(sys.argv) > 1
    and sys.argv[1].lower() in {"chinese", "cn", "zh", "zh-cn"}
    else "english"
)


def figure_text(english: str, chinese: str) -> str:
    return chinese if LANGUAGE == "chinese" else english


def configure_plot_fonts():
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["axes.unicode_minus"] = False

    if LANGUAGE == "chinese":
        chinese_font_path = None
        for family in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "PingFang SC"):
            try:
                chinese_font_path = Path(
                    font_manager.findfont(family, fallback_to_default=False)
                )
                break
            except ValueError:
                continue

        if chinese_font_path is not None:
            font_manager.fontManager.addfont(str(chinese_font_path))
            chinese_font_name = font_manager.FontProperties(
                fname=str(chinese_font_path)
            ).get_name()

            matplotlib.rcParams["font.family"] = chinese_font_name
            matplotlib.rcParams["font.sans-serif"] = [chinese_font_name]
        else:
            print(
                "WARNING: No Chinese font was found. "
                "Chinese characters in figures may not render correctly."
            )
    else:
        matplotlib.rcParams["font.family"] = [
            "Times New Roman",
            "DejaVu Serif",
        ]


configure_plot_fonts()


def localized_row_name(row_name: str) -> str:
    if LANGUAGE != "chinese":
        return row_name

    mapping = {
        "True": "真实模型",
        "Initial": "初始反演",
        "Iter. 50": "第 50 次迭代",
        "Iter. 51": "第 51 次迭代",
        "Iter. 100": "第 100 次迭代",
        "Intermediate": "中间反演",
        "Final": "最终反演",
    }
    return mapping.get(row_name, row_name)
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap



L0 = 660e3
XSIZE = 1500e3
YSIZE = 660e3

ETA0 = 1e21
KAPPA0 = 1e-6
G_PHYS = 10.0
P0 = ETA0 * KAPPA0 / L0 ** 2

T_SURFACE = 273.0
T_MANTLE = 1574.0
T_BOTTOM = 1874.0
DTEMP = T_BOTTOM - T_SURFACE

YEAR = 365.25 * 24.0 * 3600.0

RHO2_TRUE = 3350.0
A_TRUE = 21.0
N_TRUE = 3.0

SURFACE_VX_YLIM = (-5.0, 12.0)       # cm/yr
TOPO_YLIM = (-15000.0, 15000.0)          # m
TOPO_SIGN = 1.0                      # change to -1.0 if topography sign is reversed
RHO_TOPO = 3300.0                    # kg/m^3

BAD_VALUE_THRESHOLD = 1e2



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


def sigmayy_to_topography_m(sigmayy: np.ndarray) -> np.ndarray:
    sigmayy = np.asarray(sigmayy)
    return TOPO_SIGN * sigmayy * P0 / (RHO_TOPO * G_PHYS)



def _maybe_dimensionalize_coord(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)

    if x.size == 0:
        return x

    if np.nanmax(np.abs(x)) < 100.0:
        x = x * L0

    return x


def get_mesh_xy(
    data: dict[str, Any],
    field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mesh_state = data.get("mesh_state", None)

    if isinstance(mesh_state, dict) and "xp" in mesh_state and "yp" in mesh_state:
        x = _maybe_dimensionalize_coord(to_numpy(mesh_state["xp"]))
        y = _maybe_dimensionalize_coord(to_numpy(mesh_state["yp"]))

        if x.size != field.shape[1]:
            x = np.linspace(
                float(np.nanmin(x)),
                float(np.nanmax(x)),
                field.shape[1],
            )

        if y.size != field.shape[0]:
            y = np.linspace(
                float(np.nanmin(y)),
                float(np.nanmax(y)),
                field.shape[0],
            )

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


def safe_step_index(
    data: dict[str, Any],
    step_label: int,
    series_key: str,
) -> int:
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
        return maybe_temperature_kelvin(
            to_numpy(get_series(data, "all_temperature")[idx])
        )

    idx = safe_step_index(data, step_label, "all_temperature")
    T = to_numpy(get_series(data, "all_temperature")[idx])

    return maybe_temperature_kelvin(T)


def get_vx_at(data: dict[str, Any], step_label: int) -> np.ndarray:
    idx = safe_step_index(data, step_label, "all_surface_vx")
    vx = to_numpy(get_series(data, "all_surface_vx")[idx]).reshape(-1)

    return velocity_cm_per_year(vx)


def get_sigmayy_topography_at(
    data: dict[str, Any],
    step_label: int,
) -> np.ndarray | None:
    if "all_surface_sigmayy" in data:
        key = "all_surface_sigmayy"
    elif "all_surface_traction" in data:
        key = "all_surface_traction"
    else:
        return None

    idx = safe_step_index(data, step_label, key)
    sig = to_numpy(get_series(data, key)[idx]).reshape(-1)

    return sigmayy_to_topography_m(sig)


def get_initial_field_for_row(
    row_name: str,
    data: dict[str, Any],
    summary: dict[str, Any] | None,
) -> np.ndarray | None:
    if summary is None:
        return None

    if row_name == "True":
        candidate_keys = ["tkp0_ref"]
    elif row_name == "Initial":
        candidate_keys = ["tkp_bg"]
    elif row_name == "Final":
        candidate_keys = ["tkp0_final"]
    elif row_name in ("Iter. 50", "Iter. 51", "Iter. 100", "Intermediate"):
        candidate_keys = [
            "tkp0_iter50",
            "tkp0_iter51",
            "tkp0_iter100",
            "tkp0_50",
            "tkp0_51",
            "tkp0_100",
            "tkp0_mid",
        ]
    else:
        candidate_keys = []

    for key in candidate_keys:
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
    vmax = max(T_BOTTOM, data_max)

    return float(vmin), float(vmax)



def plot_temperature_with_surface_inset(
    ax: plt.Axes,
    data: dict[str, Any],
    true_data: dict[str, Any],
    T: np.ndarray,
    vx: np.ndarray,
    vx_true: np.ndarray,
    topo: np.ndarray | None,
    topo_true: np.ndarray | None,
    *,
    vmin: float,
    vmax: float,
    cmap,
    vx_ylim: tuple[float, float] = SURFACE_VX_YLIM,
    topo_ylim: tuple[float, float] = TOPO_YLIM,
    show_left_axis: bool = True,
    show_right_axis: bool = True,
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

    inset = ax.inset_axes([0.0, 1.00, 1.0, 0.25])

    x_vx = get_surface_x(data, vx.size)
    x_true = get_surface_x(true_data, vx_true.size)

    x_min = float(xe[0])
    x_max = float(xe[-1])

    inset.plot(
        x_true,
        vx_true,
        color="black",
        linewidth=1.0,
    )
    inset.plot(
        x_vx,
        vx,
        color="red",
        linewidth=1.0,
        linestyle="--",
    )

    inset.set_xlim(x_min, x_max)
    inset.set_ylim(vx_ylim)
    inset.margins(x=0.0)

    inset.set_xticks([])

    inset.set_yticks([vx_ylim[0], 0.0, vx_ylim[1]])
    inset.tick_params(axis="x", bottom=False, labelbottom=False)

    if show_left_axis:
        inset.tick_params(axis="y", labelsize=6, length=2, labelleft=True, left=True)
    else:
        inset.tick_params(axis="y", labelleft=False, left=False)
        inset.spines["left"].set_visible(False)

    inset.grid(True, axis="y", linestyle=":", linewidth=0.4)

    inset_r = inset.twinx()

    if topo_true is not None:
        x_topo_true = get_surface_x(true_data, topo_true.size)
        inset_r.plot(
            x_topo_true,
            topo_true,
            color="0.35",
            linewidth=0.9,
        )

    if topo is not None:
        x_topo = get_surface_x(data, topo.size)
        inset_r.plot(
            x_topo,
            topo,
            color="tab:orange",
            linewidth=0.9,
            linestyle="--",
        )

    inset_r.set_xlim(x_min, x_max)
    inset_r.set_ylim(topo_ylim)

    inset_r.set_yticks([topo_ylim[0], 0.0, topo_ylim[1]])

    if show_right_axis:
        inset_r.tick_params(axis="y", labelsize=6, length=2, labelright=True, right=True)
    else:
        inset_r.tick_params(axis="y", labelright=False, right=False)
        inset_r.spines["right"].set_visible(False)

    for spine in inset.spines.values():
        spine.set_linewidth(0.5)

    for spine in inset_r.spines.values():
        spine.set_linewidth(0.5)

    return im


def plot_result_figure(
    rows: list[tuple[str, dict[str, Any]]],
    step_labels: list[int],
    true_data: dict[str, Any],
    summary: dict[str, Any] | None,
    output_dir: Path,
):
    vmin, vmax = compute_temperature_limits(rows, step_labels, summary)
    cmap = make_temperature_cmap(vmin, vmax, tref=T_MANTLE)

    fig = plt.figure(
        figsize=(13.5, 9.2),
        dpi=1200,
        constrained_layout=True,
    )

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
            column_labels.append(
                figure_text("Initial", "初始时刻")
            )
        else:
            column_labels.append(
                figure_text(
                    f"{step} steps",
                    f"{step} 步",
                )
            )

    for i, (row_name, data) in enumerate(rows):
        initial_field = get_initial_field_for_row(row_name, data, summary)

        for j, step in enumerate(step_labels):
            ax = fig.add_subplot(gs[i, j])
            temp_axes.append(ax)

            T = get_temp_at(data, step, initial_field=initial_field)

            vx = get_vx_at(data, step)
            vx_true = get_vx_at(true_data, step)

            topo = get_sigmayy_topography_at(data, step)
            topo_true = get_sigmayy_topography_at(true_data, step)

            last_im = plot_temperature_with_surface_inset(
                ax,
                data,
                true_data,
                T,
                vx,
                vx_true,
                topo,
                topo_true,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                vx_ylim=SURFACE_VX_YLIM,
                topo_ylim=TOPO_YLIM,
                show_left_axis=(j == 0),
                show_right_axis=(j == len(step_labels) - 1),
            )
            if i == 0:
                ax.text(
                    0.5,
                    1.30,
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
                    localized_row_name(row_name),
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
    cbar.set_label(
        figure_text("Temperature (K)", "温度（K）")
    )

    tick_values = [T_SURFACE, 800.0, 1200.0, T_MANTLE, T_BOTTOM]
    tick_values = [v for v in tick_values if vmin <= v <= vmax]

    if vmax > T_BOTTOM:
        tick_values.append(vmax)

    tick_values = sorted(set([round(float(v), 6) for v in tick_values]))
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{v:.0f}" for v in tick_values])

    png_path = output_dir / "case_joint_snapshots.png"
    pdf_path = output_dir / "case_joint_snapshots.pdf"

    fig.savefig(png_path, dpi=1200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")



def plot_loss_panel(
    ax: plt.Axes,
    summary: dict[str, Any],
) -> None:
    history = summary.get("history", [])

    if len(history) == 0:
        ax.text(
            0.5,
            0.5,
            figure_text("No history found", "未找到迭代历史"),
            ha="center",
            va="center",
        )
        ax.axis("off")
        return

    closure = np.asarray(
        [record.get("closure", i + 1) for i, record in enumerate(history)],
        dtype=float,
    )
    total = np.asarray(
        [record.get("total_loss", np.nan) for record in history],
        dtype=float,
    )
    loss_T = np.asarray(
        [record.get("loss_T", np.nan) for record in history],
        dtype=float,
    )
    loss_vx = np.asarray(
        [record.get("loss_vx", np.nan) for record in history],
        dtype=float,
    )
    grad_dT = np.asarray(
        [record.get("grad_dT", np.nan) for record in history],
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
        & (total < BAD_VALUE_THRESHOLD)
        & (loss_T < BAD_VALUE_THRESHOLD)
        & (loss_vx < BAD_VALUE_THRESHOLD)
        & (grad_dT < BAD_VALUE_THRESHOLD)
    )

    n_all = closure.size
    n_valid = int(np.sum(valid))

    if n_valid == 0:
        ax.text(
            0.5,
            0.5,
            figure_text(
                f"No valid values below {BAD_VALUE_THRESHOLD:.0e}",
                f"没有低于 {BAD_VALUE_THRESHOLD:.0e} 的有效值",
            ),
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
        label=figure_text("total", "总损失"),
    )
    ax.semilogy(
        closure,
        loss_T,
        linewidth=1.3,
        label=figure_text("final T", "最终温度"),
    )
    ax.semilogy(
        closure,
        loss_vx,
        linewidth=1.3,
        label=figure_text(r"surface $v_x$", r"表面 $v_x$"),
    )
    ax.semilogy(
        closure,
        grad_dT,
        linewidth=1.2,
        linestyle=":",
        label=r"$\|g_T\|$",
    )

    ax.set_title(
        figure_text("Loss convergence", "损失收敛"),
        fontsize=10.5,
    )
    ax.set_ylabel(figure_text("Value", "数值"))
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend(loc="best", ncol=2, fontsize=8, frameon=False)

    ax.text(
        0.98,
        0.04,
        figure_text(
            f"kept {n_valid}/{n_all} points",
            f"保留 {n_valid}/{n_all} 个点",
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )


def plot_parameter_panel(
    ax: plt.Axes,
    closure: np.ndarray,
    value: np.ndarray,
    true_value: float,
    *,
    ylabel: str,
    title: str,
) -> None:
    ax.plot(
        closure,
        value,
        linewidth=1.5,
        label=figure_text("inverted", "反演值"),
    )
    ax.axhline(
        true_value,
        color="black",
        linewidth=1.0,
        linestyle="--",
        label=figure_text("true", "真实值"),
    )

    ax.set_title(title, fontsize=10.5)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(loc="best", fontsize=8, frameon=False)

    final_value = float(value[-1])
    ax.text(
        0.98,
        0.08,
        figure_text(
            f"final={final_value:.5g}\ntrue={true_value:.5g}",
            f"最终={final_value:.5g}\n真实={true_value:.5g}",
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )


def plot_loss_parameter_figure(
    summary: dict[str, Any] | None,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(11.2, 7.4),
        dpi=1200,
        sharex=True,
        constrained_layout=True,
    )

    if summary is None:
        for ax in axes.ravel():
            ax.axis("off")
        axes[0, 0].text(
            0.5,
            0.5,
            figure_text(
                "tkp_inversion_summary.pt not found",
                "未找到 tkp_inversion_summary.pt",
            ),
            ha="center",
            va="center",
        )
    else:
        history = summary.get("history", [])

        if len(history) == 0:
            for ax in axes.ravel():
                ax.axis("off")
            axes[0, 0].text(
                0.5,
                0.5,
                figure_text("No history found", "未找到迭代历史"),
                ha="center",
                va="center",
            )
        else:
            plot_loss_panel(axes[0, 0], summary)

            closure = np.asarray(
                [record.get("closure", i + 1) for i, record in enumerate(history)],
                dtype=float,
            )
            total = np.asarray(
                [record.get("total_loss", np.nan) for record in history],
                dtype=float,
            )
            rho2 = np.asarray(
                [record.get("rho2", np.nan) for record in history],
                dtype=float,
            )
            A = np.asarray(
                [record.get("A", np.nan) for record in history],
                dtype=float,
            )
            n = np.asarray(
                [record.get("n", np.nan) for record in history],
                dtype=float,
            )

            valid = (
                np.isfinite(closure)
                & np.isfinite(total)
                & np.isfinite(rho2)
                & np.isfinite(A)
                & np.isfinite(n)
                & (total > 0.0)
                & (total < BAD_VALUE_THRESHOLD)
            )

            n_all = closure.size
            n_valid = int(np.sum(valid))

            if n_valid == 0:
                for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
                    ax.text(
                        0.5,
                        0.5,
                        figure_text(
                            "No valid parameter values",
                            "无有效参数值",
                        ),
                        ha="center",
                        va="center",
                    )
                    ax.axis("off")
            else:
                closure_param = closure[valid]
                rho2 = rho2[valid]
                A = A[valid]
                n = n[valid]

                plot_parameter_panel(
                    axes[0, 1],
                    closure_param,
                    rho2,
                    RHO2_TRUE,
                    ylabel=r"$\rho_2$ (kg m$^{-3}$)",
                    title=figure_text(
                        r"Density parameter, $\rho_2$",
                        r"密度参数，$\rho_2$",
                    ),
                )
                plot_parameter_panel(
                    axes[1, 0],
                    closure_param,
                    A,
                    A_TRUE,
                    ylabel=r"$A$",
                    title=figure_text(
                        r"Creep prefactor, $A$",
                        r"蠕变前因子，$A$",
                    ),
                )
                plot_parameter_panel(
                    axes[1, 1],
                    closure_param,
                    n,
                    N_TRUE,
                    ylabel=r"$n$",
                    title=figure_text(
                        r"Stress exponent, $n$",
                        r"应力指数，$n$",
                    ),
                )

                axes[1, 1].text(
                    0.02,
                    0.04,
                    figure_text(
            f"kept {n_valid}/{n_all} points",
            f"保留 {n_valid}/{n_all} 个点",
        ),
                    transform=axes[1, 1].transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7,
                )

            axes[0, 0].set_xlabel("")
            axes[0, 1].set_xlabel("")
            axes[1, 0].set_xlabel(
                figure_text(
                    "Closure / function evaluation",
                    "闭包 / 函数评估次数",
                )
            )
            axes[1, 1].set_xlabel(
                figure_text(
                    "Closure / function evaluation",
                    "闭包 / 函数评估次数",
                )
            )

    for panel_label, ax in zip(
        ("(a)", "(b)", "(c)", "(d)"),
        axes.ravel(),
    ):
        if ax.axison:
            ax.text(
                0.015,
                0.97,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )

    png_path = output_dir / "case_joint_loss_parameter_convergence_2x2.png"
    pdf_path = output_dir / "case_joint_loss_parameter_convergence_2x2.pdf"

    fig.savefig(png_path, dpi=1200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")



def main():
    true_path = Path("observe_data.pt")
    initial_path = Path("observed_1.pt")
    intermediate_path = Path("observed_100.pt")
    final_path = Path("observed_501.pt")
    summary_path = Path("tkp_inversion_summary.pt")

    output_dir = Path("case_joint_composite_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    true_data = try_load_pt(true_path)
    initial_data = try_load_pt(initial_path)
    intermediate_data = try_load_pt(intermediate_path)
    final_data = try_load_pt(final_path)

    if true_data is None:
        raise FileNotFoundError("observe_data.pt is required.")

    summary = load_summary(summary_path)

    rows: list[tuple[str, dict[str, Any]]] = [("True", true_data)]

    if initial_data is not None:
        rows.append(("Initial", initial_data))

    if intermediate_data is not None:
        rows.append(("Iter. 100", intermediate_data))

    if final_data is not None:
        rows.append(("Final", final_data))

    nsteps = get_num_saved_steps(true_data)
    mid_step = max(1, nsteps // 2)
    step_labels = [0, mid_step, nsteps]

    print(f"[INFO] nsteps = {nsteps}")
    print(f"[INFO] step_labels = {step_labels}")
    print(f"[INFO] rows = {[r[0] for r in rows]}")
    print(f"[INFO] P0 = {P0:.6e} Pa")
    print(f"[INFO] TOPO_SIGN = {TOPO_SIGN}")
    print(f"[INFO] TOPO_YLIM = {TOPO_YLIM}")

    plot_result_figure(
        rows=rows,
        step_labels=step_labels,
        true_data=true_data,
        summary=summary,
        output_dir=output_dir,
    )

    plot_loss_parameter_figure(
        summary=summary,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
