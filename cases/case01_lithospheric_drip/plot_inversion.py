#!/usr/bin/env python3
'Composite figures for Case 1.'

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
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap



L0 = 500e3
KAPPA0 = 1e-6
T_SURFACE = 273.0
T_MANTLE = 1573.0
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


def get_mesh_xy(data: dict[str, Any], field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mesh_state = data.get("mesh_state", None)

    if isinstance(mesh_state, dict) and "xp" in mesh_state and "yp" in mesh_state:
        x = to_numpy(mesh_state["xp"]).reshape(-1)
        y = to_numpy(mesh_state["yp"]).reshape(-1)

        if np.nanmax(np.abs(x)) < 100.0:
            x = x * L0
        if np.nanmax(np.abs(y)) < 100.0:
            y = y * L0

        if x.size != field.shape[1]:
            x = np.linspace(x.min(), x.max(), field.shape[1])
        if y.size != field.shape[0]:
            y = np.linspace(y.min(), y.max(), field.shape[0])
    else:
        ny, nx = field.shape
        x = np.linspace(0.0, L0, nx)
        y = np.linspace(0.0, L0, ny)

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
                x = to_numpy(mesh_state[key]).reshape(-1)
                if np.nanmax(np.abs(x)) < 100.0:
                    x = x * L0
                if x.size == n:
                    return x / 1e3

    return np.linspace(0.0, L0 / 1e3, n)



def get_temp_at(
    data: dict[str, Any],
    step_label: int,
    initial_field: np.ndarray | None = None,
) -> np.ndarray:
    if step_label == 0:
        if initial_field is None:
            return maybe_temperature_kelvin(to_numpy(get_series(data, "all_temperature")[0]))
        return maybe_temperature_kelvin(initial_field)

    idx = step_label - 1
    T = to_numpy(get_series(data, "all_temperature")[idx])

    return maybe_temperature_kelvin(T)


def get_vx_at(data: dict[str, Any], step_label: int) -> np.ndarray:
    if step_label == 0:
        idx = 0
    else:
        idx = step_label - 1

    vx = to_numpy(get_series(data, "all_surface_vx")[idx]).reshape(-1)

    return velocity_cm_per_year(vx)


def load_summary(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


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
    if not (vmin < tref < vmax):
        return plt.get_cmap("turbo")

    p = (tref - vmin) / (vmax - vmin)
    p = float(np.clip(p, 0.05, 0.95))

    cmap = LinearSegmentedColormap.from_list(
        "temperature_1573_red",
        [
            (0.0, "#313695"),
            (0.50 * p, "#abd9e9"),
            (0.85 * p, "#fee090"),
            (p, "#d73027"),
            (1.0, "#7f0000"),
        ],
    )

    return cmap



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
    vx_ylim: tuple[float, float] = (-5.0, 5.0),
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
        label=figure_text("true", "真实值"),
    )
    inset.plot(
        x_vx,
        vx,
        color="red",
        linewidth=1.0,
        linestyle="--",
        label=figure_text("pred", "预测值"),
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
        ax.text(
            0.5,
            0.5,
            figure_text("No history found", "未找到迭代历史"),
            ha="center",
            va="center",
        )
        ax.axis("off")
        return

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
    grad = np.array(
        [h.get("grad_dT", np.nan) for h in history],
        dtype=float,
    )

    ax.semilogy(
        closure,
        total,
        linewidth=1.5,
        label=figure_text("total", "总损失"),
    )
    ax.semilogy(
        closure,
        loss_T,
        linewidth=1.2,
        label=figure_text("final T", "最终温度"),
    )
    ax.semilogy(
        closure,
        loss_vx,
        linewidth=1.2,
        label=figure_text("surface vx", "表面 $v_x$"),
    )
    ax.semilogy(closure, grad, linewidth=1.2, label=r"$||g_T||$")

    ax.set_xlabel(
        figure_text(
            "Closure / function evaluation",
            "闭包 / 函数评估次数",
        )
    )
    ax.set_ylabel(figure_text("Value", "数值"))
    ax.set_title(figure_text("Inversion convergence", "反演收敛过程"))
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend(ncol=4, fontsize=8)


def compute_temperature_limits(
    rows: list[tuple[str, dict[str, Any]]],
    step_labels: list[int],
    summary: dict[str, Any] | None,
) -> tuple[float, float]:
    _ = rows, step_labels, summary
    return float(T_SURFACE), float(T_MANTLE)


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
        figsize=(12.5, 12.8),
        dpi=1200,
        constrained_layout=True,
    )

    gs = GridSpec(
        nrows=4,
        ncols=3,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 1.0],
    )

    temp_axes = []
    last_im = None

    column_labels = [
        figure_text("Initial", "初始时刻"),
        figure_text("25 steps", "25 步"),
        figure_text("50 steps", "50 步"),
    ]

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
                vx_ylim=(-5.0, 5.0),
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
                    -0.06,
                    0.5,
                    {
                        "True": figure_text("True", "真实模型"),
                        "Initial": figure_text("Initial", "初始反演"),
                        "Iter. 10": figure_text("Iter. 10", "第 10 次迭代"),
                        "Final": figure_text("Final", "最终反演"),
                    }.get(row_name, row_name),
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
    cbar.set_label(figure_text("Temperature (K)", "温度（K）"))

    cbar.set_ticks([T_SURFACE, T_MANTLE])
    cbar.set_ticklabels([f"{T_SURFACE:.0f}", f"{T_MANTLE:.0f}"])

    png_path = output_dir / "case1_true_initial_iter50_final_snapshots.png"
    pdf_path = output_dir / "case1_true_initial_iter50_final_snapshots.pdf"

    fig.savefig(png_path, dpi=1200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_loss_figure(
    summary: dict[str, Any] | None,
    output_dir: Path,
):
    fig, ax = plt.subplots(
        figsize=(8.0, 3.2),
        dpi=1200,
        constrained_layout=True,
    )

    if summary is not None:
        plot_loss_panel(ax, summary)
    else:
        ax.text(
            0.5,
            0.5,
            figure_text(
                "tkp_inversion_summary.pt not found",
                "未找到 tkp_inversion_summary.pt",
            ),
            ha="center",
            va="center",
        )
        ax.axis("off")

    png_path = output_dir / "case1_loss_history.png"
    pdf_path = output_dir / "case1_loss_history.pdf"

    fig.savefig(png_path, dpi=1200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")



def main():
    true_path = Path("observe_data.pt")
    initial_path = Path("observed_1.pt")
    iter50_path = Path("observed_10.pt")
    final_path = Path("observed_201.pt")
    summary_path = Path("tkp_inversion_summary.pt")

    output_dir = Path("case1_composite_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    true_data = load_pt(true_path)
    initial_data = load_pt(initial_path)
    iter50_data = load_pt(iter50_path)
    final_data = load_pt(final_path)

    summary = load_summary(summary_path) if summary_path.exists() else None

    rows = [
        ("True", true_data),
        ("Initial", initial_data),
        ("Iter. 10", iter50_data),
        ("Final", final_data),
    ]

    step_labels = [0, 25, 50]

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
