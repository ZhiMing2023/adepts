#!/usr/bin/env python3
'Plot Benchmark 1 fields and temporal convergence.'

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager



SCRIPT_DIR = Path(__file__).resolve().parent

CSV_NAME = "benchmark1_temporal_convergence.csv"
PT_NAME = "case_res_2000_nstep_80.pt"

OUTPUT_DIR = SCRIPT_DIR / "benchmark1_postprocess_figures"

L0 = 660e3  # m



LANGUAGE = (
    "chinese"
    if len(sys.argv) > 1
    and sys.argv[1].lower() in {"chinese", "cn", "zh", "zh-cn"}
    else "english"
)


def figure_text(english: str, chinese: str) -> str:
    return chinese if LANGUAGE == "chinese" else english


def configure_plot_fonts() -> None:
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
            font_manager.fontManager.addfont(
                str(chinese_font_path)
            )

            chinese_font_name = font_manager.FontProperties(
                fname=str(chinese_font_path)
            ).get_name()

            matplotlib.rcParams["font.family"] = chinese_font_name
            matplotlib.rcParams["font.sans-serif"] = [
                chinese_font_name
            ]
        else:
            print(
                "[WARN] No Chinese font was found. "
                "Chinese characters may not render correctly."
            )
    else:
        matplotlib.rcParams["font.family"] = [
            "Times New Roman",
            "DejaVu Serif",
        ]


configure_plot_fonts()



def find_input_file(filename: str) -> Path:
    candidates = [
        SCRIPT_DIR / filename,
        SCRIPT_DIR / "benchmark1_thermal_diffusion" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find {filename}.\n"
        f"Checked:\n"
        f"  {candidates[0]}\n"
        f"  {candidates[1]}"
    )


def to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_pt(path: Path) -> dict:
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a dictionary in {path}, got {type(payload)}"
        )

    required = [
        "row",
        "T_num",
        "T_exact",
        "error",
        "xp_nd",
        "yp_nd",
    ]

    missing = [
        key for key in required
        if key not in payload
    ]

    if missing:
        raise KeyError(
            f"{path} is missing keys: {missing}\n"
            f"Available keys: {list(payload.keys())}"
        )

    return payload


def load_convergence_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    if data.size == 0:
        raise ValueError(f"CSV contains no data: {path}")

    data = np.atleast_1d(data)

    required = [
        "dt_nd",
        "L2_int",
        "Linf_int",
        "nstep",
    ]

    available = list(data.dtype.names or [])

    missing = [
        key for key in required
        if key not in available
    ]

    if missing:
        raise KeyError(
            f"{path} is missing columns: {missing}\n"
            f"Available columns: {available}"
        )

    return {
        key: np.asarray(data[key], dtype=float)
        for key in available
    }



def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(
        centers,
        dtype=float,
    ).reshape(-1)

    if centers.size == 1:
        return np.array(
            [
                centers[0] - 0.5,
                centers[0] + 0.5,
            ]
        )

    middle = 0.5 * (
        centers[:-1] + centers[1:]
    )

    first = centers[0] - 0.5 * (
        centers[1] - centers[0]
    )

    last = centers[-1] + 0.5 * (
        centers[-1] - centers[-2]
    )

    return np.concatenate(
        [[first], middle, [last]]
    )


def get_coordinates_km(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    xp_nd = to_numpy(payload["xp_nd"]).reshape(-1)
    yp_nd = to_numpy(payload["yp_nd"]).reshape(-1)

    x_km = xp_nd * L0 / 1e3
    y_km = yp_nd * L0 / 1e3

    return x_km, y_km



def plot_fields_from_pt(
    payload: dict,
    output_dir: Path,
) -> None:
    T_num = np.asarray(
        to_numpy(payload["T_num"]),
        dtype=float,
    )

    T_exact = np.asarray(
        to_numpy(payload["T_exact"]),
        dtype=float,
    )

    error = np.asarray(
        to_numpy(payload["error"]),
        dtype=float,
    )

    row = payload["row"]

    x_km, y_km = get_coordinates_km(payload)

    if T_num.shape != T_exact.shape or T_num.shape != error.shape:
        raise ValueError(
            "T_num, T_exact and error shapes are inconsistent: "
            f"T_num={T_num.shape}, "
            f"T_exact={T_exact.shape}, "
            f"error={error.shape}"
        )

    if (
        x_km.size != T_num.shape[1]
        or y_km.size != T_num.shape[0]
    ):
        raise ValueError(
            "Coordinate sizes do not match field shape: "
            f"field={T_num.shape}, "
            f"x={x_km.size}, y={y_km.size}"
        )

    x_edges = centers_to_edges(x_km)
    y_edges = centers_to_edges(y_km)

    finite_solution = np.concatenate(
        [
            T_exact[np.isfinite(T_exact)].reshape(-1),
            T_num[np.isfinite(T_num)].reshape(-1),
        ]
    )

    vmin = float(np.min(finite_solution))
    vmax = float(np.max(finite_solution))

    finite_error = error[
        np.isfinite(error)
    ]

    error_absmax = float(
        np.max(np.abs(finite_error))
    )

    if error_absmax == 0.0:
        error_absmax = 1e-16

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15.5, 4.8),
        dpi=1200,
        constrained_layout=True,
    )

    im0 = axes[0].pcolormesh(
        x_edges,
        y_edges,
        T_exact,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    axes[0].set_title(
        figure_text(
            "Analytical solution",
            "解析解",
        )
    )

    im1 = axes[1].pcolormesh(
        x_edges,
        y_edges,
        T_num,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    axes[1].set_title(
        figure_text(
            "Numerical solution",
            "数值解",
        )
    )

    im2 = axes[2].pcolormesh(
        x_edges,
        y_edges,
        error,
        shading="auto",
        cmap="coolwarm",
        vmin=-error_absmax,
        vmax=error_absmax,
    )

    L2_int = float(
        row.get(
            "L2_int",
            np.sqrt(
                np.mean(
                    error[1:-1, 1:-1] ** 2
                )
            ),
        )
    )

    Linf_int = float(
        row.get(
            "Linf_int",
            np.max(
                np.abs(
                    error[1:-1, 1:-1]
                )
            ),
        )
    )

    axes[2].set_title(
        figure_text(
            "Error",
            "误差",
        )
        + "\n"
        + rf"$L_2^{{\rm int}}={L2_int:.2e}$, "
          rf"$L_\infty^{{\rm int}}={Linf_int:.2e}$"
    )

    for panel_label, ax in zip(
        ("(a)", "(b)", "(c)"),
        axes,
    ):
        ax.set_xlabel(
            figure_text(
                "Horizontal Distance (km)",
                "水平距离（km）",
            )
        )

        ax.set_ylabel(
            figure_text(
                "Depth (km)",
                "深度（km）",
            )
        )

        ax.invert_yaxis()

        ax.text(
            0.015,
            0.98,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    cbar_solution = fig.colorbar(
        im1,
        ax=axes[:2],
        orientation="vertical",
        fraction=0.035,
        pad=0.02,
    )

    cbar_solution.set_label(
        figure_text(
            "Temperature",
            "温度",
        )
    )

    cbar_error = fig.colorbar(
        im2,
        ax=axes[2],
        orientation="vertical",
        fraction=0.06,
        pad=0.02,
    )

    cbar_error.set_label(
        figure_text(
            "Temperature error",
            "温度误差",
        )
    )

    nstep = int(
        row.get("nstep", 80)
    )

    dt_nd = float(
        row.get("dt_nd", np.nan)
    )

    fig.suptitle(
        figure_text(
            f"Thermal diffusion benchmark: "
            f"{nstep} time steps, "
            f"dt = {dt_nd:.3e}",
            f"热扩散基准测试："
            f"{nstep} 个时间步，"
            f"dt = {dt_nd:.3e}",
        ),
        fontsize=11,
    )

    png_path = (
        output_dir
        / "benchmark1_fields_res_2000_nstep_80.png"
    )

    pdf_path = (
        output_dir
        / "benchmark1_fields_res_2000_nstep_80.pdf"
    )

    fig.savefig(
        png_path,
        dpi=1200,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[SAVE] {png_path}")
    print(f"[SAVE] {pdf_path}")



def fit_loglog_slope(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > 0.0)
        & (y > 0.0)
    )

    if np.count_nonzero(mask) < 2:
        return float("nan")

    slope = np.polyfit(
        np.log(x[mask]),
        np.log(y[mask]),
        deg=1,
    )[0]

    return float(slope)


def plot_temporal_convergence_from_csv(
    table: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    dt = np.asarray(
        table["dt_nd"],
        dtype=float,
    )

    L2 = np.asarray(
        table["L2_int"],
        dtype=float,
    )

    Linf = np.asarray(
        table["Linf_int"],
        dtype=float,
    )

    nstep = np.asarray(
        table["nstep"],
        dtype=float,
    )

    order = np.argsort(dt)[::-1]

    dt = dt[order]
    L2 = L2[order]
    Linf = Linf[order]
    nstep = nstep[order]

    slope_L2 = fit_loglog_slope(
        dt,
        L2,
    )

    slope_Linf = fit_loglog_slope(
        dt,
        Linf,
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 5.7),
        dpi=1200,
        constrained_layout=True,
    )

    ax.loglog(
        dt,
        L2,
        marker="o",
        linewidth=1.5,
        markersize=5.0,
        label=figure_text(
            rf"Interior $L_2$ error, slope={slope_L2:.2f}",
            rf"内部 $L_2$ 误差，斜率={slope_L2:.2f}",
        ),
    )

    ax.loglog(
        dt,
        Linf,
        marker="s",
        linewidth=1.5,
        markersize=5.0,
        label=figure_text(
            rf"Interior $L_\infty$ error, slope={slope_Linf:.2f}",
            rf"内部 $L_\infty$ 误差，斜率={slope_Linf:.2f}",
        ),
    )

    valid_ref = (
        np.isfinite(dt)
        & np.isfinite(L2)
        & (dt > 0.0)
        & (L2 > 0.0)
    )

    if np.any(valid_ref):
        valid_indices = np.where(valid_ref)[0]
        anchor_index = valid_indices[
            np.argmax(dt[valid_ref])
        ]

        dt_anchor = dt[anchor_index]
        L2_anchor = L2[anchor_index]

        ref_first = (
            L2_anchor
            * (dt / dt_anchor)
        )

        ref_second = (
            L2_anchor
            * (dt / dt_anchor) ** 2
        )

        ax.loglog(
            dt,
            ref_first,
            linestyle="--",
            linewidth=1.2,
            label=r"$O(\Delta t)$",
        )

        ax.loglog(
            dt,
            ref_second,
            linestyle=":",
            linewidth=1.2,
            label=r"$O(\Delta t^2)$",
        )

    ax.set_xlabel(
        figure_text(
            r"Nondimensional time step, $\Delta t$",
            r"无量纲时间步长，$\Delta t$",
        )
    )

    ax.set_ylabel(
        figure_text(
            r"Error at $t_{\mathrm{end}}$",
            r"$t_{\mathrm{end}}$ 时刻误差",
        )
    )

    ax.set_title(
        figure_text(
            "Thermal diffusion temporal convergence",
            "热扩散时间收敛性",
        )
    )

    ax.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.6,
    )

    ax.legend(
        loc="best",
        fontsize=8.5,
        frameon=False,
    )

    ax.invert_xaxis()

    for dt_i, L2_i, nstep_i in zip(
        dt,
        L2,
        nstep,
    ):
        if (
            np.isfinite(dt_i)
            and np.isfinite(L2_i)
        ):
            ax.annotate(
                figure_text(
                    f"N={int(nstep_i)}",
                    f"N={int(nstep_i)}",
                ),
                xy=(dt_i, L2_i),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    png_path = (
        output_dir
        / "benchmark1_temporal_convergence.png"
    )

    pdf_path = (
        output_dir
        / "benchmark1_temporal_convergence.pdf"
    )

    fig.savefig(
        png_path,
        dpi=1200,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[INFO] fitted L2 slope   = {slope_L2:.6f}"
    )
    print(
        f"[INFO] fitted Linf slope = {slope_Linf:.6f}"
    )
    print(f"[SAVE] {png_path}")
    print(f"[SAVE] {pdf_path}")



def main() -> None:
    csv_path = find_input_file(
        CSV_NAME
    )

    pt_path = find_input_file(
        PT_NAME
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[LOAD] CSV: {csv_path}")
    print(f"[LOAD] PT : {pt_path}")

    table = load_convergence_csv(
        csv_path
    )

    payload = load_pt(
        pt_path
    )

    plot_fields_from_pt(
        payload,
        OUTPUT_DIR,
    )

    plot_temporal_convergence_from_csv(
        table,
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
