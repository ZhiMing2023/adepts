#!/usr/bin/env python3
'Plot Benchmark 4 slab-detachment snapshots.'

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm



L0 = 660e3
ETA0 = 1e21
KAPPA0 = 1e-6

YEAR = 365.25 * 24.0 * 3600.0

T0 = L0**2 / KAPPA0

TARGET_TIMES_MYR = (0.0, 6.0, 12.0)

TIME_TOLERANCE_MYR = 0.051



def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine Benchmark 4 snapshots at "
            "0, 6, and 12 Myr."
        )
    )

    parser.add_argument(
        "language",
        nargs="?",
        default="english",
        help="english or chinese",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Directory containing state_step_*.pt. "
            "If omitted, the script searches automatically."
        ),
    )

    return parser.parse_args()


ARGS = parse_args()

LANGUAGE = (
    "chinese"
    if ARGS.language.lower()
    in {"chinese", "cn", "zh", "zh-cn"}
    else "english"
)


def figure_text(
    english: str,
    chinese: str,
) -> str:
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
            font_manager.fontManager.addfont(
                str(chinese_font_path)
            )

            chinese_font_name = (
                font_manager.FontProperties(
                    fname=str(chinese_font_path)
                ).get_name()
            )

            matplotlib.rcParams["font.family"] = (
                chinese_font_name
            )

            matplotlib.rcParams["font.sans-serif"] = [
                chinese_font_name
            ]

            print(
                f"[FONT] Chinese font: "
                f"{chinese_font_name}"
            )
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



SCRIPT_DIR = Path(__file__).resolve().parent


def to_numpy(
    value: Any,
) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def load_pt(
    path: Path,
) -> dict[str, Any]:
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
            f"Expected dictionary in {path}, "
            f"got {type(payload)}."
        )

    return payload


def required_snapshot_keys():
    return (
        "time_myr",
        "C",
        "vx",
        "vy",
        "eta_p_nd",
        "epsII_p_nd",
        "mesh_xp_m",
        "mesh_yp_m",
    )


def validate_snapshot(
    payload: dict[str, Any],
    path: Path,
):
    missing = [
        key
        for key in required_snapshot_keys()
        if key not in payload
    ]

    if missing:
        raise KeyError(
            f"{path} is missing keys: {missing}\n"
            f"Available keys: {list(payload.keys())}"
        )



def inspect_snapshot_directory(
    directory: Path,
) -> dict[float, tuple[Path, float]]:
    files = sorted(
        directory.glob("state_step_*.pt")
    )

    if not files:
        return {}

    file_times = []

    for path in files:
        try:
            payload = load_pt(path)

            if "time_myr" not in payload:
                continue

            time_myr = float(
                payload["time_myr"]
            )

            if not np.isfinite(time_myr):
                continue

            file_times.append(
                (path, time_myr)
            )

        except Exception as exc:
            print(
                f"[WARN] Could not inspect "
                f"{path.name}: {exc}"
            )

    matches = {}

    for target in TARGET_TIMES_MYR:
        if not file_times:
            continue

        path, actual = min(
            file_times,
            key=lambda item: abs(
                item[1] - target
            ),
        )

        if (
            abs(actual - target)
            <= TIME_TOLERANCE_MYR
        ):
            matches[target] = (
                path,
                actual,
            )

    return matches


def candidate_directories():
    candidates = []

    if ARGS.data_dir is not None:
        explicit = Path(
            ARGS.data_dir
        ).expanduser()

        if not explicit.is_absolute():
            explicit = (
                SCRIPT_DIR
                / explicit
            )

        return [
            explicit.resolve()
        ]

    candidates.append(
        SCRIPT_DIR
    )

    candidates.append(
        SCRIPT_DIR
        / "benchmark4_slab_detachment"
    )

    candidates.extend(
        sorted(
            SCRIPT_DIR.glob(
                "benchmark4_slab_detachment_*km"
            )
        )
    )

    for child in SCRIPT_DIR.iterdir():
        if (
            child.is_dir()
            and child.name.startswith(
                "benchmark4_slab_detachment"
            )
            and child not in candidates
        ):
            candidates.append(
                child
            )

    unique = []

    for path in candidates:
        path = path.resolve()

        if path not in unique:
            unique.append(path)

    return unique


def find_snapshot_triplet():
    complete_groups = []

    for directory in candidate_directories():
        if not directory.exists():
            continue

        matches = inspect_snapshot_directory(
            directory
        )

        if all(
            target in matches
            for target in TARGET_TIMES_MYR
        ):
            complete_groups.append(
                (
                    directory,
                    matches,
                )
            )

    if not complete_groups:
        checked = "\n".join(
            f"  {path}"
            for path in candidate_directories()
        )

        raise FileNotFoundError(
            "Could not find one directory containing "
            "all three snapshots at 0, 6, and 12 Myr.\n"
            "Checked:\n"
            f"{checked}\n\n"
            "You can specify the directory explicitly with:\n"
            "  --data-dir PATH"
        )

    complete_groups.sort(
        key=lambda item: (
            item[0].name
            != "benchmark4_slab_detachment",
            str(item[0]),
        )
    )

    directory, matches = (
        complete_groups[0]
    )

    if len(complete_groups) > 1:
        print(
            "[INFO] Multiple complete Benchmark-4 "
            "directories were found."
        )
        print(
            f"[INFO] Using: {directory}"
        )
        print(
            "[INFO] Use --data-dir PATH "
            "to choose another run."
        )

    return directory, matches



def resize_2d_to_shape(
    array: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    array = np.asarray(
        array,
        dtype=float,
    )

    ny_target, nx_target = (
        target_shape
    )

    y_old = np.linspace(
        0.0,
        1.0,
        array.shape[0],
    )
    x_old = np.linspace(
        0.0,
        1.0,
        array.shape[1],
    )

    y_new = np.linspace(
        0.0,
        1.0,
        ny_target,
    )
    x_new = np.linspace(
        0.0,
        1.0,
        nx_target,
    )

    tmp = np.empty(
        (
            array.shape[0],
            nx_target,
        ),
        dtype=float,
    )

    for j in range(
        array.shape[0]
    ):
        tmp[j, :] = np.interp(
            x_new,
            x_old,
            array[j, :],
        )

    result = np.empty(
        (
            ny_target,
            nx_target,
        ),
        dtype=float,
    )

    for i in range(
        nx_target
    ):
        result[:, i] = np.interp(
            y_new,
            y_old,
            tmp[:, i],
        )

    return result


def staggered_to_p(
    array: np.ndarray,
    target_shape: tuple[int, int],
    component: str,
) -> np.ndarray:
    arr = np.asarray(
        array,
        dtype=float,
    )

    ny, nx = target_shape

    if arr.shape == target_shape:
        return arr.copy()

    if component == "vx":
        if arr.shape == (ny, nx + 1):
            return 0.5 * (
                arr[:, :-1]
                + arr[:, 1:]
            )

        if arr.shape == (ny, nx - 1):
            padded = np.pad(
                arr,
                ((0, 0), (1, 1)),
                mode="edge",
            )
            return 0.5 * (
                padded[:, :-1]
                + padded[:, 1:]
            )[:, :nx]

    if component == "vy":
        if arr.shape == (ny + 1, nx):
            return 0.5 * (
                arr[:-1, :]
                + arr[1:, :]
            )

        if arr.shape == (ny - 1, nx):
            padded = np.pad(
                arr,
                ((1, 1), (0, 0)),
                mode="edge",
            )
            return 0.5 * (
                padded[:-1, :]
                + padded[1:, :]
            )[:ny, :]

    print(
        f"[WARN] {component} shape {arr.shape} "
        f"does not match standard p-grid averaging "
        f"for target {target_shape}; "
        "using index-space linear interpolation."
    )

    return resize_2d_to_shape(
        arr,
        target_shape,
    )



def prepare_snapshot(
    path: Path,
) -> dict[str, Any]:
    payload = load_pt(
        path
    )

    validate_snapshot(
        payload,
        path,
    )

    C = np.asarray(
        to_numpy(payload["C"]),
        dtype=float,
    )

    if C.ndim != 2:
        raise ValueError(
            f"C must be 2-D in {path}; "
            f"got {C.shape}"
        )

    target_shape = C.shape

    vx_raw = np.asarray(
        to_numpy(payload["vx"]),
        dtype=float,
    )

    vy_raw = np.asarray(
        to_numpy(payload["vy"]),
        dtype=float,
    )

    vx_p = staggered_to_p(
        vx_raw,
        target_shape,
        "vx",
    )

    vy_p = staggered_to_p(
        vy_raw,
        target_shape,
        "vy",
    )

    eta_p_nd = np.asarray(
        to_numpy(
            payload["eta_p_nd"]
        ),
        dtype=float,
    )

    epsII_p_nd = np.asarray(
        to_numpy(
            payload["epsII_p_nd"]
        ),
        dtype=float,
    )

    if eta_p_nd.shape != target_shape:
        eta_p_nd = resize_2d_to_shape(
            eta_p_nd,
            target_shape,
        )

    if epsII_p_nd.shape != target_shape:
        epsII_p_nd = resize_2d_to_shape(
            epsII_p_nd,
            target_shape,
        )

    eta_phys = (
        eta_p_nd
        * ETA0
    )

    eps_phys = (
        epsII_p_nd
        / T0
    )

    speed_nd = np.sqrt(
        vx_p**2
        + vy_p**2
    )

    speed_cm_yr = (
        speed_nd
        * KAPPA0
        / L0
        * 100.0
        * YEAR
    )

    vx_cm_yr = (
        vx_p
        * KAPPA0
        / L0
        * 100.0
        * YEAR
    )

    vy_cm_yr = (
        vy_p
        * KAPPA0
        / L0
        * 100.0
        * YEAR
    )

    xp_m = np.asarray(
        to_numpy(
            payload["mesh_xp_m"]
        ),
        dtype=float,
    ).reshape(-1)

    yp_m = np.asarray(
        to_numpy(
            payload["mesh_yp_m"]
        ),
        dtype=float,
    ).reshape(-1)

    if xp_m.size != C.shape[1]:
        xp_m = np.linspace(
            float(np.nanmin(xp_m)),
            float(np.nanmax(xp_m)),
            C.shape[1],
        )

    if yp_m.size != C.shape[0]:
        yp_m = np.linspace(
            float(np.nanmin(yp_m)),
            float(np.nanmax(yp_m)),
            C.shape[0],
        )

    xp_km = xp_m / 1e3
    yp_km = yp_m / 1e3

    return {
        "path": path,
        "step": int(
            payload.get(
                "step",
                -1,
            )
        ),
        "time_myr": float(
            payload["time_myr"]
        ),
        "C": C,
        "vx_cm_yr": vx_cm_yr,
        "vy_cm_yr": vy_cm_yr,
        "eta_phys": eta_phys,
        "eps_phys": eps_phys,
        "speed_cm_yr": speed_cm_yr,
        "xp_km": xp_km,
        "yp_km": yp_km,
    }



def positive_shared_range(
    arrays,
    *,
    pmin=1.0,
    pmax=99.0,
    fallback=(1e-30, 1.0),
):
    values = []

    for arr in arrays:
        arr = np.asarray(
            arr,
            dtype=float,
        )

        valid = arr[
            np.isfinite(arr)
            & (arr > 0.0)
        ]

        if valid.size > 0:
            values.append(
                valid.reshape(-1)
            )

    if not values:
        return fallback

    values = np.concatenate(
        values
    )

    vmin = float(
        np.percentile(
            values,
            pmin,
        )
    )

    vmax = float(
        np.percentile(
            values,
            pmax,
        )
    )

    if vmax <= vmin:
        vmax = vmin * 10.0

    return (
        max(
            vmin,
            fallback[0],
        ),
        vmax,
    )



def plot_merged_snapshots(
    snapshots: list[dict[str, Any]],
    output_dir: Path,
):
    if len(snapshots) != 3:
        raise ValueError(
            "Exactly three snapshots are required."
        )

    eta_vmin, eta_vmax = (
        positive_shared_range(
            [
                snap["eta_phys"]
                for snap in snapshots
            ],
            fallback=(1e21, 1e25),
        )
    )

    eps_vmin, eps_vmax = (
        positive_shared_range(
            [
                snap["eps_phys"]
                for snap in snapshots
            ],
            fallback=(1e-18, 1e-12),
        )
    )

    speed_vmin, speed_vmax = (
        positive_shared_range(
            [
                snap["speed_cm_yr"]
                for snap in snapshots
            ],
            fallback=(1e-8, 1.0),
        )
    )

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(15.5, 15.5),
        dpi=1200,
        constrained_layout=True,
    )

    composition_images = []
    viscosity_images = []
    strain_images = []
    speed_images = []

    panel_index = 0

    for col, snap in enumerate(
        snapshots
    ):
        xp_km = snap["xp_km"]
        yp_km = snap["yp_km"]

        C = snap["C"]
        vx = snap["vx_cm_yr"]

        vy_plot = -snap["vy_cm_yr"]

        eta = snap["eta_phys"]
        eps = snap["eps_phys"]
        speed = snap["speed_cm_yr"]

        X, Y = np.meshgrid(
            xp_km,
            yp_km,
        )

        extent = [
            float(xp_km.min()),
            float(xp_km.max()),
            float(yp_km.max()),
            float(yp_km.min()),
        ]

        ax = axes[0, col]

        im0 = ax.imshow(
            C,
            origin="upper",
            extent=extent,
            aspect="equal",
            cmap="plasma",
            vmin=0.0,
            vmax=1.0,
        )

        composition_images.append(
            im0
        )

        stride_y = max(
            1,
            C.shape[0] // 22,
        )

        stride_x = max(
            1,
            C.shape[1] // 32,
        )

        ax.quiver(
            X[
                1:-1:stride_y,
                1:-1:stride_x,
            ],
            Y[
                1:-1:stride_y,
                1:-1:stride_x,
            ],
            vx[
                1:-1:stride_y,
                1:-1:stride_x,
            ],
            vy_plot[
                1:-1:stride_y,
                1:-1:stride_x,
            ],
            width=0.002,
        )

        ax.set_title(
            figure_text(
                f"{snap['time_myr']:.0f} Myr",
                f"{snap['time_myr']:.0f} Myr",
            ),
            fontsize=11,
            fontweight="bold",
        )

        ax = axes[1, col]

        im1 = ax.imshow(
            eta,
            origin="upper",
            extent=extent,
            aspect="equal",
            cmap="viridis",
            norm=LogNorm(
                vmin=eta_vmin,
                vmax=eta_vmax,
            ),
        )

        viscosity_images.append(
            im1
        )

        ax = axes[2, col]

        im2 = ax.imshow(
            eps,
            origin="upper",
            extent=extent,
            aspect="equal",
            cmap="magma",
            norm=LogNorm(
                vmin=eps_vmin,
                vmax=eps_vmax,
            ),
        )

        strain_images.append(
            im2
        )

        ax = axes[3, col]

        im3 = ax.imshow(
            speed,
            origin="upper",
            extent=extent,
            aspect="equal",
            cmap="turbo",
            norm=LogNorm(
                vmin=speed_vmin,
                vmax=speed_vmax,
            ),
        )

        speed_images.append(
            im3
        )

    row_labels = [
        figure_text(
            "Composition + velocity",
            "成分 + 速度",
        ),
        figure_text(
            "Viscosity",
            "黏度",
        ),
        figure_text(
            "Second invariant of strain rate",
            "应变率第二不变量",
        ),
        figure_text(
            "Speed",
            "速度大小",
        ),
    ]

    for row, label in enumerate(
        row_labels
    ):
        axes[row, 0].text(
            -0.12,
            0.5,
            label,
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=10,
        )

    panel_letters = [
        f"({chr(ord('a') + i)})"
        for i in range(12)
    ]

    panel_index = 0

    for row in range(4):
        for col in range(3):
            ax = axes[row, col]

            ax.set_xlim(
                snapshots[col]["xp_km"].min(),
                snapshots[col]["xp_km"].max(),
            )

            ax.set_ylim(
                snapshots[col]["yp_km"].max(),
                snapshots[col]["yp_km"].min(),
            )

            if row == 3:
                ax.set_xlabel(
                    figure_text(
                        "Horizontal Distance (km)",
                        "水平距离（km）",
                    )
                )
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(
                    figure_text(
                        "Depth (km)",
                        "深度（km）",
                    )
                )
            else:
                ax.set_yticklabels([])

            ax.text(
                0.015,
                0.98,
                panel_letters[
                    panel_index
                ],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
            )

            panel_index += 1

    cbar0 = fig.colorbar(
        composition_images[-1],
        ax=axes[0, :],
        orientation="vertical",
        fraction=0.018,
        pad=0.012,
    )

    cbar0.set_label(
        r"$C$"
    )

    cbar1 = fig.colorbar(
        viscosity_images[-1],
        ax=axes[1, :],
        orientation="vertical",
        fraction=0.018,
        pad=0.012,
    )

    cbar1.set_label(
        figure_text(
            r"Viscosity (Pa s)",
            r"黏度（Pa s）",
        )
    )

    cbar2 = fig.colorbar(
        strain_images[-1],
        ax=axes[2, :],
        orientation="vertical",
        fraction=0.018,
        pad=0.012,
    )

    cbar2.set_label(
        figure_text(
            r"Strain rate (s$^{-1}$)",
            r"应变率（s$^{-1}$）",
        )
    )

    cbar3 = fig.colorbar(
        speed_images[-1],
        ax=axes[3, :],
        orientation="vertical",
        fraction=0.018,
        pad=0.012,
    )

    cbar3.set_label(
        figure_text(
            "Speed (cm/yr)",
            "速度（cm/yr）",
        )
    )

    fig.suptitle(
        figure_text(
            "Slab detachment benchmark",
            "板片拆沉基准测试",
        ),
        fontsize=12,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        output_dir
        / "benchmark4_snapshots_0_6_12_Myr.png"
    )

    pdf_path = (
        output_dir
        / "benchmark4_snapshots_0_6_12_Myr.pdf"
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
        f"[SAVE] {png_path}"
    )

    print(
        f"[SAVE] {pdf_path}"
    )



def main():
    data_dir, matches = (
        find_snapshot_triplet()
    )

    print(
        f"[INFO] Data directory: "
        f"{data_dir}"
    )

    snapshots = []

    for target in TARGET_TIMES_MYR:
        path, actual_time = (
            matches[target]
        )

        print(
            f"[LOAD] target={target:.1f} Myr | "
            f"actual={actual_time:.3f} Myr | "
            f"file={path.name}"
        )

        snapshots.append(
            prepare_snapshot(
                path
            )
        )

    output_dir = (
        data_dir
        / "combined_snapshots_0_6_12"
    )

    plot_merged_snapshots(
        snapshots,
        output_dir,
    )


if __name__ == "__main__":
    main()
