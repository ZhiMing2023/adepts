#!/usr/bin/env python3
'Plot all Case 06 Taylor tests.'

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



SCRIPT_DIR = Path(__file__).resolve().parent

PARAMETER_CONFIGS = {
    "T": {
        "path": SCRIPT_DIR / "taylor_dT_param.pt",
        "label": r"$T_0$",
        "marker": "o",
    },
    "A": {
        "path": SCRIPT_DIR / "taylor_A.pt",
        "label": r"$A$",
        "marker": "s",
    },
    "n": {
        "path": SCRIPT_DIR / "taylor_n.pt",
        "label": r"$n$",
        "marker": "^",
    },
    "rho2": {
        "path": SCRIPT_DIR / "taylor_rho2_nd.pt",
        "label": r"$\rho_2$",
        "marker": "D",
    },
}


FIT_POINT_COUNT = 4

POSITIVE_FLOOR = 1e-300



def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()

    return np.asarray(
        value,
        dtype=float,
    ).reshape(-1)


def load_taylor_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Taylor-test file not found: {path}"
        )

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

    if (
        isinstance(payload, dict)
        and "data" in payload
        and "eps" not in payload
        and isinstance(payload["data"], dict)
    ):
        payload = payload["data"]

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a dictionary in {path}, "
            f"got {type(payload)}."
        )

    required_keys = (
        "eps",
        "R1",
    )

    missing = [
        key
        for key in required_keys
        if key not in payload
    ]

    if missing:
        raise KeyError(
            f"{path} is missing required keys: {missing}. "
            f"Available keys: {list(payload.keys())}"
        )

    return payload


def clean_taylor_data(
    payload: dict[str, Any],
    parameter_key: str,
) -> dict[str, Any]:
    eps = to_numpy(
        payload["eps"]
    )
    R1 = to_numpy(
        payload["R1"]
    )

    if eps.size != R1.size:
        raise ValueError(
            f"{parameter_key}: inconsistent lengths: "
            f"eps={eps.size}, R1={R1.size}."
        )

    valid = (
        np.isfinite(eps)
        & np.isfinite(R1)
        & (eps > 0.0)
        & (R1 > 0.0)
    )

    n_total = eps.size
    n_valid = int(
        np.sum(valid)
    )

    if n_valid < 2:
        raise ValueError(
            f"{parameter_key}: only {n_valid}/{n_total} "
            "valid Taylor-test points remain."
        )

    eps = eps[valid]
    R1 = R1[valid]

    order = np.argsort(
        eps
    )[::-1]

    return {
        "eps": eps[order],
        "R1": R1[order],
        "n_total": n_total,
        "n_valid": n_valid,
        "saved_rate_R1": payload.get(
            "rate_R1",
            None,
        ),
    }


def load_all_parameters() -> dict[str, dict[str, Any]]:
    parameters: dict[str, dict[str, Any]] = {}

    for parameter_key, config in PARAMETER_CONFIGS.items():
        payload = load_taylor_file(
            config["path"]
        )

        data = clean_taylor_data(
            payload,
            parameter_key,
        )

        parameters[parameter_key] = {
            **config,
            **data,
        }

        print(
            f"[LOAD] {parameter_key}: "
            f"file={config['path'].name}, "
            f"points={data['n_valid']}/{data['n_total']}, "
            f"eps=[{data['eps'].min():.1e}, "
            f"{data['eps'].max():.1e}]"
        )

    return parameters



def fit_loglog_order(
    eps: np.ndarray,
    remainder: np.ndarray,
    point_count: int = FIT_POINT_COUNT,
) -> float:
    nfit = min(
        int(point_count),
        eps.size,
    )

    if nfit < 2:
        return float("nan")

    x = np.log10(
        eps[:nfit]
    )
    y = np.log10(
        remainder[:nfit]
    )

    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
    ):
        return float("nan")

    slope, _ = np.polyfit(
        x,
        y,
        deg=1,
    )

    return float(slope)


def build_second_order_line(
    eps: np.ndarray,
    R1: np.ndarray,
    number_of_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    h_max = float(
        eps[0]
    )
    h_min = float(
        eps[-1]
    )

    R1_anchor = max(
        float(R1[0]),
        POSITIVE_FLOOR,
    )

    eps_reference = np.logspace(
        np.log10(h_min),
        np.log10(h_max),
        number_of_points,
    )

    R1_reference = (
        R1_anchor
        * (eps_reference / h_max) ** 2
    )

    return (
        eps_reference,
        R1_reference,
    )


def compute_axis_limits(
    parameters: dict[str, dict[str, Any]],
) -> tuple[
    tuple[float, float],
    tuple[float, float],
]:
    all_eps = []
    all_R1 = []

    for parameter in parameters.values():
        eps_reference, R1_reference = (
            build_second_order_line(
                parameter["eps"],
                parameter["R1"],
            )
        )

        all_eps.extend(
            [
                parameter["eps"],
                eps_reference,
            ]
        )

        all_R1.extend(
            [
                parameter["R1"],
                R1_reference,
            ]
        )

    eps_all = np.concatenate(
        all_eps
    )
    R1_all = np.concatenate(
        all_R1
    )

    eps_min = float(
        np.min(eps_all)
    )
    eps_max = float(
        np.max(eps_all)
    )

    R1_min = float(
        np.min(R1_all)
    )
    R1_max = float(
        np.max(R1_all)
    )

    x_limits = (
        eps_min / 1.35,
        eps_max * 1.35,
    )

    y_limits = (
        R1_min / 3.0,
        R1_max * 3.0,
    )

    return (
        x_limits,
        y_limits,
    )



def plot_taylor_tests(
    parameters: dict[str, dict[str, Any]],
) -> None:
    x_limits, y_limits = compute_axis_limits(
        parameters
    )

    fig, ax = plt.subplots(
        figsize=(8.6, 6.0),
        dpi=1200,
        constrained_layout=True,
    )

    parameter_order = (
        "T",
        "A",
        "n",
        "rho2",
    )

    for parameter_key in parameter_order:
        parameter = parameters[
            parameter_key
        ]

        fitted_order = fit_loglog_order(
            parameter["eps"],
            parameter["R1"],
        )

        R1_line, = ax.loglog(
            parameter["eps"],
            parameter["R1"],
            marker=parameter["marker"],
            linewidth=1.6,
            markersize=5.2,
            label=(
                rf"{parameter['label']}: $R_1$ "
                rf"($p={fitted_order:.2f}$)"
            ),
        )

        eps_reference, R1_reference = (
            build_second_order_line(
                parameter["eps"],
                parameter["R1"],
            )
        )

        ax.loglog(
            eps_reference,
            R1_reference,
            linestyle="--",
            linewidth=1.15,
            color=R1_line.get_color(),
            label=(
                rf"{parameter['label']}: $O(h^2)$"
            ),
        )

        print(
            f"[ORDER] {parameter_key}: "
            f"R1={fitted_order:.6f}, "
            f"fit_points="
            f"{min(FIT_POINT_COUNT, parameter['eps'].size)}"
        )

    ax.set_xlim(
        x_limits
    )
    ax.set_ylim(
        y_limits
    )

    ax.invert_xaxis()

    ax.set_xlabel(
        figure_text(
            r"Perturbation size, $h$",
            r"扰动尺度，$h$",
        )
    )

    ax.set_ylabel(
        figure_text(
            r"First-order Taylor remainder, $R_1$",
            r"一阶 Taylor 余项，$R_1$",
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
        fontsize=8.2,
        frameon=False,
        ncol=2,
    )

    png_path = (
        SCRIPT_DIR
        / "case06_taylor_all_parameters_R1.png"
    )

    pdf_path = (
        SCRIPT_DIR
        / "case06_taylor_all_parameters_R1.pdf"
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



def main() -> None:
    parameters = load_all_parameters()
    plot_taylor_tests(parameters)


if __name__ == "__main__":
    main()
