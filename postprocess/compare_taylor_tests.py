#!/usr/bin/env python3
'Compare the Case 02--05 Taylor tests.'

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


def localized_case_label(case_key: str, original: str) -> str:
    if LANGUAGE != "chinese":
        return original

    mapping = {
        "case02": r"模型 2：Picard，$k=5$",
        "case03": r"模型 3：Picard，$k=100$",
        "case04": r"模型 4：隐式，$10^{-3}$",
        "case05": r"模型 5：隐式，$10^{-8}$",
    }
    return mapping.get(case_key, original)



SCRIPT_DIR = Path(__file__).resolve().parent
CASES_DIR = SCRIPT_DIR.parent / "cases"

OUTPUT_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "taylor_tests"
)

CASE_CONFIGS = {
    "case02": {
        "path": (
            CASES_DIR
            / "case02_picard_5"
            / "taylor_test"
            / "taylor_dT_param.pt"
        ),
        "label": r"Case 2: Picard, $k=5$",
        "marker": "o",
    },
    "case03": {
        "path": (
            CASES_DIR
            / "case03_picard_100"
            / "taylor_test"
            / "taylor_dT_param.pt"
        ),
        "label": r"Case 3: Picard, $k=100$",
        "marker": "s",
    },
    "case04": {
        "path": (
            CASES_DIR
            / "case04_implicit_tol_1e-3"
            / "taylor_test"
            / "taylor_dT_param.pt"
        ),
        "label": r"Case 4: implicit, $10^{-3}$",
        "marker": "^",
    },
    "case05": {
        "path": (
            CASES_DIR
            / "case05_implicit_tol_1e-8"
            / "taylor_test"
            / "taylor_dT_param.pt"
        ),
        "label": r"Case 5: implicit, $10^{-8}$",
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


def load_taylor_file(
    path: Path,
) -> dict[str, Any]:
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
    case_key: str,
) -> dict[str, Any]:
    eps = to_numpy(payload["eps"])
    R1 = to_numpy(payload["R1"])

    if eps.size != R1.size:
        raise ValueError(
            f"{case_key}: inconsistent array lengths: "
            f"eps={eps.size}, R1={R1.size}."
        )

    valid = (
        np.isfinite(eps)
        & np.isfinite(R1)
        & (eps > 0.0)
        & (R1 > 0.0)
    )

    n_total = eps.size
    n_valid = int(np.sum(valid))

    if n_valid < 2:
        raise ValueError(
            f"{case_key}: only {n_valid}/{n_total} valid "
            "Taylor-test points remain."
        )

    eps = eps[valid]
    R1 = R1[valid]

    order = np.argsort(eps)[::-1]

    return {
        "eps": eps[order],
        "R1": R1[order],
        "n_total": n_total,
        "n_valid": n_valid,
    }


def load_all_cases() -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}

    for case_key, config in CASE_CONFIGS.items():
        payload = load_taylor_file(
            config["path"]
        )

        data = clean_taylor_data(
            payload,
            case_key,
        )

        cases[case_key] = {
            **config,
            **data,
        }

        print(
            f"[LOAD] {case_key}: "
            f"points={data['n_valid']}/{data['n_total']}, "
            f"eps=[{data['eps'].min():.1e}, "
            f"{data['eps'].max():.1e}]"
        )

    return cases



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


def interpolate_loglog(
    eps: np.ndarray,
    values: np.ndarray,
    target_eps: float,
) -> float:
    order = np.argsort(eps)

    log_eps = np.log(
        eps[order]
    )
    log_values = np.log(
        values[order]
    )

    target_log_eps = np.log(
        target_eps
    )

    value = np.exp(
        np.interp(
            target_log_eps,
            log_eps,
            log_values,
        )
    )

    return float(value)


def build_common_second_order_line(
    cases: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    common_eps_max = min(
        float(case["eps"].max())
        for case in cases.values()
    )

    global_eps_min = min(
        float(case["eps"].min())
        for case in cases.values()
    )

    anchor_values = np.asarray(
        [
            interpolate_loglog(
                case["eps"],
                case["R1"],
                common_eps_max,
            )
            for case in cases.values()
        ],
        dtype=float,
    )

    anchor_values = anchor_values[
        np.isfinite(anchor_values)
        & (anchor_values > 0.0)
    ]

    if anchor_values.size == 0:
        raise ValueError(
            "Could not determine an anchor for the O(h^2) line."
        )

    anchor_R1 = float(
        np.exp(
            np.mean(
                np.log(anchor_values)
            )
        )
    )

    eps_reference = np.logspace(
        np.log10(global_eps_min),
        np.log10(common_eps_max),
        200,
    )

    R1_reference = (
        max(anchor_R1, POSITIVE_FLOOR)
        * (eps_reference / common_eps_max) ** 2
    )

    return eps_reference, R1_reference


def compute_axis_limits(
    cases: dict[str, dict[str, Any]],
    eps_reference: np.ndarray,
    R1_reference: np.ndarray,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
]:
    all_eps = np.concatenate(
        [
            case["eps"]
            for case in cases.values()
        ]
        + [eps_reference]
    )

    all_R1 = np.concatenate(
        [
            case["R1"]
            for case in cases.values()
        ]
        + [R1_reference]
    )

    eps_min = float(
        np.min(all_eps)
    )
    eps_max = float(
        np.max(all_eps)
    )

    R1_min = float(
        np.min(all_R1)
    )
    R1_max = float(
        np.max(all_R1)
    )

    x_limits = (
        eps_min / 1.35,
        eps_max * 1.35,
    )

    y_limits = (
        R1_min / 3.0,
        R1_max * 3.0,
    )

    return x_limits, y_limits



def plot_taylor_R1_comparison(
    cases: dict[str, dict[str, Any]],
) -> None:
    eps_reference, R1_reference = (
        build_common_second_order_line(cases)
    )

    x_limits, y_limits = compute_axis_limits(
        cases,
        eps_reference,
        R1_reference,
    )

    fig, ax = plt.subplots(
        figsize=(8.2, 5.8),
        dpi=1200,
        constrained_layout=True,
    )

    for case_key in (
        "case02",
        "case03",
        "case04",
        "case05",
    ):
        case = cases[case_key]

        fitted_order = fit_loglog_order(
            case["eps"],
            case["R1"],
        )

        ax.loglog(
            case["eps"],
            case["R1"],
            marker=case["marker"],
            linewidth=1.5,
            markersize=5.0,
            label=(
                f"{localized_case_label(case_key, case['label'])} "
                rf"($p={fitted_order:.2f}$)"
            ),
        )

        print(
            f"[ORDER] {case_key}: "
            f"R1={fitted_order:.6f}, "
            f"fit_points="
            f"{min(FIT_POINT_COUNT, case['eps'].size)}"
        )

    ax.loglog(
        eps_reference,
        R1_reference,
        linestyle="--",
        linewidth=1.3,
        label=r"$O(h^2)$",
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
        fontsize=8.5,
        frameon=False,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        OUTPUT_DIR
        / "case02_case05_taylor_R1_single_panel.png"
    )

    pdf_path = (
        OUTPUT_DIR
        / "case02_case05_taylor_R1_single_panel.pdf"
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
    cases = load_all_cases()
    plot_taylor_R1_comparison(cases)


if __name__ == "__main__":
    main()
