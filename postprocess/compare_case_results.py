#!/usr/bin/env python3
'Compare the Case 02--05 inversion results.'

from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
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


def case_title(case_key: str, original: str) -> str:
    if LANGUAGE != "chinese":
        return original

    mapping = {
        "case02": r"模型 2：Picard，$k=5$",
        "case03": r"模型 3：Picard，$k=100$",
        "case04": r"模型 4：隐式，$10^{-3}$",
        "case05": r"模型 5：隐式，$10^{-8}$",
    }
    return mapping.get(case_key, original)


def case_legend(case_key: str, original: str) -> str:
    return case_title(case_key, original)



SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CASES_DIR = REPO_ROOT / "cases"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "case_comparison"

CASE_CONFIGS = {
    "case02": {
        "directory": CASES_DIR / "case02_picard_5",
        "title": r"Case 2: Picard, $k=5$",
        "legend": r"Case 2: Picard, $k=5$",
        "final_observed": "observed_201.pt",
    },
    "case03": {
        "directory": CASES_DIR / "case03_picard_100",
        "title": r"Case 3: Picard, $k=100$",
        "legend": r"Case 3: Picard, $k=100$",
        "final_observed": "observed_201.pt",
    },
    "case04": {
        "directory": CASES_DIR / "case04_implicit_tol_1e-3",
        "title": r"Case 4: implicit, $10^{-3}$",
        "legend": r"Case 4: implicit, $10^{-3}$",
        "final_observed": "observed_78.pt",
    },
    "case05": {
        "directory": CASES_DIR / "case05_implicit_tol_1e-8",
        "title": r"Case 5: implicit, $10^{-8}$",
        "legend": r"Case 5: implicit, $10^{-8}$",
        "final_observed": "observed_201.pt",
    },
}



L0 = 660e3
XSIZE = 1500e3
YSIZE = 660e3

KAPPA0 = 1e-6
YEAR = 365.25 * 24.0 * 3600.0

T_SURFACE = 273.0
T_MANTLE = 1574.0
DTEMP = T_MANTLE - T_SURFACE

TEMP_STANDARD_TICKS = [T_SURFACE, 800.0, 1200.0, T_MANTLE]

VX_YLIM = (-5.0, 20.0)
VX_YTICKS = [VX_YLIM[0], 0.0, VX_YLIM[1]]

TEMPERATURE_GHOST_CELLS = 1

BAD_VALUE_THRESHOLD = 1e2



def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_pt(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary in {path}, got {type(payload)}")

    return payload


def load_optional_pt(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return load_pt(path)


def maybe_temperature_kelvin(field: Any) -> np.ndarray:
    field_np = np.asarray(to_numpy(field), dtype=float)
    finite = field_np[np.isfinite(field_np)]

    if finite.size == 0:
        return field_np

    if np.nanmax(np.abs(finite)) <= 5.0:
        return T_SURFACE + field_np * DTEMP

    return field_np


def velocity_cm_per_year(velocity: Any) -> np.ndarray:
    return (
        np.asarray(to_numpy(velocity), dtype=float)
        * (KAPPA0 / L0)
        * YEAR
        * 100.0
    )


def get_series(data: dict[str, Any], key: str) -> list[Any]:
    value = data[key]

    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if torch.is_tensor(value):
        return list(value)

    raise TypeError(f"Unsupported time-series type for {key}: {type(value)}")


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


def get_surface_vx_at(
    data: dict[str, Any],
    step_label: int,
) -> np.ndarray:
    if "all_surface_vx" not in data:
        raise KeyError("Forward output does not contain 'all_surface_vx'.")

    idx = safe_step_index(data, step_label, "all_surface_vx")
    vx = to_numpy(get_series(data, "all_surface_vx")[idx]).reshape(-1)

    return velocity_cm_per_year(vx)


def get_final_temperature_from_forward(
    data: dict[str, Any],
) -> np.ndarray:
    if "tkp_end_out" in data:
        return maybe_temperature_kelvin(data["tkp_end_out"])

    if "all_temperature" in data:
        series = get_series(data, "all_temperature")
        if len(series) > 0:
            return maybe_temperature_kelvin(series[-1])

    raise KeyError("Forward output does not contain a final temperature field.")


def get_initial_temperature_from_forward(
    data: dict[str, Any],
) -> np.ndarray | None:
    for key in ("tkp_init_used", "tkp0", "initial_temperature"):
        if key in data:
            return maybe_temperature_kelvin(data[key])

    return None



def maybe_dimensionalize_coordinate(coord: Any) -> np.ndarray:
    coord_np = np.asarray(to_numpy(coord), dtype=float).reshape(-1)

    if coord_np.size == 0:
        return coord_np

    if np.nanmax(np.abs(coord_np)) < 100.0:
        coord_np = coord_np * L0

    return coord_np


def get_mesh_xy(
    mesh_source: dict[str, Any],
    field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mesh_state = mesh_source.get("mesh_state")

    if isinstance(mesh_state, dict) and "xp" in mesh_state and "yp" in mesh_state:
        x = maybe_dimensionalize_coordinate(mesh_state["xp"])
        y = maybe_dimensionalize_coordinate(mesh_state["yp"])

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


def get_surface_x(
    data: dict[str, Any],
    n: int,
) -> np.ndarray:
    mesh_state = data.get("mesh_state")

    if isinstance(mesh_state, dict):
        for key in ("xnode", "xp", "xvx"):
            if key not in mesh_state:
                continue

            x = maybe_dimensionalize_coordinate(mesh_state[key])

            if x.size == n:
                return x / 1e3

    return np.linspace(0.0, XSIZE / 1e3, n)


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float).reshape(-1)

    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])

    middle = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - 0.5 * (centers[1] - centers[0])
    last = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    return np.concatenate([[first], middle, [last]])


def get_physical_temperature_field(
    field: Any,
    ghost_cells: int = TEMPERATURE_GHOST_CELLS,
) -> np.ndarray:
    field_np = np.asarray(to_numpy(field), dtype=float)

    if field_np.ndim != 2:
        raise ValueError(
            f"Temperature field must be 2-D, got shape={field_np.shape}."
        )

    if ghost_cells < 0:
        raise ValueError(
            f"ghost_cells must be non-negative, got {ghost_cells}."
        )

    if ghost_cells == 0:
        return field_np

    required_size = 2 * ghost_cells + 1

    if (
        field_np.shape[0] < required_size
        or field_np.shape[1] < required_size
    ):
        raise ValueError(
            "Temperature field is too small to remove ghost cells: "
            f"shape={field_np.shape}, ghost_cells={ghost_cells}."
        )

    return field_np[
        ghost_cells:-ghost_cells,
        ghost_cells:-ghost_cells,
    ]


def get_physical_temperature_coordinates(
    mesh_source: dict[str, Any],
    full_field: np.ndarray,
    ghost_cells: int = TEMPERATURE_GHOST_CELLS,
) -> tuple[np.ndarray, np.ndarray]:
    x_km, y_km = get_mesh_xy(mesh_source, full_field)

    if ghost_cells == 0:
        return x_km, y_km

    if (
        x_km.size <= 2 * ghost_cells
        or y_km.size <= 2 * ghost_cells
    ):
        raise ValueError(
            "Coordinate arrays are too small to remove ghost cells: "
            f"x={x_km.size}, y={y_km.size}, "
            f"ghost_cells={ghost_cells}."
        )

    return (
        x_km[ghost_cells:-ghost_cells],
        y_km[ghost_cells:-ghost_cells],
    )



def pick_summary_field(
    summary: dict[str, Any],
    keys: tuple[str, ...],
) -> np.ndarray | None:
    for key in keys:
        if key in summary:
            return maybe_temperature_kelvin(summary[key])

    return None


def find_reference_forward(case_dir: Path) -> tuple[dict[str, Any], Path]:
    candidates = [
        case_dir / "observe_data.pt",
        REPO_ROOT / "observe_data.pt",
    ]

    for path in candidates:
        if path.exists():
            return load_pt(path), path

    raise FileNotFoundError(
        "Reference surface vx is required, but observe_data.pt was not found. "
        f"Checked: {candidates[0]} and {candidates[1]}."
    )


def load_case(
    case_key: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    case_dir = Path(config["directory"])

    summary_path = case_dir / "tkp_inversion_summary.pt"
    observed_1_path = case_dir / "observed_1.pt"

    final_observed_name = config.get(
        "final_observed",
        "observed_201.pt",
    )
    final_observed_path = case_dir / final_observed_name

    summary = load_pt(summary_path)
    observed_1 = load_optional_pt(observed_1_path)
    final_observed = load_pt(final_observed_path)
    reference_forward, reference_path = find_reference_forward(case_dir)

    if "all_surface_vx" not in reference_forward:
        raise KeyError(
            f"Reference file does not contain all_surface_vx: {reference_path}"
        )

    if "all_surface_vx" not in final_observed:
        raise KeyError(
            f"Final forward file does not contain all_surface_vx: "
            f"{final_observed_path}"
        )

    mesh_source = final_observed

    if "mesh_state" not in mesh_source and observed_1 is not None:
        mesh_source = observed_1

    if "mesh_state" not in mesh_source:
        mesh_source = reference_forward

    ref_step0 = pick_summary_field(
        summary,
        (
            "tkp0_ref",
            "T0_ref",
            "initial_temperature_ref",
        ),
    )

    if ref_step0 is None:
        ref_step0 = get_initial_temperature_from_forward(
            reference_forward
        )

    try:
        ref_step30 = get_final_temperature_from_forward(
            reference_forward
        )
    except KeyError:
        ref_step30 = pick_summary_field(
            summary,
            (
                "tkp_end_obs",
                "tkp_end_ref",
                "final_temperature_obs",
            ),
        )

    inv_step0 = get_initial_temperature_from_forward(
        final_observed
    )

    if inv_step0 is None:
        inv_step0 = pick_summary_field(
            summary,
            (
                "tkp0_final",
                "T0_final",
                "initial_temperature_final",
            ),
        )

    try:
        pred_step30 = get_final_temperature_from_forward(
            final_observed
        )
    except KeyError:
        pred_step30 = pick_summary_field(
            summary,
            (
                "tkp_end_final",
                "final_temperature_pred",
            ),
        )

    missing = []

    if ref_step0 is None:
        missing.append("reference step 0")
    if ref_step30 is None:
        missing.append("reference step 30")
    if inv_step0 is None:
        missing.append("inverted step 0")
    if pred_step30 is None:
        missing.append("predicted step 30")

    if missing:
        raise KeyError(
            f"{case_key} is missing required temperature fields: "
            f"{', '.join(missing)}. "
            f"Checked {summary_path}, {reference_path}, "
            f"and {final_observed_path}."
        )

    fields = {
        "ref_step0": np.asarray(ref_step0, dtype=float),
        "ref_step30": np.asarray(ref_step30, dtype=float),
        "inv_step0": np.asarray(inv_step0, dtype=float),
        "pred_step30": np.asarray(pred_step30, dtype=float),
    }

    reference_shape = fields["ref_step0"].shape

    for field_name, field in fields.items():
        if field.ndim != 2:
            raise ValueError(
                f"{case_key}/{field_name} must be 2-D, "
                f"got shape {field.shape}."
            )

        if field.shape != reference_shape:
            raise ValueError(
                f"{case_key} field shapes are inconsistent: "
                f"ref_step0={reference_shape}, "
                f"{field_name}={field.shape}."
            )

    print(
        f"[LOAD] {case_key}: "
        f"final_file={final_observed_name}, "
        f"reference_file={reference_path.name}, "
        f"shape={reference_shape}, "
        f"history={len(summary.get('history', []))}"
    )

    return {
        "key": case_key,
        "title": config["title"],
        "legend": config["legend"],
        "directory": case_dir,
        "summary": summary,
        "mesh_source": mesh_source,
        "reference_forward": reference_forward,
        "final_forward": final_observed,
        "fields": fields,
    }



def compute_global_temperature_limits(
    cases: dict[str, dict[str, Any]],
) -> tuple[float, float]:
    global_min = np.inf
    global_max = -np.inf
    n_finite = 0

    for case_key, case in cases.items():
        for field_key, field in case["fields"].items():
            physical_field = get_physical_temperature_field(field)
            finite = physical_field[np.isfinite(physical_field)]

            if finite.size == 0:
                raise ValueError(
                    f"{case_key}/{field_key} contains no finite "
                    "physical temperature values."
                )

            field_min = float(np.min(finite))
            field_max = float(np.max(finite))

            global_min = min(global_min, field_min)
            global_max = max(global_max, field_max)
            n_finite += int(finite.size)

            print(
                f"[PHYSICAL TEMP RANGE] {case_key}/{field_key}: "
                f"min={field_min:.6f} K, "
                f"max={field_max:.6f} K"
            )

    if (
        n_finite == 0
        or not np.isfinite(global_min)
        or not np.isfinite(global_max)
    ):
        raise ValueError(
            "Could not determine a finite physical temperature range."
        )

    if global_max <= global_min:
        padding = max(1.0, abs(global_min) * 1e-6)
        global_min -= padding
        global_max += padding

    print(
        f"[GLOBAL PHYSICAL TEMP RANGE] "
        f"exact min={global_min:.6f} K, "
        f"exact max={global_max:.6f} K, "
        f"finite values={n_finite}"
    )

    return global_min, global_max


def build_temperature_ticks(
    vmin: float,
    vmax: float,
) -> list[float]:
    span = vmax - vmin
    min_separation = max(1e-12, 0.035 * span)

    ticks = [float(vmin), float(vmax)]

    for value in TEMP_STANDARD_TICKS:
        value = float(value)

        if not (vmin < value < vmax):
            continue

        if all(abs(value - existing) >= min_separation for existing in ticks):
            ticks.append(value)

    return sorted(ticks)


def make_temperature_cmap(
    vmin: float,
    vmax: float,
    tref: float = T_MANTLE,
) -> LinearSegmentedColormap:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(
            f"Invalid temperature limits: vmin={vmin}, vmax={vmax}"
        )

    tref_position = (tref - vmin) / (vmax - vmin)
    tref_position = float(np.clip(tref_position, 0.05, 1.0))

    if tref_position >= 1.0 - 1e-12:
        return LinearSegmentedColormap.from_list(
            "case_temperature_no_hot_overshoot",
            [
                (0.00, "#313695"),
                (0.42, "#74add1"),
                (0.72, "#fee090"),
                (1.00, "#d73027"),
            ],
        )

    return LinearSegmentedColormap.from_list(
        "case_temperature_with_hot_overshoot",
        [
            (0.00, "#313695"),
            (0.45 * tref_position, "#74add1"),
            (0.75 * tref_position, "#fee090"),
            (tref_position, "#d73027"),
            (1.00, "#7f0000"),
        ],
    )


def print_reference_consistency(
    cases: dict[str, dict[str, Any]],
    base_case_key: str = "case02",
) -> None:
    base_case = cases[base_case_key]

    for field_key in ("ref_step0", "ref_step30"):
        base = get_physical_temperature_field(
            base_case["fields"][field_key]
        )

        for case_key, case in cases.items():
            current = get_physical_temperature_field(
                case["fields"][field_key]
            )

            if current.shape != base.shape:
                raise ValueError(
                    f"Reference shape mismatch for {field_key}: "
                    f"{case_key}={current.shape}, "
                    f"{base_case_key}={base.shape}."
                )

            diff = current - base
            rmse = float(np.sqrt(np.mean(diff * diff)))
            max_abs = float(np.max(np.abs(diff)))

            print(
                f"[REFERENCE PHYSICAL CHECK] {field_key}: "
                f"{case_key} vs {base_case_key}: "
                f"RMSE={rmse:.6e} K, "
                f"max_abs={max_abs:.6e} K"
            )


def plot_temperature_with_vx(
    ax: plt.Axes,
    *,
    field: np.ndarray,
    mesh_source: dict[str, Any],
    reference_forward: dict[str, Any],
    predicted_forward: dict[str, Any],
    step_label: int,
    cmap,
    temp_vmin: float,
    temp_vmax: float,
    panel_label: str,
    show_vx_axis: bool,
):
    full_field = np.asarray(field, dtype=float)
    field_plot = get_physical_temperature_field(full_field)

    x_km, y_km = get_physical_temperature_coordinates(
        mesh_source,
        full_field,
    )

    if (
        x_km.size != field_plot.shape[1]
        or y_km.size != field_plot.shape[0]
    ):
        raise ValueError(
            "Physical coordinate sizes do not match the cropped "
            f"temperature field: field={field_plot.shape}, "
            f"x={x_km.size}, y={y_km.size}."
        )

    x_edges = centers_to_edges(x_km)
    y_edges = centers_to_edges(y_km)

    image = ax.pcolormesh(
        x_edges,
        y_edges,
        field_plot,
        shading="auto",
        cmap=cmap,
        vmin=temp_vmin,
        vmax=temp_vmax,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        left=False,
        right=False,
        bottom=False,
        top=False,
        labelleft=False,
        labelright=False,
        labelbottom=False,
        labeltop=False,
    )

    ax.text(
        0.015,
        0.97,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="black",
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    inset = ax.inset_axes([0.0, 1.00, 1.0, 0.23])

    vx_true = get_surface_vx_at(reference_forward, step_label)
    vx_pred = get_surface_vx_at(predicted_forward, step_label)

    x_true = get_surface_x(reference_forward, vx_true.size)
    x_pred = get_surface_x(predicted_forward, vx_pred.size)

    inset.plot(
        x_pred,
        vx_pred,
        color="red",
        linewidth=0.9,
        linestyle="--",
        zorder=2,
    )
    inset.plot(
        x_true,
        vx_true,
        color="black",
        linewidth=0.9,
        zorder=3,
    )

    inset.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    inset.set_ylim(VX_YLIM)
    inset.set_xticks([])
    inset.set_yticks(VX_YTICKS)
    inset.margins(x=0.0)

    inset.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False,
    )

    if show_vx_axis:
        inset.tick_params(
            axis="y",
            left=True,
            labelleft=True,
            labelsize=5.8,
            length=2,
        )
    else:
        inset.tick_params(
            axis="y",
            left=False,
            labelleft=False,
        )
        inset.spines["left"].set_visible(False)

    inset.grid(
        True,
        axis="y",
        linestyle=":",
        linewidth=0.35,
    )

    for spine in inset.spines.values():
        spine.set_linewidth(0.45)

    return image


def plot_temperature_comparison(
    cases: dict[str, dict[str, Any]],
) -> None:
    temp_vmin, temp_vmax = compute_global_temperature_limits(cases)
    temp_ticks = build_temperature_ticks(temp_vmin, temp_vmax)
    cmap = make_temperature_cmap(
        temp_vmin,
        temp_vmax,
        tref=T_MANTLE,
    )

    fig, axes = plt.subplots(
        nrows=4,
        ncols=4,
        figsize=(15.8, 10.2),
        dpi=1200,
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.045,
        right=0.925,
        bottom=0.045,
        top=0.91,
        wspace=0.055,
        hspace=0.72,
    )

    panel_specs = [
        (0, 0, "case02", "ref_step0",    0,  "reference"),
        (0, 1, "case02", "ref_step30",  30,  "reference"),
        (0, 2, "case03", "ref_step0",    0,  "reference"),
        (0, 3, "case03", "ref_step30",  30,  "reference"),

        (1, 0, "case02", "inv_step0",    0,  "final"),
        (1, 1, "case02", "pred_step30", 30,  "final"),
        (1, 2, "case03", "inv_step0",    0,  "final"),
        (1, 3, "case03", "pred_step30", 30,  "final"),

        (2, 0, "case04", "ref_step0",    0,  "reference"),
        (2, 1, "case04", "ref_step30",  30,  "reference"),
        (2, 2, "case05", "ref_step0",    0,  "reference"),
        (2, 3, "case05", "ref_step30",  30,  "reference"),

        (3, 0, "case04", "inv_step0",    0,  "final"),
        (3, 1, "case04", "pred_step30", 30,  "final"),
        (3, 2, "case05", "inv_step0",    0,  "final"),
        (3, 3, "case05", "pred_step30", 30,  "final"),
    ]

    panel_letters = [
        f"({chr(ord('a') + index)})"
        for index in range(16)
    ]

    last_image = None

    for panel_index, spec in enumerate(panel_specs):
        row, col, case_key, field_key, step_label, prediction_kind = spec
        case = cases[case_key]

        if prediction_kind == "reference":
            predicted_forward = case["reference_forward"]
        else:
            predicted_forward = case["final_forward"]

        show_vx_axis = (col % 2 == 0)

        last_image = plot_temperature_with_vx(
            axes[row, col],
            field=case["fields"][field_key],
            mesh_source=case["mesh_source"],
            reference_forward=case["reference_forward"],
            predicted_forward=predicted_forward,
            step_label=step_label,
            cmap=cmap,
            temp_vmin=temp_vmin,
            temp_vmax=temp_vmax,
            panel_label=panel_letters[panel_index],
            show_vx_axis=show_vx_axis,
        )

    for row, text in (
        (0, figure_text("Reference", "参考模型")),
        (1, figure_text("Inversion", "反演结果")),
        (2, figure_text("Reference", "参考模型")),
        (3, figure_text("Inversion", "反演结果")),
    ):
        axes[row, 0].text(
            -0.065,
            0.5,
            text,
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=10,
        )

    case_blocks = [
        (0, 0, "case02"),
        (0, 2, "case03"),
        (2, 0, "case04"),
        (2, 2, "case05"),
    ]

    for row, start_col, case_key in case_blocks:
        left_pos = axes[row, start_col].get_position()
        right_pos = axes[row, start_col + 1].get_position()

        block_center = 0.5 * (left_pos.x0 + right_pos.x1)

        fig.text(
            block_center,
            left_pos.y1 + 0.060,
            case_title(case_key, cases[case_key]["title"]),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

        fig.text(
            0.5 * (left_pos.x0 + left_pos.x1),
            left_pos.y1 + 0.034,
            figure_text("Initial", "初始时刻"),
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

        fig.text(
            0.5 * (right_pos.x0 + right_pos.x1),
            right_pos.y1 + 0.034,
            figure_text("Step 30", "第 30 步"),
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    vx_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.0,
            label=figure_text(r"Reference $v_x$", r"参考 $v_x$"),
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=1.0,
            linestyle="--",
            label=figure_text(r"Predicted $v_x$", r"预测 $v_x$"),
        ),
    ]

    fig.legend(
        handles=vx_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    fig.text(
        0.012,
        0.5,
        figure_text(
            r"Surface $v_x$ (cm yr$^{-1}$)",
            r"表面 $v_x$（cm yr$^{-1}$）",
        ),
        rotation=90,
        ha="center",
        va="center",
        fontsize=9,
    )

    cbar_ax = fig.add_axes([0.942, 0.075, 0.012, 0.80])
    colorbar = fig.colorbar(
        last_image,
        cax=cbar_ax,
        orientation="vertical",
    )
    colorbar.set_label(figure_text("Temperature (K)", "温度（K）"))
    colorbar.set_ticks(temp_ticks)
    colorbar.set_ticklabels(
        [f"{value:.1f}" for value in temp_ticks]
    )

    png_path = (
        OUTPUT_DIR
        / "case02_case05_temperature_vx_physical_range.png"
    )
    pdf_path = (
        OUTPUT_DIR
        / "case02_case05_temperature_vx_physical_range.pdf"
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



def history_array(
    history: list[dict[str, Any]],
    key: str,
    *,
    default: float = np.nan,
) -> np.ndarray:
    return np.asarray(
        [record.get(key, default) for record in history],
        dtype=float,
    )


def plot_metric_curves(
    ax: plt.Axes,
    cases: dict[str, dict[str, Any]],
    *,
    metric_key: str,
    title: str,
    ylabel: str,
):
    for case_key in (
        "case02",
        "case03",
        "case04",
        "case05",
    ):
        case = cases[case_key]
        history = case["summary"].get("history", [])

        if len(history) == 0:
            print(f"[WARN] Empty history: {case_key}")
            continue

        closure = history_array(history, "closure")
        total = history_array(history, "total_loss")
        metric = history_array(history, metric_key)

        valid = (
            np.isfinite(closure)
            & np.isfinite(total)
            & np.isfinite(metric)
            & (total > 0.0)
            & (total < BAD_VALUE_THRESHOLD)
            & (metric > 0.0)
            & (metric < BAD_VALUE_THRESHOLD)
        )

        metric_plot = np.where(valid, metric, np.nan)

        ax.semilogy(
            closure,
            metric_plot,
            linewidth=1.35,
            label=case_legend(case_key, case["legend"]),
        )

    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.55,
    )


def plot_loss_comparison(
    cases: dict[str, dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(11.0, 7.2),
        dpi=1200,
        sharex=True,
        constrained_layout=True,
    )

    metric_specs = [
        (
            axes[0, 0],
            "total_loss",
            figure_text("Total loss", "总损失"),
            figure_text("Loss", "损失"),
        ),
        (
            axes[0, 1],
            "loss_T",
            figure_text("Final-temperature loss", "最终温度损失"),
            r"$J_T$",
        ),
        (
            axes[1, 0],
            "loss_vx",
            figure_text("Surface-velocity loss", "表面速度损失"),
            r"$J_{v_x}$",
        ),
        (
            axes[1, 1],
            "grad_dT",
            figure_text("Temperature-gradient norm", "温度梯度范数"),
            r"$\|g_T\|$",
        ),
    ]

    for ax, metric_key, title, ylabel in metric_specs:
        plot_metric_curves(
            ax,
            cases,
            metric_key=metric_key,
            title=title,
            ylabel=ylabel,
        )

    axes[1, 0].set_xlabel(
        figure_text("Closure / function evaluation", "闭包 / 函数评估次数")
    )
    axes[1, 1].set_xlabel(
        figure_text("Closure / function evaluation", "闭包 / 函数评估次数")
    )

    for panel_label, ax in zip(
        ("(a)", "(b)", "(c)", "(d)"),
        axes.ravel(),
    ):
        ax.text(
            0.015,
            0.97,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            fontsize=8,
            frameon=False,
        )

    png_path = (
        OUTPUT_DIR
        / "case02_case05_loss_comparison.png"
    )
    pdf_path = (
        OUTPUT_DIR
        / "case02_case05_loss_comparison.pdf"
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
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = {
        case_key: load_case(case_key, config)
        for case_key, config in CASE_CONFIGS.items()
    }

    print_reference_consistency(
        cases,
        base_case_key="case02",
    )

    plot_temperature_comparison(cases)
    plot_loss_comparison(cases)


if __name__ == "__main__":
    main()
