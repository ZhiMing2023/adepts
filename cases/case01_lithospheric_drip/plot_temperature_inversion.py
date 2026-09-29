#!/usr/bin/env python3
'Plot the Case 01 inversion summary.'

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


DEFAULT_T_SURFACE = 1574.0
DEFAULT_DTEMP = 500.0
DEFAULT_T_MANTLE = 1574.0
DEFAULT_XSIZE_KM = 500.0
DEFAULT_YSIZE_KM = 500.0


def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_summary(path: str | Path) -> dict[str, Any]:
    summary = torch.load(path, map_location="cpu")
    if not isinstance(summary, dict):
        raise TypeError("The summary file must contain a dictionary.")
    return summary


def as_float_array(values: list[Any], key: str) -> np.ndarray:
    out = []
    for rec in values:
        if key not in rec:
            out.append(np.nan)
        else:
            try:
                out.append(float(rec[key]))
            except Exception:
                out.append(np.nan)
    return np.asarray(out, dtype=float)


def get_history(summary: dict[str, Any]) -> list[dict[str, Any]]:
    history = summary.get("history", [])
    if history is None:
        history = []
    if not isinstance(history, list):
        raise TypeError("summary['history'] must be a list.")
    return history


def require_field(summary: dict[str, Any], key: str) -> np.ndarray:
    if key not in summary:
        raise KeyError(f"Missing field in summary: {key}")
    arr = to_numpy(summary[key]).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"summary['{key}'] must be 2D, got shape={arr.shape}")
    return arr.astype(float)


def crop_field(field: np.ndarray, remove_ghosts: bool) -> np.ndarray:
    if not remove_ghosts:
        return field
    if field.ndim != 2 or field.shape[0] <= 2 or field.shape[1] <= 2:
        return field
    return field[1:-1, 1:-1]


def maybe_to_kelvin(field: np.ndarray, *, t_surface: float, dtemp: float, dimensionless: bool) -> np.ndarray:
    if dimensionless:
        return field

    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return field

    if np.nanmax(np.abs(finite)) <= 10.0:
        return t_surface + field * dtemp
    return field


def robust_limits(field: np.ndarray, q: tuple[float, float] = (1.0, 99.0)) -> tuple[float, float]:
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, q[0]))
    vmax = float(np.nanpercentile(finite, q[1]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def symmetric_norm(field: np.ndarray, quantile: float = 99.0) -> TwoSlopeNorm:
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        scale = 1.0
    else:
        scale = float(np.nanpercentile(np.abs(finite), quantile))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
    return TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)


def field_extent(xsize_km: float, ysize_km: float) -> list[float]:
    return [0.0, float(xsize_km), float(ysize_km), 0.0]


def plot_image(
    ax: plt.Axes,
    field: np.ndarray,
    *,
    extent: list[float],
    title: str,
    label: str,
    cmap: str,
    norm: Any | None = None,
):
    im = ax.imshow(
        field,
        origin="upper",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(title)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("Depth y (km)")
    return im


def savefig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_initial_temperature_fields(
    summary: dict[str, Any],
    output_dir: Path,
    *,
    t_surface: float,
    dtemp: float,
    t_mantle: float,
    xsize_km: float,
    ysize_km: float,
    dimensionless: bool,
    remove_ghosts: bool,
    dpi: int,
) -> None:
    tkp_bg = crop_field(require_field(summary, "tkp_bg"), remove_ghosts)
    tkp_ref = crop_field(require_field(summary, "tkp0_ref"), remove_ghosts)
    tkp_final = crop_field(require_field(summary, "tkp0_final"), remove_ghosts)

    T_bg = maybe_to_kelvin(tkp_bg, t_surface=t_surface, dtemp=dtemp, dimensionless=dimensionless)
    T_ref = maybe_to_kelvin(tkp_ref, t_surface=t_surface, dtemp=dtemp, dimensionless=dimensionless)
    T_final = maybe_to_kelvin(tkp_final, t_surface=t_surface, dtemp=dtemp, dimensionless=dimensionless)

    true_anom = T_ref - T_bg
    rec_anom = T_final - T_bg
    err = T_final - T_ref

    unit = "nondim" if dimensionless else "K"
    extent = field_extent(xsize_km, ysize_km)

    fig, axes = plt.subplots(3, 3, figsize=(14.5, 13.0), constrained_layout=True)

    vmin_T, vmax_T = robust_limits(np.concatenate([T_bg.ravel(), T_ref.ravel(), T_final.ravel()]))

    fields = [
        (T_bg, "Initial background", f"Temperature ({unit})", "turbo", None, vmin_T, vmax_T),
        (T_ref, "True initial T0", f"Temperature ({unit})", "turbo", None, vmin_T, vmax_T),
        (T_final, "Recovered initial T0", f"Temperature ({unit})", "turbo", None, vmin_T, vmax_T),
        (true_anom, "True anomaly: T0 - background", f"Anomaly ({unit})", "coolwarm", symmetric_norm(true_anom), None, None),
        (rec_anom, "Recovered anomaly", f"Anomaly ({unit})", "coolwarm", symmetric_norm(np.concatenate([true_anom.ravel(), rec_anom.ravel()])), None, None),
        (err, "Initial T0 error: recovered - true", f"Error ({unit})", "coolwarm", symmetric_norm(err), None, None),
        (T_ref - t_mantle if not dimensionless else tkp_ref - np.nanmedian(tkp_ref), "True relative to mantle", f"Relative T ({unit})", "coolwarm", None, None, None),
        (T_final - t_mantle if not dimensionless else tkp_final - np.nanmedian(tkp_final), "Recovered relative to mantle", f"Relative T ({unit})", "coolwarm", None, None, None),
        (np.abs(err), "Absolute initial T0 error", f"Abs error ({unit})", "magma", None, None, None),
    ]

    for ax, (field, title, label, cmap, norm, vmin, vmax) in zip(axes.ravel(), fields):
        if title.startswith("True relative") or title.startswith("Recovered relative"):
            norm = symmetric_norm(field)
        kwargs = {"extent": extent, "title": title, "label": label, "cmap": cmap, "norm": norm}
        im = plot_image(ax, field, **kwargs)
        if vmin is not None and vmax is not None and norm is None:
            im.set_clim(vmin, vmax)
        fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)

    savefig(fig, output_dir / "initial_temperature_recovery.png", dpi=dpi)


def plot_final_temperature_fields(
    summary: dict[str, Any],
    output_dir: Path,
    *,
    t_surface: float,
    dtemp: float,
    xsize_km: float,
    ysize_km: float,
    dimensionless: bool,
    remove_ghosts: bool,
    dpi: int,
) -> None:
    if "tkp_end_final" not in summary or "tkp_end_obs" not in summary:
        print("Skip final-temperature plot: missing tkp_end_final or tkp_end_obs.")
        return

    end_final = crop_field(require_field(summary, "tkp_end_final"), remove_ghosts)
    end_obs = crop_field(require_field(summary, "tkp_end_obs"), remove_ghosts)

    T_final = maybe_to_kelvin(end_final, t_surface=t_surface, dtemp=dtemp, dimensionless=dimensionless)
    T_obs = maybe_to_kelvin(end_obs, t_surface=t_surface, dtemp=dtemp, dimensionless=dimensionless)
    err = T_final - T_obs

    unit = "nondim" if dimensionless else "K"
    extent = field_extent(xsize_km, ysize_km)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    vmin_T, vmax_T = robust_limits(np.concatenate([T_final.ravel(), T_obs.ravel()]))

    panels = [
        (T_obs, "Observed final temperature", f"Temperature ({unit})", "turbo", None, vmin_T, vmax_T),
        (T_final, "Predicted final temperature", f"Temperature ({unit})", "turbo", None, vmin_T, vmax_T),
        (err, "Final temperature error", f"Error ({unit})", "coolwarm", symmetric_norm(err), None, None),
    ]

    for ax, (field, title, label, cmap, norm, vmin, vmax) in zip(axes.ravel(), panels):
        im = plot_image(ax, field, extent=extent, title=title, label=label, cmap=cmap, norm=norm)
        if vmin is not None and vmax is not None and norm is None:
            im.set_clim(vmin, vmax)
        fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)

    savefig(fig, output_dir / "final_temperature_fit.png", dpi=dpi)


def plot_loss_history(summary: dict[str, Any], output_dir: Path, *, dpi: int) -> None:
    history = get_history(summary)
    if not history:
        print("Skip loss-history plot: empty history.")
        return

    closure = as_float_array(history, "closure")
    if np.all(~np.isfinite(closure)):
        closure = np.arange(1, len(history) + 1, dtype=float)

    loss_keys = [
        ("total_loss", "Total"),
        ("data_loss", "Data"),
        ("loss_T", "Final T"),
        ("loss_vx", "Surface vx"),
        ("loss_sigmayy", "Surface sigma_yy"),
        ("loss_C", "Composition"),
        ("reg_tkp", "T smoothness"),
        ("loss_T_bound", "T bound"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    plotted = False
    for key, label in loss_keys:
        y = as_float_array(history, key)
        if np.any(np.isfinite(y)) and np.nanmax(np.abs(y)) > 0.0:
            ax.plot(closure, y, linewidth=1.5, label=label)
            plotted = True

    if not plotted:
        print("Skip loss-history plot: no finite nonzero loss entries.")
        plt.close(fig)
        return

    ax.set_yscale("log")
    ax.set_xlabel("Closure evaluation")
    ax.set_ylabel("Loss")
    ax.set_title("LBFGS loss history")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    savefig(fig, output_dir / "loss_history.png", dpi=dpi)


def plot_gradient_history(summary: dict[str, Any], output_dir: Path, *, dpi: int) -> None:
    history = get_history(summary)
    if not history:
        print("Skip gradient-history plot: empty history.")
        return

    closure = as_float_array(history, "closure")
    if np.all(~np.isfinite(closure)):
        closure = np.arange(1, len(history) + 1, dtype=float)

    grad_keys = [
        ("grad_dT", "||grad T0||"),
        ("grad_rho2", "grad rho2"),
        ("grad_A", "grad A"),
        ("grad_n", "grad n"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    plotted = False
    for key, label in grad_keys:
        y = np.abs(as_float_array(history, key))
        if np.any(np.isfinite(y)) and np.nanmax(y) > 0.0:
            ax.plot(closure, y, linewidth=1.5, label=label)
            plotted = True

    if not plotted:
        print("Skip gradient-history plot: no finite nonzero gradient entries.")
        plt.close(fig)
        return

    ax.set_yscale("log")
    ax.set_xlabel("Closure evaluation")
    ax.set_ylabel("Gradient magnitude")
    ax.set_title("Gradient history")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    savefig(fig, output_dir / "gradient_history.png", dpi=dpi)


def first_scalar(summary: dict[str, Any], key: str) -> float | None:
    if key not in summary:
        return None
    arr = to_numpy(summary[key]).reshape(-1)
    if arr.size == 0:
        return None
    try:
        return float(arr[0])
    except Exception:
        return None


def plot_parameter_history(summary: dict[str, Any], output_dir: Path, *, dpi: int) -> None:
    history = get_history(summary)
    if not history:
        print("Skip parameter-history plot: empty history.")
        return

    closure = as_float_array(history, "closure")
    if np.all(~np.isfinite(closure)):
        closure = np.arange(1, len(history) + 1, dtype=float)

    params = [
        ("rho2", "rho2", first_scalar(summary, "rho_param_ref"), first_scalar(summary, "rho_param_final")),
        ("A", "A", first_scalar(summary, "A_param_ref"), first_scalar(summary, "A_param_final")),
        ("n", "n", first_scalar(summary, "n_param_ref"), first_scalar(summary, "n_param_final")),
    ]

    available = []
    for key, label, ref, final in params:
        y = as_float_array(history, key)
        if np.any(np.isfinite(y)):
            available.append((key, label, y, ref, final))

    if not available:
        print("Skip parameter-history plot: no parameter entries in history.")
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(8.5, 3.2 * len(available)), constrained_layout=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (key, label, y, ref, final) in zip(axes, available):
        ax.plot(closure, y, linewidth=1.5, label=label)
        if ref is not None:
            ax.axhline(ref, linestyle="--", linewidth=1.2, label="Reference")
        if final is not None:
            ax.axhline(final, linestyle=":", linewidth=1.2, label="Saved final")
        ax.set_xlabel("Closure evaluation")
        ax.set_ylabel(label)
        ax.set_title(f"Parameter history: {label}")
        ax.grid(True, linestyle=":", linewidth=0.7)
        ax.legend()

    savefig(fig, output_dir / "parameter_history.png", dpi=dpi)


def print_summary(summary: dict[str, Any], *, t_surface: float, dtemp: float, remove_ghosts: bool) -> None:
    print("Summary keys:")
    for key in sorted(summary.keys()):
        print(f"  - {key}")

    print(f"Temperature conversion: T(K) = {t_surface:.6g} + tkp * {dtemp:.6g}")
    print(f"Remove ghost layer: {remove_ghosts}")

    for key in ("tkp_bg", "tkp0_ref", "tkp0_final", "tkp_end_obs", "tkp_end_final"):
        if key in summary:
            arr = crop_field(to_numpy(summary[key]).squeeze().astype(float), remove_ghosts)
            print(
                f"{key}: shape={arr.shape}, "
                f"min={np.nanmin(arr):.6e}, max={np.nanmax(arr):.6e}, "
                f"mean={np.nanmean(arr):.6e}"
            )

    if "tkp0_ref" in summary and "tkp0_final" in summary:
        ref = crop_field(to_numpy(summary["tkp0_ref"]).squeeze().astype(float), remove_ghosts)
        final = crop_field(to_numpy(summary["tkp0_final"]).squeeze().astype(float), remove_ghosts)
        diff_nd = final - ref
        diff_K = diff_nd * dtemp
        print(
            "Initial T0 error: "
            f"RMSE_nd={np.sqrt(np.nanmean(diff_nd**2)):.6e}, "
            f"MAE_nd={np.nanmean(np.abs(diff_nd)):.6e}, "
            f"RMSE_K={np.sqrt(np.nanmean(diff_K**2)):.6e}, "
            f"MAE_K={np.nanmean(np.abs(diff_K)):.6e}"
        )

    if "tkp_end_obs" in summary and "tkp_end_final" in summary:
        obs = crop_field(to_numpy(summary["tkp_end_obs"]).squeeze().astype(float), remove_ghosts)
        final = crop_field(to_numpy(summary["tkp_end_final"]).squeeze().astype(float), remove_ghosts)
        diff_nd = final - obs
        diff_K = diff_nd * dtemp
        print(
            "Final T error: "
            f"RMSE_nd={np.sqrt(np.nanmean(diff_nd**2)):.6e}, "
            f"MAE_nd={np.nanmean(np.abs(diff_nd)):.6e}, "
            f"RMSE_K={np.sqrt(np.nanmean(diff_K**2)):.6e}, "
            f"MAE_K={np.nanmean(np.abs(diff_K)):.6e}"
        )

    history = get_history(summary)
    print(f"History length: {len(history)}")
    if history:
        last = history[-1]
        print("Last history row:")
        for key in ("closure", "total_loss", "data_loss", "loss_T", "loss_vx", "loss_sigmayy", "reg_tkp", "rho2", "A", "n", "grad_dT"):
            if key in last:
                print(f"  {key}: {last[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tkp_inversion_summary.pt.")
    parser.add_argument("--input", type=str, default="tkp_inversion_summary.pt", help="Path to tkp_inversion_summary.pt.")
    parser.add_argument("--output-dir", type=str, default="inversion_summary_figures", help="Output directory.")
    parser.add_argument("--summary-only", action="store_true", help="Only print summary; do not save figures.")
    parser.add_argument("--dimensionless", action="store_true", help="Plot nondimensional temperature values.")
    parser.add_argument("--keep-ghosts", action="store_true", help="Keep outer ghost layer in 2D fields.")
    parser.add_argument("--dpi", type=int, default=200, help="Figure resolution.")

    parser.add_argument("--t-surface", type=float, default=DEFAULT_T_SURFACE, help="Temperature offset for tkp -> K conversion.")
    parser.add_argument("--dtemp", type=float, default=DEFAULT_DTEMP, help="Temperature scale for tkp -> K conversion.")
    parser.add_argument("--t-mantle", type=float, default=DEFAULT_T_MANTLE, help="Mantle/background temperature for relative plots.")
    parser.add_argument("--xsize-km", type=float, default=DEFAULT_XSIZE_KM, help="Physical model width for plotting.")
    parser.add_argument("--ysize-km", type=float, default=DEFAULT_YSIZE_KM, help="Physical model height for plotting.")

    args = parser.parse_args()

    summary = load_summary(args.input)
    output_dir = Path(args.output_dir)
    remove_ghosts = not args.keep_ghosts

    print_summary(
        summary,
        t_surface=args.t_surface,
        dtemp=args.dtemp,
        remove_ghosts=remove_ghosts,
    )

    if args.summary_only:
        return

    plot_initial_temperature_fields(
        summary,
        output_dir,
        t_surface=args.t_surface,
        dtemp=args.dtemp,
        t_mantle=args.t_mantle,
        xsize_km=args.xsize_km,
        ysize_km=args.ysize_km,
        dimensionless=args.dimensionless,
        remove_ghosts=remove_ghosts,
        dpi=args.dpi,
    )

    plot_final_temperature_fields(
        summary,
        output_dir,
        t_surface=args.t_surface,
        dtemp=args.dtemp,
        xsize_km=args.xsize_km,
        ysize_km=args.ysize_km,
        dimensionless=args.dimensionless,
        remove_ghosts=remove_ghosts,
        dpi=args.dpi,
    )

    plot_loss_history(summary, output_dir, dpi=args.dpi)
    plot_gradient_history(summary, output_dir, dpi=args.dpi)
    plot_parameter_history(summary, output_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
