"""Run physical benchmarks in a temporary directory."""

import argparse
import importlib.util
import json
import math
import os
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_finite(value, name="result"):
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite(item, f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} is not finite")


def validate_convection(resolution, quick):
    module = load_script("isoviscous_convection")
    if quick:
        module.MAX_STEPS = 2
        module.MIN_STEPS = 0
        module.DIAG_INTERVAL = 1
        module.SAVE_FIG_INTERVAL = 1000
        module.STEADY_COUNT_TARGET = 100

    with tempfile.TemporaryDirectory(prefix="adepts_convection_") as output_dir:
        module.SAVE_DIR = output_dir
        summary = module.run_one_resolution(resolution)

    require_finite(summary)
    last = summary["last_row"]
    passed = math.isfinite(last["Nu_mean"]) and math.isfinite(last["Vrms"])
    if not quick:
        passed = passed and 4.0 <= last["Nu_mean"] <= 6.0 and 30.0 <= last["Vrms"] <= 60.0
    return {
        "benchmark": "isoviscous_convection",
        "profile": "quick" if quick else "manuscript",
        "resolution": resolution,
        "status": "passed" if passed else "failed",
        "last_row": last,
    }


def validate_slab(convergence, resolution_km, quick):
    if convergence:
        os.environ["RES_KM"] = str(resolution_km)
        module = load_script("slab_detachment_convergence")
    else:
        module = load_script("slab_detachment")
        module.resolution = resolution_km * 1e3

    if quick:
        module.num_steps = 1
        if hasattr(module, "final_time_years"):
            module.final_time_years = module.num_steps * module.dt_years
        module.diag_interval = 1
        module.plot_interval = 1000

    with tempfile.TemporaryDirectory(prefix="adepts_slab_") as output_dir:
        module.SAVE_DIR = output_dir
        module.main()
        with open(Path(output_dir) / "summary.json", encoding="utf-8") as stream:
            summary = json.load(stream)

    require_finite(summary)
    last = summary["last_diagnostics"]
    passed = isinstance(last, dict) and len(last) > 0
    return {
        "benchmark": "slab_detachment_convergence" if convergence else "slab_detachment",
        "profile": "quick" if quick else "manuscript",
        "resolution_km": resolution_km,
        "status": "passed" if passed else "failed",
        "last_diagnostics": last,
        "detachment": summary.get("detachment"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=("convection", "slab", "slab-convergence"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--resolution-km", type=float, default=10.0)
    args = parser.parse_args()

    started = time.time()
    if args.benchmark == "convection":
        result = validate_convection(args.resolution, args.quick)
    else:
        result = validate_slab(
            args.benchmark == "slab-convergence",
            args.resolution_km,
            args.quick,
        )
    result["wall_time_sec"] = time.time() - started
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
