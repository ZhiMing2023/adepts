"""Run analytical benchmarks without retaining outputs."""

import argparse
import importlib.util
import json
import tempfile
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_slope(rows, field):
    dt = np.asarray([row["dt_nd"] for row in rows])
    error = np.asarray([row[field] for row in rows])
    return float(np.polyfit(np.log(dt), np.log(error), 1)[0])


def validate_thermal(quick):
    module = load_script("thermal_diffusion")
    resolution = 20e3 if quick else 2e3
    nsteps = [2, 4, 8] if quick else [10, 20, 40, 80]
    with tempfile.TemporaryDirectory(prefix="adepts_thermal_") as output_dir:
        rows = [
            module.run_diffusion_case(
                resolution=resolution,
                t_end_nd=0.01,
                nstep=count,
                save_dir=output_dir,
                make_plots=False,
                save_tensor=False,
            )
            for count in nsteps
        ]
    slopes = {field: fit_slope(rows, field) for field in ("L2_int", "Linf_int")}
    return {
        "benchmark": "thermal_diffusion",
        "profile": "quick" if quick else "manuscript",
        "slopes": slopes,
        "status": "passed"
        if min(slopes.values()) >= (0.8 if quick else 0.9)
        else "failed",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=("thermal",))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    started = time.time()
    result = validate_thermal(args.quick)
    result["wall_time_sec"] = time.time() - started
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
