"""Run a disposable case validation."""

import argparse
import importlib.util
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    path.parent.name: path
    for path in sorted((ROOT / "cases").glob("*/run.py"))
}


def load_case(path):
    spec = importlib.util.spec_from_file_location("adepts_case", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_finite(value, name="output"):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{name}[{index}]")


def assess_taylor(results, min_slope=1.8):
    if "R1" in results:
        results = {results["param_name"]: results}

    slopes = {}
    status = {}
    for name, item in results.items():
        eps = np.asarray(item["eps"], dtype=float)
        remainder = np.asarray(item["R1"], dtype=float)
        if not np.isfinite(remainder).all() or np.any(remainder <= 0.0):
            raise ValueError(f"Taylor remainder for {name} is not positive and finite")
        slope = float(np.polyfit(np.log(eps), np.log(remainder), 1)[0])
        slopes[name] = slope
        if abs(float(item["g_dot_d"])) < 1e-12:
            status[name] = "inconclusive: directional derivative is too small"
        elif slope >= min_slope:
            status[name] = "passed"
        else:
            status[name] = "failed"
    return slopes, status


def run(case_name, steps, run_inverse, run_taylor, eps_list):
    started = time.time()
    module = load_case(CASES[case_name])
    if steps is not None:
        module.num_steps = steps

    with tempfile.TemporaryDirectory(prefix=f"adepts_{case_name}_") as work_dir:
        previous_dir = Path.cwd()
        os.chdir(work_dir)
        try:
            module.forward_main(filename="observe_data.pt")
            payload = torch.load("observe_data.pt", map_location="cpu", weights_only=False)
            observation = payload.get("data", payload)
            assert_finite(payload, "observation")

            result = {
                "case": case_name,
                "forward_steps": int(
                    observation.get("actual_num_steps", module.num_steps)
                ),
                "forward": "passed",
            }

            if run_inverse:
                summary = module.inversion_main(
                    observation_path="observe_data.pt",
                    save_path="inversion_summary.pt",
                    max_iter=1,
                    max_eval=2,
                )
                assert_finite(summary, "inversion")
                result["inverse"] = "passed"
                result["best_loss"] = float(summary["best_loss"])

            if run_taylor:
                taylor = module.taylor_test_main(
                    observation_path="observe_data.pt",
                    save_dir="taylor_test",
                    eps_list=eps_list,
                    do_plot=False,
                )
                slopes, status = assess_taylor(taylor)
                result["taylor"] = status
                result["taylor_R1_slopes"] = slopes
        finally:
            os.chdir(previous_dir)

    result["wall_time_sec"] = time.time() - started
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--inverse", action="store_true")
    parser.add_argument("--taylor", action="store_true")
    parser.add_argument("--eps", default="1e-1,1e-2,1e-3")
    args = parser.parse_args()

    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")

    try:
        eps_list = tuple(float(value) for value in args.eps.split(","))
    except ValueError as error:
        parser.error(f"invalid --eps: {error}")
    if len(eps_list) < 2 or any(value <= 0.0 for value in eps_list):
        parser.error("--eps requires at least two positive values")

    print(
        json.dumps(
            run(args.case, args.steps, args.inverse, args.taylor, eps_list),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
