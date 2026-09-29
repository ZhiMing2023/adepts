import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diffusion = load_script(
            "thermal_diffusion", ROOT / "benchmarks" / "thermal_diffusion.py"
        )
        cls.convection = load_script(
            "isoviscous_convection",
            ROOT / "benchmarks" / "isoviscous_convection.py",
        )

    def test_thermal_diffusion_smoke(self):
        with tempfile.TemporaryDirectory() as output_dir:
            row = self.diffusion.run_diffusion_case(
                resolution=110e3,
                t_end_nd=1e-3,
                nstep=2,
                save_dir=output_dir,
                make_plots=False,
            )
        self.assertTrue(np.isfinite(row["L2_int"]))
        self.assertLess(row["L2_int"], 0.1)

    def test_isoviscous_convection_smoke(self):
        with tempfile.TemporaryDirectory() as output_dir:
            self.convection.SAVE_DIR = output_dir
            self.convection.MAX_STEPS = 2
            self.convection.MIN_STEPS = 0
            self.convection.DIAG_INTERVAL = 1
            self.convection.SAVE_FIG_INTERVAL = 1000
            self.convection.STEADY_COUNT_TARGET = 100
            summary = self.convection.run_one_resolution(8)

        last = summary["last_row"]
        self.assertTrue(np.isfinite(last["Nu_top"]))
        self.assertTrue(np.isfinite(last["Vrms"]))


if __name__ == "__main__":
    unittest.main()
