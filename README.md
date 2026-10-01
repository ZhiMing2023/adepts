# ADEPTS

**An auto-differentiable framework for time-dependent nonlinear
thermo-chemical mantle convection inversion**

ADEPTS is the research code accompanying the preprint:
"ADEPTS: An auto-differentiable framework for time-dependent nonlinear
thermo-chemical mantle convection inversion" (arXiv:2609.13482). It combines
staggered-grid finite differences with automatic and implicit differentiation
for forward and inverse mantle-convection problems.
For now, it is implemented with a two-dimensional cartesian geometry, suitable
for regional-scale Earth and planetary tectonic problems.
This repository contains only source code, benchmark definitions, and case
scripts. Observations and other numerical outputs are generated locally and
are not distributed in the repository.

## Key features

- **Thermo-chemical mantle dynamics:** staggered-grid Stokes and thermal
  operators with semi-Lagrangian scalar advection.
- **Differentiable physics:** differentiable sparse solves and implicit
  gradients for nonlinear Stokes systems.
- **Inverse modelling:** gradient-based inversion of initial temperature and
  material parameters.
- **Verification:** Taylor tests, analytical benchmarks, and physical benchmark
  problems.
- **Reproducible examples:** six forward/inverse cases used in the manuscript.

## Repository layout

```text
src/adepts/       Python package
cases/            forward, Taylor-test, and inversion cases
benchmarks/       analytical and physical benchmarks
postprocess/      cross-case plotting scripts
tests/            short regression tests
```

## Environment

The reference runs use Linux under WSL, Python 3.10, CPU tensors, and
PETSc/MUMPS. Validation used PyTorch 2.9.0, NumPy 2.0.1, SciPy 1.15.3,
Matplotlib 3.10.6, and PETSc/petsc4py 3.21.4.

Create the portable environment. The environment file installs ADEPTS in
editable mode:

```bash
conda env create -f environment.yml
conda activate adepts
```

If Conda reports a Terms of Service error for an Anaconda `defaults` channel
before solving this environment, use the repository's conda-forge-only
configuration for the creation command:

```bash
CONDARC="$PWD/condarc-forge.yml" conda env create -f environment.yml
conda activate adepts
```

The environment file installs PyTorch 2.9.1 with the CPU build from
conda-forge. Its pip step only installs ADEPTS in editable mode. Verify the
installed version with `python -c "import torch; print(torch.__version__)"`.

PETSc is optional at import time. CPU solves fall back to SciPy when PETSc is
unavailable, but the archived reference validation uses PETSc/MUMPS.

## Short validation

Run the dependency-light regression layer before starting long cases:

```bash
python -m unittest discover -s tests -v
```

It checks mesh layout, staggered interpolation, a differentiable sparse solve,
and short thermal/advection-diffusion runs.

Run a disposable one-step case check without retaining generated data:

```bash
python tools/validate_case.py case01_lithospheric_drip --steps 1 --inverse --taylor
```

Repeat with the other case directory names. Each invocation runs in an
isolated temporary directory. Omit `--steps` to use the manuscript setting.

## Validation record

The cleaned tree was validated on 8 September 2026 in the reference
environment above. All validators used temporary output directories.

| Check | Configuration | Result |
| --- | --- | --- |
| Unit tests | 6 tests | passed |
| Thermal diffusion | 2 km; 10, 20, 40, 80 steps | L2 0.9905; Linf 0.9905 |
| Isoviscous convection | N=8; 2-step smoke test | passed |
| Slab detachment | 20 km; 1-step smoke test | passed |
| Slab convergence | 20 km; 1-step smoke test | passed |

The analytical entries report fitted temporal convergence orders. The
physical smoke tests check finite fields and successful nonlinear solves;
they are not substitutes for the long manuscript runs.

Each case was also run through its full default forward trajectory, one
L-BFGS update with line search, and a three-point Taylor test:

| Case | Forward steps | Best loss after the validation update | Taylor R1 order |
| --- | ---: | ---: | --- |
| 01 | 50 | 0.457927 | T0: 2.0084 |
| 02 | 30 | 0.762047 | T0: 1.9994 |
| 03 | 30 | 0.760622 | T0: 1.9992 |
| 04 | 30 | 0.768905 | T0: 2.0227 |
| 05 | 30 | 0.760625 | T0: 1.9980 |
| 06 | 30 | 1.061760 | T0: 2.0000; rho2: 1.9554; A: 1.9824; n: 1.9940 |

This inversion check verifies the complete forward/adjoint/optimizer path; it
does not claim convergence of the production inversion after one update.

## Benchmarks

Run benchmark scripts from the repository root:

```bash
python benchmarks/thermal_diffusion.py
python benchmarks/isoviscous_convection.py
python benchmarks/slab_detachment.py
RES_KM=10 python benchmarks/slab_detachment_convergence.py
RES_KM=5  python benchmarks/slab_detachment_convergence.py
```

Analytical benchmarks can also be checked without retaining output files:

```bash
python tools/validate_benchmark.py thermal
```

Physical benchmarks can be run at one resolution in a disposable directory:

```bash
python tools/validate_physical_benchmark.py convection --resolution 8 --quick
python tools/validate_physical_benchmark.py slab --resolution-km 20 --quick
python tools/validate_physical_benchmark.py slab-convergence --resolution-km 20 --quick
```

For manuscript-scale disposable runs, omit `--quick`, for example:

```bash
python tools/validate_physical_benchmark.py convection --resolution 48
python tools/validate_physical_benchmark.py slab --resolution-km 10
python tools/validate_physical_benchmark.py slab-convergence --resolution-km 5
```

The isoviscous-convection and slab-detachment benchmarks are long production
runs. Their default parameters are the manuscript parameters rather than CI
settings.

## Cases

Each case is self-contained. Run it from its own directory so generated files
remain beside that case. The required order is `forward`, `taylor`, then
`invert`; the forward run creates `observe_data.pt` used by the other modes.

```bash
cd cases/case01_lithospheric_drip
python run.py forward
python run.py taylor
python run.py invert
```

The manuscript cases are:

| Case | Directory | Purpose |
| --- | --- | --- |
| 01 | `case01_lithospheric_drip` | Lithospheric-drip inversion |
| 02 | `case02_picard_5` | Picard differentiation with 5 iterations |
| 03 | `case03_picard_100` | Picard differentiation with 100 iterations |
| 04 | `case04_implicit_tol_1e-3` | Implicit differentiation, tolerance `1e-3` |
| 05 | `case05_implicit_tol_1e-8` | Implicit differentiation, tolerance `1e-8` |
| 06 | `case06_joint_inversion` | Joint inversion of temperature and material parameters |

Use the same `forward`, `taylor`, and `invert` commands in each directory.

`python run.py forward_pre_N` reconstructs and forwards the saved L-BFGS
iterate `N` after an inversion. Case 06 applies Taylor tests to every inverted
parameter: initial temperature, density, rheological prefactor, and exponent.

## Generated files

Case and benchmark scripts write tensors, tables, and figures locally; solver
messages are written to the console. Generated artifacts are excluded by
`.gitignore`. No deleted result file is required as an input.

## Reproducibility notes

- Calculations use `torch.float64` and CPU sparse solves.
- Solver tolerances and case parameters are defined in the corresponding
  scripts.
- Numerical equivalence should be assessed with the reported tolerances, not
  by bitwise comparison across PETSc, BLAS, or operating-system versions.
- Long physical benchmarks should be recorded in the archived release
  accompanying the manuscript.
- Minor discrepancies from the published results may occur due to the
  non-deterministic nature of MUMPS direct solves and thread scheduling.

## Citation

If you use ADEPTS, please cite the preprint (and the software release once
archived). Citation metadata is also provided in `CITATION.cff`.

```bibtex
@misc{MingHu2026ADEPTS,
  author        = {Ming, Zhiying and Hu, Jiashun},
  title         = {ADEPTS: An auto-differentiable framework for time-dependent
                   nonlinear thermo-chemical mantle convection inversion},
  year          = {2026},
  eprint        = {2609.13482},
  archivePrefix = {arXiv},
  primaryClass  = {physics.geo-ph},
  doi           = {10.48550/arXiv.2609.13482},
  url           = {https://arxiv.org/abs/2609.13482}
}
```

## Contact

- Zhiying Ming, Southern University of Science and Technology 
- Jiashun Hu (corresponding author), Southern University of Science and
  Technology: `hujs@sustech.edu.cn`

## Acknowledgements

The staggered-grid discretization follows the formulation in Gerya (2019); see
the paper for the full method references.

## License

ADEPTS is licensed under `GPL-3.0-only`; see `LICENSE`. Reused code attribution
is recorded in `THIRD_PARTY_NOTICES.md`.
