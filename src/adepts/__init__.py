"""ADEPTS numerical modelling and inversion tools."""

from . import advection, interpolation, mesh, optimization, solvers, stokes, thermal
from .interpolation import interp_between_grids
from .mesh import CartesianMesh

__version__ = "0.1.0"

__all__ = [
    "CartesianMesh",
    "interp_between_grids",
    "advection",
    "mesh",
    "thermal",
    "interpolation",
    "optimization",
    "solvers",
    "stokes",
]
