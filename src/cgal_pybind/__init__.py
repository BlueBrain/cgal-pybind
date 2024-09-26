""" cgal_pybind """

from importlib.metadata import version

__version__ = version(__package__)

# pylint: disable=no-name-in-module
from ._cgal_pybind import (
    InvalidVectorError,
    Point_3,
    Polyhedron,
    SurfaceMesh,
    compute_streamlines_intersections,
    estimate_thicknesses,
    slice_volume,
)
