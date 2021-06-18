""" cgal_pybind """
from cgal_pybind.version import VERSION as __version__
# pylint: disable=no-name-in-module
from ._cgal_pybind import (
    slice_volume,
    compute_streamlines_intersections,
    Point_3,
    Polyhedron,
    SurfaceMesh
)
