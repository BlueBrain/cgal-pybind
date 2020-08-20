""" cgal_pybind """
from cgal_pybind.version import VERSION as __version__

try:
    from ._cgal_pybind import Point_3, Polyhedron
    __all__ = ['Point_3', 'Polyhedron']
except ImportError as error_msg:
    print(error_msg)
