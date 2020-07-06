""" cgal_pybind """
from cgal_pybind.version import VERSION as __version__

try:
    from ._cgal_pybind import Point_3, Polyhedron
    __all__ = ['Point_3', 'Polyhedron']
except ImportError:
    print('Cannot import C++ binding stuff')
