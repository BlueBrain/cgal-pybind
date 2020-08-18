#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_point(py::module&);
void bind_triangle_mesh(py::module&);
void bind_surface_mesh(py::module&);


PYBIND11_MODULE(_cgal_pybind, m)
{
    m.doc() = "python binding for CGAL";
    bind_point(m);
    bind_triangle_mesh(m);
    bind_surface_mesh(m);
}
