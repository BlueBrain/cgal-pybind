#include <exception> // std::runtime_error
#include <utility> // std::pair
#include <vector>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

// { for authalic
#include <CGAL/Surface_mesh_parameterization/Discrete_authalic_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Square_border_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Error_code.h>
#include <CGAL/Surface_mesh_parameterization/parameterize.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
// }

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_2 = Kernel::Point_2;
using Point_3 = Kernel::Point_3;

using SurfaceMesh = CGAL::Surface_mesh<Point_3>;

namespace {

    // From https://doc.cgal.org/latest/Surface_mesh_parameterization/index.html#title8:
    // The Discrete Authalic parameterization method has been introduced by Desbrun et al. [2].
    // It corresponds to a weak formulation of an area-preserving method, and in essence locally
    // minimizes the area distortion. A one-to-one mapping is guaranteed only if the convex combination
    // condition is fulfilled and the border is convex. This method solves two #vertices x #vertices
    // sparse linear systems. The matrix (the same for both systems) is asymmetric.

    // [2] Mathieu Desbrun, Mark Meyer, and Pierre Alliez. "Intrinsic parameterizations of surface meshes",
    // Computer Graphics Forum, 21(3):209–218, September 2002.
    // http://www.geometry.caltech.edu/pubs/DMA02.pdf
    std::vector<std::pair<double, double>>
    run_discrete_authalic(SurfaceMesh &sm)
    {
        namespace SMP = CGAL::Surface_mesh_parameterization;

        using BorderParameterizer = SMP::Square_border_arc_length_parameterizer_3<SurfaceMesh>;
        using Parameterizer = SMP::Discrete_authalic_parameterizer_3<SurfaceMesh, BorderParameterizer>;
        using halfedge_descriptor = boost::graph_traits<SurfaceMesh>::halfedge_descriptor;

        auto [bhd, length] = CGAL::Polygon_mesh_processing::longest_border(sm);

        // The input SurfaceMesh is assumed to have at least one circular boundary,
        // resembling to a not-too-distorted square.
        // Ideally, the input mesh has the topology of a disk.
        if(length == 0.) {
            throw std::runtime_error("The length of the longest border is 0.");
        }

        using vertex_descriptor = boost::graph_traits<SurfaceMesh>::vertex_descriptor;
        using UV_pmap = SurfaceMesh::Property_map<vertex_descriptor, Point_2>;
        UV_pmap uv_map = sm.add_property_map<vertex_descriptor, Point_2>("v:uv").first;

        SMP::Error_code err = SMP::parameterize(sm, Parameterizer(), bhd, uv_map);
        if(err != SMP::OK) {
            std::cerr << "Error: " << SMP::get_error_message(err) << std::endl;
            throw std::runtime_error("Error parameterize-izing");
        }

        // Make sure the ordering of the output vertices matches
        // the ordering the input SurfaceMesh vertices:
        // https://github.com/CGAL/cgal/issues/2994
        using Vertex_index_map = boost::unordered_map<vertex_descriptor, std::size_t>;
        Vertex_index_map vium;
        boost::associative_property_map<Vertex_index_map> vimap(vium);

        std::vector<std::pair<double, double>> return_vertices;
        size_t vertices_counter = 0;
        for (const auto& v : CGAL::make_range(vertices(sm)))
        {
            auto& p = get(uv_map, v);
            return_vertices.emplace_back(p.x(), p.y());
            put(vimap, v, vertices_counter++);
        }
        return return_vertices;
    }
} // anonymous namespace


void bind_surface_mesh(py::module& m)
{
    using IntIndices = std::vector<std::tuple<int, int, int>>;
     /**
     * Create SurfaceMesh class and its methods
     *
     * @param surface_mesh SurfaceMesh.
     */

    py::class_<SurfaceMesh>(m, "SurfaceMesh",
        "CGAL CGAL::Surface_mesh<Point_3> binding")
        .def(py::init())
        .def("add_vertices",
             [](SurfaceMesh& self, const std::vector<Point_3>& points) {
                 for(const auto& p : points) {
                     self.add_vertex(p);
                 }
             })
        .def("add_faces",
             [](SurfaceMesh& self, const IntIndices& indices) {
                for(auto [v0, v1, v2] : indices) {
                     self.add_face(static_cast<SurfaceMesh::Vertex_index>(v0),
                                   static_cast<SurfaceMesh::Vertex_index>(v1),
                                   static_cast<SurfaceMesh::Vertex_index>(v2));
                 }
             })
        .def("number_of_vertices", &SurfaceMesh::number_of_vertices, "number of vertices")
        .def("number_of_faces", &SurfaceMesh::number_of_faces, "number of faces")
        .def("area", [](SurfaceMesh& self) {
             return CGAL::Polygon_mesh_processing::area(self);
        }, "mesh surface area")
        .def("authalic", [](SurfaceMesh& self) {
            return py::make_tuple(run_discrete_authalic(self));
        }, "Flatten a 3D surface mesh while minimizing the area distortion locally");
}