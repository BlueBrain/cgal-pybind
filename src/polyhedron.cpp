#include <cstddef> // std::size_t
#include <string>
#include <utility> // std::pair
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/mesh_segmentation.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using std::size_t;

using IntIndices = std::vector<std::tuple<int, int, int>>;

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = CGAL::Point_3<Kernel>;
using Polyhedron = CGAL::Polyhedron_3<Kernel, CGAL::Polyhedron_items_with_id_3>;


// Property map associating a facet with an integer as id to an
// element in a vector stored internally
template<class ValueType>
struct Facet_with_id_pmap
    : public boost::put_get_helper<ValueType&, Facet_with_id_pmap<ValueType>>
{
    using Key = CGAL::Polyhedron_3<
        CGAL::Exact_predicates_inexact_constructions_kernel,
        CGAL::Polyhedron_items_with_id_3
    >::Facet_const_handle;

    Facet_with_id_pmap(std::vector<ValueType>& internal_vector)
        : internal_vector(internal_vector)
    {}

    ValueType& operator[](Key key) const { return internal_vector[key->id()]; }

private:
    std::vector<ValueType>& internal_vector;
};

// A modifier creating a Polyhedron from vectors of vertices and faces (vertex Ids).
template<class HDS>
class polyhedron_builder
    : public CGAL::Modifier_base<HDS>
{
public:
    polyhedron_builder(const std::vector<Point_3>& _vertices, const IntIndices& _tris)
        : vertices(_vertices)
        , tris(_tris)
    {}

    void operator()(HDS& hds) {
        CGAL::Polyhedron_incremental_builder_3<HDS> B(hds, true);

        B.begin_surface(vertices.size(), tris.size());

        // add the polyhedron vertices
        for(const auto& p : vertices) {
            B.add_vertex(typename HDS::Vertex::Point(p));
        }

        // add the polyhedron triangles
        for(auto [v0, v1, v2] : tris) {
            B.begin_facet();
            B.add_vertex_to_facet(v0);
            B.add_vertex_to_facet(v1);
            B.add_vertex_to_facet(v2);
            B.end_facet();
        }

        B.end_surface();
    }

private:
    const std::vector<Point_3>& vertices;
    const IntIndices &tris;
};

py::tuple contract(Polyhedron& polyhedron) {
    /**
     * Convert an interactively contracted skeleton to a skeleton curve
     *
     * @param polyhedron Polyhedron reference.
     * @return py::tuple (viewed as a Python list) containing two vectors and a map:
     * return_vertices: vector containing skeleton curve vertices described as 3D points
     * return_edges: vector containing skeleton curve edges
     * return_correspondence: map containing skeleton mapping between vertex ids and vector of surface ids.
     */

    using Skeletonization = CGAL::Mean_curvature_flow_skeletonization<Polyhedron>;

    Skeletonization mean_curve_skeletonizer(polyhedron);
    mean_curve_skeletonizer.contract_until_convergence();

    Skeletonization::Skeleton skeleton;
    mean_curve_skeletonizer.convert_to_skeleton(skeleton);

    std::vector<std::tuple<double, double, double>> return_vertices;
    for (const auto& v : CGAL::make_range(vertices(skeleton)))
    {
        return_vertices.emplace_back(skeleton[v].point[0],
                                     skeleton[v].point[1],
                                     skeleton[v].point[2]
                                    );
    }

    std::vector<std::pair<int, int>> return_edges;
    for (const auto& e : CGAL::make_range(edges(skeleton)))
    {
        return_edges.emplace_back(source(e, skeleton), target(e, skeleton));
    }

    size_t index = 0;
    for(auto& v : CGAL::make_range(vertices(polyhedron)))
    {
        v->id() = index++;
    }

    // Output skeleton points and the corresponding surface points
    std::map<size_t, std::vector<size_t>> return_correspondence;
    for (const auto& v : CGAL::make_range(vertices(skeleton)))
    {
        std::vector<size_t> vector_surface_id;
        for(const auto& vd : skeleton[v].vertices ){
            vector_surface_id.push_back(get(boost::vertex_index, polyhedron, vd));
        }
        return_correspondence.insert({v, vector_surface_id});
    }

    return py::make_tuple(return_vertices, return_edges, return_correspondence);
}

/*---------------------------------------------------*/
py::tuple segmentation(Polyhedron& polyhedron) {
    /**
     * assign a unique id to each facet in the mesh.
     * computes property-map for SDF values.
     * computes property-map for segment-ids.
     * @param polyhedron Polyhedron reference.
     * @return py::tuple (viewed as a Python list) containing two vectors:
     *     return_sdf_property_map: vector containing Shape Diameter Function property map
     *     return_segment_property_map: vector containing segment property map
     */
    // assign id field for each facet
    size_t facet_id = 0;
    for (auto facet = polyhedron.facets_begin(); facet != polyhedron.facets_end(); ++facet) {
        facet->id() = facet_id++;
    }

    /* The Shape Diameter Function provides a connection between the surface mesh and the
    * volume of the subtended 3D bounded object. More specifically, the SDF is a scalar-valued function defined on
    * facets of the mesh that measures the corresponding local object diameter.
    */

    // Create a property-map for signed distance function values
    std::vector<double> sdf_values_vec(polyhedron.size_of_facets());
    Facet_with_id_pmap<double> sdf_property_map(sdf_values_vec);

    // Computing the Shape Diameter Function over a surface mesh into sdf_property_map .
    CGAL::sdf_values(polyhedron, sdf_property_map);

    // Fill return_sdf_property_map
    std::vector<std::tuple<double>> return_sdf_property_map;
    return_sdf_property_map.reserve(polyhedron.size_of_facets());
    for(auto facet = polyhedron.facets_begin(); facet != polyhedron.facets_end(); ++facet){
        return_sdf_property_map.emplace_back(sdf_property_map[facet]);
     }

    // Create a property-map for segment-ids
    std::vector<size_t> segment_ids(polyhedron.size_of_facets());
    Facet_with_id_pmap<size_t> segment_property_map(segment_ids);

    // Computes the segmentation of a surface mesh given an SDF value per facet.
    /* CGAL::segmentation_from_sdf_values function computes the segmentation of a surface mesh given an SDF value per facet.
    * This function fills a property map which associates a segment-id (in [0, number of segments -1])
    * or a cluster-id (in [0, number_of_clusters -1]) to each facet. A segment is a set of connected facets
    * which are placed under the same cluster.
    (See Figure 66.5, https://doc.cgal.org/latest/Surface_mesh_segmentation/index.html#fig__Cluster_vs_segment.)
    */
    CGAL::segmentation_from_sdf_values(polyhedron, sdf_property_map, segment_property_map);

    // Fill return_segment_property_map
    std::vector<std::tuple<size_t>> return_segment_property_map;
    return_segment_property_map.reserve(polyhedron.size_of_facets());

    for(auto facet = polyhedron.facets_begin(); facet != polyhedron.facets_end(); ++facet){
        return_segment_property_map.emplace_back(segment_property_map[facet]);
    }

    return py::make_tuple(return_sdf_property_map, return_segment_property_map);
}


/*---------------------------------------------------*/
void bind_triangle_mesh(py::module& m)
    {
    using HalfedgeDS=Polyhedron::HalfedgeDS;
    /**
     * Create Polyhedron class and its methods
     *
     * @param polyhedron Polyhedron.
     */
    py::class_<Polyhedron>(m, "Polyhedron",
    "CGAL CGAL::Polyhedron_3<Exact_predicates_inexact_constructions_kernel, Polyhedron_items_with_id_3> binding")
        .def(py::init())
        .def("size_of_vertices", &Polyhedron::size_of_vertices, "number of vertices")
        .def("size_of_facets", &Polyhedron::size_of_facets, "number of facet")
        .def("area", [](Polyhedron& self) {
             return CGAL::Polygon_mesh_processing::area(self);
        }, "polyhedron area")
        .def("load_from_off",[](Polyhedron& self, const std::string& full_path_name) {
            std::ifstream input(full_path_name);
            input >> self;
        })
        .def("contract", &contract, "Convert an interactively contracted skeleton to a skeleton curve")
        .def("segmentation",&segmentation, "Assign a unique id to each facet in the mesh")
        .def("build",[](Polyhedron& self, const std::vector<Point_3>& vertices, const IntIndices& indices) {

            polyhedron_builder<HalfedgeDS> builder(vertices, indices);
            self.delegate(builder);
        })
    ;
}
