#include <array>
#include <cmath>
#include <optional>
#include <vector>
#include <iostream>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/MP_Float.h>
#include <CGAL/Vector_3.h>
#include <CGAL/Point_3.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace flatmap_utils {
  // The computation of streamlines intersections with a surface is a computationally expansive operation
  // performed on arrays of shape (W, H, D) where W, H and D are the integer dimensions of a brain region domain.
  // This operation is one step of the flattening process of a brain region implemented in the python module
  // atlas-building-tools/atlas_building_tools/flatmap. The computation of streamlines intersections is implemented
  // as a python-C++ binding for efficiency reasons.
  //
  // We call streamline a parametrized curve which follows the stream of a vector field, i.e., the curve time derivatives
  // are imposed by the vector field. In our discrete setting, streamlines are polygonal lines obtained by adding up vectors
  // of a discrete 3D vector field.
  using Float = float;
  using Kernel = CGAL::Simple_cartesian<Float>;
  using Vector_3 = Kernel::Vector_3;
  using Point_3 = Kernel::Point_3;

  // Helper classes for geometric computations: Vector_3, Point_3 = Vector_3, Index3 and Transform.

    /**
     * Class to perform natural operations of 3D float vectors.
     *
     * The Vector_3 class implements natural coordinate accessors and the
     * computation of the Euclidean norm.
     *
    */

  Point_3 Floor(const Point_3 &p) {
  /** Return the coordinate-wise floor of a 3D point
   *
   * @return the floored 3D point.
  */
    return Point_3(std::floor(p[0]), std::floor(p[1]), std::floor(p[2]));
  };

  /**
    * Class to manage voxel indices, i.e integer triples defining voxels
  */
  class Index3 {
    private:
      std::array<py::ssize_t, 3> vec_;
    public:
        Index3(): vec_{{0, 0, 0}} {};
        Index3(py::ssize_t x, py::ssize_t y, py::ssize_t z): vec_{{x, y, z}} {};
        Index3(const Index3 &idx): vec_(idx.vec_) {};
        py::ssize_t &operator[](int index){ return vec_[index]; };
        const py::ssize_t &operator[](int index) const { return vec_[index]; };
  };

  Index3 operator+(const Index3 &idx1, const Index3 &idx2) {
  /** Add two 3D indices
   * @param idx1 3D index
   * @param idx2 3D index
   *
   * @return the coordinate-wise sum of idx1 and idx2
  */
    return Index3(idx1[0] + idx2[0], idx1[1] + idx2[1], idx1[2] + idx2[2]);
  };


  /**
    * Class to pass from voxel indices to absolute 3D coordinates and vice-versa.
    *
    * The constructor of the Transform class takes the offset and the voxel dimensions of
    * a voxcell.VoxelData object as arguments. These allow us to define two transformations
    * one from voxel coordinates (integer triples, i.e. Index3) to 3D world coordinates
    * (Point_3) and the other one in the reverse direction.
  */
  class Transform {
    private:
      Vector_3 offset_;
      Vector_3 voxel_dimensions_;
    public:
      Transform(const Vector_3 &offset, const Vector_3 &voxel_dimensions):
        offset_(offset), voxel_dimensions_(voxel_dimensions) {};
      Point_3 PointToContinuousIndex(const Point_3 &p) const {
        const Point_3 ci(p - offset_);
        return Point_3(
          ci[0] / voxel_dimensions_[0],
          ci[1] / voxel_dimensions_[1],
          ci[2] / voxel_dimensions_[2]);
      };
      Point_3 IndexToPhysicalPoint(const Index3 &idx) const {
        return Point_3(
          voxel_dimensions_[0] * idx[0],
          voxel_dimensions_[1] * idx[1],
          voxel_dimensions_[2] * idx[2]) + offset_;
      };
      Index3 PhysicalPointToIndex(const Point_3 &p) const {
        const Point_3 &q = Floor(PointToContinuousIndex(p));
        return Index3(
          static_cast<py::ssize_t>(q[0]),
          static_cast<py::ssize_t>(q[1]),
          static_cast<py::ssize_t>(q[2])
        );
      }
  };
  /**
    * A utility wrapper around an array of type py::array_t<Float, 4> to be interpreted
    * as a 3D vector field.
    *
    * The helper method `GetVector` allows us to get access to a vector of the vector field
    * under the form of a Vector_3 object.
  */
  class Vector3Field {
    private:
      py::detail::unchecked_reference<Float, 4> vector_field_;
    public:
      Vector3Field(py::array_t<Float, 4> field): vector_field_(field.unchecked<4>()) {};
      Vector_3 GetVector(const Index3 &idx) const {
        return Vector_3(
          vector_field_(idx[0], idx[1], idx[2], 0),
          vector_field_(idx[0], idx[1], idx[2], 1),
          vector_field_(idx[0], idx[1], idx[2], 2));
      };
      size_t shape(int index) const { return vector_field_.shape(index); };
  };

  typedef std::array<std::array<Float, 8>, 8> DistanceMatrix;
  DistanceMatrix compute_distance_matrix() {
    static const std::array<Vector_3, 8> vertices = {
        Vector_3(0, 0, 0), Vector_3(1, 0, 0), Vector_3(1, 1, 0), Vector_3(0, 1, 0),
        Vector_3(0, 0, 1), Vector_3(1, 0, 1), Vector_3(1, 1, 1), Vector_3(0, 1, 1)
    };
    DistanceMatrix distance_matrix = {};
    for (int i = 0; i < 8; ++i) {
      for (int j = i + 1; j < 8; ++j) {
        distance_matrix[i][j] = std::sqrt((vertices[i] - vertices[j]).squared_length());
        distance_matrix[j][i] = distance_matrix[i][j];
      }
    }

    return distance_matrix;
  };

} // namespace::flatmap_utils


namespace flatmap {
      /**
      * Class handling the computation of streamlines intersections with the boundary
      * between two layers.
      *
      * The main method is ComputeStreamlinesIntersections. It returns the desired intersection points
      * under the form of an array of type py::array_t<Float, 4>, i.e., a mapping from voxels to 3D points.
      *
      * The input `layers` array is an array of type py::array_t<uint8_t, 3> where each voxel is labeled by a
      * 1-based layer index (uint8_t).
      *
      * The input `vector_field` array is an array of type py::array_t<uint8_t, 4> which assigns to each non-zero
      * voxel of `layers` a 3D unit vector.
      * This vector field is interpolated within voxels to get "higher resolution" streamlines.
      * This interpolation is key to obtain a "smoother" mapping from voxel origins to points on
      * the boundary surface.
      *
      * The InterpolateVector method is a custom implementation of the standard trilinear interpolation,
      * see https://en.wikipedia.org/wiki/Trilinear_interpolation for a mathematical description.
    */
    using namespace flatmap_utils;
    using Segment = std::optional<std::array<Point_3, 2>>;
    class StreamlinesIntersector {
      private:
        Vector3Field vector_field_;
        py::detail::unchecked_reference<uint8_t, 3> layers_;
        Transform transform_;
        uint8_t layer_1_;
        uint8_t layer_2_;
        Index3 region_;

        Segment FindIntersectingSegment(const Index3 &voxel_start_index, bool forward=true) const;
        Segment FindClosestVoxelOrigins(const Index3 &voxel_index) const;
        Point_3 FindClosestPoint(const Index3 &voxel_index) const;
        static Vector_3 TrilinearInterpolation(const Point_3 &origin, const std::array<Vector_3, 8> &vertex_vectors);
        Vector_3 InterpolateVector(const Point_3 &point) const;
        bool IsInsideRegion(const Index3 &idx) const {
          return (idx[0] < region_[0] && idx[1] < region_[1] && idx[2] < region_[2]) &&
            (idx[0] >= 0 && idx[1] >= 0 && idx[2] >= 0);
        };

      public:
        StreamlinesIntersector(
          py::array_t<uint8_t, 3> layers,
          py::array_t<Float, 4> vector_field,
          const Vector_3 &offset,
          const Vector_3 &voxel_dimensions,
          uint8_t layer_1, uint8_t layer_2):
            layers_(layers.unchecked<3>()),
            vector_field_(Vector3Field(vector_field)),
            transform_(Transform(offset, voxel_dimensions)),
            layer_1_(layer_1), layer_2_(layer_2) {
              region_ = Index3(vector_field_.shape(0), vector_field_.shape(1), vector_field_.shape(2));
            };
          bool IsCrossingBoundary(uint8_t origin_layer, uint8_t end_layer) const {
            return (origin_layer == layer_1_ && end_layer == layer_2_) || (origin_layer == layer_2_ && end_layer == layer_1_);
          };
        py::array_t<Float, 4> ComputeStreamlinesIntersections() const;

    };

    Segment StreamlinesIntersector::FindIntersectingSegment(
      const Index3 &voxel_index, bool forward) const
    {
    /**
     * Find the first segment of the streamline passing through the specified voxel which intersects the boundary
     * between `layer_1_` and `layer_2_`.
     *
     * The alogrithm finds the first segment of the polygonal streamline passing through the voxel with index
     * `voxel_index` which crosses the boundary between `layer_1_` and `layer_2_`. It returns the end points of
     * the segment, if such segment exists, else an empty vector.
     *
     * @param voxel_index Index of the voxel whose streamline is to be drawn.
     * @param forward if true, the forward half-streamline is drawn, otherwise the backward half-streamline is drawn.

     * @return an optional std::array<Point_3, 2>. If the desired segment has been found, the array holds the two
     * end points of the segment. Otherwise, the returned object is std::nullopt.
    */
      Index3 previous_index = voxel_index;
      Point_3 previous_point = transform_.IndexToPhysicalPoint(previous_index);
      std::array<Point_3, 2> segment;
      segment[0] = previous_point;

      // We halve the direction vector (unit) length to get a higher streamline resolution
      const Float direction = 0.5 * (forward ? 1.0 : -1.0);

      const Vector_3 &vector = vector_field_.GetVector(previous_index) * direction;
      Point_3 current_point(previous_point + vector);
      Index3 current_index = transform_.PhysicalPointToIndex(current_point);
      if (!IsInsideRegion(current_index)) return std::nullopt;
      segment[1] = current_point;
      uint8_t previous_layer = layers_(previous_index[0], previous_index[1], previous_index[2]);
      uint8_t current_layer = layers_(current_index[0], current_index[1], current_index[2]);

      // Check if the initial segment intersects the boundary
      if (IsCrossingBoundary(previous_layer, current_layer)) return segment;
      while (IsInsideRegion(current_index) && current_layer != 0) {
        const Vector_3 &vector = InterpolateVector(current_point) * direction;
        previous_point = current_point;
        current_point += vector;
        previous_index = current_index;
        current_index = transform_.PhysicalPointToIndex(current_point);
        previous_layer = layers_(previous_index[0], previous_index[1], previous_index[2]);
        current_layer = layers_(current_index[0], current_index[1], current_index[2]);
        // Check if the current segment intersects the boundary
        if (IsCrossingBoundary(previous_layer, current_layer)) {
          segment[0] = previous_point;
          segment[1] = current_point;
          return segment;
        }
      }

      return std::nullopt;
    };

    Segment StreamlinesIntersector::FindClosestVoxelOrigins(const Index3 &voxel_index) const
    /**
     * Find the voxel origins of the streamline passing through the specified voxel which are the closest to
     * the boundary between `layer_1_` and `layer_2_`.
     *
     * The alogrithm finds the first segment of the polygonal streamline passing through the voxel with index
     * `voxel_index` which crosses the boundary between `layer_1_` and `layer_2_`. It returns the end points of
     * the segment, if such segment exists, else std::nullopt.
     *
     * @param voxel_index Index of the voxel whose streamline is to be drawn.

     * @return an optional std::array<Point_3, 2>. If the desired segment has been found, the array holds the two
     * end points of the segment. Otherwise, the returned object is std::nullopt.
    */
    {
      Segment segment = FindIntersectingSegment(voxel_index);
      if (segment == std::nullopt) return FindIntersectingSegment(voxel_index, false);
      return segment;
    }

    Point_3 StreamlinesIntersector::FindClosestPoint(const Index3 &voxel_index) const
    {
    /**
     * Find on the boundary surface between `layer_1_` and `layer_2_` the closest point to the streamline
     * passing through the specified voxel.
     *
     * The alogrithm finds the first segment of the streamline passing through the voxel with index `voxel_index`
     * and which crosses the boundary between `layer_1_` and `layer_2_`. It returns the middle
     * point of this segment if such segment exists, else it returns Vector_3(NAN, NAN, NAN).
     *
     * @param voxel_index Index of the voxel whose streamline is drawn.

     * @return a 3D point approximating the intersection of the streamline passing through the voxel of index
     * `voxel_index` with the boundary between `layer_1_` and `layer_2_`.
    */
      const Segment &segment = FindClosestVoxelOrigins(voxel_index);
      Point_3 middle_point(NAN, NAN, NAN);

      if (segment != std::nullopt) {
        const std::array<Point_3, 2> &s = segment.value();
        middle_point = Point_3(
          (s[0][0] + s[1][0]) / 2.0,
          (s[0][1] + s[1][1]) / 2.0,
          (s[0][2] + s[1][2]) / 2.0
        );
      }

      return middle_point;
    }

    Vector_3 StreamlinesIntersector::TrilinearInterpolation(
      const Point_3 &origin,
      const std::array<Vector_3, 8> &vertex_vectors) {
      // Trilinear interpolation, see https://en.wikipedia.org/wiki/Trilinear_interpolation
      const Point_3 floored_origin = Floor(origin);
      const Vector_3 weights = origin - floored_origin;

      const Vector_3 v01 = vertex_vectors[0] * (1.0 - weights[0]) + vertex_vectors[1] * weights[0];
      const Vector_3 v32 = vertex_vectors[3] * (1.0 - weights[0]) + vertex_vectors[2] * weights[0];
      const Vector_3 v0 = v01 * (1.0 - weights[1]) + v32 * weights[1];

      const Vector_3 v45 = vertex_vectors[4] * (1.0 - weights[0]) + vertex_vectors[5] * weights[0];
      const Vector_3 v76 = vertex_vectors[7] * (1.0 - weights[0]) + vertex_vectors[6] * weights[0];
      const Vector_3 v1 = v45 * (1.0 - weights[1]) + v76 * weights[1];

      return v0 * (1.0 - weights[2]) + v1 * weights[2];
    }

    Vector_3 StreamlinesIntersector::InterpolateVector(const Point_3 &point) const
    /**
     * Interpolate `vector_field_` within voxels.
     *
     * This function implements the standard trilinear interpolation algorithm
     * (see the mathematical description here https://en.wikipedia.org/wiki/Trilinear_interpolation)
     * Interpolation is used to draw streamlines at sub-voxel scale.
     *
     * The vector field `vector_field_` is interpolated at `point` using a weighted average its vectors
     * evaluated on the vertices of the voxel containing `point`. The origin of a voxel is defined as the
     * voxel vertex with the lowest indices.
     *
     * @param point a 3D point with absolute coordinates (offset and voxel dimensions are thus taken into account).

     * @return a 3D vector interpolating `vector_field_` at `point` by means of trilinear interpolation.
    */
    {
      static const std::array<Index3, 8> index_offsets = {
        Index3({0, 0, 0}), Index3({1, 0, 0}), Index3({1, 1, 0}), Index3({0, 1, 0}),
        Index3({0, 0, 1}), Index3({1, 0, 1}), Index3({1, 1, 1}), Index3({0, 1, 1})
      };
      // Matrix holding the distances between any two vertices of the cube [0, 1]^3.
      static const DistanceMatrix distance_matrix = compute_distance_matrix();

      const Point_3 &origin = transform_.PointToContinuousIndex(point);
      Index3 origin_index(std::floor(origin[0]), std::floor(origin[1]), std::floor(origin[2]));
      std::array<Vector_3, 8> vertex_vectors{};
      vertex_vectors[0] = vector_field_.GetVector(origin_index);

      // Assign to each vertex of the voxel a vector from the vector field
      for (int i = 1; i < 8; ++i) {
        const Index3 &index = origin_index + index_offsets[i];
        if (IsInsideRegion(index)) vertex_vectors[i] = vector_field_.GetVector(index);
        else vertex_vectors[i] = Vector_3(NAN, NAN, NAN);
      }

      // Voxel vertices with an invalid or a missing vector are assigned a weighted average of
      // the valid vectors of their neighbours.
      for (int i = 1; i < 8; ++i) {
        if (std::isnan(vertex_vectors[i][0])) {
          Vector_3 v(0.0, 0.0, 0.0);
          Float total_weight = 0;
          for (int j = 0; j < 8; ++j) {
            if (j != i && !std::isnan(vertex_vectors[j][0])) {
              const Float d = 1.0 / distance_matrix[i][j];
              v += vertex_vectors[j] * d;
              total_weight += d;
            }
          }
          vertex_vectors[i] = v / total_weight;
        }
      }

      return TrilinearInterpolation(origin, vertex_vectors);
    }


    py::array_t<Float, 4> StreamlinesIntersector::ComputeStreamlinesIntersections() const {
    /**
       * Compute the streamlines intersection points with the surface separating two layers.
       *
       * Main method of the class StreamlinesIntersector.
       *
       * The algorithm draws for each voxel in `layers` a polygonal line passing through it. This polygonal line follows
       * the stream of `vector_field_`; we call it a streamline.
       * The first segment of the streamline L which crosses the boundary surface S between `layer_1_` and `layer_2_` is used
       * to approximate the intersection point of L with S. This intersection point is approximated by the middle point
       * of the latter segment.
       * The overall process maps every voxel in `layers_` to the intersection point of the streamline passing through it.
       * The intersection point is set to Vector_3(NAN, NAN, NAN) if no segment of the streamline crosses the boundary surface.
       *
       * @return py::array_t<flatmap::Float, 4> to be read as numpy array of shape (W, H, D, 3). This array holds for each voxel
       * V of `layers_` to the intersection point of the streamline passing through V with the boundary surface between `layer_1_`
       * and `layer_2_`.
      */
      const std::vector<py::ssize_t> dims{{layers_.shape(0), layers_.shape(1), layers_.shape(2), 3}};
      py::array_t<Float> voxel_to_point_map(dims);
      auto voxel_to_point_map_ = voxel_to_point_map.mutable_unchecked<4>();

      for (py::ssize_t i = 0; i < layers_.shape(0); ++i) {
        for (py::ssize_t j = 0; j < layers_.shape(1); ++j) {
            for (py::ssize_t k = 0; k < layers_.shape(2); ++k) {
              for (int l = 0; l < 3; ++l) voxel_to_point_map_(i, j, k, l) = NAN;
              if (layers_(i, j, k) != 0) {
                const Index3 voxel_index(i, j, k);
                const Point_3 &point = FindClosestPoint(voxel_index);
                for (int l = 0; l < 3; ++l) voxel_to_point_map_(i, j, k, l) = point[l];
              }
            }
        }
      }

      return voxel_to_point_map;
    }

} // namespace::flatmap


py::array_t<flatmap::Float, 4> compute_streamlines_intersections(
  py::array_t<uint8_t, 3> layers,
  py::array_t<flatmap::Float, 4> vector_field,
  py::array_t<flatmap::Float, 1> offset,
  py::array_t<flatmap::Float, 1> voxel_dimensions,
  uint8_t layer_1, uint8_t layer_2) {
  /**
     * Compute the streamlines intersection points with the surface separating two layers.
     *
     * The algorithm draws for each voxel in `layers` a polygonal line passing through it. This polygonal line follows
     * the stream of `vector_field`; we call it a streamline.
     * The first segment of the streamline L which crosses the boundary surface S between `layer_1` and `layer_2` is used
     * to approximate the intersection point between L and S. The intersection point is approximated by the middle point
     * of this segment.
     * This process maps every voxel in `layers` to the intersection point of its streamline.
     * The intersection point is set to Vector_3(NAN, NAN, NAN) if no segment of the streamline crosses the boundary surface.
     * @param layers uint8_t array of shape (W, H, D) where W, H and D are the 3 dimensions of the array.
     *  The value of a voxel is the index of the layer it belongs to.
     * @param vector_field float array of shape (W, H, D, 3) where (W, H, D) is the shape of `layers`.
     *  3D Vector field defined over `layers` domain used to draw the streamlines.
     * @param offset float array of shape (3,) holding the offset of the region defined by `layers`.
     *  This is used to compute point coordinates in the absolute (world) 3D frame.
     * @param voxel_dimensions float array of shape (3,) holding dimensions of voxels of the volume `layers`.
     *  This is used to compute point coordinates in the absolute (world) 3D frame.
     * @param layer_1 (non-zero) layer index of the first layer.
     * @param layer_2 (non-zero) layer index of the second layer. The layers with index `layer_1` and `layer_2` defines a
     * (voxellized) boundary surface to be intersected with the streamlines of `vector_field`.
     * Note that `offset and `voxel_dimensions` are the attributes of the voxcell.VoxelData object corresponding to
     * both `layers` and `vector_field`.
     * @return py::array_t<flatmap::Float, 4> to be read as numpy array of shape (W, H, D, 3). This array holds for each voxel
     * V of `layers` the intersection point of the streamline passing through V with the boundary surface between `layer_1` and
     * `layer_2`.
    */

  auto offset_ = offset.unchecked<1>();
  auto voxel_dimensions_ = voxel_dimensions.unchecked<1>();
  flatmap::StreamlinesIntersector intersector(
    layers, vector_field,
    flatmap_utils::Vector_3(offset_(0), offset_(1), offset_(2)),
    flatmap_utils::Vector_3(voxel_dimensions_(0), voxel_dimensions_(1), voxel_dimensions_(2)),
    layer_1, layer_2);

  return intersector.ComputeStreamlinesIntersections();
}


void bind_streamlines_intersections(py::module& m)
{
     /**
     * Compute the intersection points of streamlines with a choosen layer boundary.
     */

    m.def("compute_streamlines_intersections", &compute_streamlines_intersections,
      "A function which computes the intersection points of streamlines with a choosen layer boundary",
      py::arg("layers"), // numpy array of shape (W, H, D) and dtype np.uint8
      py::arg("vector_field"), // numpy array of shape (W, H, D, 3) and dtype np.float32
      py::arg("offset"), // numpy array of shape (3,) and dtype np.float32
      py::arg("voxel_dimensions"), // numpy array of shape (3,) and dtype np.float32,
      py::arg("layer_1"), // scalar of type np.uint8
      py::arg("layer_2") // scalar of type np.uint8
    );
}