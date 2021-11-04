#include <cmath>
#include <optional>
#include <vector>
#include <string>

#include <assert.h>

#include "utils.hpp"

// The computation of streamlines intersections with a surface is a computationally expensive operation
// performed on arrays of shape (W, H, D) where W, H and D are the integer dimensions of a brain region domain.
// This operation is one step in the flattening process of a brain region implemented in the python module
// atlas-building-tools/atlas_building_tools/flatmap. The computation of streamlines intersections is implemented
// as a python-C++ binding for efficiency reasons.
//
// We call streamline a parametrized curve which follows the stream of a vector field, i.e., the curve time derivatives
// are imposed by the vector field. In our discrete setting, streamlines are polygonal lines obtained by adding up vectors
// of a discrete 3D vector field.
//
// For the computation of streamlines, we assume that the input vector field contains no NaN values and no zero
// vectors inside the region of interest.


namespace py = pybind11;

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
    using namespace utils;
    using Segment = std::optional<std::array<Point_3, 2>>;
    class StreamlinesIntersector {
      private:
        VectorInterpolator interpolator_;
        py::array_t<uint8_t, 3> layers_;
        uint8_t layer_1_;
        uint8_t layer_2_;

        Segment FindIntersectingSegment(const Index_3 &voxel_start_index, bool forward=true) const;
        Segment FindClosestVoxelOrigins(const Index_3 &voxel_index) const;
        Point_3 FindClosestPoint(const Index_3 &voxel_index) const;

      public:
        StreamlinesIntersector(
          py::array_t<uint8_t, 3> layers,
          const Vector_3 &offset,
          const Vector_3 &voxel_dimensions,
          py::array_t<Float, 4> vector_field,
          uint8_t layer_1, uint8_t layer_2):
            layers_(layers),
            interpolator_(vector_field, offset, voxel_dimensions),
            layer_1_(layer_1), layer_2_(layer_2) {};
          bool IsCrossingBoundary(uint8_t origin_layer, uint8_t end_layer) const {
            return (origin_layer == layer_1_ && end_layer == layer_2_) || (origin_layer == layer_2_ && end_layer == layer_1_);
          };
        py::array_t<Float, 4> ComputeStreamlinesIntersections() const;

    };

    Segment StreamlinesIntersector::FindIntersectingSegment(
      const Index_3 &voxel_index, bool forward) const
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
      auto layers = layers_.unchecked<3>();
      Index_3 previous_index = voxel_index;
      Point_3 previous_point = interpolator_.IndexToPhysicalPoint(previous_index);
      std::array<Point_3, 2> segment;
      segment[0] = previous_point;

      // We halve the direction vector (unit) length to get a higher streamline resolution
      const Float direction = 0.5 * (forward ? 1.0 : -1.0);

      const Vector_3 &vector = interpolator_.GetVector(previous_index) * direction;
      // NAN and zero vectors aren't supported.
      assert(!std::isnan(vector[0]) && !std::isnan(vector[1]) && !std::isnan(vector[2]));
      assert(vector[0] != 0.0 || vector[1] != 0.0 || vector[2] != 0.0);
      Point_3 current_point(previous_point + vector);
      Index_3 current_index = interpolator_.PhysicalPointToIndex(current_point);
      if (!interpolator_.IsInsideRegion(current_index)) return std::nullopt;
      segment[1] = current_point;
      uint8_t previous_layer = layers(previous_index[0], previous_index[1], previous_index[2]);
      uint8_t current_layer = layers(current_index[0], current_index[1], current_index[2]);

      // Check if the initial segment intersects the boundary
      if (IsCrossingBoundary(previous_layer, current_layer)) return segment;
      while (interpolator_.IsInsideRegion(current_index) && current_layer != 0) {
        const Vector_3 &vector = interpolator_.InterpolateVector(current_point) * direction;
        previous_point = current_point;
        current_point += vector;
        previous_index = current_index;
        current_index = interpolator_.PhysicalPointToIndex(current_point);
        previous_layer = layers(previous_index[0], previous_index[1], previous_index[2]);
        current_layer = layers(current_index[0], current_index[1], current_index[2]);
        // Check if the current segment intersects the boundary
        if (IsCrossingBoundary(previous_layer, current_layer)) {
          segment[0] = previous_point;
          segment[1] = current_point;
          return segment;
        }
      }

      return std::nullopt;
    };

    Segment StreamlinesIntersector::FindClosestVoxelOrigins(const Index_3 &voxel_index) const
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

    Point_3 StreamlinesIntersector::FindClosestPoint(const Index_3 &voxel_index) const
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

    py::array_t<Float, 4> StreamlinesIntersector::ComputeStreamlinesIntersections() const {
    /**
       * Compute the streamlines intersection points with the surface separating two layers.
       *
       * Main method of the class StreamlinesIntersector.
       *
       * The algorithm draws for each voxel in `layers` a polygonal line passing through it. This polygonal line follows
       * the stream of `interpolator_.vector_field_`; we call it a streamline.
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
      auto layers = layers_.unchecked<3>();
      const std::vector<py::ssize_t> dims{{layers.shape(0), layers.shape(1), layers.shape(2), 3}};
      py::array_t<Float> voxel_to_point_map(dims);
      auto voxel_to_point_map_ = voxel_to_point_map.mutable_unchecked<4>();

      for (py::ssize_t i = 0; i < layers.shape(0); ++i) {
        for (py::ssize_t j = 0; j < layers.shape(1); ++j) {
            for (py::ssize_t k = 0; k < layers.shape(2); ++k) {
              for (int l = 0; l < 3; ++l) voxel_to_point_map_(i, j, k, l) = NAN;
              if (layers(i, j, k) != 0) {
                const Index_3 voxel_index(i, j, k);
                utils::ThrowOnInvalidVector(interpolator_.GetVector(voxel_index),
                  "The vector field should not contain any NAN or zero vector "
                  "on the domain defined by the layers volume."
                );
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
  py::array_t<flatmap::Float, 1> offset,
  py::array_t<flatmap::Float, 1> voxel_dimensions,
  py::array_t<flatmap::Float, 4> vector_field,
  uint8_t layer_1, uint8_t layer_2) {
  /**
     * Compute the streamlines intersection points with the surface separating two layers.
     *
     * The algorithm draws for each voxel in `layers` a polygonal line passing through it. This polygonal line follows
     * the stream of `vector_field`; we call it a streamline.
     * The first segment of the streamline L which crosses the boundary surface S between `layer_1` and `layer_2` is used
     * to approximate the intersection point between L and S: the intersection point is approximated by the middle point
     * of this segment.
     * This process maps every voxel in `layers` to the intersection point of its streamline.
     * The intersection point is set to Vector_3(NAN, NAN, NAN) if no segment of the streamline crosses the boundary surface.
     * @param layers uint8_t array of shape (W, H, D) where W, H and D are the 3 dimensions of the array.
     *  The value of a voxel is the index of the layer it belongs to.
     * @param offset float array of shape (3,) holding the offset of the region defined by `layers`.
     *  This is used to compute point coordinates in the absolute (world) 3D frame.
     * @param voxel_dimensions float array of shape (3,) holding dimensions of voxels of the volume `layers`.
     *  This is used to compute point coordinates in the absolute (world) 3D frame.
     * @param vector_field float array of shape (W, H, D, 3) where (W, H, D) is the shape of `layers`.
     *  3D Vector field defined over `layers` domain used to draw the streamlines. It should not contain any zero or NAN
     *  vectors inside the `layers` domain.
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
    layers,
    utils::Vector_3(offset_(0), offset_(1), offset_(2)),
    utils::Vector_3(voxel_dimensions_(0), voxel_dimensions_(1), voxel_dimensions_(2)),
    vector_field,
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
      py::arg("offset"), // numpy array of shape (3,) and dtype np.float32
      py::arg("voxel_dimensions"), // numpy array of shape (3,) and dtype np.float32,
      py::arg("vector_field"), // numpy array of shape (W, H, D, 3) and dtype np.float32
      py::arg("layer_1"), // scalar of type np.uint8
      py::arg("layer_2") // scalar of type np.uint8
    );
    static py::exception<utils::InvalidVectorError> exc(m, "InvalidVectorError");
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const utils::InvalidVectorError &e) {
            exc(e.what());
        }
    });
}