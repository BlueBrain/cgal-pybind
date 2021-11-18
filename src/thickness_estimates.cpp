#include <algorithm>  // std::fill
#include <cassert>    // assert
#include <cmath>      // NAN
#include <limits>     // std::numeric_limits
#include <stdexcept>  // std::invalid_argument
#include <vector>

#include "utils.hpp"

/*
Estimating the thicknesses of layers with respect to the direction vector of every voxel of 3D
region is an iterative and expensive operation performed on arrays of shape (W, H, D) where W, H and
D are the integer dimensions of a brain region domain.

This operation is used repeatedly by nse/atlas-building-tools for the computation of placement hints
(see https://bbpteam.epfl.ch/documentation/projects/placement-algorithm/latest/index.html
for the definition of placement hints and their use)
*/

namespace py = pybind11;

namespace {
using utils::Float;
using utils::Index_3;

/**
 * Class to estimate the thickness of layers along the direction vector of each voxel.
 *
 */
class ThicknessEstimator
{
  private:
    const utils::Transform transform_;
    const py::array_t<uint8_t, 3> layers_;
    const utils::Vector3Field vector_field_;
    const uint32_t layer_count_;
    const Float resolution_;
    const Index_3 region_;

    /**
     * Compute the layer thicknesses for each voxel in the region along the ray starting from a
     * voxel and whose direction is +/- the voxel direction vector.
     *
     * @param current_index Index of the voxel wrt which layer thicknesses are estimated.
     * @param forward If true, the forward ray is drawn, otherwise the backward ray is drawn.
     * @param output_counts The vector of the per layer counts to be updated
     */
    void ComputeLayerCountPerRay(Index_3 current_index,
                                 bool forward,
                                 std::vector<size_t>& output_counts) {
        const auto layers = layers_.unchecked<3>();

        const utils::Point_3 origin = transform_.IndexToVoxelCentroid(current_index);

        const int sign = forward ? 1 : -1;
        // We scale the direction vector (unit) length to increase or decrease the ray resolution
        const utils::Vector_3 direction = vector_field_.GetVector(current_index) * sign *
                                          resolution_;

        size_t count = 0;
        for (;;) {  // infinite loop
            current_index = transform_.PhysicalPointToIndex(origin + (count + 1) * direction);

            // out of bounds
            if (not utils::IsInsideRegion(region_, current_index)) {
                break;
            }

            int current_layer = layers(current_index[0], current_index[1], current_index[2]);

            // hit the void
            if (current_layer == 0) {
                break;
            }

            assert(0 <= current_layer && current_layer < output_counts.size());
            ++output_counts[current_layer];
            ++count;
        }

        // total count from the voxel center to the bottom
        if (not forward) {
            output_counts[0] = count;
        }
    };

  public:
    ThicknessEstimator(const py::array_t<uint8_t, 3>& layers,
                       const py::array_t<Float, 4>& vector_field,
                       const utils::Vector_3& offset,
                       const utils::Vector_3& voxel_dimensions,
                       uint32_t layer_count,
                       Float resolution = 0.5)
        : layers_(layers)
        , vector_field_(vector_field)
        , transform_(offset, voxel_dimensions)
        , layer_count_(layer_count)
        , resolution_(resolution)
        , region_(vector_field_.shape(0), vector_field_.shape(1), vector_field_.shape(2)){};

    /**
     * Assign to each voxel `layer_count` + 1 values of type float. The first position 0 of these
     * values corresponds to the distance of the voxel from its center to the bottom of thet last
     * layer. The rest of the positions, usually 1-6 correspond to the layer thicknesses along the
     * voxel's direction.
     *
     * The direction is obtained from  the`vector_field` and does not change for a voxel's
     * calculations.
     *
     * Main method of the class ThicknessEstimator.
     *
     * The algorithm casts for each voxel in `layers` two rays starting from the voxel centroid
     * following the +/- voxel directions in a straight line.
     *
     * The thickness of a layer is estimated by the ray segments that remain in that layer.
     *
     * @return py::array_t<Float, 4> to be read as numpy array of shape (W, H, D, <number-of-layers
     * + 1>).
     */
    py::array_t<Float, 4> Estimate() {
        const auto layers = layers_.unchecked<3>();
        const py::ssize_t total_layer_positions = layer_count_ + 1;

        py::array_t<Float, 4> thicknesses(
            {layers.shape(0), layers.shape(1), layers.shape(2), total_layer_positions});
        auto thicknesses_mut = thicknesses.mutable_unchecked<4>();

        std::vector<size_t> layer_ray_counts(total_layer_positions);

        for (size_t i = 0; i < layers.shape(0); ++i) {
            for (size_t j = 0; j < layers.shape(1); ++j) {
                for (size_t k = 0; k < layers.shape(2); ++k) {
                    // if within domain
                    if (layers(i, j, k) != 0) {
                        // reset layer_ray_counts to zero and use it to update thicknesses
                        // by making a forward (true) and backward (false) pass.
                        std::fill(layer_ray_counts.begin(), layer_ray_counts.end(), 0);

                        const Index_3 voxel_index(i, j, k);
                        ComputeLayerCountPerRay(voxel_index, true, layer_ray_counts);
                        ComputeLayerCountPerRay(voxel_index, false, layer_ray_counts);

                        for (size_t l = 0; l < total_layer_positions; ++l) {
                            thicknesses_mut(i, j, k, l) = layer_ray_counts[l] * resolution_;
                        }
                    } else {
                        for (size_t l = 0; l < total_layer_positions; ++l) {
                            thicknesses_mut(i, j, k, l) = NAN;
                        }
                    }
                }
            }
        }
        return thicknesses;
    }
};
}  // anonymous namespace


/**
 * Estimate layer thicknesses along direction vectors
 *
 * The algorithm casts for each voxel in `layers` two rays, starting from its centroid, one forward
 * following the voxel's initial direction and one backward following the opposite direction. As the
 * rays propagate in either direction, they accumulate for each layer the number of segments that
 * have crossed it. The step of the ray is determined by the resolution.
 *
 * Each voxel stores `layer_count` + 1 values corresponding to a distance to the bottom of the
 * layers followed by a thickness for each layer.
 *
 * @param layers uint8 array of shape (W, H, D) where W, H and D are the 3 dimensions of the array.
 *  The value of a voxel is 0 to determine the void or 1-6 to determine the layer number.
 * @param vector_field float array of shape (W, H, D, 3) where (W, H, D) is the shape of `layers`.
 *  3D Vector field defined over `layers` domain used to cast the rays.
 * @param offset float array of shape (3,) holding the offset of the region defined by `layers`.
 *  This is used to compute point coordinates in the absolute (world) 3D frame.
 * @param voxel_dimensions float array of shape (3,) holding dimensions of voxels of `layers`.
 *  This is used to compute point coordinates in the absolute (world) 3D frame.
 * @param layer_count scalar of type int corresponding to the number of unique layers in `layers`s
 * @param resolution uniform scaling factor applied to the vector field before computing streamline
 * lengths.
 * @return py::array_t<float, 4> to be read as a numpy array of shape (W, H, D, `layer_count` + 1).
 * This array holds for each voxel V in`layers` `layer_count` + 1 non-negative values, the first of
 * which (0) corresponds to the distance from the center to the bottom of the last layer and the
 * rest to each layer's thickness.
 */
py::array_t<Float, 4> estimate_thicknesses(py::array_t<uint8_t, 3> layers,
                                           py::array_t<Float, 4> vector_field,
                                           py::array_t<Float, 1> offset,
                                           py::array_t<Float, 1> voxel_dimensions,
                                           int layer_count,
                                           Float resolution = 0.5) {
    ThicknessEstimator estimator(layers,
                                 vector_field,
                                 utils::vector_3_from_py_array(offset),
                                 utils::vector_3_from_py_array(voxel_dimensions),
                                 layer_count,
                                 resolution);

    return estimator.Estimate();
}


void bind_estimate_thicknesses(py::module& m) {
    m.def("estimate_thicknesses",
          &estimate_thicknesses,
          R"(Estimate layer thicknesses along the direction vector of each voxel of a region.

      layers: numpy array of shape (W, H, D) and dtype np.uint8
      vector_field: numpy array of shape (W, H, D, 3) and dtype np.float32
      offset: numpy array of shape (3,) and dtype np.float32
      voxel_dimensions: numpy array of shape (3,) and dtype np.float32,
      layer_count: scalar of type int, corresponding to the unique number of layers in `layers`
      resolution: scalar of type float

      )",
          py::arg("layers"),
          py::arg("vector_field"),
          py::arg("offset"),
          py::arg("voxel_dimensions"),
          py::arg("layer_count"),
          py::arg("resolution"));
}
