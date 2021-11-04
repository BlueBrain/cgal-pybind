#include <limits> // std::numeric_limits
#include <stdexcept> // std::invalid_argument
#include <vector>

#include <assert.h>

#include "utils.hpp"

/*
The splitting of a 3D volume into slices with prescribed relative thicknesses is a computationally expensive operation
performed on arrays of shape (W, H, D) where W, H and D are the integer dimensions of a brain region domain.
This operation is used repeatedly in the computation of m-type volumetric densities done in the python module
atlas-building-tools/atlas_building_tools/cell_densities. The splitting into slices is implemented
as a python-C++ binding for efficiency reasons.

For when slicing, we assume that the input vector field contains no NaN values and no zero
vectors inside the region of interest. More: vectors from the input vector field are assumed to be approximately of the
same positive length. A common input is a field of unit direction vectors.

Note that volume slicing is also used in atlas-building-tools/atlas_building_tools/region_splitter for the splitting of
layer 2/3 into layer 2 and layer 3.
*/

namespace py = pybind11;
namespace {
using utils::Float;
using utils::Index_3;
using utils::Point_3;
using utils::Vector_3;


/**
 * Class handling the splitting of a 3D volume into slices with prescribed thicknesses.
 *
 * The input `volume` array is an array of type py::array_t<bool, 3>, i.e. a 3D binary mask .
 *
 * The input `vector_field` array is an array of type py::array_t<Float, 4> which assigns to each non-zero
 * voxel of `volume` a 3D unit vector.
 * This vector field is interpolated within voxels to get "higher resolution" streamlines.
 * The slicing algorithm is based on the length of the streamline (polygonal line in 3D space) passing through each
 * voxel. The proportion of the streamline line length run before hitting the voxel is used to determine
 * to which slice the voxel belongs.
 *
 * Every voxel of the input mask is assigned a 1-based index indicating in which slice it sits.
 */
class Slicer {
private:
    const py::array_t<bool, 3> &volume_;
    utils::VectorInterpolator interpolator_;
    std::vector<Float> thicknesses_;
    Float resolution_;

    void RescaleThicknesses() {
        // Prescribed thicknesses are turned into relative thicknesses in the range [0.0, 1.0]
        // to be compared with the length ratio of each voxel of mask `volume_`.
        Float thickness_sum = 0.0;
        for(int i = 0; i < thicknesses_.size(); ++i) {
            if (thicknesses_[i] < 0.0) {
                throw(std::invalid_argument("Negative thickness for slice with index " +
                                            std::to_string(i) +
                                            ". Slices cannot be created."
                                           ));
            }
            thickness_sum += thicknesses_[i];
        }

        if (thickness_sum == 0.0) {
            throw(std::invalid_argument("Zero thickness sum. Slices cannot be created."));
        }

        for(Float& thickness : thicknesses_) {
            thickness /= thickness_sum;
        }
    };

    /**
     * Find the slice index of a voxel.
     *
     * The alogrithm computes a streamline length ratio to decide to which slice
     * a voxel is assigned.
     *
     * @param voxel_index Index of the voxel whose streamline is to be drawn.

     * @return a slice index ranging in [1, number of slices].
     */
    int FindSliceIndex(const Index_3 &voxel_index) const
    {
        const int forward_length = ComputeStreamlineLength(voxel_index, true);
        const int backward_length = ComputeStreamlineLength(voxel_index, false);
        const double ratio =
            backward_length / (forward_length + backward_length + std::numeric_limits<Float>::epsilon());

        Float thickness = 0.0;
        int i = 0;
        for (; i < thicknesses_.size(); ++i) {
            thickness += thicknesses_[i];
            if (ratio <= thickness){
                break;
            }
        }

        return i + 1;
    }

    /**
     * Compute the length of the half-streamline starting from or ending at a voxel.
     *
     * The length of a polygonal streamline is defined as the number of vectors from the
     * input vector field (rescaled by the resolution factor) which as been added to go from
     * one end to the other.
     *
     * @param voxel_index Index of the voxel whose streamline is to be drawn.
     * @param forward if true, the forward half-streamline is drawn, otherwise the backward half-streamline is drawn.

     * @return
     */
    int ComputeStreamlineLength(const Index_3 &voxel_index, bool forward) const
    {
        const auto volume = volume_.unchecked<3>();
        Index_3 current_index = voxel_index;
        Point_3 current_point = interpolator_.IndexToPhysicalPoint(current_index);

        // We scale the direction vector (unit) length to increase or decrease the streamline resolution
        const Float direction = resolution_ * (forward ? 1.0F : -1.0F);
        int streamline_length = 0;
        bool is_in_volume = volume(current_index[0], current_index[1], current_index[2]);

        while (interpolator_.IsInsideRegion(current_index) && is_in_volume) {
            const Vector_3 vector = interpolator_.InterpolateVector(current_point) * direction;
            // NAN and zero vectors aren't supported.
            assert(!std::isnan(vector[0]) && !std::isnan(vector[1]) && !std::isnan(vector[2]));
            assert(vector[0] != 0.0 || vector[1] != 0.0 || vector[2] != 0.0);
            current_point += vector;
            ++streamline_length;
            current_index = interpolator_.PhysicalPointToIndex(current_point);
            is_in_volume = volume(current_index[0], current_index[1], current_index[2]);
        }

        return streamline_length;
    };

public:
    Slicer(const py::array_t<bool, 3>& volume,
           const Vector_3 &offset,
           const Vector_3 &voxel_dimensions,
           const py::array_t<Float, 4>& vector_field,
           std::vector<Float> &&thicknesses,
           Float resolution=0.5)
        : volume_(volume)
        , thicknesses_(thicknesses)
        , interpolator_(vector_field, offset, voxel_dimensions)
        , resolution_(resolution) {
            RescaleThicknesses();
        };

    /**
     * Assign to each voxel of the 3D binary mask a 1-based slice index.
     *
     * Main method of the class Slicer.
     *
     * The algorithm draws for each voxel in `volume` a polygonal line passing through it. This polygonal line follows
     * the stream of `interpolator_.vector_field_`; we call it a streamline.
     *
     * The length of the streamline from its starting point to the voxel divded by the full length defines a ratio
     * which is used to compute the slice index of the voxel.
     * The ratio is compared with the terms of the accumulative series of relative slice thicknesses.
     *
     * @return py::array_t<int, 3> to be read as numpy array of shape (W, H, D).
     */
    py::array_t<int, 3> CreateSlices() const
    {
        const auto volume = volume_.unchecked<3>();
        py::array_t<int> slices({volume.shape(0), volume.shape(1), volume.shape(2)});
        auto slices_ = slices.mutable_unchecked<3>();

        for (py::ssize_t i = 0; i < volume.shape(0); ++i) {
            for (py::ssize_t j = 0; j < volume.shape(1); ++j) {
                for (py::ssize_t k = 0; k < volume.shape(2); ++k) {
                    if (volume(i, j, k)) {
                        utils::ThrowOnInvalidVector(interpolator_.GetVector(Index_3(i, j, k)),
                          "The vector field should not contain any NAN or zero vectors "
                          "on the domain defined by the volume mask.");
                        slices_(i, j, k) = FindSliceIndex({i, j, k});
                    } else {
                        slices_(i, j, k) = 0;
                    }
                }
            }
        }

        return slices;
    }
};
} // namespace


/**
 * Slice a 3D volume into slices with prescribed relative thicknesses.
 *
 * The algorithm draws for each voxel in `volume` a polygonal line passing through it. This polygonal line follows
 * the stream of `vector_field`; we call it a streamline.
 * The length of the streamline passing through a voxel is used to derive the slice index of this voxel.
 * More precisely, we use the ratio length-to-voxel / total length.
 * @param volume bool array of shape (W, H, D) where W, H and D are the 3 dimensions of the array.
 *  The value of a voxel is true, if and only if the voxel belongs to the volume (3D boolean mask).
 * @param offset float array of shape (3,) holding the offset of the region defined by `volume`.
 *  This is used to compute point coordinates in the absolute (world) 3D frame.
 * @param voxel_dimensions float array of shape (3,) holding dimensions of voxels of the volume `volume`.
 *  This is used to compute point coordinates in the absolute (world) 3D frame.
 * @param vector_field float array of shape (W, H, D, 3) where (W, H, D) is the shape of `volume`.
 *  3D Vector field defined over `volume` domain used to draw the streamlines. It should not contain any zero or NAN
 *  vectors inside the `volume` domain.
 * @param thicknesses 1D vector of slice relative thicknesses. The vector's size is the number of slices to create.
 * Note that `offset and `voxel_dimensions` are the attributes of the voxcell.VoxelData object corresponding to
 * both `volume` and `vector_field`.
 * @param resolution uniform scaling factor applied to the vector field before computing streamline lengths.
 * @return py::array_t<int, 3> to be read as a numpy array of shape (W, H, D). This array holds for each voxel
 * V of `volume` the 1-based index of the slice V belongs to.
 */
py::array_t<int, 3> slice_volume(
  py::array_t<bool, 3> volume,
  py::array_t<utils::Float, 1> offset,
  py::array_t<utils::Float, 1> voxel_dimensions,
  py::array_t<utils::Float, 4> vector_field,
  py::array_t<utils::Float, 1> thicknesses,
  utils::Float resolution=0.5) {

  auto thicknesses_obj = thicknesses.request();
  auto* thicknesses_ptr = static_cast<utils::Float*>(thicknesses_obj.ptr);

  Slicer slicer_(
    volume,
    utils::vector_3_from_py_array(offset),
    utils::vector_3_from_py_array(voxel_dimensions),
    vector_field,
    std::vector<utils::Float>(thicknesses_ptr, thicknesses_ptr + thicknesses_obj.shape[0]),
    resolution
  );

  return slicer_.CreateSlices();
}

void bind_slice_volume(py::module& m)
{
    m.def("slice_volume", &slice_volume,
          R"(Split a 3D volume into slices with prescribed relative thicknesses.

      volume: numpy array of shape (W, H, D) and dtype np.uint8
      offset: numpy array of shape (3,) and dtype np.float32
      voxel_dimensions: numpy array of shape (3,) and dtype np.float32,
      vector_field: numpy array of shape (W, H, D, 3) and dtype np.float32
      thicknesses: numpy array of shape (number_of_slices,) and dtype np.float32
      resolution: scalar of type float

      The lengths of the vector field streamlines are used to determine cutoffs.
      )",
      py::arg("volume"),
      py::arg("offset"),
      py::arg("voxel_dimensions"),
      py::arg("vector_field"),
      py::arg("thicknesses"),
      py::arg("resolution")
    );
}
