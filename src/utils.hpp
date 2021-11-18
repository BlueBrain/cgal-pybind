#ifndef CGAL_PYBIND_UTILS
#define CGAL_PYBIND_UTILS

#include <array>
#include <cmath>
#include <stdexcept> // std::runtime_error

#include <CGAL/Simple_cartesian.h>
#include <CGAL/MP_Float.h>
#include <CGAL/Vector_3.h>
#include <CGAL/Point_3.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace utils {
  using Float = float;
  using Kernel = CGAL::Simple_cartesian<Float>;
  using Vector_3 = Kernel::Vector_3;
  using Point_3 = Kernel::Point_3;

  // Helper classes for geometric computations: Vector_3, Point_3 = Vector_3, Index_3 and Transform.

    /**
     * Class to perform natural operations of 3D float vectors.
     *
     * The Vector_3 class implements natural coordinate accessors and the
     * computation of the Euclidean norm.
     *
    */

  /** Return the coordinate-wise floor of a 3D point
   *
   * @return the floored 3D point.
   */
  inline Point_3 Floor(const Point_3 &p) {
      return Point_3(std::floor(p[0]), std::floor(p[1]), std::floor(p[2]));
  }

  inline Vector_3 vector_3_from_py_array(const py::array_t<utils::Float, 1>& array_){
      const auto array = array_.unchecked<1>();
      return {array(0), array(1), array(2)};
  }

  /**
    * Class to manage voxel indices, i.e integer triples defining voxels
  */
  class Index_3 {
    private:
      std::array<py::ssize_t, 3> vec_;
    public:
        Index_3(): vec_{{0, 0, 0}} {};
        Index_3(py::ssize_t x, py::ssize_t y, py::ssize_t z): vec_{{x, y, z}} {};
        Index_3(const Index_3 &idx): vec_(idx.vec_) {};
        py::ssize_t &operator[](int index){ return vec_[index]; };
        const py::ssize_t &operator[](int index) const { return vec_[index]; };
  };

  /** Add two 3D indices
   * @param idx1 3D index
   * @param idx2 3D index
   *
   * @return the coordinate-wise sum of idx1 and idx2
   */
  inline Index_3 operator+(const Index_3 &idx1, const Index_3 &idx2) {
      return Index_3(idx1[0] + idx2[0], idx1[1] + idx2[1], idx1[2] + idx2[2]);
  };

  /** Check if a 3D index lies in the 3D box whose indices are bounded above by `region`
   * @param region 3D index defining a box-shaped region whose side lengths are `region` coordinates
   * @param idx 3D index
   *
   * @return true if `index` lies in `region`, false otherwise.
   */
  inline bool IsInsideRegion(const Index_3 &region, const Index_3 &idx)
  {
      return (idx[0] < region[0] && idx[1] < region[1] && idx[2] < region[2]) &&
          (idx[0] >= 0 && idx[1] >= 0 && idx[2] >= 0);
  }

  /**
    * Class to pass from voxel indices to absolute 3D coordinates and vice-versa.
    *
    * The constructor of the Transform class takes the offset and the voxel dimensions of
    * a voxcell.VoxelData object as arguments. These allow us to define two transformations
    * one from voxel coordinates (integer triples, i.e. Index_3) to 3D world coordinates
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
      Point_3 IndexToPhysicalPoint(const Index_3 &idx) const {
        return Point_3(
          voxel_dimensions_[0] * idx[0],
          voxel_dimensions_[1] * idx[1],
          voxel_dimensions_[2] * idx[2]) + offset_;
      };
     Point_3 IndexToVoxelCentroid(const Index_3 &idx) const {
        return Point_3(
          voxel_dimensions_[0] * (idx[0] + 0.5),
          voxel_dimensions_[1] * (idx[1] + 0.5),
          voxel_dimensions_[2] * (idx[2] + 0.5)) + offset_;
     };

      Index_3 PhysicalPointToIndex(const Point_3 &p) const {
        const Point_3 &q = Floor(PointToContinuousIndex(p));
        return Index_3(
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
      Vector_3 GetVector(const Index_3 &idx) const {
        return Vector_3(
          vector_field_(idx[0], idx[1], idx[2], 0),
          vector_field_(idx[0], idx[1], idx[2], 1),
          vector_field_(idx[0], idx[1], idx[2], 2));
      };
      size_t shape(int index) const { return vector_field_.shape(index); };
  };

  typedef std::array<std::array<Float, 8>, 8> DistanceMatrix;
  DistanceMatrix compute_distance_matrix();

  class VectorInterpolator {
    private:
      Vector3Field vector_field_;
      Transform transform_;
      Index_3 region_;

    public:
      VectorInterpolator(
        py::array_t<Float, 4> vector_field,
        const Vector_3 &offset,
        const Vector_3 &voxel_dimensions
      ):
          vector_field_(Vector3Field(vector_field)),
          transform_(Transform(offset, voxel_dimensions))
      {
          region_ = Index_3(vector_field_.shape(0), vector_field_.shape(1), vector_field_.shape(2));
      };

      static Vector_3 TrilinearInterpolation(const Point_3 &origin, const std::array<Vector_3, 8> &vertex_vectors);
      Vector_3 InterpolateVector(const Point_3 &point) const;
      Vector_3 GetVector(const Index_3 &idx) const { return vector_field_.GetVector(idx); }
      bool IsInsideRegion(const Index_3 &idx) const { return utils::IsInsideRegion(region_, idx); };
      Index_3 PhysicalPointToIndex(const Point_3 &point) const { return transform_.PhysicalPointToIndex(point); };
      Point_3 IndexToPhysicalPoint(const Index_3 &idx) const { return transform_.IndexToPhysicalPoint(idx); };
  };

/*
Error raised if a 3D vector is deemed invalid, see ThrowOnInvalidVector,
*/
class InvalidVectorError : public std::runtime_error {
    public:
      InvalidVectorError(const std::string& what = ""): std::runtime_error(what) {};
};
/**
Throws an InvalidVectorError exception if the input vector is deemed invalid, i.e. is zero or
contains NANs.
   * @param v 3D vector
   * @param message additional message string appended ot the exception message.
   *
   * @throw InvalidVectorError if v is zero or contains NANs.
*/
void ThrowOnInvalidVector(const Vector_3 &v, std::string message);

} // namespace::utils

// CGAL_PYBIND_UTILS
#endif
