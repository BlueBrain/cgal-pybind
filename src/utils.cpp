

#include "utils.hpp"

namespace utils {

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

  Vector_3 VectorInterpolator::TrilinearInterpolation(
      const Point_3 &origin,
      const std::array<Vector_3, 8> &vertex_vectors)
  {
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

  Vector_3 VectorInterpolator::InterpolateVector(const Point_3 &point) const
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
     * @param region a 3D index defining the region (3D box) where interpolation is performed.

     * @return a 3D vector interpolating `vector_field_` at `point` by means of trilinear interpolation.
    */
    {
      static const std::array<Index_3, 8> index_offsets = {
        Index_3({0, 0, 0}), Index_3({1, 0, 0}), Index_3({1, 1, 0}), Index_3({0, 1, 0}),
        Index_3({0, 0, 1}), Index_3({1, 0, 1}), Index_3({1, 1, 1}), Index_3({0, 1, 1})
      };
      // Matrix holding the distances between any two vertices of the cube [0, 1]^3.
      static const DistanceMatrix distance_matrix = compute_distance_matrix();

      const Point_3 &origin = transform_.PointToContinuousIndex(point);
      Index_3 origin_index(std::floor(origin[0]), std::floor(origin[1]), std::floor(origin[2]));
      std::array<Vector_3, 8> vertex_vectors{};
      vertex_vectors[0] = GetVector(origin_index);

      // Assign to each vertex of the voxel a vector from the vector field
      for (int i = 1; i < 8; ++i) {
        const Index_3 &index = origin_index + index_offsets[i];
        if (IsInsideRegion(index)) vertex_vectors[i] = GetVector(index);
        else vertex_vectors[i] = Vector_3(NAN, NAN, NAN);
      }

      // Voxel vertices with an invalid or a missing vector are assigned a weighted average of
      // the valid vectors of their neighbours.
      for (size_t i = 1; i < 8; ++i) {
        if (std::isnan(vertex_vectors[i][0])) {
          Vector_3 v(0.0, 0.0, 0.0);
          Float total_weight = 0;
          for (size_t j = 0; j < 8; ++j) {
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
   };

void ThrowOnInvalidVector(const Vector_3 &v, std::string message) {
  if (std::isnan(v[0]) || std::isnan(v[1]) || std::isnan(v[2])) {
        message = "Found unexpected NAN vector. " + message;
        throw InvalidVectorError(message);
    }
    if (v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0) {
        message = "Found unexpected zero vector. " + message;
        throw InvalidVectorError(message);
    }
}


} // namespace::utils
