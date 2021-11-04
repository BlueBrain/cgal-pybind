import numpy as np
import numpy.testing as npt
import pytest

from cgal_pybind import InvalidVectorError, compute_streamlines_intersections


def test_compute_streamlines_intersections():
    layers = np.zeros((20, 3, 3), dtype=np.uint8)
    for i in range(4):
        layers[i * 5 : (i + 1) * 5, ...] = i + 1

    layers = np.pad(layers, 2, "constant", constant_values=0)
    direction_vectors = np.full(layers.shape + (3,), np.nan, dtype=np.float32)
    direction_vectors[2:-2, 2:-2, 2:-2] = [1.0, 0.0, 0.0]

    voxel_to_point_map = compute_streamlines_intersections(
        layers=layers,
        offset=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        voxel_dimensions=np.array([2.0, 2.0, 2.0], dtype=np.float32),
        vector_field=direction_vectors,
        layer_1=np.uint8(2),
        layer_2=np.uint8(3),
    )

    npt.assert_allclose(voxel_to_point_map[3, 3, 3], [25, 8, 9], rtol=1e-2)
    npt.assert_allclose(voxel_to_point_map[4, 3, 3], [25, 8, 9], rtol=1e-2)
    npt.assert_allclose(voxel_to_point_map[5, 3, 3], [25, 8, 9], rtol=1e-2)
    npt.assert_allclose(voxel_to_point_map[17, 3, 3], [25, 8, 9], rtol=1e-2)
    npt.assert_allclose(voxel_to_point_map[20, 3, 3], [25, 8, 9], rtol=1e-2)

    npt.assert_allclose(voxel_to_point_map[2, 4, 4], [25, 10, 11], rtol=1e-2)
    npt.assert_allclose(voxel_to_point_map[3, 4, 4], [25, 10, 11], rtol=1e-2)
    npt.assert_allclose(voxel_to_point_map[4, 4, 4], [25, 10, 11], rtol=1e-2)
    npt.assert_allclose(voxel_to_point_map[13, 4, 4], [25, 10, 11], rtol=1e-2)
    npt.assert_allclose(voxel_to_point_map[15, 4, 4], [25, 10, 11], rtol=1e-2)


def test_compute_streamlines_intersections_exception():
    layers = np.zeros((8, 3, 3), dtype=np.uint8)
    for i in range(4):
        layers[i * 2 : (i + 1) * 2, ...] = i + 1

    layers = np.pad(layers, 2, "constant", constant_values=0)
    direction_vectors = np.full(layers.shape + (3,), np.nan, dtype=np.float32)

    direction_vectors[3, 2, 2] = [np.nan] * 3
    with pytest.raises(InvalidVectorError, match="N[Aa]N vector"):
        compute_streamlines_intersections(
            layers=layers,
            offset=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            voxel_dimensions=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            vector_field=direction_vectors,
            layer_1=np.uint8(2),
            layer_2=np.uint8(3),
        )
    direction_vectors[3, 2, 2] = [0.0] * 3
    with pytest.raises(InvalidVectorError, match="[Zz]ero vector"):
        compute_streamlines_intersections(
            layers=layers,
            offset=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            voxel_dimensions=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            vector_field=direction_vectors,
            layer_1=np.uint8(2),
            layer_2=np.uint8(3),
        )
