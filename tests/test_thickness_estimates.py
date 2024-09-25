import sys

import numpy as np
import numpy.testing as npt
import pytest

from cgal_pybind import estimate_thicknesses

np.set_printoptions(threshold=sys.maxsize, suppress=True)


@pytest.fixture
def layers():

    layer_nums = [1, 2, 3, 4, 5, 6]
    rows_per_layer = 2

    values = np.zeros((len(layer_nums) * rows_per_layer, 1, 1), dtype=np.uint8)

    for i, layer in enumerate(layer_nums):
        values[i * rows_per_layer : (i + 1) * rows_per_layer, ...] = layer

    # reverse it
    values[:, ...] = values[::-1, ...]

    values = np.pad(values, 2, "constant", constant_values=0)

    return values


@pytest.fixture
def vector_field(layers):

    values = np.full(layers.shape + (3,), np.nan, dtype=np.float32)

    values[2:-2, 2:-2, 2:-2] = [1.0, 0.0, 0.0]

    return values


def test_thickness_estimates__horizontal_layers(layers, vector_field):
    layer_count = 6
    resolution = 0.5

    thicknesses = estimate_thicknesses(
        layers,
        vector_field,
        offset=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        voxel_dimensions=np.array([10.0, 10.0, 10.0], dtype=np.float32),
        layer_count=layer_count,
        resolution=resolution,
    )

    assert thicknesses.shape == layers.shape + (layer_count + 1,)

    # there is only one central column with values
    central_column = thicknesses[2:-2, 2, 2, :]

    # the first column corresponds to the distance to the bottom of layer 6
    npt.assert_allclose(
        central_column[:, 0],
        [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 105.0, 115.0],
    )

    # the rest of the columns correspond to layer thicknesses 1 to 6
    npt.assert_allclose(central_column[:, 1], 20.0, atol=resolution)
    npt.assert_allclose(central_column[:, 2], 20.0, atol=resolution)
    npt.assert_allclose(central_column[:, 3], 20.0, atol=resolution)
    npt.assert_allclose(central_column[:, 4], 20.0, atol=resolution)
    npt.assert_allclose(central_column[:, 5], 20.0, atol=resolution)
    npt.assert_allclose(central_column[:, 6], 20.0, atol=resolution)


def test_thickness_estimates__diagonal_layers():
    """See diagonal_grid_layers.eps for layer structure used in
    this test.
    """

    N = 29

    layers = np.zeros((N, N, 1), dtype=np.uint8)

    # diagonal layers from bottom left (0, 0) [layer 6] to top right (N, N) [layer 1]

    for i in range(N):
        for j in range(N):

            if i - j in {-28, -27, -26, -25, -24, -23}:
                layers[i, j, :] = 0
            elif i - j in {-22, -21, -20, -19, -18, -17, -16}:
                layers[i, j, :] = 6
            elif i - j in {-15, -14, -13, -12, -11, -10, -9}:
                layers[i, j, :] = 5
            elif i - j in {-8, -7, -6, -5, -4, -3, -2}:
                layers[i, j, :] = 4
            elif i - j in {-1, 0, 1, 2, 3, 4, 5}:
                layers[i, j, :] = 3
            elif i - j in {6, 7, 8, 9, 10, 11, 12}:
                layers[i, j, :] = 2
            elif i - j in {13, 14, 15, 16, 17, 18, 19}:
                layers[i, j, :] = 1
            elif i - j in {20, 21, 22, 23, 24, 25, 26, 27, 28}:
                layers[i, j, :] = 0

            if i + j < 21 or i + j > 35:
                layers[i, j, :] = 0

    layers = np.flip(layers, axis=1)
    layers = np.pad(layers, 2, "constant", constant_values=0)

    diag_vector = np.array([1.0, 1.0, 0.0])
    diag_vector /= np.linalg.norm(diag_vector)

    vector_field = np.full(layers.shape + (3,), fill_value=np.nan, dtype=np.float32)
    vector_field[2:-2, 2:-2, 2:-2, :] = diag_vector

    # because we want to calculate distances along the diagonal it is nice to choose
    # the voxel size so that the diagonal is a rational number:
    # the xy diagonal of the voxel will be l = sqrt(2) * x = 2 * 10 = 20um
    voxel_dimensions = np.array([10.0, 10.0, 10.0], dtype=np.float32) * np.sqrt(2, dtype=np.float32)

    # to avoid overlapping with the grid we use an irrational resolution
    # if we used a multiple of the
    resolution = 0.05 * np.sqrt(2)

    thicknesses = estimate_thicknesses(
        layers,
        vector_field,
        offset=np.array([5.0, 5.0, 5.0], dtype=np.float32),
        voxel_dimensions=voxel_dimensions,
        layer_count=6,
        resolution=0.05 * np.sqrt(2),
    )

    # There are 5 columns in the third dimension, two columns of void,
    # one central column with layer data, and two columns of void

    slice_thicknesses = thicknesses[:, :, 2, :]
    slice_layers = layers[:, :, 2]

    # we shouldn't get any zero thicknesses
    assert not np.isclose(slice_thicknesses, 0.0).any()

    # voxel in layer 1
    npt.assert_allclose(
        slice_thicknesses[19, 26], [350.0, 80.0, 60.0, 80.0, 60.0, 80.0, 60.0], atol=0.1
    )

    # voxel in layer 2
    npt.assert_allclose(
        slice_thicknesses[18, 25], [330.0, 80.0, 60.0, 80.0, 60.0, 80.0, 60.0], atol=0.1
    )

    # voxel in layer 3
    npt.assert_allclose(
        slice_thicknesses[16, 18], [250.0, 60.0, 80.0, 60.0, 80.0, 60.0, 80.0], atol=0.1
    )

    # voxel in layer 4
    npt.assert_allclose(
        slice_thicknesses[9, 15], [150.0, 60.0, 80.0, 60.0, 80.0, 60.0, 80.0], atol=0.1
    )

    # voxel in layer 5
    npt.assert_allclose(
        slice_thicknesses[11, 11], [130.0, 60.0, 80.0, 60.0, 80.0, 60.0, 80.0], atol=0.1
    )

    # voxel in layer 6
    npt.assert_allclose(
        slice_thicknesses[3, 7], [10.0, 60.0, 80.0, 60.0, 80.0, 60.0, 80.0], atol=0.1
    )
