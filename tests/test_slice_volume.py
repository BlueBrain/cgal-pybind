from cgal_pybind import slice_volume
import numpy as np
import numpy.testing as npt


def test_slice_volume():
    volume = np.ones((18, 2, 2), dtype=bool)
    volume = np.pad(volume, 2, 'constant', constant_values=False)
    direction_vectors = np.full(volume.shape + (3,), np.nan, dtype=np.float32)
    direction_vectors[2:-2, 2:-2, 2:-2] = [1.0, 0.0, 0.0]

    # Uniform slice thickness
    slices = slice_volume(
        volume=volume,
        offset=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        voxel_dimensions=np.array([2.0, 2.0, 2.0], dtype=np.float32),
        vector_field=direction_vectors,
        thicknesses=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        resolution=0.5,
    )
    expected = np.zeros(volume.shape, dtype=int)
    expected[2:5, 2:-2, 2:-2] = 1
    expected[5:8, 2:-2, 2:-2] = 2
    expected[8:11, 2:-2, 2:-2] = 3
    expected[11:14, 2:-2, 2:-2] = 4
    expected[14:17, 2:-2, 2:-2] = 5
    expected[17:20, 2:-2, 2:-2] = 6

    npt.assert_array_equal(slices, expected)

    # Distinct slice thicknesses
    slices = slice_volume(
        volume=volume,
        offset=np.array([2.0, 3.0, 1.0], dtype=np.float32),
        voxel_dimensions=np.array([2.0, 2.0, 2.0], dtype=np.float32),
        vector_field=direction_vectors,
        thicknesses=np.array([2.0, 2.0, 5.0, 5.0, 2.0, 2.0], dtype=np.float32),
        resolution=0.5,
    )
    expected = np.zeros(volume.shape, dtype=int)
    expected[2:4, 2:-2, 2:-2] = 1
    expected[4:6, 2:-2, 2:-2] = 2
    expected[6:11, 2:-2, 2:-2] = 3
    expected[11:16, 2:-2, 2:-2] = 4
    expected[16:18, 2:-2, 2:-2] = 5
    expected[18:20, 2:-2, 2:-2] = 6

    npt.assert_array_equal(slices, expected)