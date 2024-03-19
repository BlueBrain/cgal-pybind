from pathlib import Path

import numpy as np
import pytest
import trimesh
from numpy import testing as npt

from cgal_pybind import Polyhedron

TESTS_DIR = Path(__file__).parent


@pytest.fixture
def mesh():
    return trimesh.load(TESTS_DIR / "capsule.stl")


@pytest.fixture(scope="function")
def polyhedron(mesh):
    polyhedron = Polyhedron()
    polyhedron.build(mesh.vertices, mesh.faces)
    return polyhedron


def test_polyhedron_build(mesh, polyhedron):

    npt.assert_allclose(mesh.vertices, polyhedron.vertices)
    npt.assert_array_equal(mesh.faces, polyhedron.faces)

    npt.assert_array_equal(polyhedron.vertex_ids, np.arange(len(mesh.vertices), dtype=int))
    npt.assert_array_equal(polyhedron.face_ids, np.arange(len(mesh.faces), dtype=int))


def test_polyhedron_contract(polyhedron):

    (
        vertices,
        edges,
        correspondence,
    ) = polyhedron.contract()

    assert isinstance(vertices, np.ndarray)
    assert vertices.ndim == 2 and vertices.shape[1] == 3

    assert isinstance(edges, np.ndarray)
    assert edges.ndim == 2 and edges.shape[1] == 2

    assert len(correspondence.keys()) == vertices.shape[0]


def test_polyhedron_segmentation(polyhedron):

    shape_diameters, segment_ids = polyhedron.segmentation()

    assert isinstance(shape_diameters, np.ndarray)
    assert shape_diameters.ndim == 1
    assert isinstance(segment_ids, np.ndarray)
    assert segment_ids.ndim == 1

    assert shape_diameters.size == 4032
    assert segment_ids.size == 4032
