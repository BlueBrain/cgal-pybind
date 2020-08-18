from cgal_pybind import SurfaceMesh, Point_3
import trimesh
import numpy as np


def test_contract():
    mesh = trimesh.primitives.Capsule(
        transform=np.array([[1, 0, 0, 2], [0, 1, 0, 2], [0, 0, 1, 2], [0, 0, 0, 1],])
    )
    surface_mesh = SurfaceMesh()
    mesh_vertices = [Point_3(v[0], v[1], v[2]) for v in mesh.vertices]
    surface_mesh.add_vertices(mesh_vertices)
    surface_mesh.add_faces([tuple(f) for f in mesh.faces])

    vertices, edges = surface_mesh.contract()
    vertices = np.array(vertices).reshape((-1, 3))

    assert vertices.shape[0] > len(edges)


def test_authalic():
    # A pyramid without basis
    mesh = trimesh.Trimesh(
        vertices=[[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0], [0, 0, 1]],
        faces=[[4, 0, 1], [4, 1, 2], [4, 2, 3], [4, 3, 0]],
    )
    surface_mesh = SurfaceMesh()
    surface_mesh.add_vertices([Point_3(v[0], v[1], v[2]) for v in mesh.vertices])
    surface_mesh.add_faces([tuple(f) for f in mesh.faces])
    vertices = np.array(surface_mesh.authalic()[0])
    assert vertices.shape[0] == mesh.vertices.shape[0]
    assert np.all(vertices >= 0.0) and np.all(vertices <= 1.0)
