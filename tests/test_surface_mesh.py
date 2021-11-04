import numpy as np
import trimesh

from cgal_pybind import Point_3, SurfaceMesh


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
