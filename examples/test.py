""" cgal binding example """

import sys
import cgal_pybind
import numpy as np
import trimesh


def contract(_polyhedron):
    """
    Convert an interactively contracted skeleton to a skeleton curve
    Args:
        polyhedron (cgal_pybind.Polyhedron): Polyhedral surfaces in three dimensions
        to be skeletonized.
    Return:
        vertices (numpy.array)[nb_vertex, 3] : Vertices cartesian coordinates
        edges (numpy.array)[nb_vertex, 2] : vertex source and vertex target per vertex
        correspondence : map { key=vertex_id, values=list(surface_id) }
    """
    _vertices, _edges, _correspondence = _polyhedron.contract()
    # need to shape this right on exit
    _vertices = np.asarray(_vertices).reshape((-1, 3))
    _edges = np.asarray(_edges)
    return _vertices, _edges, _correspondence


def segmentation(_polyhedron):
    """
    Assign a unique id to each facet in the mesh.
    Computes property-map for SDF values.
    Computes property-map for segment-ids.e
    Args:
        _polyhedron (cgal_pybind.Polyhedron): Polyhedral surfaces in three dimensions
        to be skeletonized.
    Return:
        sdf_property_map (numpy.array[nb_face]) : 1 diameter per face
        segment_property_map (numpy.array[nb_face) : segment_cluster_id per face
    """
    _sdf_property_map, _segment_property_map = _polyhedron.segmentation()
    _sdf_property_map = np.asarray(_sdf_property_map).reshape(-1)
    _segment_property_map = np.asarray(_segment_property_map).reshape(-1)
    return _sdf_property_map, _segment_property_map


def build_from_file(filename, _polyhedron):
    """
    Construct a Polyhedron from OFF file.
    Args:
        filename (str): path to OFF file
        _polyhedron (cgal_pybind.Polyhedron) : Polyhedron ton construct
    Return:
        mesh (path – Data as a native trimesh Path object)
    """
    print('Build Polyhedron from OFF file {}'.format(filename))
    _polyhedron.load_from_off(filename)
    return trimesh.load_mesh(filename)


def build_capsule_mesh():
    """
    Construct a Polyhedron from a trimesh Capsule
    Args:
    Return:
        mesh (path – Data as a native trimesh Path object)
    """
    _mesh = trimesh.primitives.Capsule(transform=np.array([[1, 0, 0, 2],
                                                           [0, 1, 0, 2],
                                                           [0, 0, 1, 2],
                                                           [0, 0, 0, 1], ]))
    # TODO: add_vertices/add_faces should be more efficient
    _vertices = [cgal_pybind.Point_3(v[0], v[1], v[2]) for v in _mesh.vertices]
    _face_indices = [tuple(f) for f in _mesh.faces]
    return _mesh, _vertices, _face_indices


if __name__ == '__main__':
    mesh = None
    polyhedron = cgal_pybind.Polyhedron()
    if len(sys.argv) > 1:
        mesh = build_from_file(sys.argv[1], polyhedron)
    else:
        mesh, vertices, face_indices = build_capsule_mesh()
        polyhedron.build(vertices, face_indices)

    print('size_of_vertices: ', polyhedron.size_of_vertices())
    print('size_of_facets: ', polyhedron.size_of_facets())
    print('area: ', polyhedron.area())
    print('\n===> 1/ Contract geometry <====\n')
    skeleton_vertices, skeleton_edges, correspondence = contract(polyhedron)
    print('\n------Skeleton vertices and edges (atrocyde-skeleton.txt)------')
    print('skeleton_vertices.shape', skeleton_vertices.shape)
    print('skeleton_edges.shape', skeleton_edges.shape)

    print('''\n------(atrocyde-correspondence.txt) ------\n
    A mapping file between the skeleton points (id) and the surface
                          faces (id) generated by the contraction process which gives
                          information about the the faces of the surface that were collapsed
                          into a skeleton point
    ''')
    print('len(correspondence.keys()):', len(correspondence.keys()))

    print('\n\n===> 2/ Segment geometry <====\n')
    sdf_property_map, segment_property_map = segmentation(polyhedron)
    print('------ sdf values (atrocyde-sdf.txt) ------')
    print('(sdf_property_map).shape {}'.format(sdf_property_map.shape))
    print('(sdf_property_map).dtype {}'.format(sdf_property_map.dtype))
    print('\n------Segmentation of a surface mesh given an'
          ' SDF value per facet atrocyde-sdf.txt------')
    print('(segment_property_map).shape {}'.format(segment_property_map.shape))
    print(segment_property_map)

    # display the skeleton shifted over, so one can see it
    path_visual = trimesh.load_path(skeleton_vertices[skeleton_edges, :] + [2, 2, 2])
    scene = trimesh.Scene([path_visual, mesh])
    scene.show()
