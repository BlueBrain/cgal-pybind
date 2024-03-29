cmake_minimum_required(VERSION 3.9.0)

project(_cgal_pybind)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TODO(mgevaert): add EXTLIB_FROM_SUBMODULES option
add_subdirectory(extlib EXCLUDE_FROM_ALL)

pybind11_add_module(_cgal_pybind
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cgal_bind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/point.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/polyhedron.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/surface_mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/streamlines_intersections.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slice_volume.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/thickness_estimates.cpp
    )

find_package(Eigen3 CONFIG REQUIRED)

target_link_libraries(_cgal_pybind PRIVATE Eigen3::Eigen)

find_package(CGAL CONFIG REQUIRED)
target_link_libraries(_cgal_pybind PRIVATE CGAL::CGAL)
