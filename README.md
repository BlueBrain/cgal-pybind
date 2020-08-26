# cgal-pybind

 [CGAL](http://cgal.org) Python binding with [pybind11](https://pybind11.readthedocs.io)

## Methods:
* Skeletonization/contraction
* Segmentation
* Discrete authalic parametrization of surface mesh

## Installation
```bash
$> git clone https://github.com/CGAL/cgal.git $PATH_TO_GIT
$> git clone https://github.com/mgeplf/cgal-pybind.git
$> cd cgal-pybind
$> git submodule init
$> git submodule update
$> export CGAL_DIR=$PATH_TO_GIT
$> pip install .
```

## Test
```bash
$> pip install tox
$> tox
```

## Requirements
* cmake > 3.0.9
* C++ compiler (with C++11 support)
* CGAL header

## Unit test requirement:
* trimesh Python package
