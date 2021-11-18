# cgal-pybind

 [CGAL](http://cgal.org) Python binding with [pybind11](https://pybind11.readthedocs.io)

## CGAL Methods:
* Skeletonization/contraction
* Segmentation
* Discrete authalic parametrization of surface mesh

## BBP Atlas tools (CGAL-free)
* Volume slicer (used to split the layer 2/3 of the AIBS mouse isocortex)
* Streamlines intersector (used for flat mapping of brain regions)
* Thickness estimator (used for placement hints computation)
* Utils: a 3D vector field interpolator (trilinear interpolation)

## Installation
```bash
$> git clone https://github.com/CGAL/cgal.git $PATH_TO_GIT
$> git clone https://bbpgitlab.epfl.ch/nse/cgal-pybind
$> cd cgal-pybind
$> git submodule init
$> git submodule update
$> export CGAL_DIR=$PATH_TO_GIT
$> pip install .
```

## Tests
```bash
$> pip install tox
$> tox
```

## Requirements
* cmake > 3.0.9
* C++ compiler (with C++17 support)
* Boost
* CGAL header
* Eigen3

## Unit tests requirement:
* trimesh Python package

## Developer instructions
Before each `git commit`, execute `tox -e format`. This will format code
according to the `tox.ini` configuration.