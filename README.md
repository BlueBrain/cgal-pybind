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

## Tests
```bash
$> pip install tox
$> tox
```

## Requirements
* cmake > 3.0.9
* C++ compiler (with C++11 support)
* CGAL header

## Unit tests requirement:
* trimesh Python package

## Developer instructions
Before each `git commit`, execute `tox -e format`. This will format code
according to `tox.ini` configuration.