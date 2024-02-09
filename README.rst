Overview:
=========

Some CGAL_ Python bindings built with pybind11_.
This includes some auxiliary functions, as well, as mentioned in the CGAL-free portion.

CGAL Methods:
=============

* Skeletonization/contraction
* Segmentation
* Discrete authalic parametrization of surface mesh

BBP Atlas tools (CGAL-free)
===========================

* Volume slicer (used to split the layer 2/3 of the AIBS mouse isocortex)
* Streamlines intersector (used for flat mapping of brain regions)
* Thickness estimator (used for placement hints computation)
* Utils: a 3D vector field interpolator (trilinear interpolation)

Installation
============

If one is on a Linux platform, one should be able to use the compiled Python wheels.
This is the recommended way.
Otherwise, use the developer instructions to try and install it that way.

.. code-block:: bash

    $ pip install cgal-pybind

Examples
========

Several examples are available in the rendered ipython notebook:
https://github.com/BlueBrain/cgal-pybind/blob/main/examples/Examples.ipynb

Tests
=====

.. code-block:: bash

    pip install tox
    tox

Requirements
============

* cmake > 3.0.9
* C++ compiler (with C++17 support)
* Boost
* CGAL header
* Eigen3

Unit tests requirement:
=======================

* trimesh Python package

Developer instructions
======================

It is assumed that the develop has installed the C++ requirements as noted above.

CGAL_ in header only format can be used, but it still will require dependencies.
These include things like *eigen3-devel*, *gmp-devel*, *mpfr-devel* on RedHat like Linux systems.
On modern Debian systems, *libcgal-dev* and *libeigen3-dev*  can be used instead of downloading the CGAL library.
More information can be found on the CGAL_ installation getting started https://doc.cgal.org/latest/Manual/usage.html

.. code-block:: bash

    # if CGAL header-only is used, if using system packages, skip the following two lines.
    git clone https://github.com/CGAL/cgal.git $TARGET_CGAL_DIRECTORY
    export CGAL_DIR=$TARGET_CGAL_DIRECTORY

    git clone https://github.com/BlueBrain/cgal-pybind
    cd cgal-pybind
    git submodule init
    git submodule update
    pip install .



Acknowledgements
================

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

For license and authors, see LICENSE.txt and AUTHORS.txt respectively.

Copyright (c) 2022-2024 Blue Brain Project/EPFL

.. _CGAL: https://www.cgal.org/
.. _pybind11: https://pybind11.readthedocs.io
