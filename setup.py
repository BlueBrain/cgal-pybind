import os
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

REQUIRED_TRIMESH_VERSION = "trimesh>=2.38.10"  # For unit tests only


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg,
                       '-GNinja',
                       ]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

# nearly verbatim from how h5py handles is
install_requires = [
    # we only really aim to support numpy & python combinations for which
    # there are wheels on pypi (e.g. numpy >=1.17.5 for python 3.8).
    # but we don't want to duplicate the information in oldest-supported-numpy
    # here, and if you can build an older numpy on a newer python
    # numpy 1.14.5 is the first with wheels for python 3.7, our minimum python.
    "numpy >=1.14.5",
]

extras_require = {"tests": [REQUIRED_TRIMESH_VERSION, "pytest", ]}

setup(
    name="cgal-pybind",
    url="https://github.com/BlueBrain/cgal-pybind",
    author="Blue Brain Project, EPFL",
    description="Python bindings for some CGAL classes",
    packages=[
        "cgal_pybind",
    ],
    ext_modules=[
        CMakeExtension("cgal_pybind._cgal_pybind"),
    ],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=install_requires,
    extras_require=extras_require,

    setup_requires=[
        'setuptools_scm',
    ],
    use_scm_version={
        "local_scheme": "no-local-version",
        },

    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
