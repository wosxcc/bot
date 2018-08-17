
##把pyx文件变成pyd文件后直接放到python的环境下
import sys
sys.path.insert(0, "..")

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
# ext_module = cythonize("TestOMP.pyx")
ext_module = Extension(
    "bbox",
    ["bbox.pyx"],
    extra_compile_args=["/openmp"],
    extra_link_args=["/openmp"],

)

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_module],
    include_dirs=[np.get_include()]
)
