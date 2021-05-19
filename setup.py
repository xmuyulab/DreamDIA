from distutils.core import setup, Extension
from Cython.Build import cythonize

ld = Extension(name="tools_cython", sources=["tools_cython.pyx"])
setup(ext_modules=cythonize(ld),include_dirs=[np.get_include()])