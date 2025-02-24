"""
╔═════════════════════════════════════════════════════╗
║                   File: setup.py                    ║
╠═════════════════════════════════════════════════════╣
║           Description: DreamDIA installation        ║
╠═════════════════════════════════════════════════════╣
║                Author: Wenxian Yang                 ║
║           Contact: mingxuan.gao@utoronto.ca         ║
╚═════════════════════════════════════════════════════╝
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

ld = Extension(name="tools_cython", sources=["tools_cython.pyx"])
setup(ext_modules=cythonize(ld),include_dirs=[np.get_include()])