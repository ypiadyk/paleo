from distutils.core import setup, Extension
import numpy

# To install SWIG:
# sudo apt install swig

# To compile/install labeler:
# python3 setup.py build_ext --inplace

setup(ext_modules=[Extension("_matcher_module",
      sources=["matcher.c", "matcher.i"],
      extra_compile_args=["/O2"],
      include_dirs=[numpy.get_include()])])
