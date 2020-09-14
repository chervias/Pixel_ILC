from distutils.core import setup, Extension
from distutils.sysconfig import get_python_lib
import os

numpy_inc = os.path.join(get_python_lib(plat_specific=1), 'numpy/core/include')

module1 =  Extension('PixelILC',
sources = ['source/pixel_ILC.c','source/pixel_ILC_mod.c','source/query_disc_wrapper.cpp'],
include_dirs = [numpy_inc,'source','/home/chervias/Software/anaconda3/envs/healpy/include/healpix_cxx/'],
libraries=['gsl','gslcblas','fftw3','healpix_cxx','cxxsupport','sharp','fftpack','c_utils','cfitsio'],
library_dirs = ["lib"],
extra_compile_args=['-fPIC','-Wall','-g'])

setup (name = 'PixelILC',
	   version = '0.1',
	   url='',
	   description = '',
	   ext_modules = [module1]
	  )
