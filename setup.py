from __future__ import print_function

import os
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy
import bnp

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)


libraries = []
if os.name == 'posix':
    libraries.append('m')


extensions = [
    Extension("bnp.utils.fast_expectation",
              ["bnp/utils/fast_expectation.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries),
    Extension("bnp.utils.extmath",
              ["bnp/utils/extmath.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries),
]

setup(
    name='bnp',
    version=bnp.__version__,
    url='https://github.com/chyikwei/bnp/',
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': build_ext},
    test_suite='nose.collector',
    tests_require=['nose'],
    install_requires=INSTALL_REQUIRES,
    packages=['bnp', 'bnp.utils'],
    author='Chyi-Kwei Yau',
    author_email='chyikwei.yau@gmail.com',
    description='Bayesian Nonparametric models with Python'
)
