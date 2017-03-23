#from distutils.core import setup
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy
import bnp

extensions = [
    Extension("bnp.utils.fast_expectation",
              ["bnp/utils/fast_expectation.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=["m"]),
]

setup(
    name='bnp',
    version=bnp.__version__,
    url='https://github.com/chyikwei/bnp/',
    install_requires=[
        'numpy>=1.11.2',
        'scipy>=0.18.1',
        'scikit-learn>=0.18.1',
        'Cython>=0.20.2'
    ],
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': build_ext},
    test_suite='nose.collector',
    tests_require=['nose'],
    packages=['bnp', 'bnp.utils'],
    author='Chyi-Kwei Yau',
    author_email='chyikwei.yau@gmail.com',
    description='Bayesian Nonparametric model implementation with Python'
)
