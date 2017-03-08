from setuptools import setup

import bnp

setup(
    name='bnp',
    version=bnp.__version__,
    url='https://github.com/chyikwei/bnp/',
    install_requires=[
        'numpy>=1.11.2',
        'scipy>=0.18.1',
        'scikit-learn>=0.18.1'
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    packages=['bnp', 'bnp.utils'],
    author='Chyi-Kwei Yau',
    author_email='chyikwei.yau@gmail.com',
    description='Bayesian Nonparametric model implementation with Python'
)