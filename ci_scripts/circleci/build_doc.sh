#!/bin/bash

sudo -E apt-get -yq remove texlive-binaries --purge
sudo apt-get update
sudo apt-get install libatlas-dev libatlas3gf-base
sudo apt-get install build-essential python-dev python-setuptools
# install numpy first as it is a compile time dependency for other packages
# Installing required packages for `make -C doc check command` to work.
sudo -E apt-get -yq update
sudo -E apt-get -yq --no-install-suggests --no-install-recommends --force-yes install dvipng texlive-latex-base texlive-latex-extra

# Install dependencies with miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
   -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
export PATH="$MINICONDA_PATH/bin:$PATH"
conda update --yes --quiet conda

# Configure the conda environment and put it in the path using the
# provided versions
conda create -n $CONDA_ENV_NAME --yes --quiet python="${PYTHON_VERSION:-*}" \
  setuptools numpy scipy cython nose coverage matplotlib sphinx pillow \
  sphinx_rtd_theme numpydoc scikit-learn

source activate testenv
pip install sphinx-gallery
python setup.py clean
python setup.py develop
# The pipefail is requested to propagate exit code
set -o pipefail && cd doc && make html 2>&1 | tee ~/log.txt