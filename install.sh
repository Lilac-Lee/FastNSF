#!/usr/bin/env bash

ENVIRONMENT_FILE=requirements.yml

echo "Using ${ENVIRONMENT_FILE} ..."

# Ensure mamba is installed.
conda install -y mamba

# Create library environment.
mamba env create -f ${ENVIRONMENT_FILE} \
&& eval "$(conda shell.bash hook)" \
&& conda activate fast_nsf

# NOTE: install pip dependencies that require special arguments
pip install FastGeodis --no-build-isolation

# TODO: make code a package
