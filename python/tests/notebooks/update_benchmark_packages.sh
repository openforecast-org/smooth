#!/usr/bin/env bash
set -e

PIP="${PIP:-pip}"

$PIP install --upgrade \
    numpy \
    pandas \
    statsforecast \
    sktime \
    skforecast \
    aeon \
    fcompdata
