# smooth

[![License: LGPL-2.1](https://img.shields.io/badge/License-LGPL--2.1-blue.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)

R:

[![CRAN_Status_Badge](https://www.r-pkg.org/badges/version/smooth)](https://cran.r-project.org/package=smooth)
[![Downloads](https://cranlogs.r-pkg.org/badges/smooth)](https://cran.r-project.org/package=smooth)
[![R-CMD-check](https://github.com/openforecast-org/smooth/actions/workflows/test.yml/badge.svg)](https://github.com/openforecast-org/smooth/actions/workflows/test.yml)

[![Conda version](https://img.shields.io/conda/v/conda-forge/r-smooth)](https://anaconda.org/conda-forge/r-smooth)
[![Conda downloads](https://img.shields.io/conda/dn/conda-forge/r-smooth)](https://anaconda.org/conda-forge/r-smooth)

Python:

[![PyPI version](https://img.shields.io/pypi/v/smooth.svg)](https://pypi.org/project/smooth/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/smooth.svg)](https://pypi.org/project/smooth/)
[![Python versions](https://img.shields.io/pypi/pyversions/smooth.svg)](https://pypi.org/project/smooth/)
[![Python CI](https://github.com/openforecast-org/smooth/actions/workflows/python_ci.yml/badge.svg)](https://github.com/openforecast-org/smooth/actions/workflows/python_ci.yml)
[![SLSA Build Level 3](https://slsa.dev/images/gh-badge-level3.svg)](https://slsa.dev)

Python wheels on PyPI ship with [PEP 740 attestations](https://peps.python.org/pep-0740/) — SLSA Build Level 3 provenance, signed via [Sigstore](https://www.sigstore.dev/) on the GitHub Actions runner that built them. Verifiable client-side with [`pypi-attestations`](https://pypi.org/project/pypi-attestations/).

The **smooth** package implements Single Source of Error (SSOE) state-space models for forecasting and time series analysis, available for both R and Python.

![hex-sticker of the smooth package for R](https://github.com/openforecast-org/smooth/blob/master/man/figures/smooth-web.png?raw=true) ![hex-sticker of the smooth package for Python](https://github.com/openforecast-org/smooth/blob/master/python/img/smooth-python-web.png?raw=true)

Both the R and Python versions of **smooth** depend on the [**greybox**](https://github.com/openforecast-org/greybox) package for distributions, information criteria, and supporting utilities (in Python this also provides the LOWESS smoother). It is installed automatically with **smooth**.


## Installation

**R (CRAN):**
```r
install.packages("smooth")
```

**R (github):**
```r
if (!require("remotes")) install.packages("remotes")
remotes::install_github("openforecast-org/smooth")
```


**Python (PyPI):**
```bash
pip install smooth
```

**Python (github, dev):**
```bash
pip install "git+https://github.com/openforecast-org/smooth.git@master#subdirectory=python"
```

For development versions and system requirements, see the [Installation wiki page](https://github.com/openforecast-org/smooth/wiki/Installation).

## Quick Examples

### R

```r
library(smooth)

# ADAM - the recommended function for most tasks
model <- adam(y, model="ZXZ", lags=12)
forecast(model, h=12)

# Exponential Smoothing
model <- es(y, model="ZXZ", lags=12)

# Automatic model selection for ETS+ARIMA and distributions
model <- auto.adam(y, model="ZZZ",
                   orders=list(ar=2, i=2, ma=2, select=TRUE))
```

### Python

```python
from smooth import ADAM, ES

# ADAM model
model = ADAM(model="ZXZ", lags=12)
model.fit(y)
model.predict(h=12)

# Exponential Smoothing
model = ES(model="ZXZ")
model.fit(y)
```

## Documentation

Full documentation is available on the **[GitHub Wiki](https://github.com/openforecast-org/smooth/wiki)**, including:

- [ADAM](https://github.com/openforecast-org/smooth/wiki/ADAM) - Main unified ETS/ARIMA framework
- [Function reference](https://github.com/openforecast-org/smooth/wiki) - All functions and methods
- [Installation guide](https://github.com/openforecast-org/smooth/wiki/Installation) - Dependencies and troubleshooting

**Book:** Svetunkov, I. (2023). *Forecasting and Analytics with the Augmented Dynamic Adaptive Model (ADAM)*. Chapman and Hall/CRC. Online: https://openforecast.org/adam/
