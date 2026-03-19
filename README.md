# smooth

[![License: LGPL-2.1](https://img.shields.io/badge/License-LGPL--2.1-blue.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)

R:

[![CRAN_Status_Badge](https://www.r-pkg.org/badges/version/smooth)](https://cran.r-project.org/package=smooth)
[![Downloads](https://cranlogs.r-pkg.org/badges/smooth)](https://cran.r-project.org/package=smooth)
[![R-CMD-check](https://github.com/config-i1/smooth/actions/workflows/test.yml/badge.svg)](https://github.com/config-i1/smooth/actions/workflows/test.yml)

Python:

[![PyPI version](https://img.shields.io/pypi/v/smooth.svg)](https://pypi.org/project/smooth/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/smooth.svg)](https://pypi.org/project/smooth/)
[![Python CI](https://github.com/config-i1/smooth/actions/workflows/python_ci.yml/badge.svg)](https://github.com/config-i1/smooth/actions/workflows/python_ci.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/smooth.svg)](https://pypi.org/project/smooth/)


The **smooth** package implements Single Source of Error (SSOE) state-space models for forecasting and time series analysis, available for both R and Python.

![hex-sticker of the smooth package for R](https://github.com/config-i1/smooth/blob/master/man/figures/smooth-web.png?raw=true) ![hex-sticker of the smooth package for Python](https://github.com/config-i1/smooth/blob/master/python/img/smooth-python-web.png?raw=true)


## Installation

**R (CRAN):**
```r
install.packages("smooth")
```

**R (github):**
```r
if (!require("remotes")) install.packages("remotes")
remotes::install_github("config-i1/smooth")
```


**Python (PyPI):**
```bash
# Not yet available
```

**Python (github, release):**
Choose the wheel file for your platform from the [latest release](https://github.com/config-i1/smooth/releases/tag/v4.4.0) then run:
```bash
pip install <wheel-file-for-your-platform>
```
For example, for Windows with Python 3.13:

```bash
pip install https://github.com/config-i1/smooth/releases/download/v4.4.0/smooth-1.0.0-cp313-cp313-win_amd64.whl
```


**Python (github, dev):**
```bash
pip install "git+https://github.com/config-i1/smooth.git@master#subdirectory=python"
```

For development versions and system requirements, see the [Installation wiki page](https://github.com/config-i1/smooth/wiki/Installation).

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

Full documentation is available on the **[GitHub Wiki](https://github.com/config-i1/smooth/wiki)**, including:

- [ADAM](https://github.com/config-i1/smooth/wiki/ADAM) - Main unified ETS/ARIMA framework
- [Function reference](https://github.com/config-i1/smooth/wiki) - All functions and methods
- [Installation guide](https://github.com/config-i1/smooth/wiki/Installation) - Dependencies and troubleshooting

**Book:** Svetunkov, I. (2023). *Forecasting and Analytics with the Augmented Dynamic Adaptive Model (ADAM)*. Chapman and Hall/CRC. Online: https://openforecast.org/adam/
