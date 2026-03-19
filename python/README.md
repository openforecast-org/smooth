# smooth (Python)

[![PyPI version](https://img.shields.io/pypi/v/smooth.svg)](https://pypi.org/project/smooth/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/smooth.svg)](https://pypi.org/project/smooth/)
[![Python CI](https://github.com/config-i1/smooth/actions/workflows/python_ci.yml/badge.svg)](https://github.com/config-i1/smooth/actions/workflows/python_ci.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/smooth.svg)](https://pypi.org/project/smooth/)
[![License: LGPL-2.1](https://img.shields.io/badge/License-LGPL--2.1-blue.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)


Python implementation of the **smooth** package for time series forecasting using Single Source of Error (SSOE) state-space models.

![hex-sticker of the smooth package for Python](https://github.com/config-i1/smooth/blob/master/python/img/smooth-python-web.png?raw=true)


## Installation

**Python (PyPI):**
```bash
# Not yet available
```

**From GitHub (source):**
```bash
pip install "git+https://github.com/config-i1/smooth.git@master#subdirectory=python"
```

**From wheels:**
Check the wheels for your system [the latest release](https://github.com/config-i1/smooth/releases/tag/v4.4.0).

For example, for Windows with Python 3.13:
```bash
pip install https://github.com/config-i1/smooth/releases/download/v4.4.0/smooth-1.0.0-cp313-cp313-win_amd64.whl
```

See the [Installation Guide](https://github.com/config-i1/smooth/wiki/Installation) for platform-specific instructions.


## System Requirements

If installing from source, this package requires compilation of C++ extensions. Before installing, ensure you have:
- **C++ compiler** (g++, clang++, or MSVC)
- **CMake** >= 3.25
- **Armadillo** linear algebra library

## Quick Example

```python
import numpy as np
from smooth import ADAM

# Sample data
y = np.array([10, 12, 15, 13, 16, 18, 20, 19, 22, 25, 28, 30,
              11, 13, 16, 14, 17, 19, 21, 20, 23, 26, 29, 31])

# Fit ADAM model with additive error, no trend, no seasonality
model = ADAM(model="ANN")
model.fit(y)

# Generate forecasts
forecasts = model.predict(h=12)

# With seasonal component (monthly data, annual seasonality)
model = ADAM(model="ANA", lags=[1, 12])
model.fit(y)
forecasts = model.predict(h=12)
```

## Documentation

- [GitHub Wiki](https://github.com/config-i1/smooth/wiki) - Full documentation
- [ADAM](https://github.com/config-i1/smooth/wiki/ADAM) - Main unified ETS/ARIMA framework
- [Installation Guide](https://github.com/config-i1/smooth/wiki/Installation) - Dependencies and troubleshooting

**Book:** Svetunkov, I. (2023). *Forecasting and Analytics with the Augmented Dynamic Adaptive Model (ADAM)*. Chapman and Hall/CRC. Online: https://openforecast.org/adam/

## See Also

- [R package on CRAN](https://cran.r-project.org/package=smooth) - Production-ready R implementation
