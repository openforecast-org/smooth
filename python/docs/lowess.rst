lowess
======

LOWESS (Locally Weighted Scatterplot Smoothing) for robust nonparametric regression.

.. currentmodule:: smooth

.. autofunction:: smooth.lowess

Overview
--------

LOWESS is a nonparametric regression method that combines polynomial regression
with local weighting. It is particularly useful for:

- **Smoothing noisy data** while preserving local patterns
- **Robust estimation** that is resistant to outliers
- **Exploratory data analysis** to reveal underlying trends

Output is deterministic: the returned ``x`` values are sorted in ascending
order and ``y`` contains the smoothed values aligned to that sorted ``x``.

Example Usage
-------------

Basic smoothing:

.. code-block:: python

   from smooth import lowess
   import numpy as np

   # Generate noisy data
   x = np.linspace(0, 2*np.pi, 50)
   y = np.sin(x) + np.random.randn(50) * 0.3

   # Apply LOWESS smoothing
   result = lowess(x, y)

   # Access smoothed values
   x_smooth = result['x']  # Sorted x values
   y_smooth = result['y']  # Smoothed y values

Adjusting smoothness:

.. code-block:: python

   # More smoothing (larger span)
   result_smooth = lowess(x, y, f=0.8)

   # Less smoothing (smaller span)
   result_rough = lowess(x, y, f=0.2)

Handling outliers:

.. code-block:: python

   # Add outliers
   y_outliers = y.copy()
   y_outliers[10] = 5  # Outlier

   # LOWESS is robust to outliers due to iterative reweighting
   result = lowess(x, y_outliers, iter=3)  # Default iterations

   # More iterations for heavily contaminated data
   result_robust = lowess(x, y_outliers, iter=5)

Using 2D input (single array containing both x and y):

.. code-block:: python

   # Combine x and y into 2D array
   xy = np.column_stack([x, y])

   # Call with single argument
   result = lowess(xy)

Parameters
----------

**x** : array-like
    X values. Can be 1D array or 2D array with x in first column. *(required)*

**y** : array-like, optional
    Y values. Optional if x is 2D array containing both x and y. Default: ``None``

**f** : float, optional
    Smoother span (fraction of points). Larger values = smoother. Default: ``2/3``

**iter** : int, optional
    Number of robustifying iterations. More = more robust. Default: ``3``

**delta** : float, optional
    Distance threshold for interpolation. Points within delta are treated as
    the same point. Default: ``0.01 * range(x)``

Returns
-------

The function returns a dictionary with two keys:

**x** : ndarray
    Sorted x values.

**y** : ndarray
    Smoothed y values corresponding to sorted x.

Algorithm
---------

LOWESS uses Cleveland's (1979) algorithm:

1. **Local Fitting**: At each point, fit a weighted linear regression using
   nearby points. Weights decrease with distance using a tricube function.

2. **Robustness Iterations**: Recompute weights based on residuals to
   downweight outliers. Repeat ``iter`` times.

3. **Interpolation**: For efficiency, only compute fits at a subset of
   points and interpolate between them (controlled by ``delta``).

The tricube weight function is:

.. math::

   w(u) = (1 - |u|^3)^3 \quad \text{for } |u| < 1

References
----------

Cleveland, W.S. (1979) "Robust Locally Weighted Regression and Smoothing
Scatterplots". *Journal of the American Statistical Association* 74(368): 829-836.

See Also
--------

- :doc:`msdecompose` - Uses LOWESS internally for trend extraction
