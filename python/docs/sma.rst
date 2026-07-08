SMA
===

Simple moving average in single-source-of-error state space form, with
optional automatic order selection.

.. currentmodule:: smooth

.. autoclass:: SMA
   :members: fit, predict

Overview
--------

``SMA(m)`` is the classical simple moving average expressed as an AR(``m``)
state-space model with every autoregressive coefficient fixed at ``1/m``.
Wrapping the moving average inside the ADAM framework — rather than treating
it as a rolling window — gives several practical advantages:

- **Multi-step forecasts** are produced recursively from the state vector,
  not by repeating the last observation.
- **Forecast variance and prediction intervals** follow directly from the
  state-space recursion, so intervals widen with the horizon as they should.
- **Automatic order selection** is available: when ``order`` is left
  unspecified, ``SMA`` evaluates a range of orders and picks the one that
  minimises an information criterion (default ``AICc``).
- **Full ADAM API** is inherited — ``.fit()``, ``.predict()``, residual
  diagnostics, and the ``ForecastResult`` return type all behave exactly as
  for :class:`ADAM`.

Internally ``SMA`` calls :class:`ADAM` with ``model="NNN"``,
``ar_order=m``, ``arma={"ar": [1/m] * m}``, ``initial="backcasting"``,
``loss="MSE"``, and ``distribution="dnorm"``. These are fixed and cannot be
overridden — use :class:`ADAM` or :class:`MSARIMA` directly if you need
arbitrary AR coefficients.

Example Usage
-------------

Fixed order:

.. code-block:: python

   import numpy as np
   from smooth import SMA

   y = np.cumsum(np.random.randn(60)) + 100

   model = SMA(order=4, h=5)
   model.fit(y)
   print(model)              # e.g. "SMA(4)"

   fc = model.predict(h=5, interval="prediction", level=0.95)
   fc.mean
   fc.lower, fc.upper

Automatically selected order:

.. code-block:: python

   model = SMA(h=5)          # order=None → auto-select
   model.fit(y)
   print(model.model)        # e.g. "SMA(3)"
   print(model.ICs_)         # {1: ic_1, 2: ic_2, ...} for evaluated orders

With holdout for validation:

.. code-block:: python

   model = SMA(h=12, holdout=True)
   model.fit(y)              # last 12 observations reserved as test set
   fc = model.predict(h=12)

Order selection
---------------

When ``order`` is ``None``, ``SMA`` searches over orders
``1 … min(200, T)`` using the criterion given by ``ic`` (one of ``"AIC"``,
``"AICc"``, ``"BIC"``, ``"BICc"``). Two search modes are available:

- ``fast=True`` (default) — **ternary search**, which converges to a local
  IC minimum without evaluating every order. Fast on long series. When the
  input is a pandas ``Series`` with a ``DatetimeIndex``, the inferred
  seasonal period is also evaluated explicitly so seasonal moving averages
  are never missed.
- ``fast=False`` — **sequential scan** of every candidate order. Slower but
  guaranteed to find the global minimum within the search range.

The IC values from the search are stored on the fitted object as ``ICs_``
(a dict keyed by order).

References
----------

Svetunkov, I., & Petropoulos, F. (2017). *Old dog, new tricks: a modelling
view of simple moving averages.* International Journal of Production
Research. https://doi.org/10.1080/00207543.2017.1380326

See Also
--------

- :class:`ADAM` — Underlying unified state-space framework
- :class:`MSARIMA` — General multiple-seasonal ARIMA wrapper
- :class:`ES` — Exponential smoothing
