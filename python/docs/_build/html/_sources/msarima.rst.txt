MSARIMA and AutoMSARIMA
=======================

Multiple Seasonal ARIMA with fixed or automatically-selected orders.

.. currentmodule:: smooth

MSARIMA
-------

.. autoclass:: MSARIMA
   :members: fit, predict

``MSARIMA`` fits a pure ARIMA model (no ETS components) with explicitly
specified orders.

.. code-block:: python

   from smooth import MSARIMA

   # ARIMA(0,1,1) — default
   model = MSARIMA()
   model.fit(y)
   print(model)

   # SARIMA(1,1,1)(1,1,1)[12]
   model = MSARIMA(
       orders={"ar": [1, 1], "i": [1, 1], "ma": [1, 1]},
       lags=[1, 12],
   )
   model.fit(y)
   fc = model.predict(h=12, interval="prediction", level=0.95)

   # With drift term
   model = MSARIMA(ar_order=1, i_order=1, ma_order=1, constant=True)
   model.fit(y)

AutoMSARIMA
-----------

.. autoclass:: AutoMSARIMA
   :members: fit, predict

``AutoMSARIMA`` wraps :class:`AutoADAM` with ``model="NNN"`` and
``distribution="dnorm"`` fixed, mirroring R's ``auto.msarima()``.
The parameters ``model``, ``distribution``, and ``arima_select`` are fixed
and cannot be overridden — passing them raises ``ValueError``.

.. code-block:: python

   from smooth import AutoMSARIMA

   # Automatic seasonal ARIMA
   model = AutoMSARIMA(lags=[1, 12])
   model.fit(y)
   print(model)   # AutoMSARIMA: ARIMA([p,P],[d,D],[q,Q])

   # Reduce search space for speed
   model = AutoMSARIMA(
       lags=[1, 12],
       ar_order=[2, 1],
       i_order=[2, 1],
       ma_order=[2, 1],
   )
   model.fit(y)
   fc = model.predict(h=24)

   # With external regressors
   model = AutoMSARIMA(lags=[1, 12], regressors="select")
   model.fit(y, X=X)

See Also
--------

- :class:`AutoADAM` — Full automatic ETS + ARIMA + distribution selection
- :class:`ADAM` — Base unified framework
- :doc:`msdecompose` — Multiple seasonal decomposition
