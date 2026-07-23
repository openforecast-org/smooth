OM, OMG and AutoOM
==================

Occurrence models for intermittent demand forecasting.

.. currentmodule:: smooth

OM
--

.. autoclass:: OM
   :members: fit, predict

``OM`` fits a single-component occurrence model to a binary demand-occurrence
series. The occurrence type (``"fixed"``, ``"logistic"``, ``"odds-ratio"``,
``"inverse-odds-ratio"``, ``"general"``) can be set explicitly or selected
automatically with ``occurrence="auto"``. When ``occurrence="auto"``, ``.fit()``
returns the best ``OM`` or ``OMG`` instance directly.

.. code-block:: python

   from smooth import OM
   import numpy as np

   # Binary occurrence series (1 = demand observed, 0 = zero demand)
   y = np.array([0, 1, 0, 0, 1, 1, 0, 1] * 10, dtype=float)

   # Explicit occurrence type
   model = OM(model="ANN", occurrence="logistic", lags=[1])
   model.fit(y)
   print(model)

   # Automatic type selection — returns best OM or OMG
   best = OM(model="ANN", occurrence="auto", lags=[1]).fit(y)
   print(best.occurrence)
   fc = best.predict(h=8)

OMG
---

.. autoclass:: OMG
   :members: fit, predict

``OMG`` is the general two-component occurrence model with separate A-side and
B-side ETS models. It is more flexible than a single ``OM`` but requires more
data to estimate reliably.

.. code-block:: python

   from smooth import OMG

   model = OMG(model="ANN", lags=[1])
   model.fit(y)
   print(model)

   fc = model.predict(h=8)
   print(fc.mean)

   # Access component diagnostics
   print(model.fitted)
   print(model.b_value)

actuals semantics
~~~~~~~~~~~~~~~~~

- ``OMG.actuals`` returns the **binary occurrence indicator** built from the
  raw input series (``(y != 0).astype(float)``), with the same shape as the
  series passed to ``fit``. Mirrors ``OM.actuals`` and R's ``actuals.omg``.
- ``OMG.model_a.actuals`` / ``OMG.model_b.actuals`` return the **latent
  unobservable** value the sub-model was implicitly fitting before the link
  function, *not* the binary indicator: ``fitted + residuals``. OM stores
  residuals additively (``ot - fitted``) regardless of error type, so the
  same formula recovers the latent value for both ``Etype="A"`` and
  ``Etype="M"`` sub-models. This is the reconstruction useful for diagnosing
  the sub-model in isolation (e.g. inspecting how well its continuous
  component fits its target).

AutoOM
------

.. autoclass:: AutoOM
   :members: fit

``AutoOM`` tests multiple occurrence types and returns the best ``OM`` or
``OMG`` instance, selected by information criterion (default: AICc).

.. code-block:: python

   from smooth import AutoOM

   # Test a subset of occurrence types
   model = AutoOM(model="ANN", lags=[1],
                  occurrence=["logistic", "odds-ratio", "general"])
   best = model.fit(y)
   print(best.occurrence)    # e.g. "odds-ratio"
   print(best.model_name)

   # Default: all supported types
   best = AutoOM(model="ANN", lags=[1]).fit(y)
   fc = best.predict(h=8)

See Also
--------

- :class:`ADAM` — Full ETS + ARIMA model (the demand-size component)
- :class:`AutoADAM` — Automatic ETS + distribution + ARIMA selection
- :class:`AutoMSARIMA` — Automatic pure-ARIMA selection
