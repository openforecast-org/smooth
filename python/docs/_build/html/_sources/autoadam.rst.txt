AutoADAM
========

Automatic ADAM model selection with distribution and ARIMA order selection.

.. currentmodule:: smooth

.. autoclass:: AutoADAM
   :members: fit, predict, rstandard, rstudent, outlierdummy

Overview
--------

``AutoADAM`` extends :class:`ADAM` with automatic selection of:

1. **Distribution** — tests multiple error distributions and selects by IC
2. **ARIMA orders** — when ``arima_select=True`` (default), selects AR/I/MA orders
3. **Outlier detection** — optionally detects and includes outlier dummies

Example Usage
-------------

Distribution selection:

.. code-block:: python

   from smooth import AutoADAM

   model = AutoADAM(model="ZZZ",
                    distribution=["dnorm", "dlaplace", "ds"])
   model.fit(y)
   print(model)

With ARIMA order selection:

.. code-block:: python

   model = AutoADAM(model="ZZZ",
                    lags=[1, 12],
                    ar_order=[3, 1],
                    i_order=[2, 1],
                    ma_order=[3, 1])
   model.fit(y)
   fc = model.predict(h=24)

With outlier detection:

.. code-block:: python

   model = AutoADAM(model="ZZZ", lags=[1, 12],
                    outliers="use", level=0.99)
   model.fit(y)

Diagnostics after fitting:

.. code-block:: python

   # Standardised residuals
   std_res = model.rstandard()

   # Studentised residuals
   stud_res = model.rstudent()

   # Outlier dummy variables (if outliers detected)
   dummies = model.outlierdummy()

See Also
--------

- :class:`ADAM` — Base ADAM class
- :class:`AutoMSARIMA` — Automatic pure-ARIMA selection
- :doc:`msdecompose` — Multiple seasonal decomposition
