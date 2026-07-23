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
2. **ARIMA orders** — when ``arima_select=True`` or ``orders={..., "select": True}``
   (see ARIMA precedence below), selects AR/I/MA orders
3. **Outlier detection** — optionally detects and includes outlier dummies

ARIMA orders precedence
-----------------------

``AutoADAM`` (and :class:`ADAM`) share the following rule for resolving the
ARIMA-order arguments:

- If ``orders`` (dict) is supplied, it is used and the three scalar arguments
  ``ar_order`` / ``i_order`` / ``ma_order`` are **ignored** (a ``UserWarning``
  is emitted). Order selection runs iff ``orders.get("select", arima_select)``
  is true.
- Otherwise, if any of ``ar_order`` / ``i_order`` / ``ma_order`` has a
  non-zero value, those are used. They are interpreted as **upper search
  bounds** when ``arima_select=True``, or as **fixed** orders when
  ``arima_select=False`` (the default).
- With no ARIMA spec at all (``orders=None`` and all three scalars zero/None),
  the model is pure ETS and no ARIMA selection is performed.

``lags`` accepts either a scalar (``lags=12``) or a list (``lags=[12]``).

Verbose progress
----------------

When constructed with ``verbose=1``, ``AutoADAM`` prints the distribution
loop's progress in real time:

.. code-block:: text

   Evaluating models with different distributions... dnorm, dlaplace, ds, Done!
   Selected distribution: dlaplace
   Selected ARIMA orders: AR=[1], I=[1], MA=[1]

Example Usage
-------------

Distribution selection:

.. code-block:: python

   from smooth import AutoADAM

   model = AutoADAM(model="ZZZ",
                    distribution=["dnorm", "dlaplace", "ds"])
   model.fit(y)
   print(model)

With ARIMA order selection (using the ``orders`` dict — recommended):

.. code-block:: python

   model = AutoADAM(model="ZZZ",
                    lags=[1, 12],
                    orders={"ar": [3, 1], "i": [2, 1], "ma": [3, 1], "select": True})
   model.fit(y)
   fc = model.predict(h=24)

Equivalent using scalar bounds with ``arima_select=True``:

.. code-block:: python

   model = AutoADAM(model="ZZZ",
                    lags=[1, 12],
                    ar_order=[3, 1],
                    i_order=[2, 1],
                    ma_order=[3, 1],
                    arima_select=True)   # required: default is False
   model.fit(y)

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
