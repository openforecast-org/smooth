.. smooth documentation master file, created by
   sphinx-quickstart on Fri Jan 16 12:46:32 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

smooth documentation
====================

.. currentmodule:: smooth

Classes
-------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   ADAM
   AutoADAM
   ES
   SMA
   MSARIMA
   AutoMSARIMA
   OM
   OMG
   AutoOM

ADAM Methods
------------

.. autosummary::
   :toctree: _autosummary
   :template: method.rst

   ADAM.fit
   ADAM.predict
   ADAM.predict_intervals
   ADAM.select_best_model
   ADAM.summary
   ADAM.plot

ADAM Diagnostics
----------------

.. autosummary::
   :toctree: _autosummary
   :template: method.rst

   ADAM.rstandard
   ADAM.rstudent
   ADAM.outlierdummy

AutoADAM Methods
----------------

.. autosummary::
   :toctree: _autosummary
   :template: method.rst

   AutoADAM.fit
   AutoADAM.predict

ES Methods
----------

.. autosummary::
   :toctree: _autosummary
   :template: method.rst

   ES.fit
   ES.predict
   ES.predict_intervals
   ES.select_best_model
   ES.summary

SMA Methods
-----------

.. autosummary::
   :toctree: _autosummary
   :template: method.rst

   SMA.fit
   SMA.predict
   SMA.predict_intervals
   SMA.summary

MSARIMA Methods
---------------

.. autosummary::
   :toctree: _autosummary
   :template: method.rst

   MSARIMA.fit
   MSARIMA.predict

AutoMSARIMA Methods
-------------------

.. autosummary::
   :toctree: _autosummary
   :template: method.rst

   AutoMSARIMA.fit
   AutoMSARIMA.predict

OM Methods
----------

.. autosummary::
   :toctree: _autosummary
   :template: method.rst

   OM.fit
   OM.predict

OMG Methods
-----------

.. autosummary::
   :toctree: _autosummary
   :template: method.rst

   OMG.fit
   OMG.predict

AutoOM Methods
--------------

.. autosummary::
   :toctree: _autosummary
   :template: method.rst

   AutoOM.fit

Utility Functions
-----------------

- :doc:`msdecompose` - Multiple seasonal decomposition for time series
- :doc:`lowess` - LOWESS (Locally Weighted Scatterplot Smoothing)

Optimization Settings
---------------------

The ADAM and ES classes use the NLopt library for parameter optimization. You can
customize the optimization behavior via the ``nlopt_kwargs`` parameter:

.. code-block:: python

   from smooth import ADAM

   model = ADAM(
       model="AAN",
       nlopt_kwargs={
           "print_level": 1,        # Print optimization progress
           "xtol_rel": 1e-8,        # Relative parameter tolerance
           "algorithm": "NLOPT_LN_SBPLX"  # Use Subplex algorithm
       }
   )
   model.fit(y)

**Available parameters:**

+--------------+--------------------------------------------------------------+--------------------+
| Parameter    | Description                                                  | Default            |
+==============+==============================================================+====================+
| print_level  | Verbosity level. When >0, prints B and CF on every           | 0                  |
|              | iteration.                                                   |                    |
+--------------+--------------------------------------------------------------+--------------------+
| xtol_rel     | Relative tolerance on parameters. Stops when changes         | 1e-6               |
|              | < xtol_rel * \|params\|.                                     |                    |
+--------------+--------------------------------------------------------------+--------------------+
| xtol_abs     | Absolute tolerance on parameters. Stops when changes         | 1e-8               |
|              | < xtol_abs.                                                  |                    |
+--------------+--------------------------------------------------------------+--------------------+
| ftol_rel     | Relative tolerance on cost function. Stops when changes      | 1e-8               |
|              | < ftol_rel * \|CF\|.                                         |                    |
+--------------+--------------------------------------------------------------+--------------------+
| ftol_abs     | Absolute tolerance on cost function. Stops when changes      | 0                  |
|              | < ftol_abs.                                                  |                    |
+--------------+--------------------------------------------------------------+--------------------+
| algorithm    | NLopt algorithm name. Use "LN\_" prefix for derivative-free. | NLOPT_LN_NELDERMEAD|
|              | Options: NLOPT_LN_NELDERMEAD, NLOPT_LN_SBPLX,                |                    |
|              | NLOPT_LN_COBYLA, NLOPT_LN_BOBYQA.                            |                    |
+--------------+--------------------------------------------------------------+--------------------+

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   api
   autoadam
   sma
   msarima
   om
   msdecompose
   lowess
