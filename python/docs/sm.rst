Scale Model (sm)
================

A scale model lets the error's scale vary over time instead of being a single
number. It is a second ADAM, fitted to a transform of a fitted model's
residuals, and scored by the *original* model's log-likelihood — the
state-space counterpart of a GARCH or GAMLSS scale equation.

.. currentmodule:: smooth

Where an ordinary ADAM assumes

.. math::

   y_t = \mu_t + \sigma \varepsilon_t

with a constant :math:`\sigma`, a scale model estimates :math:`\sigma_t`
dynamically, using an ETS, ARIMA or regression structure of its own.

Quick start
-----------

.. code-block:: python

   from smooth import ADAM

   location = ADAM(model="MNN", lags=[1, 12], distribution="dnorm")
   location.fit(y)

   scale_model = location.sm()        # fit the scale model
   location.scale_model = scale_model # adopt it (R's implant())

   forecast = location.predict(h=12, interval="prediction")

Once attached, the scale model changes the model's log-likelihood, its
parameter count, and the width of its prediction intervals at every horizon.

No ``implant()`` in Python
--------------------------

R needs two steps, because it cannot modify a fitted object in place:

.. code-block:: r

   scaleModel <- sm(locationModel)
   mergedModel <- implant(locationModel, scaleModel)   # returns a new object

Python assigns instead, and there is **no** ``implant()`` function:

.. code-block:: python

   location.scale_model = location.sm()

The assignment does everything R's ``implant()`` does — it validates that what
you are attaching really is a scale model, and switches the location model's
``loglik`` and ``nparam`` over to it. Setting ``scale_model = None`` detaches
it again, which R has no equivalent for.

Because ``sm()`` returns the model rather than attaching it, you still get R's
two-step control: inspect the scale model first, adopt it only if you want it.

Requirements
------------

``sm()`` raises ``ValueError`` unless:

- the location model was estimated with ``loss="likelihood"``; and
- its distribution is one of ``dnorm``, ``dlaplace``, ``ds``, ``dgnorm``,
  ``dlnorm``, ``dgamma``, ``dinvgauss``.

``dalaplace`` and the log-variants are not supported — R's own loss function
has no branch for them either, so they are rejected rather than silently
mis-scored.

What the scale model is fitted to
---------------------------------

The response is a transform of the location model's residuals, chosen so that
its mean is the quantity that distribution calls a scale:

+---------------+--------------------------------------+
| Distribution  | Scale response                       |
+===============+======================================+
| ``dnorm``     | :math:`e_t^2`                        |
+---------------+--------------------------------------+
| ``dlaplace``  | :math:`|e_t|`                        |
+---------------+--------------------------------------+
| ``ds``        | :math:`0.5\sqrt{|e_t|}`              |
+---------------+--------------------------------------+
| ``dgnorm``    | :math:`(\beta|e_t|^\beta)^{1/\beta}` |
+---------------+--------------------------------------+
| ``dlnorm``    | :math:`\log(e_t)^2`                  |
+---------------+--------------------------------------+
| ``dgamma``    | :math:`(e_t-1)^2`                    |
+---------------+--------------------------------------+
| ``dinvgauss`` | :math:`(e_t-1)^2/e_t`                |
+---------------+--------------------------------------+

For ``dgamma``, ``dinvgauss`` and the log-domain distributions ``e_t`` is the
ratio ``y_t / fitted_t``, which is what ``residuals`` returns for them.

Reading the result
------------------

+---------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| Accessor                  | Meaning                                                                                                                        |
+===========================+================================================================================================================================+
| ``model.scale_model``     | The attached scale model, or ``None``.                                                                                         |
+---------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| ``model.extract_scale()`` | The fitted scale at each observation, mapped out of the space ``sm()`` fitted it in. A scalar when no scale model is attached. |
+---------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| ``model.extract_sigma()`` | The standard deviation that scale implies, which differs per distribution.                                                     |
+---------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| ``model.loglik``          | Switches to the scale model's once attached.                                                                                   |
+---------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| ``model.nparam``          | Gains the scale model's parameters, losing the one constant scale they replace.                                                |
+---------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| ``scale_model.fitted``    | The fitted scale, in the space it was fitted in.                                                                               |
+---------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| ``scale_model.residuals`` | Standardised residuals of the *location* model.                                                                                |
+---------------------------+--------------------------------------------------------------------------------------------------------------------------------+

``extract_scale()`` is not the same as ``scale_model.fitted``: ``sm()`` fits
squared residuals for ``dnorm``, so the scale is their square root.

Choosing a scale model
----------------------

+--------------------------------+-------------------------------+
| Pattern in the residuals       | Try                           |
+================================+===============================+
| Constant variance              | No scale model needed         |
+--------------------------------+-------------------------------+
| Slowly changing variance       | ``model.sm(model="MNN")``     |
+--------------------------------+-------------------------------+
| Seasonal variance              | ``model.sm(model="MNM")``     |
+--------------------------------+-------------------------------+
| Unsure                         | ``model.sm()`` — selects      |
|                                | among multiplicative ETS      |
+--------------------------------+-------------------------------+
| Variance driven by covariates  | ``model.sm(model="NNN", X=X)``|
+--------------------------------+-------------------------------+

A scale model adds parameters, so compare information criteria before and
after attaching one — ``model.aicc`` accounts for it automatically.

Worked example
--------------

.. code-block:: python

   import numpy as np
   from smooth import ADAM

   location = ADAM(model="ANN", lags=[1], distribution="dnorm")
   location.fit(y)

   plain_aicc = location.aicc
   plain = location.predict(h=12, interval="prediction", level=0.95)

   location.scale_model = location.sm()

   print(location.aicc - plain_aicc)          # negative: the scale model helps

   varying = location.predict(h=12, interval="prediction", level=0.95)
   width_plain = np.asarray(plain.upper) - np.asarray(plain.lower)
   width_varying = np.asarray(varying.upper) - np.asarray(varying.lower)
   # width_varying tracks the estimated scale instead of being flat

   sigma_t = location.extract_sigma()          # one value per observation

API
---

.. autosummary::
   :toctree: _autosummary
   :template: method.rst

   ADAM.sm
   ADAM.extract_scale
   ADAM.extract_sigma

Differences from R
------------------

+---------------------------------+--------------------------------+-------------------------------------+
| Behaviour                       | R                              | Python                              |
+=================================+================================+=====================================+
| Fit a scale model               | ``sm(model)``                  | ``model.sm()``                      |
+---------------------------------+--------------------------------+-------------------------------------+
| Attach it                       | ``implant(location, scale)``   | ``location.scale_model = scale``    |
+---------------------------------+--------------------------------+-------------------------------------+
| Detach it                       | —                              | ``location.scale_model = None``     |
+---------------------------------+--------------------------------+-------------------------------------+
| Is a scale model attached?      | ``is.scale(model$scale)``      | ``model.scale_model is not None``   |
+---------------------------------+--------------------------------+-------------------------------------+
| The scale                       | ``extractScale(model)``        | ``model.extract_scale()``           |
+---------------------------------+--------------------------------+-------------------------------------+
| The implied sigma               | ``extractSigma(model)``        | ``model.extract_sigma()``           |
+---------------------------------+--------------------------------+-------------------------------------+
| Where the scale lives           | ``model$scale`` holds either a | ``model.scale`` stays a float;      |
|                                 | number or a model              | the model lives in                  |
|                                 |                                | ``model.scale_model``               |
+---------------------------------+--------------------------------+-------------------------------------+
| Explanatory variables           | ``formula=`` / ``data=``       | ``X=``                              |
+---------------------------------+--------------------------------+-------------------------------------+

R overloads a single ``$scale`` slot to hold either a number or a model, and
tells them apart with ``is.scale()``. Python keeps ``scale`` a float and gives
the model its own attribute, so the return type never changes underfoot.

Diagnostics on both sides use the same numbers: the fitted scale, the
prediction-interval bounds and the standardised residuals all agree with R.

References
----------

- Svetunkov, I. (2023). *Forecasting and Analytics with the Augmented Dynamic
  Adaptive Model (ADAM)*. Chapman and Hall/CRC.
  `Chapter 17: the scale model <https://openforecast.org/adam/ADAMscaleModel.html>`_.
