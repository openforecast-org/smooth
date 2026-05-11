# Python: utils Module

**Path**: python/src/smooth/adam_general/core/utils/

## Structure

| File | Purpose |
|------|---------|
| cost_functions.py | CF, log_Lik_ADAM |
| ic.py | ic_function (AIC, AICc, BIC, BICc) |
| var_covar.py | sigma, covar_anal, var_anal, matrix_power_wrap |
| utils.py | msdecompose, lowess_r, calculate_likelihood, scaler |
| polynomials.py | adam_polynomialiser |
| distributions.py | rlaplace, rs, rgnorm, generate_errors |
| n_param.py | create_n_param, count_internal_params |
| printing.py | model_summary, _format_* |

## cost_functions.CF

Main cost function for optimization. Calls filler(B), adam_fitter(), returns loss (e.g. -logLik or MSE).

## var_covar

- sigma: Scale parameter from residuals
- covar_anal: Analytical covariance (R covarAnal)
- var_anal: Analytical variance (R adamVarAnal)
- matrix_power_wrap: Matrix power for variance (R matrixPowerWrap)

## utils

- msdecompose: Multiple seasonal decomposition (R msdecompose)
- lowess_r: R-compatible LOWESS
- scaler: Error scaling by distribution

## R Mapping

- R likelihoodFunction → log_Lik_ADAM / CF
- R ICFunction → ic_function
- R covarAnal, adamVarAnal → covar_anal, var_anal
- R adam_polynomialiser (C++) → adam_polynomialiser (polynomials.py)
