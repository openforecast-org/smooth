# Python: estimator Module

**Path**: python/src/smooth/adam_general/core/estimator/

## Structure

| File | Purpose |
|------|---------|
| estimator.py | estimator() |
| selector.py | selector(), _form_model_pool, _estimate_model, _run_branch_and_bound |
| optimization.py | _setup_optimization_parameters, _create_objective_function, _run_optimization |
| two_stage.py | _run_two_stage_estimator |
| initial_values.py | _process_initial_values |

## estimator

1. initialiser() for B, Bl, Bu
2. _setup_arima_polynomials if ARIMA
3. _create_objective_function wraps CF
4. _run_optimization (NLopt)
5. _calculate_loglik
6. _generate_forecasts (fitted values)
7. Returns adam_estimated dict

## selector

For model="ZZZ", "XXX", etc.:
- _form_model_pool or _build_models_pool_from_components
- _estimate_model for each candidate (or _run_branch_and_bound)
- Select by ic_function (AICc, etc.)

## optimization

- _configure_optimizer: NLopt setup
- _create_objective_function: Wraps CF, handles penalties
- _run_optimization: opt.optimize

## R Mapping

- R optimization in adam() (nloptr/nlminb) → optimization._run_optimization
- R model selection (adam() when modelDo=="select") → selector
