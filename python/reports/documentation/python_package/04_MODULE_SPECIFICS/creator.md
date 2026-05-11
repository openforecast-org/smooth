# Python: creator Module

**Path**: python/src/smooth/adam_general/core/creator/

## Structure

| File | Purpose |
|------|---------|
| architector.py | architector, adam_profile_creator |
| creator.py | creator, _setup_matrices, _setup_measurement_vector, _setup_persistence_vector |
| filler.py | filler |
| initialiser.py | initialiser |
| initialization.py | _initialize_states, _initialize_ets_states, _initialize_arima_states |

## architector

Inputs: model_type_dict, arima_checked, lags_dict, etc.  
Outputs: components_dict, lags_dict (updated), matrices_dict (empty template), profiles.

Calls: _setup_components, _setup_lags, _create_profiles, adam_profile_creator.

## creator

Inputs: model_type_dict, lags_dict, observations_dict, explanatory_checked, etc.  
Builds: mat_vt, mat_wt, mat_f, vec_g, index_lookup_table, profiles_recent_table.

Calls: _extract_model_parameters, _setup_matrices, _setup_measurement_vector, _setup_persistence_vector, _handle_polynomial_setup, _initialize_states.

## filler

Inputs: B, matrices_dict, model_type_dict, lags_dict, ...  
Updates matrices_dict in-place with values from B (persistence, phi, initials, ARIMA, xreg).

Called by: CF (cost_functions), preparator._fill_matrices_if_needed.

## initialiser

Returns: B (initial vector), Bl (lower), Bu (upper), B_names.  
Called by estimator before optimization.

## R Mapping

- R architector (adam.R ~656) → architector
- R creator (adam.R ~750) → creator
- R filler (adam.R ~1194) → filler
- R initialiser (adam.R ~1402) → initialiser
