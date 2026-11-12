# Analysis Workflow Map

This document maps the main analysis steps to the scripts in this repository.

## Core data processing and hierarchy construction

- `analysis/main_analysis.py`
  - Reads an authorized local workbook specified by `ANALYSIS_WORKBOOK`.
  - Uses a local ignored column map specified by `ANALYSIS_COLUMN_MAP` or `data/column_map.json`.
  - Applies adult-valid record filtering, range checks, MAU-grade parsing, urine-protein categorization, and the four-level MAU-proteinuria hierarchy.
  - Produces aggregate sample-flow, baseline, threshold-model, PPO, component, selection, wave, and renal-function summaries under `outputs/main_analysis/`.

## Threshold-specific serum uric acid association

- `analysis/main_analysis.py`
  - Fits cumulative logistic models for Phenotype >=1, >=2, and >=3.
  - Fits the partial proportional odds model used to test threshold heterogeneity.
  - Fits conventional component endpoints, including MAU>=80 mg/L and urine protein >=1+.

## Upper-threshold component analyses

- `analysis/main_analysis.py`
  - Compares Phenotype >=3 with urine protein >=1+.
  - Re-estimates Phenotype >=3 after excluding MAU150-only records.
- `analysis/mau_axis_decomposition.py`
  - Reconstructs the MAU-only axis among protein-negative or trace records.

## Renal-function context

- `analysis/main_analysis.py`
  - Calculates eGFR and fits same-visit renal-function context models.
- `analysis/renal_function_context_sensitivity.py`
  - Refits threshold, component, PPO, and IPW models after adding eGFR or serum creatinine to the adjustment set.

## Selection and IPW analyses

- `analysis/main_analysis.py`
  - Fits complete-case selection and IPW threshold models after Phenotype and serum uric acid are observed.
- `analysis/test_ordering_selection.py`
  - Summarizes stage profiles along the analysis filtering path.
- `analysis/cumulative_selection_ipw.py`
  - Runs cumulative selection/IPW sensitivity analyses across earlier observable transitions.
- `analysis/observable_selection_ipw.py`
  - Separates observable transitions into parseable urine-protein results, valid MAU-grade phenotype, serum uric acid availability, and complete covariates.
- `analysis/ipw_diagnostics.py`
  - Provides IPW diagnostics, weight trimming checks, balance summaries, and bootstrap robustness checks.

## Possible repeated records and collection-wave sensitivity

- `analysis/repeat_record_sensitivity.py`
  - Uses pseudo-cluster robust variance based on annual-wave records where exact age and height are locally available.
- `analysis/repeat_record_collapse_calibration.py`
  - Collapses annual-wave pseudo-clusters to one record under random and max-Phenotype rules.
- `analysis/main_analysis.py`
  - Runs collection-wave stratified and leave-one-wave analyses.

## Same-sample urinary context

- `analysis/same_sample_urinalysis_sensitivity.py`
  - Fits threshold models before and after conditioning on urine specific gravity, urine pH, and occult blood.
  - Fits stepwise conditioning models for Phenotype >=1.
  - Fits threshold models using raw and residualized standardized serum uric acid. Residualized models use the extended same-sample urinalysis context defined in the script output summary.
