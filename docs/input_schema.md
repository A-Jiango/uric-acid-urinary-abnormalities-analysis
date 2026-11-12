# Expected Input Structure

The scripts expect an authorized Excel workbook containing annual or multi-year worksheets named in `YYYY` or `YYYY-YYYY` format.

The analysis workflow requires the variables needed to construct:

- Adult-valid records.
- Sex and age covariates.
- Serum uric acid exposure.
- Microalbuminuria level.
- Semiquantitative urine protein category.
- Systolic blood pressure.
- Fasting plasma glucose.
- Body mass index.
- Serum creatinine and estimated glomerular filtration rate.
- Same-sample urine environment measures used in sensitivity analyses.

Some optional analyses also use additional local mappings when available, including urine creatinine strip grade, routine urinalysis availability fields, diastolic blood pressure, broader anthropometry fields, and supplemental blood chemistry items. These keys appear in `config/column_map.example.json`.

Copy `config/column_map.example.json` to a local path, fill in the workbook's source-column labels, and set `ANALYSIS_COLUMN_MAP` if the mapping is stored outside `data/column_map.json`.

## Renal-function-context sensitivity

`analysis/renal_function_context_sensitivity.py` reuses the main analytic rows and serum creatinine field to calculate concurrent eGFR with the CKD-EPI creatinine equation. It then refits aggregate models after adding either eGFR or serum creatinine to the main adjustment set.

The script writes aggregate model summaries under `outputs/renal_function_context_sensitivity/` by default:

- `renal_function_context_models.tsv`
- `renal_function_context_summary.tsv`

These output files are generated locally.
