# Uric Acid and Urinary Abnormalities Analysis

This repository contains public analysis code for:

**Serum uric acid associations across a semi-quantitative MAU-proteinuria screening hierarchy: a repeated cross-sectional study**

The code implements aggregate analyses for construction of the MAU-proteinuria screening hierarchy, threshold-specific serum uric acid models, partial proportional odds modeling, upper-threshold component analyses, renal-function-context models, selection/IPW checks, possible repeated-record sensitivity analyses, collection-wave checks, and same-sample urinary-context sensitivity analyses.

## Data availability

The source data are not included in this repository. They consist of institution-based health examination records containing privacy-sensitive clinical information and are subject to institutional data-access restrictions.

The scripts read an authorized local workbook through the `ANALYSIS_WORKBOOK` environment variable and use a local column map for source-column labels. Generated model summaries are written under `outputs/` by default.

## Repository contents

- `analysis/`: Python scripts for the main analysis and sensitivity analyses.
- `config/column_map.example.json`: Public placeholder map for local source-column labels.
- `docs/`: Notes on data access, expected input structure, and analysis workflow.
- `requirements.txt`: Python package dependencies used by the scripts.

## Running the analysis

Install dependencies in a local Python environment:

```bash
pip install -r requirements.txt
```

Set the path to an authorized workbook before running any script:

```bash
export ANALYSIS_WORKBOOK="/path/to/authorized_workbook"
```

Create a local column map from the example file, using source-column labels from the authorized workbook:

```bash
mkdir -p data
cp config/column_map.example.json data/column_map.json
```

The local `data/column_map.json` file is ignored by Git. You can also point to another local mapping file:

```bash
export ANALYSIS_COLUMN_MAP="/path/to/local_column_map.json"
```

Run the main analysis:

```bash
python analysis/main_analysis.py
```

Optional sensitivity analyses can be run individually:

```bash
python analysis/test_ordering_selection.py
python analysis/repeat_record_sensitivity.py
python analysis/cumulative_selection_ipw.py
python analysis/repeat_record_collapse_calibration.py
python analysis/ipw_diagnostics.py
python analysis/mau_axis_decomposition.py
python analysis/same_sample_urinalysis_sensitivity.py
python analysis/observable_selection_ipw.py
python analysis/renal_function_context_sensitivity.py
```

See `docs/analysis_map.md` for the relationship between analysis steps and scripts.

## License

The analysis code is released under the MIT License. The license does not grant access to the underlying health examination data.
