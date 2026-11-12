from __future__ import annotations

import csv
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", REPO_ROOT / "outputs" / "renal_function_context_sensitivity"))
BASE_PATH = Path(__file__).resolve().with_name("main_analysis.py")
AGE_SOURCE = os.environ.get("ANALYSIS_AGE_SOURCE", "exact")


def load_base_module():
    spec = importlib.util.spec_from_file_location("main_analysis", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base analysis module from {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_tsv(path: Path, rows: list[list[object]], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def design_with_context(base, rows, context: str | None = None):
    # Context variables are added after the primary adjustment covariates.
    design = base.design_primary(rows, exposure="ua_z")
    cols = [
        np.ones(len(rows), dtype=float),
        design["ua_z"],
        design["female"],
        design["age_z"],
        design["sbp_z"],
        design["fpg_z"],
        design["bmi_z"],
    ]
    names = ["intercept", "ua_z", "female", "age_z", "sbp_z", "fpg_z", "bmi_z"]
    if context == "egfr":
        egfr_z, _, _ = base.zscore([row["egfr"] for row in rows])
        cols.append(egfr_z)
        names.append("egfr_z")
    elif context == "serum_creatinine":
        scr_z, _, _ = base.zscore([row["scr"] for row in rows])
        cols.append(scr_z)
        names.append("scr_z")
    cols.extend(design["year_cols"])
    names.extend([f"year_{name}" for name in design["year_names"]])
    return base.build_matrix(cols), names, design


def logistic_result(base, x, names, y, weights=None):
    # The serum uric acid coefficient is extracted by name from each model matrix.
    model = base.fit_weighted_logistic(x, y, weights) if weights is not None else base.fit_logistic(x, y)
    idx = names.index("ua_z")
    beta = float(model["beta"][idx])
    se = float(model["se"][idx])
    lcl, ucl = base.ci_from_beta(beta, se)
    return math.exp(beta), lcl, ucl, base.p_from_z(beta / se)


def threshold_models(base, rows, context: str | None = None, weights=None):
    # Threshold endpoints are refit under each renal-function adjustment context.
    x, names, _ = design_with_context(base, rows, context)
    out = []
    for threshold in (1, 2, 3):
        y = np.array([1.0 if row["phenotype"] >= threshold else 0.0 for row in rows], dtype=float)
        or_value, lcl, ucl, pvalue = logistic_result(base, x, names, y, weights)
        out.append([f"Phenotype >={threshold}, independent threshold", context or "base", len(rows), int(y.sum()), or_value, lcl, ucl, pvalue])
    return out


def binary_endpoint(base, rows, label, outcome_fn, context: str | None = None):
    # Component endpoints use the same context-augmented covariate structure.
    x, names, _ = design_with_context(base, rows, context)
    y = np.array([1.0 if outcome_fn(row) else 0.0 for row in rows], dtype=float)
    or_value, lcl, ucl, pvalue = logistic_result(base, x, names, y)
    return [label, context or "base", len(rows), int(y.sum()), or_value, lcl, ucl, pvalue]


def ppo_egfr(base, rows):
    # The eGFR-adjusted PPO model keeps threshold-specific serum uric acid slopes.
    _, _, design = design_with_context(base, rows, None)
    egfr_z, _, _ = base.zscore([row["egfr"] for row in rows])
    y = np.array([row["phenotype"] for row in rows], dtype=int)
    x_po, _, _ = design_with_context(base, rows, "egfr")
    po = base.fit_ordered_logit(x_po[:, 1:], y, 4)
    ua_col = design["ua_z"]
    common_cols = base.build_matrix(
        [design["female"], design["age_z"], design["sbp_z"], design["fpg_z"], design["bmi_z"], egfr_z] + design["year_cols"]
    )
    n_categories = 4

    def unpack(params):
        first_cut = params[0]
        increments = np.exp(params[1 : n_categories - 1])
        cutpoints = [first_cut]
        for increment in increments:
            cutpoints.append(cutpoints[-1] + increment)
        cutpoints = np.asarray(cutpoints, dtype=float)
        ua_betas = np.asarray(params[n_categories - 1 : n_categories - 1 + (n_categories - 1)], dtype=float)
        beta_common = np.asarray(params[n_categories - 1 + (n_categories - 1) :], dtype=float)
        return cutpoints, ua_betas, beta_common

    def probabilities(params):
        # Cumulative probabilities are converted to category probabilities by differencing.
        cutpoints, ua_betas, beta_common = unpack(params)
        base_eta = common_cols @ beta_common
        cdf = np.zeros((len(rows), n_categories - 1), dtype=float)
        for idx in range(n_categories - 1):
            cdf[:, idx] = base.sigmoid(cutpoints[idx] - (base_eta + ua_col * ua_betas[idx]))
        probs = np.zeros((len(rows), n_categories), dtype=float)
        probs[:, 0] = cdf[:, 0]
        for idx in range(1, n_categories - 1):
            probs[:, idx] = cdf[:, idx] - cdf[:, idx - 1]
        probs[:, n_categories - 1] = 1.0 - cdf[:, n_categories - 2]
        return np.clip(probs, 1e-12, 1.0)

    def objective(params):
        probs = probabilities(params)
        return -np.log(probs[np.arange(len(rows)), y]).sum()

    init = np.concatenate(
        [
            [po["cutpoints"][0]],
            np.log(np.clip(np.diff(po["cutpoints"]), 1e-3, None)),
            np.repeat(po["beta"][0], n_categories - 1),
            po["beta"][1:],
        ]
    )
    result = minimize(objective, init, method="BFGS", options={"maxiter": 900})
    if not result.success:
        result = minimize(objective, result.x, method="L-BFGS-B", options={"maxiter": 1400})
    if not result.success:
        raise RuntimeError(f"eGFR-adjusted PPO failed: {result.message}")
    stat, lrt_p = base.lrt_pvalue(po["ll"], -objective(result.x), 2)
    hessian = base.finite_difference_hessian(objective, result.x)
    cov = base.covariance_from_hessian(hessian)
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    _, ua_betas, _ = unpack(result.x)
    ua_se = se[n_categories - 1 : n_categories - 1 + (n_categories - 1)]
    rows_out = []
    for threshold, beta, beta_se in zip((1, 2, 3), ua_betas, ua_se):
        lcl, ucl = base.ci_from_beta(float(beta), float(beta_se))
        rows_out.append([f"Phenotype >={threshold}, PPO", "egfr", len(rows), "", math.exp(float(beta)), lcl, ucl, base.p_from_z(float(beta / beta_se))])
    return stat, lrt_p, rows_out


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    base = load_base_module()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = base.load_rows(age_source=AGE_SOURCE)
    primary_rows = base.dataset_for_primary(rows)
    egfr_rows = base.primary_with_egfr(primary_rows)
    if len(egfr_rows) != len(primary_rows):
        raise RuntimeError("All primary records are expected to have serum creatinine for eGFR calculation.")

    result_rows = []
    for context in (None, "egfr", "serum_creatinine"):
        result_rows.extend(threshold_models(base, egfr_rows, context=context))

    mau_axis_rows = [row for row in egfr_rows if row.get("prot_detail") in {"neg", "trace"}]

    for context in (None, "egfr"):
        result_rows.append(binary_endpoint(base, egfr_rows, "Urine protein >=1+", lambda row: row["prot1"] == 1, context=context))
        result_rows.append(
            binary_endpoint(
                base,
                mau_axis_rows,
                "MAU-only axis >=80, protein negative/trace",
                lambda row: row["mau_level"] >= 80.0,
                context=context,
            )
        )

    ppo_stat, ppo_p, ppo_rows = ppo_egfr(base, egfr_rows)
    result_rows.extend(ppo_rows)

    # IPW models reuse complete-case selection weights and add eGFR to the outcome model.
    selection = base.selection_analysis(rows)
    weights = selection["weights"]
    result_rows.extend(threshold_models(base, egfr_rows, context="egfr", weights=weights))
    for row in result_rows[-3:]:
        row[0] = row[0].replace("independent threshold", "complete-case IPW")

    write_tsv(
        OUTPUT_DIR / "renal_function_context_models.tsv",
        [[fmt(value) for value in row] for row in result_rows],
        ["analysis", "context", "n", "events", "or", "lcl", "ucl", "p"],
    )
    write_tsv(
        OUTPUT_DIR / "renal_function_context_summary.tsv",
        [
            ["age_source", AGE_SOURCE],
            ["primary_n", len(primary_rows)],
            ["egfr_rows_n", len(egfr_rows)],
            ["egfr_mean", f"{np.mean([row['egfr'] for row in egfr_rows]):.6g}"],
            ["egfr_sd", f"{np.std([row['egfr'] for row in egfr_rows], ddof=0):.6g}"],
            ["egfr_adjusted_ppo_lrt_stat", f"{ppo_stat:.6g}"],
            ["egfr_adjusted_ppo_lrt_p", f"{ppo_p:.6g}"],
        ],
        ["metric", "value"],
    )


if __name__ == "__main__":
    main()
