from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np

import extended_rows
import main_analysis as base


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", REPO_ROOT / "outputs" / "observable_selection_ipw"))
AGE_SOURCE = os.environ.get("ANALYSIS_AGE_SOURCE", "exact")


def safe_exp(value: float) -> float:
    if value >= 700:
        return float("inf")
    if value <= -700:
        return 0.0
    return math.exp(value)


def ci_from_beta(beta: float, se: float) -> tuple[float, float]:
    return safe_exp(beta - 1.96 * se), safe_exp(beta + 1.96 * se)


def clip_prob(values: np.ndarray, lo: float = 1e-6, hi: float = 1.0 - 1e-6) -> np.ndarray:
    # Bounded probabilities keep inverse-probability weights numerically stable.
    return np.clip(np.asarray(values, dtype=float), lo, hi)


def write_tsv(path: Path, rows: list[list[object]], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def zscore_fill(rows, key: str) -> np.ndarray:
    # Missing values are represented by a zero-filled standardized value plus an indicator.
    observed = [row[key] for row in rows if row.get(key) is not None]
    if not observed:
        return np.zeros(len(rows), dtype=float)
    arr = np.asarray(observed, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    if std == 0.0:
        std = 1.0
    return np.asarray([0.0 if row.get(key) is None else (float(row[key]) - mean) / std for row in rows], dtype=float)


def binary_col(rows, key: str) -> np.ndarray:
    return np.asarray([1.0 if row.get(key) else 0.0 for row in rows], dtype=float)


def add_year_terms(rows, columns, names):
    year_cols, year_names, year_base = base.year_dummies([row["year"] for row in rows])
    columns.extend(year_cols)
    names.extend([f"year_{name}" for name in year_names])
    return year_base


def add_value_with_missing(rows, key: str, label: str, columns, names):
    # Separate availability and magnitude terms allow both patterns to enter the model.
    observed = np.asarray([1.0 if row.get(key) is not None else 0.0 for row in rows], dtype=float)
    observed_rate = float(np.mean(observed))
    if observed_rate not in {0.0, 1.0}:
        columns.append(observed)
        names.append(f"{label}_obs")
    if np.any(observed == 1.0):
        columns.append(zscore_fill(rows, key))
        names.append(f"{label}_z")


def add_binary_if_variable(rows, key: str, label: str, columns, names):
    # Constant binary terms are omitted so the design matrix remains full rank.
    col = binary_col(rows, key)
    if float(np.mean(col)) not in {0.0, 1.0}:
        columns.append(col)
        names.append(label)


def add_phenotype_terms(rows, columns, names):
    for level in (1, 2, 3):
        col = np.asarray([1.0 if row.get("phenotype") == level else 0.0 for row in rows], dtype=float)
        columns.append(col)
        names.append(f"phenotype_{level}")


def add_protein_terms(rows, columns, names):
    # Protein-strip categories are used as observable predictors before MAU-grade selection.
    for detail in ("trace", "1+", "2plus_or_more"):
        if detail == "2plus_or_more":
            col = np.asarray([1.0 if row.get("prot_detail") in {"2+", "3+", "4+"} else 0.0 for row in rows], dtype=float)
        else:
            col = np.asarray([1.0 if row.get("prot_detail") == detail else 0.0 for row in rows], dtype=float)
        if float(np.mean(col)) not in {0.0, 1.0}:
            columns.append(col)
            names.append(f"protein_{detail}")


def year_is_annual(year_value: str) -> bool:
    return "-" not in str(year_value)


def add_common_terms(rows, columns, names):
    # Age, sex, and collection wave are included in each observable-stage model.
    age_z, _, _ = base.zscore([row["age_low"] for row in rows])
    columns.append(age_z)
    names.append("age_z")
    columns.append(np.asarray([row["female"] for row in rows], dtype=float))
    names.append("female")
    return add_year_terms(rows, columns, names)


def add_nonurine_package_terms(rows, columns, names):
    # Package-level availability terms summarize whether non-urine measurements were recorded.
    for key, label in (
        ("kidney_blood_any", "kidney_blood_any"),
        ("lipid_any", "lipid_any"),
        ("liver_any", "liver_any"),
        ("anthropometry_any", "anthropometry_any"),
        ("vital_sign_any", "vital_sign_any"),
    ):
        add_binary_if_variable(rows, key, label, columns, names)
    for key, label in (
        ("urea", "urea"),
        ("scr", "scr"),
        ("fpg", "fpg"),
        ("sbp", "sbp"),
        ("bmi", "bmi"),
        ("tg", "tg"),
        ("alt", "alt"),
    ):
        add_value_with_missing(rows, key, label, columns, names)


def build_stage_matrix(rows, stage: str, numerator: bool, spec: str):
    # Numerator models use baseline terms; denominator models add stage-observable predictors.
    columns = [np.ones(len(rows), dtype=float)]
    names = ["intercept"]
    year_base = add_common_terms(rows, columns, names)

    if stage == "A1_mau_grade" and not numerator:
        add_protein_terms(rows, columns, names)
    if stage in {"B_sua", "C_complete"}:
        add_phenotype_terms(rows, columns, names)
    if stage == "C_complete":
        ua_z, _, _ = base.zscore([row["ua"] for row in rows])
        columns.append(ua_z)
        names.append("ua_z")

    if not numerator:
        if stage in {"A0_protein", "A1_mau_grade"}:
            add_nonurine_package_terms(rows, columns, names)
        if stage == "A1_mau_grade" and spec == "module_proxy":
            add_binary_if_variable(rows, "urine_creat_available", "urine_creat_available", columns, names)
            add_value_with_missing(rows, "urine_creat", "urine_creat", columns, names)
            add_value_with_missing(rows, "urine_ph", "urine_ph", columns, names)
            add_value_with_missing(rows, "urine_sg", "urine_sg", columns, names)
        if stage == "B_sua":
            for key, label in (("sbp", "sbp"), ("fpg", "fpg"), ("bmi", "bmi"), ("scr", "scr")):
                add_value_with_missing(rows, key, label, columns, names)
        if stage == "C_complete":
            add_value_with_missing(rows, "scr", "scr", columns, names)

    return base.build_matrix(columns), names, year_base


def fit_stage(rows, stage: str, outcome_fn, spec: str, numerator: bool):
    # A stage model estimates the probability of moving to the next observable subset.
    x, names, year_base = build_stage_matrix(rows, stage, numerator=numerator, spec=spec)
    y = np.asarray([1.0 if outcome_fn(row) else 0.0 for row in rows], dtype=float)
    model = base.fit_logistic(x, y, maxiter=2000)
    probs = clip_prob(base.sigmoid(x @ model["beta"]))
    table = []
    for idx, name in enumerate(names[1:], start=1):
        beta = float(model["beta"][idx])
        se = float(model["se"][idx])
        lcl, ucl = ci_from_beta(beta, se)
        table.append(
            [
                stage,
                "numerator" if numerator else "denominator",
                year_base,
                name,
                f"{safe_exp(beta):.3f}",
                f"{lcl:.3f}",
                f"{ucl:.3f}",
                f"{base.p_from_z(beta / se):.6g}",
            ]
        )
    return {"rows": rows, "y": y, "probs": probs, "table": table, "names": names}


def weighted_effective_n(weights: np.ndarray) -> float:
    # Effective sample size summarizes the spread of positive weights.
    denom = float(np.sum(weights**2))
    return 0.0 if denom == 0.0 else float(np.sum(weights) ** 2 / denom)


def weight_summary(weights: np.ndarray) -> dict[str, float]:
    weights = np.asarray(weights, dtype=float)
    return {
        "min": float(np.min(weights)),
        "p25": float(np.quantile(weights, 0.25)),
        "median": float(np.quantile(weights, 0.50)),
        "p75": float(np.quantile(weights, 0.75)),
        "p95": float(np.quantile(weights, 0.95)),
        "p99": float(np.quantile(weights, 0.99)),
        "max": float(np.max(weights)),
        "sum": float(np.sum(weights)),
        "ess": weighted_effective_n(weights),
    }


def cap_weights(weights: np.ndarray, method: str) -> tuple[np.ndarray, float | None]:
    if method == "none":
        return weights.copy(), None
    cap = float(np.quantile(weights, 0.99 if method == "p99" else 0.95))
    return np.minimum(weights, cap), cap


def fit_weighted_threshold(primary_rows, weights, threshold: int):
    design = base.design_primary(primary_rows, exposure="ua_z")
    x, _ = base.logistic_matrix(design, exposure="ua_z", include_female=True, use_high_sbp=False, interaction=None)
    y = np.asarray([1.0 if row["phenotype"] >= threshold else 0.0 for row in primary_rows], dtype=float)
    weights = np.asarray(weights, dtype=float)
    model = base.fit_weighted_logistic(x, y, weights)
    beta = float(model["beta"][1])
    pred = base.sigmoid(x @ model["beta"])
    # Sandwich covariance is based on weighted score residuals.
    bread = x.T @ ((weights * pred * (1.0 - pred))[:, None] * x)
    score = (weights * (y - pred))[:, None] * x
    meat = score.T @ score
    bread_inv = np.linalg.pinv(bread)
    cov_hc0 = bread_inv @ meat @ bread_inv
    n_rows, n_cols = x.shape
    cov_hc1 = cov_hc0 * (n_rows / max(n_rows - n_cols, 1))
    se = math.sqrt(max(float(cov_hc1[1, 1]), 0.0))
    lcl, ucl = ci_from_beta(beta, se)
    return {
        "threshold": threshold,
        "n": len(primary_rows),
        "events": int(np.sum(y)),
        "or": safe_exp(beta),
        "lcl": lcl,
        "ucl": ucl,
        "p": base.p_from_z(beta / se),
    }


def fit_all_thresholds(primary_rows, weights):
    return [fit_weighted_threshold(primary_rows, weights, threshold) for threshold in (1, 2, 3)]


def run_suite(rows, scope: str, spec: str):
    # Stabilized weights combine numerator and denominator probabilities across stages.
    adult = [row for row in rows if row.get("adult_valid")]
    protein_rows = [row for row in adult if row.get("protein_available")]
    phenotype_rows = [row for row in protein_rows if row.get("mau_grade_available") and row.get("phenotype") is not None]
    phua_rows = [row for row in phenotype_rows if row.get("ua") is not None]
    primary_rows = [row for row in phua_rows if all(row.get(key) is not None for key in ("sbp", "fpg", "bmi"))]

    stage_defs = (
        ("A0_protein", adult, lambda row: row.get("protein_available")),
        ("A1_mau_grade", protein_rows, lambda row: row.get("mau_grade_available") and row.get("phenotype") is not None),
        ("B_sua", phenotype_rows, lambda row: row.get("ua") is not None),
        ("C_complete", phua_rows, lambda row: all(row.get(key) is not None for key in ("sbp", "fpg", "bmi"))),
    )

    stage_results = {}
    stage_table_rows = []
    stage_count_rows = [
        [scope, spec, "adult_valid", len(adult)],
        [scope, spec, "protein_available", len(protein_rows)],
        [scope, spec, "phenotype_available", len(phenotype_rows)],
        [scope, spec, "phenotype_plus_sua", len(phua_rows)],
        [scope, spec, "primary_complete", len(primary_rows)],
    ]

    for stage, stage_rows, outcome_fn in stage_defs:
        den = fit_stage(stage_rows, stage, outcome_fn, spec=spec, numerator=False)
        num = fit_stage(stage_rows, stage, outcome_fn, spec=spec, numerator=True)
        stage_results[(stage, "den")] = den
        stage_results[(stage, "num")] = num
        stage_count_rows.append([scope, spec, f"{stage}_risk_set", len(stage_rows)])
        stage_count_rows.append([scope, spec, f"{stage}_selected", int(np.sum(den["y"]))])
        for result_rows in (den["table"], num["table"]):
            for row in result_rows:
                stage_table_rows.append([scope, spec, *row])

    prob_maps = {}
    for stage, _, _ in stage_defs:
        for kind in ("den", "num"):
            result = stage_results[(stage, kind)]
            prob_maps[(stage, kind)] = {id(row): prob for row, prob in zip(result["rows"], result["probs"])}

    weights = []
    for row in primary_rows:
        weight = 1.0
        for stage in ("A0_protein", "A1_mau_grade", "B_sua", "C_complete"):
            weight *= prob_maps[(stage, "num")][id(row)] / prob_maps[(stage, "den")][id(row)]
        weights.append(weight)
    weights = np.asarray(weights, dtype=float)

    threshold_rows = []
    weight_rows = []
    for trim in ("none", "p99", "p95"):
        capped, cap = cap_weights(weights, trim)
        summary = weight_summary(capped)
        for item in fit_all_thresholds(primary_rows, capped):
            threshold_rows.append(
                [
                    scope,
                    spec,
                    trim,
                    "" if cap is None else f"{cap:.6f}",
                    f"{summary['ess']:.2f}",
                    item["threshold"],
                    item["n"],
                    item["events"],
                    f"{item['or']:.6f}",
                    f"{item['lcl']:.6f}",
                    f"{item['ucl']:.6f}",
                    f"{item['p']:.8g}",
                ]
            )
        weight_rows.append(
            [
                scope,
                spec,
                trim,
                "" if cap is None else f"{cap:.6f}",
                *[f"{summary[key]:.8f}" for key in ("min", "p25", "median", "p75", "p95", "p99", "max", "sum", "ess")],
            ]
        )

    return {
        "stage_counts": stage_count_rows,
        "stage_models": stage_table_rows,
        "thresholds": threshold_rows,
        "weights": weight_rows,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = extended_rows.load_enriched_rows(age_source=AGE_SOURCE)
    annual_rows = [row for row in rows if year_is_annual(row.get("year", ""))]

    suites = []
    for spec in ("package_proxy", "module_proxy"):
        suites.append(run_suite(rows, "full_scope", spec))
        suites.append(run_suite(annual_rows, "annual_waves_only", spec))

    stage_counts = [row for suite in suites for row in suite["stage_counts"]]
    stage_models = [row for suite in suites for row in suite["stage_models"]]
    thresholds = [row for suite in suites for row in suite["thresholds"]]
    weights = [row for suite in suites for row in suite["weights"]]

    write_tsv(OUTPUT_DIR / "observable_selection_stage_counts.tsv", stage_counts, ["scope", "spec", "stage", "n"])
    write_tsv(
        OUTPUT_DIR / "observable_selection_stage_models.tsv",
        stage_models,
        ["scope", "spec", "stage", "model_part", "year_base", "term", "or", "lcl", "ucl", "p"],
    )
    write_tsv(
        OUTPUT_DIR / "observable_selection_thresholds.tsv",
        thresholds,
        ["scope", "spec", "trim", "cap", "ess", "threshold", "n", "events", "or", "lcl", "ucl", "p"],
    )
    write_tsv(
        OUTPUT_DIR / "observable_selection_weight_summary.tsv",
        weights,
        ["scope", "spec", "trim", "cap", "min", "p25", "median", "p75", "p95", "p99", "max", "sum", "ess"],
    )

if __name__ == "__main__":
    main()
