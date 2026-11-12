from __future__ import annotations

import csv
import importlib.util
import math
import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", REPO_ROOT / "outputs" / "cumulative_selection_ipw"))
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


def safe_exp(value: float) -> float:
    if value >= 700:
        return float("inf")
    if value <= -700:
        return 0.0
    return math.exp(value)


def ci_from_beta(beta: float, se: float) -> tuple[float, float]:
    return safe_exp(beta - 1.96 * se), safe_exp(beta + 1.96 * se)


def clip_prob(values: np.ndarray, lo: float = 1e-6, hi: float = 1.0 - 1e-6) -> np.ndarray:
    # Probability clipping prevents unstable inverse-probability weights.
    return np.clip(np.asarray(values, dtype=float), lo, hi)


def zscore_fill(rows: list[dict[str, object]], key: str) -> np.ndarray:
    # Missing continuous predictors are filled at zero after observed-value standardization.
    observed = [row[key] for row in rows if row.get(key) is not None]
    if not observed:
        return np.zeros(len(rows), dtype=float)
    arr = np.asarray(observed, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    if std == 0:
        std = 1.0
    out = []
    for row in rows:
        value = row.get(key)
        out.append(0.0 if value is None else (float(value) - mean) / std)
    return np.asarray(out, dtype=float)


def add_year_terms(base, rows: list[dict[str, object]], columns: list[np.ndarray], names: list[str]) -> str:
    year_cols, year_names, year_base = base.year_dummies([row["year"] for row in rows])
    columns.extend(year_cols)
    names.extend([f"year_{name}" for name in year_names])
    return year_base


def add_phenotype_terms(rows: list[dict[str, object]], columns: list[np.ndarray], names: list[str]) -> None:
    # Phenotype indicators carry already-observed urinary category information forward.
    for level in (1, 2, 3):
        columns.append(np.asarray([1.0 if row.get("phenotype") == level else 0.0 for row in rows], dtype=float))
        names.append(f"phenotype_{level}")


def add_value_with_missing(rows: list[dict[str, object]], key: str, label: str, columns: list[np.ndarray], names: list[str]) -> None:
    # Availability and value terms are separated to represent missingness patterns.
    values = [row.get(key) for row in rows]
    observed = np.asarray([1.0 if value is not None else 0.0 for value in values], dtype=float)
    zfilled = zscore_fill(rows, key)
    observed_rate = float(np.mean(observed))
    if observed_rate not in {0.0, 1.0}:
        columns.append(observed)
        names.append(f"{label}_obs")
    if np.any(observed == 1.0):
        columns.append(zfilled)
        names.append(f"{label}_z")


def stage_feature_pairs(stage: str) -> list[tuple[str, str]]:
    # Later selection stages condition on variables that are observable at that stage.
    if stage == "A":
        return [("ua", "ua"), ("sbp", "sbp"), ("fpg", "fpg"), ("bmi", "bmi"), ("scr", "scr")]
    if stage == "B":
        return [("sbp", "sbp"), ("fpg", "fpg"), ("bmi", "bmi"), ("scr", "scr")]
    if stage == "C":
        return [("scr", "scr")]
    return []


def availability_cell_counts(rows: list[dict[str, object]], outcome_fn, key: str) -> dict[tuple[int, int], int]:
    counts = {(obs, out): 0 for obs in (0, 1) for out in (0, 1)}
    for row in rows:
        obs = 1 if row.get(key) is not None else 0
        out = 1 if outcome_fn(row) else 0
        counts[(obs, out)] += 1
    return counts


def tight_feature_pairs(
    rows: list[dict[str, object]], stage: str, outcome_fn, min_cell: int = 5
) -> tuple[list[tuple[str, str]], list[tuple[str, dict[tuple[int, int], int]]]]:
    # Sparse availability-by-outcome cells are excluded from tighter selection models.
    kept: list[tuple[str, str]] = []
    dropped: list[tuple[str, dict[tuple[int, int], int]]] = []
    for key, label in stage_feature_pairs(stage):
        counts = availability_cell_counts(rows, outcome_fn, key)
        if min(counts.values()) < min_cell:
            dropped.append((label, counts))
            continue
        kept.append((key, label))
    return kept, dropped


def build_stage_matrix(base, rows, stage: str, reduced: bool, outcome_fn, analysis_spec: str):
    # Denominator models include observed auxiliary predictors; numerator models are reduced.
    columns = [np.ones(len(rows), dtype=float)]
    names = ["intercept"]
    dropped: list[tuple[str, dict[tuple[int, int], int]]] = []

    age_z, _, _ = base.zscore([row["age_low"] for row in rows])
    columns.append(age_z)
    names.append("age_z")

    female = np.asarray([row["female"] for row in rows], dtype=float)
    columns.append(female)
    names.append("female")

    year_base = add_year_terms(base, rows, columns, names)

    if stage in {"B", "C"}:
        add_phenotype_terms(rows, columns, names)

    if stage == "C":
        ua_z, _, _ = base.zscore([row["ua"] for row in rows])
        columns.append(ua_z)
        names.append("ua_z")

    if not reduced:
        if analysis_spec == "tight":
            feature_pairs, dropped = tight_feature_pairs(rows, stage, outcome_fn)
        else:
            feature_pairs = stage_feature_pairs(stage)
        for key, label in feature_pairs:
            add_value_with_missing(rows, key, label, columns, names)

    x = base.build_matrix(columns)
    return x, names, year_base, dropped


def fit_stage_model(base, rows, stage: str, reduced: bool, outcome_fn, analysis_spec: str):
    # Each stage model estimates the probability of passing one analysis filter.
    x, names, year_base, dropped = build_stage_matrix(base, rows, stage, reduced, outcome_fn, analysis_spec)
    y = np.asarray([1.0 if outcome_fn(row) else 0.0 for row in rows], dtype=float)
    model = base.fit_logistic(x, y)
    probs = clip_prob(base.sigmoid(x @ model["beta"]))
    table = []
    for idx, name in enumerate(names[1:], start=1):
        beta = float(model["beta"][idx])
        se = float(model["se"][idx])
        lcl, ucl = ci_from_beta(beta, se)
        table.append([name, f"{safe_exp(beta):.3f}", f"{lcl:.3f}", f"{ucl:.3f}", f"{base.p_from_z(beta / se):.6g}"])
    return {
        "x": x,
        "y": y,
        "model": model,
        "probs": probs,
        "rows": rows,
        "names": names,
        "year_base": year_base,
        "table": table,
        "dropped": dropped,
    }


def weighted_effective_n(weights: np.ndarray) -> float:
    # ESS decreases as inverse-probability weights become more concentrated.
    weights = np.asarray(weights, dtype=float)
    denom = float(np.sum(weights**2))
    return 0.0 if denom == 0 else float(np.sum(weights) ** 2 / denom)


def cap_weights(weights: np.ndarray, method: str) -> tuple[np.ndarray, float | None]:
    weights = np.asarray(weights, dtype=float)
    if method == "none":
        return weights.copy(), None
    if method == "p99":
        cap = float(np.quantile(weights, 0.99))
    elif method == "p95":
        cap = float(np.quantile(weights, 0.95))
    else:
        raise ValueError(f"Unknown trim method: {method}")
    return np.minimum(weights, cap), cap


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


def fit_weighted_threshold_with_cov(base, primary_rows, weights, threshold: int) -> dict[str, float]:
    design = base.design_primary(primary_rows, exposure="ua_z")
    x, _ = base.logistic_matrix(design, exposure="ua_z", include_female=True, use_high_sbp=False, interaction=None)
    y = np.asarray([1.0 if row["phenotype"] >= threshold else 0.0 for row in primary_rows], dtype=float)
    weights = np.asarray(weights, dtype=float)
    model = base.fit_weighted_logistic(x, y, weights)
    beta = float(model["beta"][1])
    pred = base.sigmoid(x @ model["beta"])
    # HC1 covariance uses weighted scores and a small-sample degrees-of-freedom factor.
    bread = x.T @ ((weights * pred * (1.0 - pred))[:, None] * x)
    score = (weights * (y - pred))[:, None] * x
    meat = score.T @ score
    bread_inv = np.linalg.pinv(bread)
    cov_hc0 = bread_inv @ meat @ bread_inv
    n, p = x.shape
    cov_hc1 = cov_hc0 * (n / max(n - p, 1))
    hc1_se = math.sqrt(max(float(cov_hc1[1, 1]), 0.0))
    hc1_lcl, hc1_ucl = ci_from_beta(beta, hc1_se)
    return {
        "threshold": threshold,
        "n": len(primary_rows),
        "events": int(np.sum(y)),
        "or": safe_exp(beta),
        "beta": beta,
        "hc1_lcl": hc1_lcl,
        "hc1_ucl": hc1_ucl,
        "hc1_p": base.p_from_z(beta / hc1_se),
    }


def fit_all_thresholds(base, primary_rows, weights):
    return [fit_weighted_threshold_with_cov(base, primary_rows, weights, threshold) for threshold in (1, 2, 3)]


def naive_thresholds(base, primary_rows):
    return base.fit_threshold_models(primary_rows)


def one_stage_ipw_thresholds(base, rows):
    selection = base.selection_analysis(rows)
    primary_rows = base.dataset_for_primary(rows)
    return fit_all_thresholds(base, primary_rows, selection["weights"])


def year_is_annual(year_value: str) -> bool:
    return "-" not in str(year_value)


def run_suite(base, rows, label: str, analysis_spec: str):
    # Cumulative weights multiply stabilized probabilities across sequential filters.
    adult_rows = [row for row in rows if row.get("adult_valid")]
    stage_a_outcome = lambda row: row.get("phenotype") is not None
    stage_b_outcome = lambda row: row.get("ua") is not None
    stage_c_outcome = lambda row: all(row.get(key) is not None for key in ("sbp", "fpg", "bmi"))

    stage_a = fit_stage_model(base, adult_rows, "A", reduced=False, outcome_fn=stage_a_outcome, analysis_spec=analysis_spec)
    phenotype_rows = [row for row in adult_rows if row.get("phenotype") is not None]
    stage_b = fit_stage_model(base, phenotype_rows, "B", reduced=False, outcome_fn=stage_b_outcome, analysis_spec=analysis_spec)
    phua_rows = [row for row in phenotype_rows if row.get("ua") is not None]
    stage_c = fit_stage_model(base, phua_rows, "C", reduced=False, outcome_fn=stage_c_outcome, analysis_spec=analysis_spec)

    stage_a_num = fit_stage_model(base, adult_rows, "A", reduced=True, outcome_fn=stage_a_outcome, analysis_spec=analysis_spec)
    stage_b_num = fit_stage_model(base, phenotype_rows, "B", reduced=True, outcome_fn=stage_b_outcome, analysis_spec=analysis_spec)
    stage_c_num = fit_stage_model(base, phua_rows, "C", reduced=True, outcome_fn=stage_c_outcome, analysis_spec=analysis_spec)

    p_a_den = {id(row): p for row, p in zip(stage_a["rows"], stage_a["probs"])}
    p_b_den = {id(row): p for row, p in zip(stage_b["rows"], stage_b["probs"])}
    p_c_den = {id(row): p for row, p in zip(stage_c["rows"], stage_c["probs"])}
    p_a_num = {id(row): p for row, p in zip(stage_a_num["rows"], stage_a_num["probs"])}
    p_b_num = {id(row): p for row, p in zip(stage_b_num["rows"], stage_b_num["probs"])}
    p_c_num = {id(row): p for row, p in zip(stage_c_num["rows"], stage_c_num["probs"])}

    primary_rows = base.dataset_for_primary(rows)
    cumulative_weights = np.asarray(
        [
            (p_a_num[id(row)] / p_a_den[id(row)])
            * (p_b_num[id(row)] / p_b_den[id(row)])
            * (p_c_num[id(row)] / p_c_den[id(row)])
            for row in primary_rows
        ],
        dtype=float,
    )

    naive = naive_thresholds(base, primary_rows)
    one_stage = one_stage_ipw_thresholds(base, rows)
    cumulative = fit_all_thresholds(base, primary_rows, cumulative_weights)

    trimmed_results = []
    for trim in ("none", "p99", "p95"):
        capped, cap = cap_weights(cumulative_weights, trim)
        summary = weight_summary(capped)
        for item in fit_all_thresholds(base, primary_rows, capped):
            trimmed_results.append(
                [
                    label,
                    analysis_spec,
                    trim,
                    "" if cap is None else f"{cap:.6f}",
                    f"{summary['ess']:.2f}",
                    int(item["threshold"]),
                    int(item["n"]),
                    int(item["events"]),
                    f"{item['or']:.6f}",
                    f"{item['hc1_lcl']:.6f}",
                    f"{item['hc1_ucl']:.6f}",
                    f"{item['hc1_p']:.8g}",
                ]
            )

    comparison_rows = []
    for threshold in (1, 2, 3):
        naive_item = next(item for item in naive if int(item["threshold"]) == threshold)
        one_stage_item = next(item for item in one_stage if int(item["threshold"]) == threshold)
        cumulative_item = next(item for item in cumulative if int(item["threshold"]) == threshold)
        comparison_rows.append(
            [
                label,
                analysis_spec,
                threshold,
                f"{naive_item['or']:.6f}",
                f"{naive_item['lcl']:.6f}",
                f"{naive_item['ucl']:.6f}",
                f"{naive_item['p']:.8g}",
                f"{one_stage_item['or']:.6f}",
                f"{one_stage_item['hc1_lcl']:.6f}",
                f"{one_stage_item['hc1_ucl']:.6f}",
                f"{one_stage_item['hc1_p']:.8g}",
                f"{cumulative_item['or']:.6f}",
                f"{cumulative_item['hc1_lcl']:.6f}",
                f"{cumulative_item['hc1_ucl']:.6f}",
                f"{cumulative_item['hc1_p']:.8g}",
            ]
        )

    stage_counts = [
        [label, analysis_spec, "adult_valid", len(adult_rows)],
        [label, analysis_spec, "phenotype_available", len(phenotype_rows)],
        [label, analysis_spec, "phenotype_plus_ua", len(phua_rows)],
        [label, analysis_spec, "primary_complete", len(primary_rows)],
    ]

    stage_tables = []
    for stage_name, reduced_name, result in (
        ("A_den", "full", stage_a),
        ("B_den", "full", stage_b),
        ("C_den", "full", stage_c),
        ("A_num", "reduced", stage_a_num),
        ("B_num", "reduced", stage_b_num),
        ("C_num", "reduced", stage_c_num),
    ):
        for row in result["table"]:
            stage_tables.append([label, analysis_spec, stage_name, reduced_name, result["year_base"], *row])
        for dropped_name, counts in result["dropped"]:
            count_text = ",".join(f"{obs}{out}={value}" for (obs, out), value in counts.items())
            stage_tables.append(
                [label, analysis_spec, stage_name, reduced_name, result["year_base"], f"dropped_{dropped_name}", count_text, "", "", ""]
            )

    ws = weight_summary(cumulative_weights)
    weight_rows = [[label, analysis_spec, metric, f"{value:.8f}"] for metric, value in ws.items()]

    return {
        "comparison_rows": comparison_rows,
        "trimmed_rows": trimmed_results,
        "stage_counts": stage_counts,
        "stage_tables": stage_tables,
        "weight_rows": weight_rows,
    }


def main() -> None:
    base = load_base_module()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = base.load_rows(age_source=AGE_SOURCE)
    annual_rows = [row for row in rows if row.get("adult_valid") and year_is_annual(row["year"])]
    annual_rows = annual_rows + [row for row in rows if not row.get("adult_valid") and year_is_annual(row["year"])]

    suites = []
    for analysis_spec in ("full", "tight"):
        suites.append(run_suite(base, rows, "full_scope", analysis_spec))
        suites.append(run_suite(base, annual_rows, "annual_waves_only", analysis_spec))

    all_counts = [row for suite in suites for row in suite["stage_counts"]]
    all_stage_tables = [row for suite in suites for row in suite["stage_tables"]]
    all_weight_rows = [row for suite in suites for row in suite["weight_rows"]]
    all_trimmed_rows = [row for suite in suites for row in suite["trimmed_rows"]]
    all_comparison_rows = [row for suite in suites for row in suite["comparison_rows"]]

    write_tsv(OUTPUT_DIR / "stage_counts.tsv", all_counts, ["scope", "analysis_spec", "stage", "n"])
    write_tsv(
        OUTPUT_DIR / "stage_models.tsv",
        all_stage_tables,
        ["scope", "analysis_spec", "model", "spec", "year_base", "term", "or", "lcl", "ucl", "p"],
    )
    write_tsv(OUTPUT_DIR / "weight_summary.tsv", all_weight_rows, ["scope", "analysis_spec", "metric", "value"])
    write_tsv(
        OUTPUT_DIR / "threshold_comparison.tsv",
        all_comparison_rows,
        [
            "scope",
            "analysis_spec",
            "threshold",
            "naive_or",
            "naive_lcl",
            "naive_ucl",
            "naive_p",
            "one_stage_ipw_or",
            "one_stage_ipw_lcl",
            "one_stage_ipw_ucl",
            "one_stage_ipw_p",
            "cumulative_ipw_or",
            "cumulative_ipw_lcl",
            "cumulative_ipw_ucl",
            "cumulative_ipw_p",
        ],
    )
    write_tsv(
        OUTPUT_DIR / "trimming.tsv",
        all_trimmed_rows,
        ["scope", "analysis_spec", "trim", "cap", "ess", "threshold", "n", "events", "or", "lcl", "ucl", "p"],
    )

if __name__ == "__main__":
    main()
