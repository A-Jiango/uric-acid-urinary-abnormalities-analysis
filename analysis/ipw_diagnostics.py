from __future__ import annotations

import csv
import importlib.util
import math
import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", REPO_ROOT / "outputs" / "ipw_diagnostics"))
BASE_PATH = Path(__file__).resolve().with_name("main_analysis.py")
SEED = 12345
DEFAULT_BOOTSTRAP_B = 1000
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


def quantile(values, q):
    arr = np.asarray(values, dtype=float)
    return float(np.quantile(arr, q))


def ci_from_beta(beta: float, se: float) -> tuple[float, float]:
    return math.exp(beta - 1.96 * se), math.exp(beta + 1.96 * se)


def weighted_effective_n(weights: np.ndarray) -> float:
    # Effective sample size summarizes concentration of positive weights.
    weights = np.asarray(weights, dtype=float)
    denom = float(np.sum(weights**2))
    return 0.0 if denom == 0 else float(np.sum(weights) ** 2 / denom)


def cap_weights(weights: np.ndarray, method: str) -> tuple[np.ndarray, float | None]:
    # Weight caps replace only the upper tail of the weight distribution.
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


def fit_weighted_threshold_with_cov(base, primary_rows, weights, threshold: int) -> dict[str, float]:
    design = base.design_primary(primary_rows, exposure="ua_z")
    x, _ = base.logistic_matrix(design, exposure="ua_z", include_female=True, use_high_sbp=False, interaction=None)
    y = np.array([1.0 if r["phenotype"] >= threshold else 0.0 for r in primary_rows], dtype=float)
    weights = np.asarray(weights, dtype=float)
    model = base.fit_weighted_logistic(x, y, weights)
    beta = float(model["beta"][1])
    naive_se = float(model["se"][1])
    pred = base.sigmoid(x @ model["beta"])
    # HC covariance is computed from the weighted score residuals.
    bread = x.T @ ((weights * pred * (1.0 - pred))[:, None] * x)
    score = (weights * (y - pred))[:, None] * x
    meat = score.T @ score
    bread_inv = np.linalg.pinv(bread)
    cov_hc0 = bread_inv @ meat @ bread_inv
    n, p = x.shape
    cov_hc1 = cov_hc0 * (n / max(n - p, 1))
    hc0_se = math.sqrt(max(float(cov_hc0[1, 1]), 0.0))
    hc1_se = math.sqrt(max(float(cov_hc1[1, 1]), 0.0))
    hc1_lcl, hc1_ucl = ci_from_beta(beta, hc1_se)
    return {
        "threshold": threshold,
        "n": float(len(primary_rows)),
        "events": float(np.sum(y)),
        "beta": beta,
        "or": math.exp(beta),
        "naive_se": naive_se,
        "naive_p": base.p_from_z(beta / naive_se),
        "hc0_se": hc0_se,
        "hc0_p": base.p_from_z(beta / hc0_se),
        "hc1_se": hc1_se,
        "hc1_lcl": hc1_lcl,
        "hc1_ucl": hc1_ucl,
        "hc1_p": base.p_from_z(beta / hc1_se),
    }


def fit_all_thresholds_with_cov(base, primary_rows, weights) -> list[dict[str, float]]:
    return [fit_weighted_threshold_with_cov(base, primary_rows, weights, threshold) for threshold in (1, 2, 3)]


def selection_weight_summary(weights: np.ndarray) -> dict[str, float]:
    # Quantiles and ESS describe the scale and concentration of the weights.
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


def weighted_mean(values, weights=None) -> float:
    # Weighted summaries are computed on complete-case rows after IPW.
    arr = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.mean(arr))
    weights = np.asarray(weights, dtype=float)
    return float(np.average(arr, weights=weights))


def weighted_var(values, weights=None) -> float:
    arr = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.var(arr, ddof=0))
    weights = np.asarray(weights, dtype=float)
    mean = np.average(arr, weights=weights)
    return float(np.average((arr - mean) ** 2, weights=weights))


def smd(target_values, complete_values, weights=None) -> float:
    # SMD compares target rows with complete rows before or after weighting.
    target_mean = weighted_mean(target_values)
    target_var = weighted_var(target_values)
    comp_mean = weighted_mean(complete_values, weights)
    comp_var = weighted_var(complete_values, weights)
    denom = math.sqrt((target_var + comp_var) / 2.0)
    return 0.0 if denom == 0 else float((comp_mean - target_mean) / denom)


def balance_rows(base, phua_rows, primary_rows, weights):
    # Balance diagnostics compare the phenotype-plus-exposure set with complete cases.
    weights = np.asarray(weights, dtype=float)
    diagnostics = []

    def add_variable(name, target_values, complete_values):
        target_values = np.asarray(target_values, dtype=float)
        complete_values = np.asarray(complete_values, dtype=float)
        before = smd(target_values, complete_values)
        after = smd(target_values, complete_values, weights)
        diagnostics.append(
            [
                name,
                f"{weighted_mean(target_values):.6f}",
                f"{weighted_mean(complete_values):.6f}",
                f"{weighted_mean(complete_values, weights):.6f}",
                f"{before:.6f}",
                f"{after:.6f}",
                f"{abs(before):.6f}",
                f"{abs(after):.6f}",
            ]
        )

    add_variable("age", [r["age_low"] for r in phua_rows], [r["age_low"] for r in primary_rows])
    add_variable("ua", [r["ua"] for r in phua_rows], [r["ua"] for r in primary_rows])
    add_variable("female", [r["female"] for r in phua_rows], [r["female"] for r in primary_rows])
    for phenotype in (1, 2, 3):
        add_variable(
            f"phenotype_{phenotype}",
            [1.0 if r["phenotype"] == phenotype else 0.0 for r in phua_rows],
            [1.0 if r["phenotype"] == phenotype else 0.0 for r in primary_rows],
        )
    add_variable(
        "phenotype_ge2",
        [1.0 if r["phenotype"] >= 2 else 0.0 for r in phua_rows],
        [1.0 if r["phenotype"] >= 2 else 0.0 for r in primary_rows],
    )
    for year in base.YEARS:
        add_variable(
            f"year_{year}",
            [1.0 if r["year"] == year else 0.0 for r in phua_rows],
            [1.0 if r["year"] == year else 0.0 for r in primary_rows],
        )
    return diagnostics


def stratified_bootstrap_indices(rows, rng):
    # Resample within collection waves so bootstrap samples keep wave composition.
    by_year: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_year.setdefault(row["year"], []).append(idx)
    sampled = []
    for year in sorted(by_year):
        indices = np.asarray(by_year[year], dtype=int)
        sampled.extend(rng.choice(indices, size=len(indices), replace=True).tolist())
    return sampled


def bootstrap_ipw(base, phua_rows, b: int, seed: int):
    # Bootstrap replicates refit selection and threshold models in each resample.
    rng = np.random.default_rng(seed)
    replicates = []
    failures = 0
    for rep in range(1, b + 1):
        sample = [phua_rows[i] for i in stratified_bootstrap_indices(phua_rows, rng)]
        try:
            selection = base.selection_analysis(sample)
            primary = base.dataset_for_primary(sample)
            results = base.fit_threshold_models(primary, weights=selection["weights"])
            beta = {item["threshold"]: math.log(item["or"]) for item in results}
            replicates.append(
                {
                    "replicate": rep,
                    "beta_ge1": beta[1],
                    "beta_ge2": beta[2],
                    "beta_ge3": beta[3],
                    "diff_ge2_ge1": beta[2] - beta[1],
                    "diff_ge3_ge1": beta[3] - beta[1],
                    "diff_ge3_ge2": beta[3] - beta[2],
                }
            )
        except Exception:
            failures += 1
    return replicates, failures


def summarize_bootstrap(base, original: dict[str, float], replicates: list[dict[str, float]]):
    # Bootstrap summaries are reported on log-OR and exponentiated scales.
    rows = []
    for key, label, scale in [
        ("beta_ge1", "phenotype >= 1", "or"),
        ("beta_ge2", "phenotype >= 2", "or"),
        ("beta_ge3", "phenotype >= 3", "or"),
        ("diff_ge2_ge1", "logOR >=2 minus >=1", "ratio_of_ors"),
        ("diff_ge3_ge1", "logOR >=3 minus >=1", "ratio_of_ors"),
        ("diff_ge3_ge2", "logOR >=3 minus >=2", "ratio_of_ors"),
    ]:
        values = np.asarray([r[key] for r in replicates], dtype=float)
        estimate = original[key]
        se = float(np.std(values, ddof=1))
        normal_p = base.p_from_z(estimate / se) if se > 0 else float("nan")
        lower = float(np.quantile(values, 0.025))
        upper = float(np.quantile(values, 0.975))
        tail_p = 2.0 * min(float(np.mean(values <= 0.0)), float(np.mean(values >= 0.0)))
        rows.append(
            [
                key,
                label,
                scale,
                f"{estimate:.8f}",
                f"{math.exp(estimate):.6f}",
                f"{np.mean(values):.8f}",
                f"{se:.8f}",
                f"{normal_p:.8g}",
                f"{tail_p:.8g}",
                f"{lower:.8f}",
                f"{upper:.8f}",
                f"{math.exp(lower):.6f}",
                f"{math.exp(upper):.6f}",
                len(values),
            ]
        )
    return rows


def main():
    base = load_base_module()
    b = int(os.environ.get("IPW_BOOTSTRAP_B", DEFAULT_BOOTSTRAP_B))
    seed = int(os.environ.get("IPW_BOOTSTRAP_SEED", SEED))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = base.load_rows(age_source=AGE_SOURCE)
    phua_rows = base.dataset_for_phenotype_plus_ua(rows)
    primary_rows = base.dataset_for_primary(rows)
    selection = base.selection_analysis(rows)
    weights = np.asarray(selection["weights"], dtype=float)

    robust = fit_all_thresholds_with_cov(base, primary_rows, weights)
    write_tsv(
        OUTPUT_DIR / "robust_variance_check.tsv",
        [
            [
                int(r["threshold"]),
                int(r["n"]),
                int(r["events"]),
                f"{r['or']:.6f}",
                f"{r['beta']:.8f}",
                f"{r['naive_se']:.8f}",
                f"{r['naive_p']:.8g}",
                f"{r['hc0_se']:.8f}",
                f"{r['hc0_p']:.8g}",
                f"{r['hc1_se']:.8f}",
                f"{r['hc1_lcl']:.6f}",
                f"{r['hc1_ucl']:.6f}",
                f"{r['hc1_p']:.8g}",
            ]
            for r in robust
        ],
        [
            "threshold",
            "n",
            "events",
            "or",
            "log_or",
            "naive_se",
            "naive_p",
            "hc0_se",
            "hc0_p",
            "hc1_se",
            "hc1_lcl",
            "hc1_ucl",
            "hc1_p",
        ],
    )

    weight_summary = selection_weight_summary(weights)
    write_tsv(
        OUTPUT_DIR / "weight_summary.tsv",
        [[key, f"{value:.8f}"] for key, value in weight_summary.items()],
        ["metric", "value"],
    )

    trimming_rows = []
    for method in ("none", "p99", "p95"):
        capped, cap = cap_weights(weights, method)
        results = fit_all_thresholds_with_cov(base, primary_rows, capped)
        summary = selection_weight_summary(capped)
        for r in results:
            trimming_rows.append(
                [
                    method,
                    "" if cap is None else f"{cap:.8f}",
                    f"{summary['max']:.8f}",
                    f"{summary['ess']:.2f}",
                    int(r["threshold"]),
                    f"{r['or']:.6f}",
                    f"{r['hc1_lcl']:.6f}",
                    f"{r['hc1_ucl']:.6f}",
                    f"{r['hc1_p']:.8g}",
                ]
            )
    write_tsv(
        OUTPUT_DIR / "weight_trimming_sensitivity.tsv",
        trimming_rows,
        ["trim", "cap", "max_weight", "ess", "threshold", "or", "hc1_lcl", "hc1_ucl", "hc1_p"],
    )

    write_tsv(
        OUTPUT_DIR / "covariate_balance.tsv",
        balance_rows(base, phua_rows, primary_rows, weights),
        [
            "variable",
            "target_mean",
            "complete_unweighted_mean",
            "complete_weighted_mean",
            "smd_before",
            "smd_after",
            "abs_smd_before",
            "abs_smd_after",
        ],
    )

    original = {
        "beta_ge1": robust[0]["beta"],
        "beta_ge2": robust[1]["beta"],
        "beta_ge3": robust[2]["beta"],
        "diff_ge2_ge1": robust[1]["beta"] - robust[0]["beta"],
        "diff_ge3_ge1": robust[2]["beta"] - robust[0]["beta"],
        "diff_ge3_ge2": robust[2]["beta"] - robust[1]["beta"],
    }
    replicates, failures = bootstrap_ipw(base, phua_rows, b=b, seed=seed)
    write_tsv(
        OUTPUT_DIR / "bootstrap_replicates.tsv",
        [
            [
                r["replicate"],
                f"{r['beta_ge1']:.8f}",
                f"{r['beta_ge2']:.8f}",
                f"{r['beta_ge3']:.8f}",
                f"{r['diff_ge2_ge1']:.8f}",
                f"{r['diff_ge3_ge1']:.8f}",
                f"{r['diff_ge3_ge2']:.8f}",
            ]
            for r in replicates
        ],
        ["replicate", "beta_ge1", "beta_ge2", "beta_ge3", "diff_ge2_ge1", "diff_ge3_ge1", "diff_ge3_ge2"],
    )
    boot_summary = summarize_bootstrap(base, original, replicates)
    write_tsv(
        OUTPUT_DIR / "bootstrap_summary.tsv",
        boot_summary,
        [
            "estimand",
            "label",
            "scale",
            "estimate_log",
            "estimate_exp",
            "bootstrap_mean_log",
            "bootstrap_se",
            "normal_p",
            "tail_p",
            "percentile_lcl_log",
            "percentile_ucl_log",
            "percentile_lcl_exp",
            "percentile_ucl_exp",
            "successful_replicates",
        ],
    )

if __name__ == "__main__":
    main()
