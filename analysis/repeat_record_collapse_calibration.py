from __future__ import annotations

import csv
import importlib.util
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", REPO_ROOT / "outputs" / "repeat_record_collapse_calibration"))
BASE_PATH = Path(__file__).resolve().with_name("main_analysis.py")
AGE_SOURCE = os.environ.get("ANALYSIS_AGE_SOURCE", "exact")
ANNUAL_WAVE_RE = re.compile(r"^\d{4}$")
RNG_SEED = 12345
N_MONTE_CARLO = 1000


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


def ci_from_beta(beta: float, se: float) -> tuple[float, float]:
    return math.exp(beta - 1.96 * se), math.exp(beta + 1.96 * se)


def annual_primary_rows(base, rows):
    # Collapse calibration is restricted to annual waves with repeated-record fields.
    primary_rows = base.dataset_for_primary(rows)
    annual = []
    for row in primary_rows:
        year = str(row["year"])
        if ANNUAL_WAVE_RE.fullmatch(year):
            year_num = int(year)
            if 2021 <= year_num <= 2025:
                annual.append(row)
    return annual


def primary_cluster_labels(rows: list[dict[str, object]]) -> list[tuple[object, ...]]:
    # Pseudo-clusters use coarse fields available in annual waves; rows without
    # all fields remain singletons.
    labels: list[tuple[object, ...]] = []
    for idx, row in enumerate(rows):
        year = int(str(row["year"]))
        sex = row.get("sex")
        age_exact = row.get("age_exact")
        height = row.get("height")
        if sex is not None and age_exact is not None and height is not None:
            birth_year = int(round(year - float(age_exact)))
            height_cm = int(round(float(height)))
            labels.append(("linked", sex, birth_year, height_cm))
        else:
            labels.append(("single", idx))
    return labels


def build_fixed_design(base, rows: list[dict[str, object]]):
    # The fixed design is reused across all record-selection draws.
    n = len(rows)
    age_z, _, _ = base.zscore([row["age_low"] for row in rows])
    ua_z, _, _ = base.zscore([row["ua"] for row in rows])
    sbp_z, _, _ = base.zscore([row["sbp"] for row in rows])
    fpg_z, _, _ = base.zscore([row["fpg"] for row in rows])
    bmi_z, _, _ = base.zscore([row["bmi"] for row in rows])
    female = np.asarray([row["female"] for row in rows], dtype=float)
    year_cols, _, _ = base.year_dummies([row["year"] for row in rows])
    x = base.build_matrix([np.ones(n, dtype=float), ua_z, female, age_z, sbp_z, fpg_z, bmi_z] + year_cols)
    phenotype = np.asarray([int(row["phenotype"]) for row in rows], dtype=int)
    return {"x": x, "phenotype": phenotype}


def fit_thresholds_on_subset(base, design, indices: np.ndarray):
    # Subset fits use the same design columns and vary only the retained rows.
    x = design["x"][indices]
    phenotype = design["phenotype"][indices]
    rows = []
    for threshold in (1, 2, 3):
        y = np.asarray(phenotype >= threshold, dtype=float)
        model = base.fit_logistic(x, y)
        beta = float(model["beta"][1])
        se = float(model["se"][1])
        lcl, ucl = ci_from_beta(beta, se)
        rows.append(
            {
                "threshold": threshold,
                "n": len(indices),
                "events": int(np.sum(y)),
                "or": math.exp(beta),
                "lcl": lcl,
                "ucl": ucl,
                "p": base.p_from_z(beta / se),
            }
        )
    return rows


def group_indices(labels: list[tuple[object, ...]]):
    # Group labels define the candidate records available to each pseudo-cluster.
    grouped: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        grouped[label].append(idx)
    return grouped


def max_phenotype_choice(rows, grouped):
    # Select the highest observed phenotype within each pseudo-cluster.
    picks = []
    for indices in grouped.values():
        best = max(indices, key=lambda idx: (rows[idx]["phenotype"], rows[idx]["ua"], -idx))
        picks.append(best)
    return np.asarray(sorted(picks), dtype=int)


def monte_carlo_random_choices(grouped, rng, reps: int):
    # Draw one record per pseudo-cluster repeatedly to vary the retained row.
    clusters = list(grouped.values())
    draws = []
    for _ in range(reps):
        chosen = [indices[int(rng.integers(0, len(indices)))] if len(indices) > 1 else indices[0] for indices in clusters]
        draws.append(np.asarray(sorted(chosen), dtype=int))
    return draws


def summarize_draws(results_by_threshold):
    # Monte Carlo summaries describe the distribution of ORs across random draws.
    rows = []
    for threshold, ors in sorted(results_by_threshold.items()):
        arr = np.asarray(ors, dtype=float)
        rows.append(
            [
                threshold,
                f"{np.median(arr):.6f}",
                f"{np.quantile(arr, 0.025):.6f}",
                f"{np.quantile(arr, 0.975):.6f}",
                f"{np.mean(arr > 1.0):.6f}",
            ]
        )
    return rows


def main() -> None:
    base = load_base_module()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = base.load_rows(age_source=AGE_SOURCE)
    annual_rows = annual_primary_rows(base, rows)
    labels = primary_cluster_labels(annual_rows)
    grouped = group_indices(labels)
    cluster_sizes = Counter(labels)
    multi_sizes = [size for size in cluster_sizes.values() if size > 1]

    design = build_fixed_design(base, annual_rows)
    baseline_indices = np.arange(len(annual_rows), dtype=int)
    baseline = fit_thresholds_on_subset(base, design, baseline_indices)

    rng = np.random.default_rng(RNG_SEED)
    random_draws = monte_carlo_random_choices(grouped, rng, N_MONTE_CARLO)
    random_or_by_threshold = {1: [], 2: [], 3: []}
    random_rows = []
    for rep, indices in enumerate(random_draws, start=1):
        fitted = fit_thresholds_on_subset(base, design, indices)
        for item in fitted:
            threshold = int(item["threshold"])
            random_or_by_threshold[threshold].append(item["or"])
            random_rows.append(
                [
                    rep,
                    threshold,
                    int(item["n"]),
                    int(item["events"]),
                    f"{item['or']:.6f}",
                    f"{item['lcl']:.6f}",
                    f"{item['ucl']:.6f}",
                    f"{item['p']:.8g}",
                ]
            )

    maxphen_indices = max_phenotype_choice(annual_rows, grouped)
    maxphen = fit_thresholds_on_subset(base, design, maxphen_indices)

    summary_rows = [
        ["age_source", AGE_SOURCE],
        ["rng_seed", RNG_SEED],
        ["monte_carlo_reps", N_MONTE_CARLO],
        ["annual_primary_records", len(annual_rows)],
        ["clusters_total", len(cluster_sizes)],
        ["singleton_clusters", sum(1 for size in cluster_sizes.values() if size == 1)],
        ["multi_record_clusters", len(multi_sizes)],
        ["records_in_multi_record_clusters", int(sum(multi_sizes))],
        ["max_cluster_size", max(cluster_sizes.values()) if cluster_sizes else 0],
    ]

    baseline_rows = [
        [int(item["threshold"]), int(item["n"]), int(item["events"]), f"{item['or']:.6f}", f"{item['lcl']:.6f}", f"{item['ucl']:.6f}", f"{item['p']:.8g}"]
        for item in baseline
    ]
    maxphen_rows = [
        [int(item["threshold"]), int(item["n"]), int(item["events"]), f"{item['or']:.6f}", f"{item['lcl']:.6f}", f"{item['ucl']:.6f}", f"{item['p']:.8g}"]
        for item in maxphen
    ]
    monte_summary_rows = summarize_draws(random_or_by_threshold)

    write_tsv(OUTPUT_DIR / "summary.tsv", summary_rows, ["metric", "value"])
    write_tsv(OUTPUT_DIR / "baseline.tsv", baseline_rows, ["threshold", "n", "events", "or", "lcl", "ucl", "p"])
    write_tsv(
        OUTPUT_DIR / "random_draws.tsv",
        random_rows,
        ["replicate", "threshold", "n", "events", "or", "lcl", "ucl", "p"],
    )
    write_tsv(
        OUTPUT_DIR / "random_summary.tsv",
        monte_summary_rows,
        ["threshold", "or_median", "or_p2_5", "or_p97_5", "p_or_gt_1"],
    )
    write_tsv(OUTPUT_DIR / "maxphen.tsv", maxphen_rows, ["threshold", "n", "events", "or", "lcl", "ucl", "p"])

if __name__ == "__main__":
    main()
