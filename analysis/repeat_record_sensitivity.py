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
OUTPUT_DIR = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", REPO_ROOT / "outputs" / "repeat_record_sensitivity"))
BASE_PATH = Path(__file__).resolve().with_name("main_analysis.py")
AGE_SOURCE = os.environ.get("ANALYSIS_AGE_SOURCE", "exact")
ANNUAL_WAVE_RE = re.compile(r"^\d{4}$")


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


def pseudo_cluster_labels(rows: list[dict[str, object]]) -> list[tuple[object, ...]]:
    # Annual-wave pseudo-clusters are defined from coarse stable fields when present.
    labels: list[tuple[object, ...]] = []
    for idx, row in enumerate(rows):
        year = str(row["year"])
        sex = row.get("sex")
        age_exact = row.get("age_exact")
        height = row.get("height")
        if ANNUAL_WAVE_RE.fullmatch(year):
            year_num = int(year)
            if 2021 <= year_num <= 2025 and sex is not None and age_exact is not None and height is not None:
                birth_year = int(round(year_num - float(age_exact)))
                height_cm = int(round(float(height)))
                labels.append(("linked", sex, birth_year, height_cm))
                continue
        labels.append(("single", idx))
    return labels


def cluster_robust_threshold(base, primary_rows, cluster_labels, threshold: int) -> dict[str, float]:
    # Cluster-robust covariance aggregates score residuals within pseudo-clusters.
    design = base.design_primary(primary_rows, exposure="ua_z")
    x, _ = base.logistic_matrix(design, exposure="ua_z", include_female=True, use_high_sbp=False, interaction=None)
    y = np.array([1.0 if row["phenotype"] >= threshold else 0.0 for row in primary_rows], dtype=float)
    model = base.fit_logistic(x, y)
    beta = float(model["beta"][1])
    naive_se = float(model["se"][1])
    pred = base.sigmoid(x @ model["beta"])
    bread = x.T @ ((pred * (1.0 - pred))[:, None] * x)
    bread_inv = np.linalg.pinv(bread)

    # The meat matrix is formed from cluster-level score sums.
    grouped_scores: dict[tuple[object, ...], np.ndarray] = defaultdict(lambda: np.zeros(x.shape[1], dtype=float))
    for label, row_x, row_y, row_p in zip(cluster_labels, x, y, pred):
        grouped_scores[label] += (row_y - row_p) * row_x
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for score in grouped_scores.values():
        meat += np.outer(score, score)

    n = x.shape[0]
    p = x.shape[1]
    g = len(grouped_scores)
    # Finite-sample correction follows the usual cluster sandwich adjustment.
    correction = 1.0
    if g > 1 and n > p:
        correction = (g / (g - 1.0)) * ((n - 1.0) / (n - p))
    cov = correction * (bread_inv @ meat @ bread_inv)
    cluster_se = math.sqrt(max(float(cov[1, 1]), 0.0))
    cluster_lcl, cluster_ucl = ci_from_beta(beta, cluster_se)
    naive_lcl, naive_ucl = ci_from_beta(beta, naive_se)
    return {
        "threshold": threshold,
        "n": float(n),
        "events": float(np.sum(y)),
        "or": math.exp(beta),
        "naive_lcl": naive_lcl,
        "naive_ucl": naive_ucl,
        "naive_p": base.p_from_z(beta / naive_se),
        "cluster_lcl": cluster_lcl,
        "cluster_ucl": cluster_ucl,
        "cluster_p": base.p_from_z(beta / cluster_se),
        "clusters": float(g),
        "correction": correction,
    }


def main() -> None:
    base = load_base_module()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = base.load_rows(age_source=AGE_SOURCE)
    primary_rows = base.dataset_for_primary(rows)
    cluster_labels = pseudo_cluster_labels(primary_rows)
    cluster_sizes = Counter(cluster_labels)
    multi_sizes = [size for size in cluster_sizes.values() if size > 1]
    records_in_multi = int(sum(multi_sizes))
    summary_rows = [
        ["age_source", AGE_SOURCE],
        ["primary_records", len(primary_rows)],
        ["clusters_total", len(cluster_sizes)],
        ["singleton_clusters", sum(1 for size in cluster_sizes.values() if size == 1)],
        ["multi_record_clusters", len(multi_sizes)],
        ["records_in_multi_record_clusters", records_in_multi],
        ["max_cluster_size", max(cluster_sizes.values()) if cluster_sizes else 0],
    ]
    write_tsv(
        OUTPUT_DIR / "cluster_summary.tsv",
        summary_rows,
        ["metric", "value"],
    )

    threshold_rows = []

    for threshold in (1, 2, 3):
        result = cluster_robust_threshold(base, primary_rows, cluster_labels, threshold)
        threshold_rows.append(
            [
                int(result["threshold"]),
                int(result["n"]),
                int(result["events"]),
                f"{result['or']:.6f}",
                f"{result['naive_lcl']:.6f}",
                f"{result['naive_ucl']:.6f}",
                f"{result['naive_p']:.8g}",
                f"{result['cluster_lcl']:.6f}",
                f"{result['cluster_ucl']:.6f}",
                f"{result['cluster_p']:.8g}",
                f"{result['clusters']:.0f}",
                f"{result['correction']:.8f}",
            ]
        )

    write_tsv(
        OUTPUT_DIR / "threshold_sensitivity.tsv",
        threshold_rows,
        [
            "threshold",
            "n",
            "events",
            "or",
            "naive_lcl",
            "naive_ucl",
            "naive_p",
            "cluster_lcl",
            "cluster_ucl",
            "cluster_p",
            "clusters",
            "small_sample_correction",
        ],
    )


if __name__ == "__main__":
    main()
