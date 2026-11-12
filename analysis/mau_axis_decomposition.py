from __future__ import annotations

import csv
import importlib.util
import math
import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", REPO_ROOT / "outputs" / "mau_axis_decomposition"))
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


def main() -> None:
    base = load_base_module()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = base.load_rows(age_source=AGE_SOURCE)
    primary_rows = base.dataset_for_primary(rows)

    mau_axis_rows = []
    for row in primary_rows:
        # The MAU-only axis is evaluated among protein-negative or trace records.
        if row.get("prot1") != 0:
            continue
        mau_level = row.get("mau_level")
        # Adjacent low-grade MAU values are collapsed to an ordered three-level axis.
        if mau_level == 10:
            axis = 0
        elif mau_level == 30:
            axis = 1
        elif mau_level in (80, 150):
            axis = 2
        else:
            continue
        record = dict(row)
        record["mau_axis"] = axis
        mau_axis_rows.append(record)

    count_rows = []
    for axis in (0, 1, 2):
        count_rows.append([axis, sum(1 for row in mau_axis_rows if row["mau_axis"] == axis)])
    write_tsv(
        OUTPUT_DIR / "mau_axis_counts.tsv",
        count_rows,
        ["mau_axis", "n"],
    )

    design = base.design_primary(mau_axis_rows, exposure="ua_z")
    x, _ = base.logistic_matrix(design, exposure="ua_z", include_female=True, use_high_sbp=False, interaction=None)

    threshold_rows = []
    for threshold in (1, 2):
        # Threshold models treat the ordered MAU-only axis as cumulative binary endpoints.
        y = np.array([1.0 if row["mau_axis"] >= threshold else 0.0 for row in mau_axis_rows], dtype=float)
        model = base.fit_logistic(x, y)
        beta = float(model["beta"][1])
        se = float(model["se"][1])
        lcl, ucl = base.ci_from_beta(beta, se)
        pvalue = base.p_from_z(beta / se)
        threshold_rows.append(
            [
                threshold,
                len(mau_axis_rows),
                int(np.sum(y)),
                f"{math.exp(beta):.6f}",
                f"{lcl:.6f}",
                f"{ucl:.6f}",
                f"{pvalue:.8g}",
            ]
        )
    write_tsv(
        OUTPUT_DIR / "mau_axis_threshold_models.tsv",
        threshold_rows,
        ["threshold", "n", "events", "or", "lcl", "ucl", "p"],
    )

    # The ordered-logit summary uses the full three-level MAU-only axis.
    y_ord = np.array([row["mau_axis"] for row in mau_axis_rows], dtype=int)
    ordered = base.fit_ordered_logit(design["x"], y_ord, 3)
    beta = float(ordered["beta"][0])
    se = float(ordered["beta_se"][0])
    lcl, ucl = base.ci_from_beta(beta, se)
    pvalue = base.p_from_z(beta / se)
    ordered_rows = [[len(mau_axis_rows), f"{math.exp(beta):.6f}", f"{lcl:.6f}", f"{ucl:.6f}", f"{pvalue:.8g}"]]
    write_tsv(
        OUTPUT_DIR / "mau_axis_ordered_summary.tsv",
        ordered_rows,
        ["n", "or", "lcl", "ucl", "p"],
    )


if __name__ == "__main__":
    main()
