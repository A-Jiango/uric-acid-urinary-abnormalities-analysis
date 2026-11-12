from __future__ import annotations

import csv
import importlib.util
import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", REPO_ROOT / "outputs" / "test_ordering_selection"))
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


def summarize_stage(label: str, rows: list[dict[str, object]]) -> list[object]:
    # Stage summaries track sample size and basic covariate composition.
    ages = np.asarray([row["age_low"] for row in rows], dtype=float)
    females = np.asarray([row["female"] for row in rows], dtype=float)
    return [label, len(rows), f"{ages.mean():.6f}", f"{100.0 * females.mean():.6f}"]


def main() -> None:
    base = load_base_module()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = base.load_rows(age_source=AGE_SOURCE)
    # The stages follow the ordered filtering path used by the analysis scripts.
    adult_rows = [row for row in rows if row.get("adult_valid")]
    phenotype_rows = [row for row in adult_rows if row.get("phenotype") is not None]
    phenotype_plus_ua_rows = base.dataset_for_phenotype_plus_ua(rows)
    primary_rows = base.dataset_for_primary(rows)

    tsv_rows = [
        ["raw_records", len(rows), "", ""],
        summarize_stage("adult_valid", adult_rows),
        summarize_stage("valid_phenotype", phenotype_rows),
        summarize_stage("phenotype_plus_ua", phenotype_plus_ua_rows),
        summarize_stage("primary_complete", primary_rows),
    ]
    write_tsv(
        OUTPUT_DIR / "stage_profile.tsv",
        tsv_rows,
        ["stage", "n", "mean_age", "female_pct"],
    )

if __name__ == "__main__":
    main()
