from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

import extended_rows
import main_analysis as base


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", REPO_ROOT / "outputs" / "same_sample_urinalysis_sensitivity"))
AGE_SOURCE = os.environ.get("ANALYSIS_AGE_SOURCE", "exact")
THRESHOLDS = (1, 2, 3)
# Three-measure context is used for paired threshold conditioning; the extended
# context is used only for residualizing serum uric acid.
THREE_MEASURE_CONTEXT_TERMS = ("sg_z", "ph_z", "occult_pos")
EXTENDED_URINALYSIS_CONTEXT_TERMS = (
    "sg_z",
    "low_pH",
    "occult_pos",
    "nitrite_pos",
    "leuk_pos",
    "bilirubin_pos",
    "urobilinogen_abn",
    "ketone_pos",
    "glucose_pos",
)


def write_tsv(path: Path, rows: list[list[object]], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def finite_rows(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    # Complete-row filtering is keyed to the variables required by each model family.
    out: list[dict[str, object]] = []
    for row in rows:
        if all(row.get(key) is not None for key in keys):
            out.append(row)
    return out


def add_zscore(rows: list[dict[str, object]], source: str, target: str) -> tuple[float, float]:
    # Standardized copies keep model coefficients on comparable scales.
    values = [float(row[source]) for row in rows]
    scores, mean, sd = base.zscore(values)
    for row, value in zip(rows, scores):
        row[target] = float(value)
    return float(mean), float(sd)


def ols_fit(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    # Pseudoinverse OLS handles collinearity in expanded urinalysis context matrices.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.linalg.pinv(x.T @ x) @ (x.T @ y)
    pred = x @ beta
    return {"beta": beta, "pred": pred, "resid": y - pred}


def prepare_rows() -> list[dict[str, object]]:
    # Keep rows with the same-sample urine measures needed for paired models.
    column_map = base.load_column_map()
    missing_columns = [
        key
        for key in (
            "urine_specific_gravity",
            "urine_ph",
            "hematuria",
        )
        if not column_map.get(key)
    ]
    if missing_columns:
        raise ValueError(f"Column map is missing same-sample urinary keys: {', '.join(missing_columns)}")

    source_rows = extended_rows.load_enriched_rows(age_source=AGE_SOURCE)
    required = (
        "phenotype",
        "ua",
        "female",
        "age_low",
        "year",
        "sbp",
        "fpg",
        "bmi",
        "urine_sg",
        "urine_ph",
        "hematuria_protein_like_positive",
    )
    rows = finite_rows([dict(row) for row in source_rows if row.get("adult_valid")], required)
    rows = [
        row
        for row in rows
        if 1.0 <= float(row["urine_sg"]) <= 1.06 and 4.0 <= float(row["urine_ph"]) <= 9.5
    ]
    for row in rows:
        row["occult_pos"] = float(row["hematuria_protein_like_positive"])

    for source, target in (
        ("ua", "ua_z"),
        ("age_low", "age_z"),
        ("sbp", "sbp_z"),
        ("fpg", "fpg_z"),
        ("bmi", "bmi_z"),
        ("urine_sg", "sg_z"),
        ("urine_ph", "ph_z"),
    ):
        add_zscore(rows, source, target)
    return rows


def prepare_extended_urinalysis_rows() -> list[dict[str, object]]:
    # Broader urinalysis context is reserved for exposure residualization.
    column_map = base.load_column_map()
    missing_columns = [
        key
        for key in (
            "urine_specific_gravity",
            "urine_ph",
            "hematuria",
            "leukocyte_esterase",
            "nitrite",
            "bilirubin",
            "urobilinogen",
            "ketone",
            "urine_glucose",
        )
        if not column_map.get(key)
    ]
    if missing_columns:
        raise ValueError(f"Column map is missing same-sample urinary keys: {', '.join(missing_columns)}")

    source_rows = extended_rows.load_enriched_rows(age_source=AGE_SOURCE)
    required = (
        "phenotype",
        "ua",
        "female",
        "age_low",
        "year",
        "sbp",
        "fpg",
        "bmi",
        "urine_sg",
        "urine_ph",
        "hematuria_positive",
        "leuk_positive",
        "nitrite_positive",
        "bilirubin_positive",
        "urobilinogen_abnormal",
        "ketone_positive",
        "urine_glucose_positive",
    )
    rows = finite_rows([dict(row) for row in source_rows if row.get("adult_valid")], required)
    rows = [
        row
        for row in rows
        if 1.0 <= float(row["urine_sg"]) <= 1.06 and 4.0 <= float(row["urine_ph"]) <= 9.5
    ]
    for row in rows:
        row["occult_pos"] = float(row["hematuria_positive"])
        row["leuk_pos"] = float(row["leuk_positive"])
        row["nitrite_pos"] = float(row["nitrite_positive"])
        row["bilirubin_pos"] = float(row["bilirubin_positive"])
        row["urobilinogen_abn"] = float(row["urobilinogen_abnormal"])
        row["ketone_pos"] = float(row["ketone_positive"])
        row["glucose_pos"] = float(row["urine_glucose_positive"])
        row["low_pH"] = 1.0 if float(row["urine_ph"]) <= 5.5 else 0.0

    for source, target in (
        ("ua", "ua_z"),
        ("age_low", "age_z"),
        ("sbp", "sbp_z"),
        ("fpg", "fpg_z"),
        ("bmi", "bmi_z"),
        ("urine_sg", "sg_z"),
        ("urine_ph", "ph_z"),
    ):
        add_zscore(rows, source, target)
    return rows


def base_covariate_matrix(rows: list[dict[str, object]]) -> tuple[np.ndarray, list[str], str]:
    # Residualization starts from the same base covariates used in threshold models.
    year_cols, year_names, base_year = year_terms(rows)
    columns = [
        np.ones(len(rows), dtype=float),
        np.asarray([float(row["female"]) for row in rows], dtype=float),
        np.asarray([float(row["age_z"]) for row in rows], dtype=float),
        np.asarray([float(row["sbp_z"]) for row in rows], dtype=float),
        np.asarray([float(row["fpg_z"]) for row in rows], dtype=float),
        np.asarray([float(row["bmi_z"]) for row in rows], dtype=float),
        *year_cols,
    ]
    names = ["intercept", "female", "age_z", "sbp_z", "fpg_z", "bmi_z", *year_names]
    return base.build_matrix(columns), names, base_year


def stratified_folds(rows: list[dict[str, object]], n_folds: int = 5, seed: int = 20260418) -> list[list[int]]:
    # Fold assignment is stratified by collection wave.
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        grouped.setdefault(str(row["year"]), []).append(idx)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for indices in grouped.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        for pos, idx in enumerate(shuffled):
            folds[pos % n_folds].append(idx)
    return folds


def add_residualized_ua(rows: list[dict[str, object]], context_terms: tuple[str, ...]) -> float:
    # Cross-fitted residuals evaluate each row with a projection fitted on other folds.
    y = np.asarray([float(row["ua_z"]) for row in rows], dtype=float)
    x_base, _, _ = base_covariate_matrix(rows)
    x_context = np.column_stack(
        [x_base] + [np.asarray([float(row[key]) for row in rows], dtype=float) for key in context_terms]
    )
    pred = np.zeros(len(rows), dtype=float)
    for fold in stratified_folds(rows):
        holdout = np.asarray(fold, dtype=int)
        train = np.ones(len(rows), dtype=bool)
        train[holdout] = False
        model = ols_fit(x_context[train], y[train])
        pred[holdout] = x_context[holdout] @ model["beta"]
    resid = y - pred
    for row, value in zip(rows, resid):
        row["ua_resid_z"] = float(value)

    full_model = ols_fit(x_context, y)
    base_model = ols_fit(x_base, y)
    ss_base = float(base_model["resid"] @ base_model["resid"])
    ss_full = float(full_model["resid"] @ full_model["resid"])
    return 0.0 if ss_base <= 0 else 1.0 - ss_full / ss_base


def year_terms(rows: list[dict[str, object]]) -> tuple[list[np.ndarray], list[str], str]:
    years = [str(row["year"]) for row in rows]
    cols, names, base_year = base.year_dummies(years)
    return cols, [f"year_{name}" for name in names], str(base_year)


def design_matrix(
    rows: list[dict[str, object]],
    ua_key: str,
    context_terms: tuple[str, ...] = (),
) -> tuple[np.ndarray, list[str], str]:
    # Urinary context terms are appended after the base adjustment covariates.
    year_cols, year_names, base_year = year_terms(rows)
    columns: list[np.ndarray] = [
        np.ones(len(rows), dtype=float),
        np.asarray([float(row[ua_key]) for row in rows], dtype=float),
        np.asarray([float(row["female"]) for row in rows], dtype=float),
        np.asarray([float(row["age_z"]) for row in rows], dtype=float),
        *year_cols,
        np.asarray([float(row["sbp_z"]) for row in rows], dtype=float),
        np.asarray([float(row["fpg_z"]) for row in rows], dtype=float),
        np.asarray([float(row["bmi_z"]) for row in rows], dtype=float),
    ]
    names = ["intercept", ua_key, "female", "age_z", *year_names, "sbp_z", "fpg_z", "bmi_z"]

    for term in context_terms:
        columns.append(np.asarray([float(row[term]) for row in rows], dtype=float))
        names.append(term)

    return base.build_matrix(columns), names, base_year


def logistic_effect(model: dict[str, object], index: int) -> tuple[float, float, float, float]:
    beta = float(model["beta"][index])
    se = float(model["se"][index])
    lower, upper = base.ci_from_beta(beta, se)
    p_value = base.p_from_z(beta / se)
    return float(np.exp(np.clip(beta, -50.0, 50.0))), lower, upper, p_value


def fit_threshold(
    rows: list[dict[str, object]],
    threshold: int,
    ua_key: str,
    context_terms: tuple[str, ...] = (),
) -> dict[str, object]:
    # The same binary threshold equation is used for raw and residualized exposure.
    y = np.asarray([1 if int(row["phenotype"]) >= threshold else 0 for row in rows], dtype=int)
    x, names, base_year = design_matrix(rows, ua_key=ua_key, context_terms=context_terms)
    model = base.fit_logistic(x, y)
    ua_index = names.index(ua_key)
    odds_ratio, lower, upper, p_value = logistic_effect(model, ua_index)
    return {
        "threshold": f"Phenotype >={threshold}",
        "ua_term": ua_key,
        "context_terms": "+".join(context_terms) if context_terms else "none",
        "n": len(rows),
        "events": int(y.sum()),
        "or": odds_ratio,
        "lcl": lower,
        "ucl": upper,
        "p": p_value,
        "base_year": base_year,
    }


def format_model_row(model: dict[str, object]) -> list[object]:
    return [
        model["threshold"],
        model["ua_term"],
        model["context_terms"],
        model["n"],
        model["events"],
        f"{model['or']:.6f}",
        f"{model['lcl']:.6f}",
        f"{model['ucl']:.6f}",
        f"{model['p']:.6g}",
        model["base_year"],
    ]


def threshold_models(rows: list[dict[str, object]]) -> list[list[object]]:
    # Each threshold is fitted with base covariates and with added urine context.
    out: list[list[object]] = []
    for threshold in THRESHOLDS:
        out.append(format_model_row(fit_threshold(rows, threshold, ua_key="ua_z")))
        out.append(format_model_row(fit_threshold(rows, threshold, ua_key="ua_z", context_terms=THREE_MEASURE_CONTEXT_TERMS)))
    return out


def stepwise_models(rows: list[dict[str, object]]) -> list[list[object]]:
    # Stepwise conditioning adds same-sample urine measures in prespecified blocks.
    specs = (
        ("base", ()),
        ("+ urine specific gravity", ("sg_z",)),
        ("+ urine pH", ("ph_z",)),
        ("+ urine occult blood", ("occult_pos",)),
        ("+ urine specific gravity + urine pH", ("sg_z", "ph_z")),
        ("+ all three variables", ("sg_z", "ph_z", "occult_pos")),
    )
    out: list[list[object]] = []
    for label, terms in specs:
        model = fit_threshold(rows, 1, ua_key="ua_z", context_terms=terms)
        out.append(
            [
                label,
                model["n"],
                model["events"],
                f"{model['or']:.6f}",
                f"{model['lcl']:.6f}",
                f"{model['ucl']:.6f}",
                f"{model['p']:.6g}",
                model["base_year"],
            ]
        )
    return out


def residualized_models(rows: list[dict[str, object]]) -> list[list[object]]:
    # Raw and residualized exposure models are paired at the same thresholds.
    out: list[list[object]] = []
    for threshold in (2, 3):
        raw = fit_threshold(rows, threshold, ua_key="ua_z")
        residual = fit_threshold(rows, threshold, ua_key="ua_resid_z")
        out.append(format_model_row(raw))
        out.append(format_model_row(residual))
    return out


def summarize_rows(rows: list[dict[str, object]], residual_r2: float) -> list[list[object]]:
    years = sorted({str(row["year"]) for row in rows})
    counts = {threshold: sum(1 for row in rows if int(row["phenotype"]) >= threshold) for threshold in THRESHOLDS}
    return [
        ["age_source", AGE_SOURCE],
        ["n_complete_same_sample", len(rows)],
        ["collection_waves", ",".join(years)],
        ["phenotype_ge1_events", counts[1]],
        ["phenotype_ge2_events", counts[2]],
        ["phenotype_ge3_events", counts[3]],
        ["sua_residual_r2_from_urinary_context", f"{residual_r2:.6f}"],
    ]


def main() -> None:
    rows = prepare_rows()
    extended_rows_for_residualization = prepare_extended_urinalysis_rows()
    residual_r2 = add_residualized_ua(extended_rows_for_residualization, EXTENDED_URINALYSIS_CONTEXT_TERMS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_tsv(
        OUTPUT_DIR / "threshold_models.tsv",
        threshold_models(rows),
        ["threshold", "ua_term", "context_terms", "n", "events", "or", "lcl", "ucl", "p", "base_year"],
    )
    write_tsv(
        OUTPUT_DIR / "stepwise_conditioning_phenotype_ge1.tsv",
        stepwise_models(rows),
        ["model", "n", "events", "or", "lcl", "ucl", "p", "base_year"],
    )
    write_tsv(
        OUTPUT_DIR / "residualized_sua_models.tsv",
        residualized_models(extended_rows_for_residualization),
        ["threshold", "ua_term", "context_terms", "n", "events", "or", "lcl", "ucl", "p", "base_year"],
    )
    summary_rows = summarize_rows(rows, residual_r2)
    summary_rows.extend(
        [
            ["n_extended_urinalysis_residualized", len(extended_rows_for_residualization)],
            [
                "extended_urinalysis_residualized_context",
                "urine specific gravity, low-pH indicator, occult blood, leukocyte esterase, nitrite, bilirubin, urobilinogen, ketone, urine glucose",
            ],
        ]
    )
    write_tsv(OUTPUT_DIR / "summary.tsv", summary_rows, ["metric", "value"])


if __name__ == "__main__":
    main()
