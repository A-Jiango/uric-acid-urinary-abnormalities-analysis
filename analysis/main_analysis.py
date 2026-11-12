from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from scipy.optimize import minimize
from scipy.stats import chi2, norm, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.environ["ANALYSIS_WORKBOOK"]) if os.environ.get("ANALYSIS_WORKBOOK") else None
OUTPUT_DIR = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", REPO_ROOT / "outputs" / "main_analysis"))
COLUMN_MAP_PATH = Path(os.environ.get("ANALYSIS_COLUMN_MAP", REPO_ROOT / "data" / "column_map.json"))
SHEET_NAME_RE = re.compile(r"^(\d{4})(?:-(\d{4}))?$")
YEARS: list[str] = []
MAU_LEVELS = [10.0, 30.0, 80.0, 150.0]
MAX_PLAUSIBLE_AGE = 120
REQUIRED_COLUMN_KEYS = (
    "sex",
    "age_band",
    "fpg",
    "serum_uric_acid",
    "sbp",
    "height",
    "triglycerides",
    "hdl_cholesterol",
    "bmi",
    "urine_protein",
    "mau",
    "serum_creatinine",
)


def load_column_map(path: Path = COLUMN_MAP_PATH) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            "Set ANALYSIS_COLUMN_MAP to a local JSON column map, or create data/column_map.json from config/column_map.example.json."
        )
    with path.open(encoding="utf-8") as f:
        column_map = json.load(f)
    missing = [key for key in REQUIRED_COLUMN_KEYS if not column_map.get(key)]
    if missing:
        raise ValueError(f"Column map is missing required keys: {', '.join(missing)}")
    return column_map


def require_data_path() -> Path:
    if DATA_PATH is None:
        raise FileNotFoundError("Set ANALYSIS_WORKBOOK to an authorized local workbook before running the analysis.")
    return DATA_PATH


def parse_age(value):
    if value is None:
        return None
    match = re.match(r"^(\d+)-(\d+)$", str(value).strip())
    return int(match.group(1)) if match else None


def parse_exact_age(value):
    # Exact ages are accepted as integer ages within a plausible adult range.
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "NULL", "无效", "nan"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not number.is_integer():
        return None
    age = int(number)
    if age < 1 or age > MAX_PLAUSIBLE_AGE:
        return None
    return age


def parse_float(value):
    # Numeric fields are parsed before variable-specific range checks.
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text in {"", "NULL", "无效", "nan"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number == 0:
        return None
    return number


def clean_range(value, low=None, high=None):
    # Range filters remove implausible values before model construction.
    if value is None:
        return None
    if low is not None and value < low:
        return None
    if high is not None and value > high:
        return None
    return value


def normalize_sex(value):
    text = str(value).strip()
    return text if text in {"男", "女"} else None


def normalize_protein(value):
    # Collapse strip-style protein entries to the ordered categories used below.
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "NULL"}:
        return None
    text = (
        text.replace("＋", "+")
        .replace("－", "-")
        .replace("—", "-")
        .replace("–", "-")
        .replace("（", "(")
        .replace("）", ")")
        .replace(" ", "")
        .lower()
    )
    if "阴性" in text or text in {"-", "(-)", "negative", "neg", "normal"}:
        return "neg"
    if "trace" in text or "+-" in text or "±" in text:
        return "trace"
    if "1+" in text or "+1" in text:
        return "1+"
    if "2+" in text or "+2" in text:
        return "2+"
    if "3+" in text or "+3" in text:
        return "3+"
    if "4+" in text or "+4" in text:
        return "4+"
    return None


def protein_group(prot_detail):
    # Three-level grouping keeps negative, trace, and positive strip results separate.
    if prot_detail is None:
        return None
    if prot_detail == "neg":
        return "negative"
    if prot_detail == "trace":
        return "trace"
    return "1plus_or_more"


def phenotype_level(mau_level, prot_detail):
    # The hierarchy combines the MAU grade with the urine-protein strip result.
    if mau_level not in MAU_LEVELS or prot_detail is None:
        return None
    if mau_level == 150.0 or prot_detail in {"1+", "2+", "3+", "4+"}:
        return 3
    if mau_level == 80.0 and prot_detail in {"neg", "trace"}:
        return 2
    if (mau_level == 30.0 and prot_detail in {"neg", "trace"}) or (mau_level == 10.0 and prot_detail == "trace"):
        return 1
    if mau_level == 10.0 and prot_detail == "neg":
        return 0
    return None


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def zscore(values):
    # Continuous covariates are standardized so coefficients represent per-SD changes.
    arr = np.asarray(values, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=0)
    if std == 0:
        return np.zeros_like(arr), mean, std
    return (arr - mean) / std, mean, std


def year_dummies(values):
    # The first sorted collection wave is the reference category.
    levels = sorted(set(values))
    base = levels[0]
    cols = []
    names = []
    for level in levels[1:]:
        cols.append(np.array([1.0 if v == level else 0.0 for v in values], dtype=float))
        names.append(level)
    return cols, names, base


def build_matrix(columns):
    return np.column_stack(columns).astype(float)


def ci_from_beta(beta, se):
    lower = float(np.exp(np.clip(beta - 1.96 * se, -50.0, 50.0)))
    upper = float(np.exp(np.clip(beta + 1.96 * se, -50.0, 50.0)))
    return lower, upper


def p_from_z(z_value):
    return float(2.0 * norm.sf(abs(z_value)))


def finite_difference_hessian(objective, params, step_scale=1e-4):
    # Central differences approximate the likelihood curvature for covariance estimates.
    params = np.asarray(params, dtype=float)
    n_params = len(params)
    steps = step_scale * np.maximum(1.0, np.abs(params))
    base = float(objective(params))
    hessian = np.zeros((n_params, n_params), dtype=float)

    for i in range(n_params):
        step_i = steps[i]
        e_i = np.zeros(n_params, dtype=float)
        e_i[i] = step_i
        f_plus = float(objective(params + e_i))
        f_minus = float(objective(params - e_i))
        hessian[i, i] = (f_plus - 2.0 * base + f_minus) / (step_i**2)

        for j in range(i + 1, n_params):
            step_j = steps[j]
            e_j = np.zeros(n_params, dtype=float)
            e_j[j] = step_j
            f_pp = float(objective(params + e_i + e_j))
            f_pm = float(objective(params + e_i - e_j))
            f_mp = float(objective(params - e_i + e_j))
            f_mm = float(objective(params - e_i - e_j))
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step_i * step_j)
            hessian[i, j] = value
            hessian[j, i] = value

    return 0.5 * (hessian + hessian.T)


def covariance_from_hessian(hessian):
    hessian = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    try:
        return np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(hessian)


def fit_logistic(x, y, maxiter=500):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    beta0 = np.zeros(x.shape[1], dtype=float)

    def objective(beta):
        # Stable Bernoulli negative log-likelihood for a logit link.
        linear = x @ beta
        return np.logaddexp(0.0, linear).sum() - np.dot(y, linear)

    def gradient(beta):
        linear = x @ beta
        return x.T @ (sigmoid(linear) - y)

    result = minimize(objective, beta0, jac=gradient, method="BFGS", options={"maxiter": maxiter})
    if not result.success:
        result = minimize(objective, beta0, jac=gradient, method="L-BFGS-B", options={"maxiter": maxiter})
    if not result.success:
        raise RuntimeError(f"logistic model failed: {result.message}")

    pred = sigmoid(x @ result.x)
    hessian = x.T @ ((pred * (1.0 - pred))[:, None] * x)
    try:
        cov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hessian)
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    return {
        "beta": result.x,
        "se": se,
        "ll": -objective(result.x),
        "pred": pred,
        "n": len(y),
        "converged": True,
    }


def fit_weighted_logistic(x, y, weights, maxiter=500):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    beta0 = np.zeros(x.shape[1], dtype=float)

    def objective(beta):
        linear = x @ beta
        loss = np.logaddexp(0.0, linear) - y * linear
        return float(np.dot(weights, loss))

    def gradient(beta):
        linear = x @ beta
        return x.T @ (weights * (sigmoid(linear) - y))

    result = minimize(objective, beta0, jac=gradient, method="BFGS", options={"maxiter": maxiter})
    if not result.success:
        result = minimize(objective, beta0, jac=gradient, method="L-BFGS-B", options={"maxiter": maxiter})
    if not result.success:
        raise RuntimeError(f"weighted logistic model failed: {result.message}")

    pred = sigmoid(x @ result.x)
    # Inverse-probability weights enter the score equations; the sandwich
    # covariance keeps uncertainty tied to the weighted estimating equations.
    hessian = x.T @ ((weights * pred * (1.0 - pred))[:, None] * x)
    try:
        bread = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(hessian)
    scores = (weights * (y - pred))[:, None] * x
    meat = scores.T @ scores
    cov = bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    return {
        "beta": result.x,
        "se": se,
        "ll": -objective(result.x),
        "pred": pred,
        "n": len(y),
        "converged": True,
    }


def initial_cutpoints(y, n_categories):
    # Empirical cumulative proportions provide ordered-logit starting cutpoints.
    counts = np.bincount(y, minlength=n_categories).astype(float)
    probs = counts / counts.sum()
    cum = np.clip(np.cumsum(probs)[:-1], 1e-4, 1 - 1e-4)
    cuts = np.log(cum / (1.0 - cum))
    return cuts


def unpack_ordered_params(params, n_categories):
    # Exponentiated increments keep ordered-logit cutpoints monotone.
    first_cut = params[0]
    increments = np.exp(params[1 : n_categories - 1])
    cutpoints = [first_cut]
    for inc in increments:
        cutpoints.append(cutpoints[-1] + inc)
    cutpoints = np.asarray(cutpoints, dtype=float)
    beta = np.asarray(params[n_categories - 1 :], dtype=float)
    return cutpoints, beta


def ordered_probabilities(cutpoints, eta):
    # Category probabilities are recovered from adjacent cumulative logits.
    n = len(eta)
    k = len(cutpoints) + 1
    cdf = np.zeros((n, k - 1), dtype=float)
    for j, cut in enumerate(cutpoints):
        cdf[:, j] = sigmoid(cut - eta)
    probs = np.zeros((n, k), dtype=float)
    probs[:, 0] = cdf[:, 0]
    for j in range(1, k - 1):
        probs[:, j] = cdf[:, j] - cdf[:, j - 1]
    probs[:, k - 1] = 1.0 - cdf[:, k - 2]
    return np.clip(probs, 1e-12, 1.0)


def fit_ordered_logit(x, y, n_categories, maxiter=500):
    # Proportional-odds logit uses one slope vector across ordered categories.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    init_cuts = initial_cutpoints(y, n_categories)
    init_params = np.concatenate(
        [
            [init_cuts[0]],
            np.log(np.clip(np.diff(init_cuts), 1e-3, None)),
            np.zeros(x.shape[1], dtype=float),
        ]
    )

    def objective(params):
        cutpoints, beta = unpack_ordered_params(params, n_categories)
        eta = x @ beta
        probs = ordered_probabilities(cutpoints, eta)
        return -np.log(probs[np.arange(len(y)), y]).sum()

    attempts = [
        ("BFGS", {"maxiter": maxiter}),
        ("L-BFGS-B", {"maxiter": maxiter}),
        ("Powell", {"maxiter": maxiter}),
    ]
    result = None
    best_fun = None
    for method, options in attempts:
        current = minimize(objective, init_params, method=method, options=options)
        if best_fun is None or current.fun < best_fun:
            result = current
            best_fun = current.fun
        if current.success:
            result = current
            break
        init_params = current.x

    if result is None or (not result.success and not np.isfinite(result.fun)):
        raise RuntimeError("ordered logit failed without a finite solution")

    hessian = finite_difference_hessian(objective, result.x)
    cov = covariance_from_hessian(hessian)
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    cutpoints, beta = unpack_ordered_params(result.x, n_categories)
    eta = x @ beta
    probs = ordered_probabilities(cutpoints, eta)
    beta_se = se[n_categories - 1 :]
    return {
        "params": result.x,
        "cutpoints": cutpoints,
        "beta": beta,
        "beta_se": beta_se,
        "ll": -objective(result.x),
        "pred_probs": probs,
        "n": len(y),
        "converged": True,
    }


def lrt_pvalue(ll_base, ll_full, df):
    stat = max(0.0, 2.0 * (ll_full - ll_base))
    return stat, float(chi2.sf(stat, df))


def auc(scores, labels):
    # Rank-based AUC is used for binary selection-model discrimination summaries.
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def mean_sd(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def fit_ols(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xtx = x.T @ x
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    sigma2 = float((resid @ resid) / max(1, len(y) - x.shape[1]))
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 1e-12, None))
    return {
        "beta": beta,
        "se": se,
        "pred": x @ beta,
        "n": len(y),
    }


def percent(count, total):
    return 100.0 * count / total if total else 0.0


def available_values(rows, key):
    return [r.get(key) for r in rows if r.get(key) is not None]


def smd_continuous(a_values, b_values):
    # Standardized mean differences compare groups on a scale-free metric.
    a = np.asarray([v for v in a_values if v is not None], dtype=float)
    b = np.asarray([v for v in b_values if v is not None], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    pool = math.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2.0)
    return 0.0 if pool == 0 else float((a.mean() - b.mean()) / pool)


def smd_binary(a_values, b_values):
    a = np.asarray([v for v in a_values if v is not None], dtype=float)
    b = np.asarray([v for v in b_values if v is not None], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return None
    pa = float(a.mean())
    pb = float(b.mean())
    pool = math.sqrt((pa * (1.0 - pa) + pb * (1.0 - pb)) / 2.0)
    return 0.0 if pool == 0 else float((pa - pb) / pool)


def format_mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return ""
    mean, sd = mean_sd(values)
    return f"{mean:.2f} ({sd:.2f})"


def format_count_pct(values):
    values = [v for v in values if v is not None]
    if not values:
        return ""
    count = int(sum(values))
    return f"{count} ({percent(count, len(values)):.2f})"


def sheet_sort_key(name):
    match = SHEET_NAME_RE.fullmatch(str(name).strip())
    if match is None:
        return (float("inf"), float("inf"), str(name))
    start = int(match.group(1))
    end = int(match.group(2) or start)
    return (start, end, str(name))


def load_analysis_sheet_names(workbook):
    names = [name for name in workbook.sheetnames if SHEET_NAME_RE.fullmatch(str(name).strip())]
    if not names:
        raise ValueError("Input workbook does not contain analysis worksheets named YYYY or YYYY-YYYY")
    return sorted(names, key=sheet_sort_key)


def load_rows(age_source="exact"):
    global YEARS
    wb = load_workbook(require_data_path(), read_only=True, data_only=True)
    YEARS = load_analysis_sheet_names(wb)
    column_map = load_column_map()
    rows = []
    for year in YEARS:
        ws = wb[year]
        header = list(next(ws.iter_rows(min_row=1, max_row=1, values_only=True)))
        idx = {key: header.index(column_map[key]) for key in REQUIRED_COLUMN_KEYS}
        exact_age_column = column_map.get("exact_age")
        exact_age_idx = header.index(exact_age_column) if exact_age_column and exact_age_column in header else None
        for row in ws.iter_rows(min_row=2, values_only=True):
            sex = normalize_sex(row[idx["sex"]])
            age_band_low = parse_age(row[idx["age_band"]])
            age_exact = parse_exact_age(row[exact_age_idx]) if exact_age_idx is not None else None
            if age_source == "band_lower":
                analysis_age = age_band_low
            elif age_source == "exact":
                analysis_age = age_exact if age_exact is not None else age_band_low if exact_age_idx is None else None
            else:
                raise ValueError(f"unsupported age_source: {age_source}")
            adult_valid = sex is not None and analysis_age is not None and analysis_age >= 20

            if not adult_valid:
                rows.append(
                    {
                        "year": year,
                        "adult_valid": False,
                    }
                )
                continue

            ua = clean_range(parse_float(row[idx["serum_uric_acid"]]), 100, 1000)
            fpg = clean_range(parse_float(row[idx["fpg"]]), 2.5, 30)
            sbp = clean_range(parse_float(row[idx["sbp"]]), 70, 260)
            height = clean_range(parse_float(row[idx["height"]]), 100, 220)
            tg = clean_range(parse_float(row[idx["triglycerides"]]), 0.2, 30)
            hdl = clean_range(parse_float(row[idx["hdl_cholesterol"]]), 0.2, 5)
            bmi = clean_range(parse_float(row[idx["bmi"]]), 10, 80)
            mau = clean_range(parse_float(row[idx["mau"]]), 1, 5000)
            scr = clean_range(parse_float(row[idx["serum_creatinine"]]), 20, 2000)
            # The MAU hierarchy uses the predefined semi-quantitative grade values.
            mau_level = mau if mau in MAU_LEVELS else None
            prot_detail = normalize_protein(row[idx["urine_protein"]])
            prot_group3 = protein_group(prot_detail)
            phenotype = phenotype_level(mau_level, prot_detail)

            hu = None
            if ua is not None:
                if sex == "男":
                    hu = 1 if ua >= 420 else 0
                else:
                    hu = 1 if ua >= 360 else 0

            if hdl is None:
                hdl_low = None
            elif sex == "男":
                hdl_low = 1 if hdl < 1.0 else 0
            else:
                hdl_low = 1 if hdl < 1.3 else 0

            meta_count = None
            if bmi is not None and tg is not None and hdl is not None:
                meta_count = int(bmi >= 24) + int(tg >= 1.7) + hdl_low

            rows.append(
                {
                    "year": year,
                    "adult_valid": True,
                    "sex": sex,
                    "female": 1 if sex == "女" else 0,
                    # analysis_age keeps a stable key used by downstream design functions.
                    "age_low": analysis_age,
                    "age_exact": age_exact,
                    "age_band_low": age_band_low,
                    "age_band_raw": row[idx["age_band"]],
                    "ua": ua,
                    "hu": hu,
                    "sbp": sbp,
                    "high_sbp": 1 if sbp is not None and sbp >= 130 else 0 if sbp is not None else None,
                    "fpg": fpg,
                    "height": height,
                    "high_fpg": 1 if fpg is not None and fpg >= 5.6 else 0 if fpg is not None else None,
                    "bmi": bmi,
                    "tg": tg,
                    "hdl": hdl,
                    "meta_count": meta_count,
                    "scr": scr,
                    "mau": mau,
                    "mau_level": mau_level,
                    "mau30": 1 if mau is not None and mau >= 30 else 0 if mau is not None else None,
                    "mau80": 1 if mau is not None and mau >= 80 else 0 if mau is not None else None,
                    "prot_detail": prot_detail,
                    "prot_group3": prot_group3,
                    "prot1": 1 if prot_detail in {"1+", "2+", "3+", "4+"} else 0 if prot_detail is not None else None,
                    "phenotype": phenotype,
                }
            )
    return rows


def write_tsv(path, rows, header):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def sample_flow(rows):
    overall = Counter()
    by_year = defaultdict(Counter)
    for row in rows:
        year = row["year"]
        overall["raw"] += 1
        by_year[year]["raw"] += 1
        if not row["adult_valid"]:
            continue
        overall["adult_valid"] += 1
        by_year[year]["adult_valid"] += 1
        if row.get("ua") is not None:
            overall["valid_ua"] += 1
            by_year[year]["valid_ua"] += 1
        if row.get("mau_level") is not None:
            overall["valid_mau4"] += 1
            by_year[year]["valid_mau4"] += 1
        if row.get("prot_group3") is not None:
            overall["valid_protein"] += 1
            by_year[year]["valid_protein"] += 1
        if row.get("phenotype") is not None:
            overall["valid_phenotype"] += 1
            by_year[year]["valid_phenotype"] += 1
        if row.get("phenotype") is not None and row.get("ua") is not None:
            overall["phenotype_plus_ua"] += 1
            by_year[year]["phenotype_plus_ua"] += 1
        if all(row.get(key) is not None for key in ["phenotype", "ua", "sbp", "fpg", "bmi"]):
            overall["primary_model_complete"] += 1
            by_year[year]["primary_model_complete"] += 1
    return overall, by_year


def dataset_for_primary(rows):
    # Primary models require the urinary phenotype, exposure, and adjustment covariates.
    return [
        row
        for row in rows
        if row.get("adult_valid")
        and all(row.get(key) is not None for key in ["phenotype", "ua", "sbp", "fpg", "bmi", "sex", "age_low", "year"])
    ]


def dataset_for_phenotype_plus_ua(rows):
    # This subset supports selection modeling before complete-case covariates are required.
    return [row for row in rows if row.get("adult_valid") and row.get("phenotype") is not None and row.get("ua") is not None]


def logistic_matrix(primary_design, exposure="ua_z", include_female=True, use_high_sbp=False, interaction=None):
    # Column order is kept stable so the exposure coefficient has a fixed index.
    n = len(primary_design["ua_z"])
    cols = [np.ones(n, dtype=float)]
    names = ["intercept"]

    exposure_col = primary_design[exposure]
    cols.append(exposure_col)
    names.append(exposure)

    if use_high_sbp:
        cols.append(primary_design["high_sbp"])
        names.append("high_sbp")

    if include_female:
        cols.append(primary_design["female"])
        names.append("female")

    cols.extend([primary_design["age_z"], primary_design["sbp_z"], primary_design["fpg_z"], primary_design["bmi_z"]])
    names.extend(["age_z", "sbp_z", "fpg_z", "bmi_z"])

    cols.extend(primary_design["year_cols"])
    names.extend([f"year_{name}" for name in primary_design["year_names"]])

    if interaction == "sex":
        cols.append(exposure_col * primary_design["female"])
        names.append(f"{exposure}_x_female")
    elif interaction == "high_sbp":
        cols.append(exposure_col * primary_design["high_sbp"])
        names.append(f"{exposure}_x_high_sbp")

    return build_matrix(cols), names


def design_primary(rows, exposure="ua_z"):
    # Continuous adjustment covariates are standardized within the modeled subset.
    age_z, age_mean, age_sd = zscore([r["age_low"] for r in rows])
    ua_z, ua_mean, ua_sd = zscore([r["ua"] for r in rows])
    sbp_z, sbp_mean, sbp_sd = zscore([r["sbp"] for r in rows])
    fpg_z, fpg_mean, fpg_sd = zscore([r["fpg"] for r in rows])
    bmi_z, bmi_mean, bmi_sd = zscore([r["bmi"] for r in rows])
    female = np.array([r["female"] for r in rows], dtype=float)
    hu = np.array([r["hu"] for r in rows], dtype=float)
    year_cols, year_names, year_base = year_dummies([r["year"] for r in rows])

    named = {
        "ua_z": ua_z,
        "hu": hu,
        "female": female,
        "age_z": age_z,
        "sbp_z": sbp_z,
        "fpg_z": fpg_z,
        "bmi_z": bmi_z,
    }
    names = [exposure, "female", "age_z", "sbp_z", "fpg_z", "bmi_z"] + [f"year_{name}" for name in year_names]
    columns = [named[exposure], female, age_z, sbp_z, fpg_z, bmi_z] + year_cols
    return {
        "x": build_matrix(columns),
        "names": names,
        "ua_z": ua_z,
        "hu": hu,
        "female": female,
        "high_sbp": np.array([r["high_sbp"] for r in rows], dtype=float),
        "age_z": age_z,
        "sbp_z": sbp_z,
        "fpg_z": fpg_z,
        "bmi_z": bmi_z,
        "year_cols": year_cols,
        "year_names": year_names,
        "year_base": year_base,
        "scale": {
            "ua_mean": ua_mean,
            "ua_sd": ua_sd,
            "age_mean": age_mean,
            "age_sd": age_sd,
            "sbp_mean": sbp_mean,
            "sbp_sd": sbp_sd,
            "fpg_mean": fpg_mean,
            "fpg_sd": fpg_sd,
            "bmi_mean": bmi_mean,
            "bmi_sd": bmi_sd,
        },
    }


def egfr_ckdepi_2021(scr_umol, age_low, sex):
    scr = scr_umol / 88.4
    if sex == "女":
        k = 0.7
        alpha = -0.241
        sex_mult = 1.012
    else:
        k = 0.9
        alpha = -0.302
        sex_mult = 1.0
    return 142.0 * (min(scr / k, 1.0) ** alpha) * (max(scr / k, 1.0) ** -1.2) * (0.9938**age_low) * sex_mult


def primary_with_egfr(primary_rows):
    enriched = []
    for row in primary_rows:
        if row.get("scr") is None:
            continue
        record = dict(row)
        record["egfr"] = egfr_ckdepi_2021(record["scr"], record["age_low"], record["sex"])
        record["egfr_lt90"] = 1 if record["egfr"] < 90 else 0
        record["egfr_lt60"] = 1 if record["egfr"] < 60 else 0
        enriched.append(record)
    return enriched


def selection_analysis(rows):
    # Selection weights are estimated among rows with Phenotype and serum uric
    # acid available, then carried to the complete-covariate analysis subset.
    phua_rows = dataset_for_phenotype_plus_ua(rows)
    complete_rows = dataset_for_primary(rows)

    age_z, _, _ = zscore([r["age_low"] for r in phua_rows])
    ua_z, _, _ = zscore([r["ua"] for r in phua_rows])
    year_cols, year_names, _ = year_dummies([r["year"] for r in phua_rows])
    x = build_matrix(
        [
            np.ones(len(phua_rows), dtype=float),
            age_z,
            ua_z,
            np.array([r["female"] for r in phua_rows], dtype=float),
            np.array([1.0 if r["phenotype"] == 1 else 0.0 for r in phua_rows], dtype=float),
            np.array([1.0 if r["phenotype"] == 2 else 0.0 for r in phua_rows], dtype=float),
            np.array([1.0 if r["phenotype"] == 3 else 0.0 for r in phua_rows], dtype=float),
        ]
        + year_cols
    )
    names = ["intercept", "age_z", "ua_z", "female", "phenotype_1", "phenotype_2", "phenotype_3"] + [
        f"year_{name}" for name in year_names
    ]
    y = np.array(
        [1.0 if all(r.get(key) is not None for key in ["sbp", "fpg", "bmi", "sex", "age_low", "year"]) else 0.0 for r in phua_rows],
        dtype=float,
    )
    model = fit_logistic(x, y)
    probs = sigmoid(x @ model["beta"])
    complete_weights = np.array([1.0 / prob for prob, flag in zip(probs, y) if flag == 1.0], dtype=float)
    weight_summary = {
        "min": float(np.min(complete_weights)),
        "p25": float(np.quantile(complete_weights, 0.25)),
        "median": float(np.median(complete_weights)),
        "p75": float(np.quantile(complete_weights, 0.75)),
        "max": float(np.max(complete_weights)),
    }

    incomplete_rows = [r for r in phua_rows if not all(r.get(key) is not None for key in ["sbp", "fpg", "bmi", "sex", "age_low", "year"])]
    compare = {
        "n_complete": len(complete_rows),
        "n_incomplete": len(incomplete_rows),
        "completion_rate_pct": percent(len(complete_rows), len(phua_rows)),
        "complete_mean_age": float(np.mean([r["age_low"] for r in complete_rows])),
        "incomplete_mean_age": float(np.mean([r["age_low"] for r in incomplete_rows])),
        "complete_mean_ua": float(np.mean([r["ua"] for r in complete_rows])),
        "incomplete_mean_ua": float(np.mean([r["ua"] for r in incomplete_rows])),
        "complete_female_pct": 100.0 * float(np.mean([r["female"] for r in complete_rows])),
        "incomplete_female_pct": 100.0 * float(np.mean([r["female"] for r in incomplete_rows])),
    }

    def smd_cont(key):
        a = np.asarray([r[key] for r in complete_rows], dtype=float)
        b = np.asarray([r[key] for r in incomplete_rows], dtype=float)
        pool = math.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2.0)
        return 0.0 if pool == 0 else float((a.mean() - b.mean()) / pool)

    def smd_bin(fn):
        pa = float(np.mean([1.0 if fn(r) else 0.0 for r in complete_rows]))
        pb = float(np.mean([1.0 if fn(r) else 0.0 for r in incomplete_rows]))
        pool = math.sqrt((pa * (1.0 - pa) + pb * (1.0 - pb)) / 2.0)
        return 0.0 if pool == 0 else float((pa - pb) / pool)

    compare["smd_age"] = smd_cont("age_low")
    compare["smd_ua"] = smd_cont("ua")
    compare["smd_female"] = smd_bin(lambda r: r["female"] == 1)
    compare["smd_phenotype_ge2"] = smd_bin(lambda r: r["phenotype"] >= 2)
    compare["smd_phenotype_3"] = smd_bin(lambda r: r["phenotype"] == 3)

    by_year = []
    for year in YEARS:
        subset = [r for r in phua_rows if r["year"] == year]
        complete_subset = [r for r in subset if all(r.get(key) is not None for key in ["sbp", "fpg", "bmi", "sex", "age_low", "year"])]
        by_year.append([year, len(subset), len(complete_subset), f"{percent(len(complete_subset), len(subset)):.2f}"])

    by_phenotype = []
    for phenotype in range(4):
        subset = [r for r in phua_rows if r["phenotype"] == phenotype]
        complete_subset = [r for r in subset if all(r.get(key) is not None for key in ["sbp", "fpg", "bmi", "sex", "age_low", "year"])]
        by_phenotype.append([phenotype, len(subset), len(complete_subset), f"{percent(len(complete_subset), len(subset)):.2f}"])

    selection_rows = []
    for idx, name in enumerate(names[1:], start=1):
        beta = model["beta"][idx]
        se = model["se"][idx]
        lcl, ucl = ci_from_beta(beta, se)
        selection_rows.append([name, f"{math.exp(beta):.3f}", f"{lcl:.3f}", f"{ucl:.3f}", f"{p_from_z(beta / se):.6f}"])

    return {
        "phua_rows": phua_rows,
        "selection_model": model,
        "selection_compare": compare,
        "weight_summary": weight_summary,
        "year_rows": by_year,
        "phenotype_rows": by_phenotype,
        "selection_rows": selection_rows,
        "weights": complete_weights,
    }


def early_measurement_selection_analysis(rows):
    adult_rows = [r for r in rows if r.get("adult_valid")]
    phenotype_rows = [r for r in adult_rows if r.get("phenotype") is not None]
    no_phenotype_rows = [r for r in adult_rows if r.get("phenotype") is None]

    feature_specs = [
        ("Age, years", "age_low", "continuous"),
        ("Female sex", "female", "binary"),
        ("Serum uric acid, umol/L", "ua", "continuous"),
        ("Systolic blood pressure, mmHg", "sbp", "continuous"),
        ("Fasting plasma glucose, mmol/L", "fpg", "continuous"),
        ("BMI, kg/m2", "bmi", "continuous"),
        ("Serum creatinine, umol/L", "scr", "continuous"),
    ]

    comparison_rows = []
    availability_rows = []
    for label, key, kind in feature_specs:
        available_values_yes = available_values(phenotype_rows, key)
        available_values_no = available_values(no_phenotype_rows, key)
        availability_rows.append(
            [
                label,
                len(available_values_yes),
                f"{percent(len(available_values_yes), len(phenotype_rows)):.2f}",
                len(available_values_no),
                f"{percent(len(available_values_no), len(no_phenotype_rows)):.2f}",
            ]
        )
        if kind == "continuous":
            summary_yes = format_mean_sd([r.get(key) for r in phenotype_rows])
            summary_no = format_mean_sd([r.get(key) for r in no_phenotype_rows])
            smd = smd_continuous([r.get(key) for r in phenotype_rows], [r.get(key) for r in no_phenotype_rows])
        else:
            summary_yes = format_count_pct([r.get(key) for r in phenotype_rows])
            summary_no = format_count_pct([r.get(key) for r in no_phenotype_rows])
            smd = smd_binary([r.get(key) for r in phenotype_rows], [r.get(key) for r in no_phenotype_rows])
        comparison_rows.append(
            [
                label,
                len(available_values_yes),
                summary_yes,
                len(available_values_no),
                summary_no,
                "" if smd is None else f"{smd:.3f}",
            ]
        )

    by_year_rows = []
    for year in YEARS:
        subset = [r for r in adult_rows if r["year"] == year]
        available_n = sum(1 for r in subset if r.get("phenotype") is not None)
        by_year_rows.append([year, len(subset), available_n, f"{percent(available_n, len(subset)):.2f}"])

    y = np.array([1.0 if r.get("phenotype") is not None else 0.0 for r in adult_rows], dtype=float)
    age_z, _, _ = zscore([r["age_low"] for r in adult_rows])
    female = np.array([r["female"] for r in adult_rows], dtype=float)
    year_cols, year_names, year_base = year_dummies([r["year"] for r in adult_rows])
    x = build_matrix([np.ones(len(adult_rows), dtype=float), age_z, female] + year_cols)
    names = ["intercept", "age_z", "female"] + [f"year_{name}" for name in year_names]
    model = fit_logistic(x, y)
    basic_model_rows = []
    for idx, name in enumerate(names[1:], start=1):
        beta = model["beta"][idx]
        se = model["se"][idx]
        lcl, ucl = ci_from_beta(beta, se)
        basic_model_rows.append([name, f"{math.exp(beta):.3f}", f"{lcl:.3f}", f"{ucl:.3f}", f"{p_from_z(beta / se):.6f}"])

    covariate_keys = ["ua", "sbp", "fpg", "bmi", "scr"]
    covariate_rows = [r for r in adult_rows if all(r.get(key) is not None for key in covariate_keys)]
    covariate_model_rows = []
    covariate_model_summary = {
        "n": len(covariate_rows),
        "phenotype_available_n": sum(1 for r in covariate_rows if r.get("phenotype") is not None),
        "year_base": "",
    }
    if covariate_rows:
        y_covariate = np.array([1.0 if r.get("phenotype") is not None else 0.0 for r in covariate_rows], dtype=float)
        columns = [np.ones(len(covariate_rows), dtype=float)]
        covariate_names = ["intercept"]
        for key in ["age_low", "ua", "sbp", "fpg", "bmi", "scr"]:
            z, _, _ = zscore([r[key] for r in covariate_rows])
            columns.append(z)
            covariate_names.append(f"{key}_z")
        columns.append(np.array([r["female"] for r in covariate_rows], dtype=float))
        covariate_names.append("female")
        covariate_year_cols, covariate_year_names, covariate_year_base = year_dummies([r["year"] for r in covariate_rows])
        columns.extend(covariate_year_cols)
        covariate_names.extend([f"year_{name}" for name in covariate_year_names])
        covariate_model_summary["year_base"] = covariate_year_base
        covariate_model = fit_logistic(build_matrix(columns), y_covariate)
        for idx, name in enumerate(covariate_names[1:], start=1):
            beta = covariate_model["beta"][idx]
            se = covariate_model["se"][idx]
            lcl, ucl = ci_from_beta(beta, se)
            covariate_model_rows.append(
                [name, f"{math.exp(beta):.3f}", f"{lcl:.3f}", f"{ucl:.3f}", f"{p_from_z(beta / se):.6f}"]
            )

    return {
        "adult_n": len(adult_rows),
        "phenotype_available_n": len(phenotype_rows),
        "phenotype_unavailable_n": len(no_phenotype_rows),
        "comparison_rows": comparison_rows,
        "availability_rows": availability_rows,
        "by_year_rows": by_year_rows,
        "basic_model_rows": basic_model_rows,
        "basic_model_year_base": year_base,
        "covariate_model_rows": covariate_model_rows,
        "covariate_model_summary": covariate_model_summary,
    }


def fit_threshold_models(primary_rows, weights=None):
    # Cumulative thresholds model Phenotype >=1, >=2, and >=3 as binary endpoints.
    design = design_primary(primary_rows, exposure="ua_z")
    x, _ = logistic_matrix(design, exposure="ua_z", include_female=True, use_high_sbp=False, interaction=None)
    result_rows = []
    for threshold in [1, 2, 3]:
        y = np.array([1.0 if r["phenotype"] >= threshold else 0.0 for r in primary_rows], dtype=float)
        if weights is None:
            model = fit_logistic(x, y)
        else:
            model = fit_weighted_logistic(x, y, weights)
        beta = model["beta"][1]
        se = model["se"][1]
        lcl, ucl = ci_from_beta(beta, se)
        result_rows.append(
            {
                "threshold": threshold,
                "label": f"phenotype >= {threshold}",
                "n": len(primary_rows),
                "events": int(y.sum()),
                "or": math.exp(beta),
                "lcl": lcl,
                "ucl": ucl,
                "p": p_from_z(beta / se),
            }
        )
    return result_rows


def fit_partial_proportional_odds(primary_rows):
    # Covariates share one coefficient vector; serum uric acid is allowed to
    # vary across cumulative thresholds.
    design = design_primary(primary_rows, exposure="ua_z")
    y = np.array([r["phenotype"] for r in primary_rows], dtype=int)
    po = fit_ordered_logit(design["x"], y, 4)
    ua_col = design["ua_z"]
    common_cols = build_matrix(
        [design["female"], design["age_z"], design["sbp_z"], design["fpg_z"], design["bmi_z"]] + design["year_cols"]
    )
    n = len(y)
    n_categories = 4

    def unpack(params):
        first_cut = params[0]
        increments = np.exp(params[1 : n_categories - 1])
        cutpoints = [first_cut]
        for inc in increments:
            cutpoints.append(cutpoints[-1] + inc)
        cutpoints = np.asarray(cutpoints, dtype=float)
        ua_betas = np.asarray(params[n_categories - 1 : n_categories - 1 + (n_categories - 1)], dtype=float)
        beta_common = np.asarray(params[n_categories - 1 + (n_categories - 1) :], dtype=float)
        return cutpoints, ua_betas, beta_common

    def probabilities(params):
        # Each cumulative logit gets its own serum uric acid slope.
        cutpoints, ua_betas, beta_common = unpack(params)
        base = common_cols @ beta_common
        cdf = np.zeros((n, n_categories - 1), dtype=float)
        for j in range(n_categories - 1):
            eta_j = base + ua_col * ua_betas[j]
            cdf[:, j] = sigmoid(cutpoints[j] - eta_j)
        probs = np.zeros((n, n_categories), dtype=float)
        probs[:, 0] = cdf[:, 0]
        for j in range(1, n_categories - 1):
            probs[:, j] = cdf[:, j] - cdf[:, j - 1]
        probs[:, n_categories - 1] = 1.0 - cdf[:, n_categories - 2]
        return np.clip(probs, 1e-12, 1.0)

    def objective(params):
        probs = probabilities(params)
        return -np.log(probs[np.arange(n), y]).sum()

    init = np.concatenate(
        [
            [po["cutpoints"][0]],
            np.log(np.clip(np.diff(po["cutpoints"]), 1e-3, None)),
            np.repeat(po["beta"][0], n_categories - 1),
            po["beta"][1:],
        ]
    )
    result = minimize(objective, init, method="BFGS", options={"maxiter": 800})
    if not result.success:
        result = minimize(objective, result.x, method="L-BFGS-B", options={"maxiter": 1200})
    if not result.success:
        raise RuntimeError(f"partial proportional odds model failed: {result.message}")

    ppo_ll = -objective(result.x)
    stat, pvalue = lrt_pvalue(po["ll"], ppo_ll, 2)
    hessian = finite_difference_hessian(objective, result.x)
    cov = covariance_from_hessian(hessian)
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    cutpoints, ua_betas, beta_common = unpack(result.x)
    ua_start = n_categories - 1
    ua_stop = ua_start + (n_categories - 1)
    ua_se = se[ua_start:ua_stop]
    threshold_results = []
    for threshold, beta, beta_se in zip([1, 2, 3], ua_betas, ua_se):
        lcl, ucl = ci_from_beta(beta, beta_se)
        threshold_results.append(
            {
                "threshold": threshold,
                "label": f"ppo_threshold_{threshold}",
                "or": math.exp(beta),
                "lcl": lcl,
                "ucl": ucl,
                "p": p_from_z(beta / beta_se),
                "se": beta_se,
            }
        )
    return {
        "po_model": po,
        "ppo_ll": ppo_ll,
        "lrt_stat": stat,
        "lrt_p": pvalue,
        "ua_betas": ua_betas,
        "ua_or_by_threshold": [math.exp(beta) for beta in ua_betas],
        "ua_threshold_results": threshold_results,
        "cutpoints": cutpoints,
        "beta_common": beta_common,
    }


def wave_robustness(primary_rows):
    # Collection-wave checks refit the same threshold model within each wave
    # and after removing one wave at a time.
    overall = threshold_model(primary_rows, 2)
    by_year_rows = []
    leave_one_out_rows = []
    for year in YEARS:
        subset = [r for r in primary_rows if r["year"] == year]
        result = threshold_model(subset, 2)
        by_year_rows.append(
            [year, result["n"], result["events"], f"{result['or']:.3f}", f"{result['lcl']:.3f}", f"{result['ucl']:.3f}", f"{result['p']:.6f}"]
        )

        loo_subset = [r for r in primary_rows if r["year"] != year]
        loo_result = threshold_model(loo_subset, 2)
        leave_one_out_rows.append(
            [year, loo_result["n"], loo_result["events"], f"{loo_result['or']:.3f}", f"{loo_result['lcl']:.3f}", f"{loo_result['ucl']:.3f}", f"{loo_result['p']:.6f}"]
        )

    design = design_primary(primary_rows, exposure="ua_z")
    y = np.array([1.0 if r["phenotype"] >= 2 else 0.0 for r in primary_rows], dtype=float)
    x_base, names_base = logistic_matrix(design, exposure="ua_z", include_female=True, use_high_sbp=False, interaction=None)
    extra_cols = [design["ua_z"] * col for col in design["year_cols"]]
    x_int = build_matrix([x_base[:, i] for i in range(x_base.shape[1])] + extra_cols)
    base_model = fit_logistic(x_base, y)
    int_model = fit_logistic(x_int, y)
    _, heterogeneity_p = lrt_pvalue(base_model["ll"], int_model["ll"], len(extra_cols))
    return {
        "overall": overall,
        "by_year_rows": by_year_rows,
        "leave_one_out_rows": leave_one_out_rows,
        "year_interaction_p": heterogeneity_p,
    }


def level3_component_summary(primary_rows):
    # Level 3 is decomposed into its two construction components.
    level3_rows = [r for r in primary_rows if r["phenotype"] == 3]
    prot1_n = sum(r["prot1"] == 1 for r in level3_rows)
    mau150_n = sum(r["mau_level"] == 150 for r in level3_rows)
    both_n = sum(r["prot1"] == 1 and r["mau_level"] == 150 for r in level3_rows)
    prot1_only_n = sum(r["prot1"] == 1 and r["mau_level"] != 150 for r in level3_rows)
    mau150_only_n = sum(r["prot1"] != 1 and r["mau_level"] == 150 for r in level3_rows)
    return {
        "level3_n": len(level3_rows),
        "prot1_n": prot1_n,
        "mau150_n": mau150_n,
        "both_n": both_n,
        "prot1_only_n": prot1_only_n,
        "mau150_only_n": mau150_only_n,
        "rows": [
            ["level3_total", len(level3_rows)],
            ["level3_with_proteinuria_1plus", prot1_n],
            ["level3_with_mau150", mau150_n],
            ["level3_both", both_n],
            ["level3_proteinuria_only", prot1_only_n],
            ["level3_mau150_only", mau150_only_n],
        ],
    }


def fit_binary_outcome(rows, outcome_values):
    # Binary component models reuse the primary adjustment set for comparability.
    design = design_primary(rows, exposure="ua_z")
    x, _ = logistic_matrix(design, exposure="ua_z", include_female=True, use_high_sbp=False, interaction=None)
    y = np.asarray(outcome_values, dtype=float)
    model = fit_logistic(x, y)
    beta = model["beta"][1]
    se = model["se"][1]
    lcl, ucl = ci_from_beta(beta, se)
    return {
        "n": len(rows),
        "events": int(y.sum()),
        "or": math.exp(beta),
        "lcl": lcl,
        "ucl": ucl,
        "p": p_from_z(beta / se),
    }


def level3_contrast_analysis(primary_rows, ppo_results):
    # Component contrasts fit related endpoints under the same covariate structure.
    phenotype_ge3 = fit_binary_outcome(primary_rows, [1.0 if r["phenotype"] >= 3 else 0.0 for r in primary_rows])
    proteinuria = fit_binary_outcome(primary_rows, [1.0 if r["prot1"] == 1 else 0.0 for r in primary_rows])
    no_mau150_only_rows = [
        r for r in primary_rows if not (r["phenotype"] == 3 and r["mau_level"] == 150 and r["prot1"] != 1)
    ]
    no_mau150_only = fit_binary_outcome(
        no_mau150_only_rows,
        [1.0 if r["phenotype"] >= 3 else 0.0 for r in no_mau150_only_rows],
    )

    mau_axis_rows = []
    for row in primary_rows:
        if row.get("prot1") != 0:
            continue
        mau_level = row.get("mau_level")
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
    mau_only = fit_binary_outcome(mau_axis_rows, [1.0 if r["mau_axis"] >= 2 else 0.0 for r in mau_axis_rows])

    ppo_ge3 = next(item for item in ppo_results["ua_threshold_results"] if item["threshold"] == 3)
    rows = [
        ["Urine protein >=1+", proteinuria["n"], proteinuria["events"], proteinuria["or"], proteinuria["lcl"], proteinuria["ucl"], proteinuria["p"]],
        ["Phenotype >=3 independent threshold", phenotype_ge3["n"], phenotype_ge3["events"], phenotype_ge3["or"], phenotype_ge3["lcl"], phenotype_ge3["ucl"], phenotype_ge3["p"]],
        ["PPO Phenotype >=3", len(primary_rows), phenotype_ge3["events"], ppo_ge3["or"], ppo_ge3["lcl"], ppo_ge3["ucl"], ppo_ge3["p"]],
        ["Phenotype >=3 excluding MAU150-only records", no_mau150_only["n"], no_mau150_only["events"], no_mau150_only["or"], no_mau150_only["lcl"], no_mau150_only["ucl"], no_mau150_only["p"]],
        ["MAU-only axis >=80 among protein negative/trace", mau_only["n"], mau_only["events"], mau_only["or"], mau_only["lcl"], mau_only["ucl"], mau_only["p"]],
    ]
    return {
        "rows": rows,
        "proteinuria": proteinuria,
        "phenotype_ge3": phenotype_ge3,
        "no_mau150_only": no_mau150_only,
        "mau_only": mau_only,
    }


def ua_unit_effects(primary_rows, ppo_results):
    # Unit conversions express per-SD model coefficients on fixed serum uric acid increments.
    design = design_primary(primary_rows, exposure="ua_z")
    ua_sd = design["scale"]["ua_sd"]
    rows = []
    for item in ppo_results["ua_threshold_results"]:
        threshold = item["threshold"]
        if threshold == 1:
            continue
        beta = math.log(item["or"])
        beta_l = math.log(item["lcl"])
        beta_u = math.log(item["ucl"])
        for unit in [10, 50, 100]:
            rows.append(
                [
                    f"Phenotype >={threshold}",
                    unit,
                    f"{math.exp(beta * unit / ua_sd):.3f}",
                    f"{math.exp(beta_l * unit / ua_sd):.3f}",
                    f"{math.exp(beta_u * unit / ua_sd):.3f}",
                ]
            )

    common_cols = build_matrix(
        [design["female"], design["age_z"], design["sbp_z"], design["fpg_z"], design["bmi_z"]] + design["year_cols"]
    )
    base = common_cols @ ppo_results["beta_common"]
    probability_rows = []
    delta_z = 100.0 / ua_sd
    for threshold_index, threshold in enumerate([1, 2, 3]):
        if threshold == 1:
            continue
        eta_base = base + design["ua_z"] * ppo_results["ua_betas"][threshold_index]
        eta_plus = base + (design["ua_z"] + delta_z) * ppo_results["ua_betas"][threshold_index]
        cutpoint = ppo_results["cutpoints"][threshold_index]
        prob_base = 1.0 - sigmoid(cutpoint - eta_base)
        prob_plus = 1.0 - sigmoid(cutpoint - eta_plus)
        probability_rows.append(
            [
                f"Phenotype >={threshold}",
                f"{100.0 * float(np.mean(prob_base)):.3f}",
                f"{100.0 * float(np.mean(prob_plus)):.3f}",
                f"{100.0 * float(np.mean(prob_plus - prob_base)):.3f}",
            ]
        )

    return {
        "ua_sd": ua_sd,
        "unit_rows": rows,
        "probability_rows": probability_rows,
    }


def renal_function_context(primary_rows):
    # Models here describe the concurrent renal-function context of the thresholds.
    context_rows = primary_with_egfr(primary_rows)
    design = design_primary(context_rows, exposure="ua_z")
    covariate_cols = [np.ones(len(context_rows), dtype=float)]
    covariate_names = ["intercept"]
    for phenotype in [1, 2, 3]:
        covariate_cols.append(np.array([1.0 if r["phenotype"] == phenotype else 0.0 for r in context_rows], dtype=float))
        covariate_names.append(f"phenotype_{phenotype}")
    covariate_cols.extend([design["female"], design["age_z"], design["sbp_z"], design["fpg_z"], design["bmi_z"]])
    covariate_names.extend(["female", "age_z", "sbp_z", "fpg_z", "bmi_z"])
    covariate_cols.extend(design["year_cols"])
    covariate_names.extend([f"year_{name}" for name in design["year_names"]])
    x = build_matrix(covariate_cols)

    binary_rows = []
    binary_results = {}
    for outcome in ["egfr_lt90", "egfr_lt60"]:
        y = np.array([r[outcome] for r in context_rows], dtype=float)
        model = fit_logistic(x, y)
        binary_results[outcome] = {}
        for idx, phenotype in enumerate([1, 2, 3], start=1):
            beta = model["beta"][idx]
            se = model["se"][idx]
            lcl, ucl = ci_from_beta(beta, se)
            binary_results[outcome][f"phenotype_{phenotype}"] = {
                "or": math.exp(beta),
                "lcl": lcl,
                "ucl": ucl,
                "p": p_from_z(beta / se),
            }
            binary_rows.append(
                [
                    outcome,
                    f"phenotype_{phenotype}",
                    len(context_rows),
                    int(y.sum()),
                    f"{math.exp(beta):.3f}",
                    f"{lcl:.3f}",
                    f"{ucl:.3f}",
                    f"{p_from_z(beta / se):.6f}",
                ]
            )

    y_cont = np.array([r["egfr"] for r in context_rows], dtype=float)
    ols_model = fit_ols(x, y_cont)
    continuous_rows = []
    continuous_results = {}
    for idx, phenotype in enumerate([1, 2, 3], start=1):
        beta = ols_model["beta"][idx]
        se = ols_model["se"][idx]
        z_value = beta / se
        pvalue = p_from_z(z_value)
        continuous_results[f"phenotype_{phenotype}"] = {"beta": beta, "se": se, "p": pvalue}
        continuous_rows.append([f"phenotype_{phenotype}", f"{beta:.3f}", f"{se:.3f}", f"{pvalue:.6f}"])

    descriptive_rows = []
    for phenotype in range(4):
        subset = [r for r in context_rows if r["phenotype"] == phenotype]
        descriptive_rows.append(
            [
                phenotype,
                len(subset),
                f"{np.mean([r['scr'] for r in subset]):.2f}",
                f"{np.mean([r['egfr'] for r in subset]):.2f}",
                f"{100.0 * np.mean([r['egfr_lt90'] for r in subset]):.2f}",
                f"{100.0 * np.mean([r['egfr_lt60'] for r in subset]):.2f}",
            ]
        )

    return {
        "context_rows": context_rows,
        "binary_rows": binary_rows,
        "binary_results": binary_results,
        "continuous_rows": continuous_rows,
        "continuous_results": continuous_results,
        "descriptive_rows": descriptive_rows,
    }


def threshold_model(primary_rows, threshold):
    design = design_primary(primary_rows, exposure="ua_z")
    x, _ = logistic_matrix(design, exposure="ua_z", include_female=True, use_high_sbp=False, interaction=None)
    y = np.array([1 if r["phenotype"] >= threshold else 0 for r in primary_rows], dtype=float)
    model = fit_logistic(x, y)
    beta = model["beta"][1]
    se = model["se"][1]
    or_value = math.exp(beta)
    lcl, ucl = ci_from_beta(beta, se)
    return {
        "threshold": threshold,
        "label": f"phenotype >= {threshold}",
        "n": len(primary_rows),
        "events": int(y.sum()),
        "event_pct": float(100.0 * y.mean()),
        "or": or_value,
        "lcl": lcl,
        "ucl": ucl,
        "p": p_from_z(beta / se),
        "auc": auc(model["pred"], y),
        "design": design,
        "model": model,
    }


def cumulative_phenotype_results(primary_rows):
    thresholds = [threshold_model(primary_rows, threshold) for threshold in [1, 2, 3]]

    return {
        "thresholds": thresholds,
        "primary": thresholds[1],
    }


def stratified_severe_result(primary_rows, label, subset_fn, include_female):
    subset = [r for r in primary_rows if subset_fn(r)]
    design = design_primary(subset, exposure="ua_z")
    x, _ = logistic_matrix(design, exposure="ua_z", include_female=include_female, use_high_sbp=False, interaction=None)
    y = np.array([1 if r["phenotype"] >= 2 else 0 for r in subset], dtype=float)
    model = fit_logistic(x, y)
    beta = model["beta"][1]
    se = model["se"][1]
    lcl, ucl = ci_from_beta(beta, se)
    return {
        "label": label,
        "n": len(subset),
        "or": math.exp(beta),
        "lcl": lcl,
        "ucl": ucl,
        "p": p_from_z(beta / se),
    }


def primary_interactions(primary_rows):
    design = design_primary(primary_rows, exposure="ua_z")
    y = np.array([1 if r["phenotype"] >= 2 else 0 for r in primary_rows], dtype=float)

    x_sex_base, _ = logistic_matrix(design, exposure="ua_z", include_female=True, use_high_sbp=False, interaction=None)
    x_sex_int, _ = logistic_matrix(design, exposure="ua_z", include_female=True, use_high_sbp=False, interaction="sex")
    m0 = fit_logistic(x_sex_base, y)
    m1 = fit_logistic(x_sex_int, y)
    _, sex_p = lrt_pvalue(m0["ll"], m1["ll"], 1)

    x_sbp_base, _ = logistic_matrix(design, exposure="ua_z", include_female=True, use_high_sbp=True, interaction=None)
    x_sbp_int, _ = logistic_matrix(design, exposure="ua_z", include_female=True, use_high_sbp=True, interaction="high_sbp")
    s0 = fit_logistic(x_sbp_base, y)
    s1 = fit_logistic(x_sbp_int, y)
    _, sbp_p = lrt_pvalue(s0["ll"], s1["ll"], 1)

    return {
        "sex_lrt_p": sex_p,
        "sbp_lrt_p": sbp_p,
        "male": stratified_severe_result(primary_rows, "male", lambda r: r["sex"] == "男", include_female=False),
        "female": stratified_severe_result(primary_rows, "female", lambda r: r["sex"] == "女", include_female=False),
        "sbp_lt130": stratified_severe_result(primary_rows, "sbp_lt130", lambda r: r["high_sbp"] == 0, include_female=True),
        "sbp_ge130": stratified_severe_result(primary_rows, "sbp_ge130", lambda r: r["high_sbp"] == 1, include_female=True),
        "sex_model": m1,
        "sbp_model": s1,
        "design": design,
    }


def adjusted_binary_results(primary_rows, main_design):
    x, _ = logistic_matrix(main_design, exposure="ua_z", include_female=True, use_high_sbp=False, interaction=None)
    outcomes = [("mau80", "MAU >= 80"), ("prot1", "Urine protein >=1+"), ("phenotype_ge2", "phenotype level >= 2")]
    result_rows = []

    for key, label in outcomes:
        if key == "phenotype_ge2":
            y = np.array([1 if r["phenotype"] >= 2 else 0 for r in primary_rows], dtype=float)
        else:
            y = np.array([r[key] for r in primary_rows], dtype=float)
        model = fit_logistic(x, y)
        beta = model["beta"][1]
        se = model["se"][1]
        or_value = math.exp(beta)
        lcl, ucl = ci_from_beta(beta, se)
        pvalue = p_from_z(beta / se)
        result_rows.append(
            {
                "outcome": label,
                "n": int(len(y)),
                "events": int(y.sum()),
                "event_pct": float(100.0 * y.mean()),
                "or": or_value,
                "lcl": lcl,
                "ucl": ucl,
                "p": pvalue,
                "auc": auc(model["pred"], y),
            }
        )
    return result_rows


def boundary_summary(rows):
    table_rows = []
    stability_rows = []
    phenotype_year_rows = []
    phenotype_counts = Counter()

    complete = [r for r in rows if r.get("phenotype") is not None]
    for row in complete:
        phenotype_counts[row["phenotype"]] += 1

    for year in YEARS:
        year_rows = [r for r in rows if r.get("year") == year and r.get("mau_level") is not None and r.get("prot_group3") is not None]
        for level in MAU_LEVELS:
            subset = [r for r in year_rows if r["mau_level"] == level]
            trace_pct = percent(sum(1 for r in subset if r["prot_group3"] == "trace"), len(subset))
            prot1_pct = percent(sum(1 for r in subset if r["prot_group3"] == "1plus_or_more"), len(subset))
            table_rows.append([year, int(level), len(subset), f"{trace_pct:.2f}", f"{prot1_pct:.2f}"])

        rho_rows = [r for r in year_rows if r["mau_level"] is not None and r["prot_group3"] is not None]
        x = [MAU_LEVELS.index(r["mau_level"]) + 1 for r in rho_rows]
        y = [{"negative": 0, "trace": 1, "1plus_or_more": 3}[r["prot_group3"]] for r in rho_rows]
        rho = spearmanr(x, y).correlation
        stability_rows.append([year, f"{rho:.4f}"])

        year_complete = [r for r in complete if r["year"] == year]
        year_total = len(year_complete)
        for phenotype in range(4):
            count = sum(1 for r in year_complete if r["phenotype"] == phenotype)
            phenotype_year_rows.append([year, phenotype, count, f"{percent(count, year_total):.2f}"])

    return {
        "boundary_rows": table_rows,
        "stability_rows": stability_rows,
        "phenotype_year_rows": phenotype_year_rows,
        "phenotype_counts": phenotype_counts,
    }


def table1_by_phenotype(primary_rows):
    rows = []
    total_n = len(primary_rows)
    egfr_rows = primary_with_egfr(primary_rows)
    for level in range(4):
        subset = [r for r in primary_rows if r["phenotype"] == level]
        egfr_subset = [r for r in egfr_rows if r["phenotype"] == level]
        ua_mean, ua_sd = mean_sd([r["ua"] for r in subset])
        sbp_mean, sbp_sd = mean_sd([r["sbp"] for r in subset])
        fpg_mean, fpg_sd = mean_sd([r["fpg"] for r in subset])
        bmi_mean, bmi_sd = mean_sd([r["bmi"] for r in subset])
        age_mean, age_sd = mean_sd([r["age_low"] for r in subset])
        egfr_mean, egfr_sd = mean_sd([r["egfr"] for r in egfr_subset])
        female_n = sum(1 for r in subset if r["sex"] == "女")
        hu_n = sum(1 for r in subset if r["hu"] == 1)
        row = [
            f"level_{level}",
            len(subset),
            f"{percent(len(subset), total_n):.2f}",
            f"{age_mean:.2f} ({age_sd:.2f})",
            f"{female_n} ({percent(female_n, len(subset)):.2f})",
            f"{ua_mean:.2f} ({ua_sd:.2f})",
            f"{hu_n} ({percent(hu_n, len(subset)):.2f})",
            f"{sbp_mean:.2f} ({sbp_sd:.2f})",
            f"{fpg_mean:.2f} ({fpg_sd:.2f})",
            f"{bmi_mean:.2f} ({bmi_sd:.2f})",
            f"{egfr_mean:.2f} ({egfr_sd:.2f})",
        ]
        rows.append(row)
    return rows


def build_summary_json(
    flow,
    boundary,
    early_selection,
    cumulative_results,
    interactions,
    binary_results,
    selection_results,
    ppo_results,
    ipw_results,
    renal_context_results,
    level3_results,
    level3_contrast,
    ua_effects,
    wave_results,
):
    overall, by_year = flow
    return {
        "sample_flow": {
            "overall": dict(overall),
            "by_year": {year: dict(by_year[year]) for year in YEARS},
        },
        "phenotype_counts": {str(k): int(v) for k, v in boundary["phenotype_counts"].items()},
        "cumulative_thresholds": {
            item["label"]: {
                "n": item["n"],
                "events": item["events"],
                "event_pct": item["event_pct"],
                "or": item["or"],
                "lcl": item["lcl"],
                "ucl": item["ucl"],
                "p": item["p"],
            }
            for item in cumulative_results["thresholds"]
        },
        "interactions": {
            "sex_lrt_p": interactions["sex_lrt_p"],
            "sex_male_or": interactions["male"]["or"],
            "sex_female_or": interactions["female"]["or"],
            "sbp_lrt_p": interactions["sbp_lrt_p"],
            "sbp_low_or": interactions["sbp_lt130"]["or"],
            "sbp_high_or": interactions["sbp_ge130"]["or"],
        },
        "binary_results": binary_results,
        "selection": {
            "early_measurement": {
                "adult_n": early_selection["adult_n"],
                "phenotype_available_n": early_selection["phenotype_available_n"],
                "phenotype_unavailable_n": early_selection["phenotype_unavailable_n"],
                "basic_model_year_base": early_selection["basic_model_year_base"],
                "covariate_model_summary": early_selection["covariate_model_summary"],
            },
            "compare": selection_results["selection_compare"],
            "weight_summary": selection_results["weight_summary"],
            "by_year": selection_results["year_rows"],
            "by_phenotype": selection_results["phenotype_rows"],
            "selection_model": selection_results["selection_rows"],
        },
        "non_uniformity": {
            "ppo_lrt_stat": ppo_results["lrt_stat"],
            "ppo_lrt_p": ppo_results["lrt_p"],
            "ua_or_by_threshold": ppo_results["ua_or_by_threshold"],
            "ua_threshold_results": ppo_results["ua_threshold_results"],
        },
        "ipw_results": ipw_results,
        "renal_function_context": {
            "binary_results": renal_context_results["binary_results"],
            "continuous_results": renal_context_results["continuous_results"],
            "descriptive_rows": renal_context_results["descriptive_rows"],
        },
        "level3_components": level3_results,
        "level3_contrast": level3_contrast,
        "ua_unit_effects": ua_effects,
        "wave_robustness": {
            "year_specific": wave_results["by_year_rows"],
            "leave_one_out": wave_results["leave_one_out_rows"],
            "year_interaction_p": wave_results["year_interaction_p"],
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    flow = sample_flow(rows)
    boundary = boundary_summary(rows)
    primary_rows = dataset_for_primary(rows)
    cumulative_results = cumulative_phenotype_results(primary_rows)
    interactions = primary_interactions(primary_rows)
    binary_results = adjusted_binary_results(primary_rows, cumulative_results["primary"]["design"])
    early_selection = early_measurement_selection_analysis(rows)
    selection_results = selection_analysis(rows)
    ppo_results = fit_partial_proportional_odds(primary_rows)
    ipw_results = fit_threshold_models(primary_rows, weights=selection_results["weights"])
    renal_context_results = renal_function_context(primary_rows)
    level3_results = level3_component_summary(primary_rows)
    level3_contrast = level3_contrast_analysis(primary_rows, ppo_results)
    ua_effects = ua_unit_effects(primary_rows, ppo_results)
    wave_results = wave_robustness(primary_rows)

    write_tsv(
        OUTPUT_DIR / "sample_flow_by_year.tsv",
        [
            [
                year,
                flow[1][year]["raw"],
                flow[1][year]["adult_valid"],
                flow[1][year]["valid_ua"],
                flow[1][year]["valid_phenotype"],
                flow[1][year]["phenotype_plus_ua"],
                flow[1][year]["primary_model_complete"],
            ]
            for year in YEARS
        ],
        ["year", "raw", "adult_valid", "valid_ua", "valid_phenotype", "phenotype_plus_ua", "primary_model_complete"],
    )

    write_tsv(
        OUTPUT_DIR / "threshold_boundary_by_year.tsv",
        boundary["boundary_rows"],
        ["year", "mau_level", "n", "trace_pct", "protein_1plus_pct"],
    )
    write_tsv(
        OUTPUT_DIR / "analysis_stability_summary.tsv",
        boundary["stability_rows"],
        ["year", "spearman_mau_vs_protein_score"],
    )
    write_tsv(
        OUTPUT_DIR / "phenotype_distribution_by_year.tsv",
        boundary["phenotype_year_rows"],
        ["year", "phenotype", "count", "pct"],
    )
    write_tsv(
        OUTPUT_DIR / "baseline_characteristics_by_phenotype.tsv",
        table1_by_phenotype(primary_rows),
        [
            "phenotype",
            "n",
            "pct_of_primary_sample",
            "age_mean_sd",
            "female_n_pct",
            "ua_mean_sd",
            "hyperuricemia_n_pct",
            "sbp_mean_sd",
            "fpg_mean_sd",
            "bmi_mean_sd",
            "egfr_mean_sd",
        ],
    )
    write_tsv(
        OUTPUT_DIR / "test_ordering_selection_comparison.tsv",
        early_selection["comparison_rows"],
        ["variable", "phenotype_available_valid_n", "phenotype_available_summary", "phenotype_unavailable_valid_n", "phenotype_unavailable_summary", "smd"],
    )
    write_tsv(
        OUTPUT_DIR / "variable_availability_by_test_ordering.tsv",
        early_selection["availability_rows"],
        ["variable", "phenotype_available_valid_n", "phenotype_available_valid_pct", "phenotype_unavailable_valid_n", "phenotype_unavailable_valid_pct"],
    )
    write_tsv(
        OUTPUT_DIR / "phenotype_availability_by_year.tsv",
        early_selection["by_year_rows"],
        ["year", "adult_valid_n", "phenotype_available_n", "phenotype_available_pct"],
    )
    write_tsv(
        OUTPUT_DIR / "test_ordering_selection_model_basic.tsv",
        early_selection["basic_model_rows"],
        ["term", "or", "lcl", "ucl", "p"],
    )
    write_tsv(
        OUTPUT_DIR / "test_ordering_selection_model_extended.tsv",
        early_selection["covariate_model_rows"],
        ["term", "or", "lcl", "ucl", "p"],
    )
    write_tsv(
        OUTPUT_DIR / "complete_case_selection_by_year.tsv",
        selection_results["year_rows"],
        ["year", "phenotype_plus_ua_n", "complete_case_n", "complete_case_pct"],
    )
    write_tsv(
        OUTPUT_DIR / "complete_case_selection_by_phenotype.tsv",
        selection_results["phenotype_rows"],
        ["phenotype", "phenotype_plus_ua_n", "complete_case_n", "complete_case_pct"],
    )
    write_tsv(
        OUTPUT_DIR / "complete_case_selection_model.tsv",
        selection_results["selection_rows"],
        ["term", "or", "lcl", "ucl", "p"],
    )
    write_tsv(
        OUTPUT_DIR / "upper_category_components.tsv",
        level3_results["rows"],
        ["component", "count"],
    )
    write_tsv(
        OUTPUT_DIR / "upper_category_contrast_models.tsv",
        [
            [label, n, events, f"{or_value:.3f}", f"{lcl:.3f}", f"{ucl:.3f}", f"{pvalue:.6f}"]
            for label, n, events, or_value, lcl, ucl, pvalue in level3_contrast["rows"]
        ],
        ["analysis", "n", "events", "or", "lcl", "ucl", "p"],
    )
    write_tsv(
        OUTPUT_DIR / "uric_acid_unit_effects.tsv",
        ua_effects["unit_rows"],
        ["threshold", "sua_change_umol_l", "or", "lcl", "ucl"],
    )
    write_tsv(
        OUTPUT_DIR / "marginal_probability_change.tsv",
        ua_effects["probability_rows"],
        ["threshold", "base_probability_pct", "plus_100_umol_l_probability_pct", "absolute_difference_percentage_points"],
    )
    write_tsv(
        OUTPUT_DIR / "egfr_context_descriptive.tsv",
        renal_context_results["descriptive_rows"],
        ["phenotype", "n", "mean_scr", "mean_egfr", "egfr_lt90_pct", "egfr_lt60_pct"],
    )
    write_tsv(
        OUTPUT_DIR / "egfr_binary_models.tsv",
        renal_context_results["binary_rows"],
        ["outcome", "term", "n", "events", "or", "lcl", "ucl", "p"],
    )
    write_tsv(
        OUTPUT_DIR / "egfr_continuous_models.tsv",
        renal_context_results["continuous_rows"],
        ["term", "beta", "se", "p"],
    )
    write_tsv(
        OUTPUT_DIR / "threshold_heterogeneity_lrt.tsv",
        (
            [
                ["ppo_lrt_stat", f"{ppo_results['lrt_stat']:.6f}"],
                ["ppo_lrt_p", f"{ppo_results['lrt_p']:.6f}"],
            ]
            + [
                [f"ppo_or_ge{item['threshold']}", f"{item['or']:.3f}"]
                for item in ppo_results["ua_threshold_results"]
            ]
            + [
                [f"ppo_lcl_ge{item['threshold']}", f"{item['lcl']:.3f}"]
                for item in ppo_results["ua_threshold_results"]
            ]
            + [
                [f"ppo_ucl_ge{item['threshold']}", f"{item['ucl']:.3f}"]
                for item in ppo_results["ua_threshold_results"]
            ]
            + [
                [f"ppo_p_ge{item['threshold']}", f"{item['p']:.6f}"]
                for item in ppo_results["ua_threshold_results"]
            ]
        ),
        ["metric", "value"],
    )
    write_tsv(
        OUTPUT_DIR / "partial_proportional_odds_thresholds.tsv",
        [
            [
                item["label"],
                item["threshold"],
                f"{item['or']:.3f}",
                f"{item['lcl']:.3f}",
                f"{item['ucl']:.3f}",
                f"{item['p']:.6f}",
                f"{item['se']:.6f}",
            ]
            for item in ppo_results["ua_threshold_results"]
        ],
        ["term", "threshold", "or", "lcl", "ucl", "p", "se"],
    )
    write_tsv(
        OUTPUT_DIR / "ipw_threshold_models.tsv",
        [
            [
                item["label"],
                item["n"],
                item["events"],
                f"{item['or']:.3f}",
                f"{item['lcl']:.3f}",
                f"{item['ucl']:.3f}",
                f"{item['p']:.6f}",
            ]
            for item in ipw_results
        ],
        ["term", "n", "events", "or", "lcl", "ucl", "p"],
    )
    write_tsv(
        OUTPUT_DIR / "wave_stratified_models.tsv",
        wave_results["by_year_rows"],
        ["year", "n", "events", "or", "lcl", "ucl", "p"],
    )
    write_tsv(
        OUTPUT_DIR / "leave_one_wave_sensitivity.tsv",
        wave_results["leave_one_out_rows"],
        ["excluded_wave", "n", "events", "or", "lcl", "ucl", "p"],
    )
    model_rows = [
        [
            "graded_phenotype",
            item["label"],
            item["n"],
            f"{item['or']:.3f}",
            f"{item['lcl']:.3f}",
            f"{item['ucl']:.3f}",
            f"{item['p']:.6f}",
        ]
        for item in cumulative_results["thresholds"]
    ]
    model_rows.extend(
        [
            ["interaction", "ua_x_sex_on_phenotype>=2", cumulative_results["primary"]["n"], "", "", "", f"{interactions['sex_lrt_p']:.6f}"],
            [
                "interaction",
                "phenotype>=2_male",
                interactions["male"]["n"],
                f"{interactions['male']['or']:.3f}",
                f"{interactions['male']['lcl']:.3f}",
                f"{interactions['male']['ucl']:.3f}",
                f"{interactions['male']['p']:.6f}",
            ],
            [
                "interaction",
                "phenotype>=2_female",
                interactions["female"]["n"],
                f"{interactions['female']['or']:.3f}",
                f"{interactions['female']['lcl']:.3f}",
                f"{interactions['female']['ucl']:.3f}",
                f"{interactions['female']['p']:.6f}",
            ],
            ["interaction", "ua_x_sbp_on_phenotype>=2", cumulative_results["primary"]["n"], "", "", "", f"{interactions['sbp_lrt_p']:.6f}"],
            [
                "interaction",
                "phenotype>=2_sbp_lt130",
                interactions["sbp_lt130"]["n"],
                f"{interactions['sbp_lt130']['or']:.3f}",
                f"{interactions['sbp_lt130']['lcl']:.3f}",
                f"{interactions['sbp_lt130']['ucl']:.3f}",
                f"{interactions['sbp_lt130']['p']:.6f}",
            ],
            [
                "interaction",
                "phenotype>=2_sbp_ge130",
                interactions["sbp_ge130"]["n"],
                f"{interactions['sbp_ge130']['or']:.3f}",
                f"{interactions['sbp_ge130']['lcl']:.3f}",
                f"{interactions['sbp_ge130']['ucl']:.3f}",
                f"{interactions['sbp_ge130']['p']:.6f}",
            ],
        ]
    )
    for item in binary_results:
        model_rows.append(
            [
                "binary_validation",
                item["outcome"],
                item["n"],
                f"{item['or']:.3f}",
                f"{item['lcl']:.3f}",
                f"{item['ucl']:.3f}",
                f"{item['p']:.6f}",
            ]
        )
    model_rows.extend(
        [
            ["non_uniformity", "ppo_lrt_stat", primary_rows.__len__(), "", "", "", f"{ppo_results['lrt_stat']:.6f}"],
            ["non_uniformity", "ppo_lrt_p", primary_rows.__len__(), "", "", "", f"{ppo_results['lrt_p']:.6f}"],
        ]
    )
    for item in ppo_results["ua_threshold_results"]:
        model_rows.append(
            [
                "non_uniformity",
                f"ppo_ua_or_ge{item['threshold']}",
                primary_rows.__len__(),
                f"{item['or']:.3f}",
                f"{item['lcl']:.3f}",
                f"{item['ucl']:.3f}",
                f"{item['p']:.6f}",
            ]
        )
    for item in ipw_results:
        model_rows.append(
            [
                "ipw_sensitivity",
                item["label"],
                item["n"],
                f"{item['or']:.3f}",
                f"{item['lcl']:.3f}",
                f"{item['ucl']:.3f}",
                f"{item['p']:.6f}",
            ]
        )
    for row in renal_context_results["binary_rows"]:
        model_rows.append(["egfr_context", row[0] + "_" + row[1], row[2], row[4], row[5], row[6], row[7]])

    write_tsv(OUTPUT_DIR / "primary_model_summary.tsv", model_rows, ["section", "term", "n", "or", "lcl", "ucl", "p"])

    summary = build_summary_json(
        flow,
        boundary,
        early_selection,
        cumulative_results,
        interactions,
        binary_results,
        selection_results,
        ppo_results,
        ipw_results,
        renal_context_results,
        level3_results,
        level3_contrast,
        ua_effects,
        wave_results,
    )
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
