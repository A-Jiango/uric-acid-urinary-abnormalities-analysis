from __future__ import annotations

from pathlib import Path

from openpyxl import load_workbook

import main_analysis as base


# Plausibility ranges are applied only when optional fields are present.
OPTIONAL_NUMERIC_RANGES: dict[str, tuple[float | None, float | None]] = {
    "weight": (20.0, 300.0),
    "waist": (30.0, 250.0),
    "dbp": (30.0, 200.0),
    "urea": (0.5, 80.0),
    "total_cholesterol": (1.0, 20.0),
    "ldl_cholesterol": (0.2, 15.0),
    "hba1c": (3.0, 25.0),
    "homocysteine": (1.0, 200.0),
    "crp": (0.01, 300.0),
    "hscrp": (0.01, 300.0),
    "ast": (1.0, 2000.0),
    "alt": (1.0, 2000.0),
    "ggt": (1.0, 2000.0),
    "urine_creatinine_grade": (0.1, 1000.0),
    "urine_specific_gravity": (1.0, 1.1),
    "urine_ph": (1.0, 14.0),
}

OPTIONAL_TEXT_KEYS = (
    "leukocyte_esterase",
    "nitrite",
    "urobilinogen",
    "hematuria",
    "bilirubin",
    "ketone",
    "urine_glucose",
    "urine_color",
    "urine_transparency",
)

STRIP_TEXT_KEYS = (
    "leukocyte_esterase",
    "nitrite",
    "urobilinogen",
    "hematuria",
    "bilirubin",
    "ketone",
    "urine_glucose",
)


def raw_float(value):
    # Optional numeric fields are parsed without treating zero as automatically missing.
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text in {"", "NULL", "无效", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def nonblank(value) -> bool:
    # Availability flags distinguish blank placeholders from recorded entries.
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text not in {"0", "0.0", "NULL", "null", "None", "none", "无效", "nan"}


def parse_strip_positive(value):
    # Routine urinalysis text values are converted only when the signal is clear.
    if value is None:
        return None
    text = (
        str(value)
        .strip()
        .replace("＋", "+")
        .replace("－", "-")
        .replace("—", "-")
        .replace("–", "-")
        .replace("（", "(")
        .replace("）", ")")
        .replace(" ", "")
        .lower()
    )
    if text in {"", "0", "0.0", "null", "none", "nan", "无效"}:
        return None
    if "阴性" in text or text in {"-", "(-)", "negative", "neg", "normal"}:
        return 0
    if "阳性" in text or "异常" in text or "+" in text or "trace" in text or "±" in text:
        return 1
    return None


def parse_protein_like_positive(value):
    # Some strip fields use the same semi-quantitative scale as urine protein.
    detail = base.normalize_protein(value)
    if detail is None:
        return None
    return 0 if detail == "neg" else 1


def optional_index(header: list[object], column_map: dict[str, str], key: str):
    # Optional fields are skipped when the local workbook has no mapped column.
    mapped = column_map.get(key)
    if not mapped or mapped not in header:
        return None
    return header.index(mapped)


def read_numeric(row, idx: dict[str, int | None], key: str, low=None, high=None):
    # Numeric optional fields use the same parser and plausibility checks as core fields.
    column_idx = idx.get(key)
    if column_idx is None:
        return None
    return base.clean_range(base.parse_float(row[column_idx]), low, high)


def read_text_flag(row, idx: dict[str, int | None], key: str) -> bool:
    # Text availability is recorded separately from positive/negative interpretation.
    column_idx = idx.get(key)
    if column_idx is None:
        return False
    return nonblank(row[column_idx])


def read_strip_positive(row, idx: dict[str, int | None], key: str):
    column_idx = idx.get(key)
    if column_idx is None:
        return None
    return parse_strip_positive(row[column_idx])


def read_protein_like_positive(row, idx: dict[str, int | None], key: str):
    column_idx = idx.get(key)
    if column_idx is None:
        return None
    return parse_protein_like_positive(row[column_idx])


def load_enriched_rows(age_source: str = "exact") -> list[dict[str, object]]:
    # Enriched rows extend the core analytic record with optional measurement fields.
    workbook = load_workbook(base.require_data_path(), read_only=True, data_only=True)
    column_map = base.load_column_map()
    years = base.load_analysis_sheet_names(workbook)
    rows: list[dict[str, object]] = []

    for year in years:
        worksheet = workbook[year]
        header = list(next(worksheet.iter_rows(min_row=1, max_row=1, values_only=True)))
        required_idx = {key: header.index(column_map[key]) for key in base.REQUIRED_COLUMN_KEYS}
        exact_age_idx = optional_index(header, column_map, "exact_age")
        optional_idx = {key: optional_index(header, column_map, key) for key in OPTIONAL_NUMERIC_RANGES}
        optional_idx.update({key: optional_index(header, column_map, key) for key in OPTIONAL_TEXT_KEYS})

        for row in worksheet.iter_rows(min_row=2, values_only=True):
            sex = base.normalize_sex(row[required_idx["sex"]])
            age_band_low = base.parse_age(row[required_idx["age_band"]])
            age_exact = base.parse_exact_age(row[exact_age_idx]) if exact_age_idx is not None else None

            if age_source == "band_lower":
                analysis_age = age_band_low
            elif age_source == "exact":
                analysis_age = age_exact if age_exact is not None else age_band_low if exact_age_idx is None else None
            else:
                raise ValueError(f"unsupported age_source: {age_source}")

            adult_valid = sex is not None and analysis_age is not None and analysis_age >= 20
            if not adult_valid:
                rows.append({"year": year, "adult_valid": False})
                continue

            ua = base.clean_range(base.parse_float(row[required_idx["serum_uric_acid"]]), 100, 1000)
            fpg = base.clean_range(base.parse_float(row[required_idx["fpg"]]), 2.5, 30)
            sbp = base.clean_range(base.parse_float(row[required_idx["sbp"]]), 70, 260)
            height = base.clean_range(base.parse_float(row[required_idx["height"]]), 100, 220)
            tg = base.clean_range(base.parse_float(row[required_idx["triglycerides"]]), 0.2, 30)
            hdl = base.clean_range(base.parse_float(row[required_idx["hdl_cholesterol"]]), 0.2, 5)
            bmi = base.clean_range(base.parse_float(row[required_idx["bmi"]]), 10, 80)
            scr = base.clean_range(base.parse_float(row[required_idx["serum_creatinine"]]), 20, 2000)

            mau_raw = raw_float(row[required_idx["mau"]])
            mau = base.clean_range(mau_raw, 1, 5000) if mau_raw is not None else None
            mau_level = mau if mau in base.MAU_LEVELS else None

            prot_detail = base.normalize_protein(row[required_idx["urine_protein"]])
            phenotype = base.phenotype_level(mau_level, prot_detail)

            weight = read_numeric(row, optional_idx, "weight", *OPTIONAL_NUMERIC_RANGES["weight"])
            waist = read_numeric(row, optional_idx, "waist", *OPTIONAL_NUMERIC_RANGES["waist"])
            dbp = read_numeric(row, optional_idx, "dbp", *OPTIONAL_NUMERIC_RANGES["dbp"])
            urea = read_numeric(row, optional_idx, "urea", *OPTIONAL_NUMERIC_RANGES["urea"])
            tc = read_numeric(row, optional_idx, "total_cholesterol", *OPTIONAL_NUMERIC_RANGES["total_cholesterol"])
            ldl = read_numeric(row, optional_idx, "ldl_cholesterol", *OPTIONAL_NUMERIC_RANGES["ldl_cholesterol"])
            hba1c = read_numeric(row, optional_idx, "hba1c", *OPTIONAL_NUMERIC_RANGES["hba1c"])
            hcy = read_numeric(row, optional_idx, "homocysteine", *OPTIONAL_NUMERIC_RANGES["homocysteine"])
            crp = read_numeric(row, optional_idx, "crp", *OPTIONAL_NUMERIC_RANGES["crp"])
            hscrp = read_numeric(row, optional_idx, "hscrp", *OPTIONAL_NUMERIC_RANGES["hscrp"])
            urine_creat = read_numeric(
                row,
                optional_idx,
                "urine_creatinine_grade",
                *OPTIONAL_NUMERIC_RANGES["urine_creatinine_grade"],
            )
            urine_sg = read_numeric(
                row,
                optional_idx,
                "urine_specific_gravity",
                *OPTIONAL_NUMERIC_RANGES["urine_specific_gravity"],
            )
            urine_ph = read_numeric(row, optional_idx, "urine_ph", *OPTIONAL_NUMERIC_RANGES["urine_ph"])

            leuk_present = read_text_flag(row, optional_idx, "leukocyte_esterase")
            nitrite_present = read_text_flag(row, optional_idx, "nitrite")
            urobilinogen_present = read_text_flag(row, optional_idx, "urobilinogen")
            hematuria_present = read_text_flag(row, optional_idx, "hematuria")
            hematuria_positive = read_strip_positive(row, optional_idx, "hematuria")
            hematuria_protein_like_positive = read_protein_like_positive(row, optional_idx, "hematuria")
            bilirubin_present = read_text_flag(row, optional_idx, "bilirubin")
            ketone_present = read_text_flag(row, optional_idx, "ketone")
            urine_glucose_present = read_text_flag(row, optional_idx, "urine_glucose")
            urine_color_present = read_text_flag(row, optional_idx, "urine_color")
            urine_transparency_present = read_text_flag(row, optional_idx, "urine_transparency")
            # Binary strip variables are harmonized before downstream context models.
            strip_binary = {
                key: read_strip_positive(row, optional_idx, key)
                for key in STRIP_TEXT_KEYS
            }

            alt = read_numeric(row, optional_idx, "alt", *OPTIONAL_NUMERIC_RANGES["alt"])
            ast = read_numeric(row, optional_idx, "ast", *OPTIONAL_NUMERIC_RANGES["ast"])
            ggt = read_numeric(row, optional_idx, "ggt", *OPTIONAL_NUMERIC_RANGES["ggt"])

            hu = None
            if ua is not None:
                if sex == "男":
                    hu = 1 if ua >= 420 else 0
                else:
                    hu = 1 if ua >= 360 else 0

            # These availability flags summarize whether same-sample urine
            # context was recorded; they are not individual identifiers.
            routine_urine_flags = (
                prot_detail is not None,
                urine_ph is not None,
                urine_sg is not None,
                hematuria_present,
                leuk_present,
                nitrite_present,
                bilirubin_present,
                urobilinogen_present,
                ketone_present,
                urine_glucose_present,
                urine_color_present,
                urine_transparency_present,
            )

            rows.append(
                {
                    "year": year,
                    "adult_valid": True,
                    "sex": sex,
                    "female": 1 if sex == "女" else 0,
                    "age_low": analysis_age,
                    "age_exact": age_exact,
                    "age_band_low": age_band_low,
                    "ua": ua,
                    "hu": hu,
                    "sbp": sbp,
                    "high_sbp": 1 if sbp is not None and sbp >= 130 else 0 if sbp is not None else None,
                    "dbp": dbp,
                    "fpg": fpg,
                    "high_fpg": 1 if fpg is not None and fpg >= 5.6 else 0 if fpg is not None else None,
                    "height": height,
                    "weight": weight,
                    "waist": waist,
                    "bmi": bmi,
                    "tg": tg,
                    "hdl": hdl,
                    "tc": tc,
                    "ldl": ldl,
                    "scr": scr,
                    "urea": urea,
                    "hba1c": hba1c,
                    "hcy": hcy,
                    "ast": ast,
                    "alt": alt,
                    "ggt": ggt,
                    "crp": crp,
                    "hscrp": hscrp,
                    "mau_raw": mau_raw,
                    "mau": mau,
                    "mau_level": mau_level,
                    "mau_zero": 1 if mau_raw == 0 else 0,
                    "prot_detail": prot_detail,
                    "prot_group3": base.protein_group(prot_detail),
                    "prot1": 1 if prot_detail in {"1+", "2+", "3+", "4+"} else 0 if prot_detail is not None else None,
                    "phenotype": phenotype,
                    "urine_creat": urine_creat,
                    "urine_sg": urine_sg,
                    "urine_ph": urine_ph,
                    "leuk_present": leuk_present,
                    "leuk_positive": strip_binary["leukocyte_esterase"],
                    "nitrite_present": nitrite_present,
                    "nitrite_positive": strip_binary["nitrite"],
                    "urobilinogen_present": urobilinogen_present,
                    "urobilinogen_abnormal": strip_binary["urobilinogen"],
                    "hematuria_present": hematuria_present,
                    "hematuria_positive": hematuria_positive,
                    "hematuria_protein_like_positive": hematuria_protein_like_positive,
                    "bilirubin_present": bilirubin_present,
                    "bilirubin_positive": strip_binary["bilirubin"],
                    "ketone_present": ketone_present,
                    "ketone_positive": strip_binary["ketone"],
                    "urine_glucose_present": urine_glucose_present,
                    "urine_glucose_positive": strip_binary["urine_glucose"],
                    "urine_color_present": urine_color_present,
                    "urine_transparency_present": urine_transparency_present,
                    "protein_available": prot_detail is not None,
                    "mau_grade_available": mau_level is not None,
                    "urine_creat_available": urine_creat is not None,
                    "routine_urine_any": any(routine_urine_flags),
                    "routine_urine_core_complete": all(
                        (
                            prot_detail is not None,
                            urine_ph is not None,
                            urine_sg is not None,
                            hematuria_present,
                            urine_color_present,
                            urine_transparency_present,
                        )
                    ),
                    "kidney_blood_any": any(value is not None for value in (ua, urea, scr)),
                    "kidney_blood_all": all(value is not None for value in (ua, urea, scr)),
                    "main_model_covariate_package": all(value is not None for value in (ua, fpg, sbp, bmi)),
                    "metabolic_blood_any": any(value is not None for value in (fpg, hba1c, tg, hdl, tc, ldl, hcy)),
                    "anthropometry_any": any(value is not None for value in (height, weight, waist, bmi)),
                    "vital_sign_any": any(value is not None for value in (sbp, dbp)),
                    "lipid_any": any(value is not None for value in (tg, hdl, tc, ldl)),
                    "liver_any": any(value is not None for value in (ast, alt, ggt)),
                }
            )

    return rows
