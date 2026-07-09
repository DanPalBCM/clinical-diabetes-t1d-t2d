"""
T1D Analysis: Overall Characteristics + Severe Hypoglycemia Outcome Comparison
===============================================================================

Cell 1 — Preprocessing (reusable: run before any analysis)
Cell 2 — Characteristic table generation & export
"""

# %% [markdown]
# ## Cell 1 — Load Data & Preprocess

# %%
"""
Preprocessing for T1D Cohort
=============================
Reusable preprocessing steps (adapted from T2D):
  1. BMI unit conversion: raw oz_av/in² → standard kg/m² (÷16)
  2. SDOH categorical binarization (keyword/exact matching)
  3. Collapse remaining high-cardinality SDOH categoricals to top-N + Other
"""

import os
import warnings
import gc
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency
from foundry.transforms import Dataset

warnings.filterwarnings("ignore")

# ============================================================================
# PREPROCESSING FUNCTIONS (same logic as T2D)
# ============================================================================

_BMI_COLS = ["bmi_at_diagnosis", "bmi_at_2_years", "bmi_at_5_years"]
BMI_RAW_TO_KG_M2 = 1.0 / 16.0

_SDOH_COLLAPSE_TOP_N = [
    "socio_education_level_parents_guardian",
    "socio_employment_status_parents_guardian",
    "socio_financial_strain",
    "socio_insurance_status",
    "socio_social_family_support",
]

_SDOH_BINARIZE = {
    "socio_physical_activity": {
        "new_col": "socio_physical_activity_binary",
        "method": "keyword",
        "inactive_keywords": [
            "sedentary", "inactive", "no exercise", "not exercise",
            "does not exercise", "not physically active", "not active",
            "not currently active", "none", "no extracurricular",
            "not interested", "does not engage", "not involved in clubs",
            "not involved in sports", "not involved in extracurricular",
            "0 days/week", "0 times per week", "no regular exercise",
            "does not participate", "not currently involved",
            "stopped exercising", "not on file", "none reported",
            "low - mostly stays home", "low - primarily sedentary",
            "low - no extracurricular", "low - no interest",
            "low activity; 1/10", "low (lacks motivation",
            "limited - plays video games", "limited - watching tv",
            "limited - homebody", "limited - mostly at home",
            "sedentary/no exercise", "sedentary/very limited",
            "sedentary/limited", "minimal/sedentary", "minimal/none",
            "none/inactive", "inactive/sedentary",
            "previously active", "previously highly active",
            "previously started", "previously swimming",
            "previously participated", "currently no regular",
            "currently reduced", "currently avoids",
            "no:", "little to no regular",
        ],
        "active_keywords": [
            "active", "sport", "football", "basketball", "soccer",
            "baseball", "softball", "volleyball", "tennis", "golf",
            "track", "swim", "dance", "cheer", "gym", "exercise",
            "workout", "work out", "working out", "lifting",
            "weight", "run", "jog", "walk", "bike", "bik",
            "bicycle", "karate", "boxing", "yoga", "zumba",
            "kickbox", "wrestling", "rowing", "cross country",
            "gymnastics", "martial", "mma", "pe ", "p.e.",
            "rotc", "jrotc", "band", "marching", "color guard",
            "colorguard", "plays ", "playing ", "regular",
            "daily", "weekly", "times per week", "days/week",
            "x/week", "x per week", "days per week", "minutes",
            "hour", "5 times", "7 times", "4-5x", "3x", "2-3",
            "3-4", "4x", "5x", "5-6", "5-7",
            "high", "very active", "highly active", "athlete",
            "physically active", "incorporated exercise",
            "increased physical", "increasing", "exercising",
            "treadmill", "elliptical", "cardio", "aerobic",
            "jump rope", "trampoline", "skateboard", "skate",
            "outside", "park", "recess", "conditioning",
            "power lifting", "powerlifting", "shot put", "discus",
            "choir", "orchestra", "theater", "theatre",
        ],
        "default": np.nan,
    },
    "socio_social_family_support": {
        "new_col": "socio_social_family_support_binary",
        "method": "exact",
        "positive": ["Adequate", "Strong", "Excellent",
                     "Family, Friends/peers",
                     "Extended family, family and friends/peers",
                     "Family, Friends/peers, School",
                     "Limited to Adequate"],
        "negative": ["Limited", "None", "Minimal"],
    },
    "socio_financial_strain": {
        "new_col": "socio_financial_strain_binary",
        "method": "exact",
        "positive": ["Moderate Risk", "High Risk", "Severe", "Medium Risk"],
        "negative": ["Low Risk"],
    },
    "socio_employment_status_parents_guardian": {
        "new_col": "socio_parental_employment_binary",
        "method": "keyword",
        "active_keywords": ["employed", "both employed", "student"],
        "inactive_keywords": ["unemployed", "disabled", "retired", "mixed"],
        "default": np.nan,
    },
    "socio_education_level_parents_guardian": {
        "new_col": "socio_parental_education_binary",
        "method": "exact",
        "positive": ["High School", "Some College", "College",
                     "Graduate", "Post Graduate", "High School/College",
                     "Some College (mother); Graduate (father)",
                     "High School (Mother), Some College (Father)",
                     "Mother: Some College, Father: High School",
                     "Mother: College, Father: High School",
                     "Mother: Some College; Father: College",
                     "Mother: High School; Father: Some College",
                     "Mother: College; Father: High School",
                     "Mother: High School, Father: Some College",
                     "Mother: Some College; Father: High School",
                     "Mother: High School; Father: Some College; Primary caregiver (grandmother): Some College"],
        "negative": ["Elementary", "Some High School",
                     "Mother: Elementary, Father: High School",
                     "Mother: Some High School (9th Grade); Father: High School (12th Grade)",
                     "Mother: High School; Father: Elementary",
                     "Mother: High School (incomplete, grade 9); Father: Some College",
                     "Some College (Mother), Elementary (Father)",
                     "Mother: High School, Father: Elementary"],
    },
    "socio_insurance_status": {
        "new_col": "socio_insurance_category",
        "method": "exact_categorical",
        "mapping": {
            "Private": "Private", "Multiple": "Private",
            "Medicaid": "Government", "CHIP": "Government",
            "TRICARE WEST": "Government", "Medicare": "Government",
            "Uninsured": "Uninsured", "Self-Pay": "Uninsured",
        },
    },
}

DEFAULT_TOP_N = 5


def convert_bmi_to_standard(df):
    df = df.copy()
    for col in _BMI_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * BMI_RAW_TO_KG_M2
    return df


def collapse_sdoh_categories(df, top_n=DEFAULT_TOP_N, columns=None):
    df = df.copy()
    if columns is None:
        columns = _SDOH_COLLAPSE_TOP_N
    for col in columns:
        if col not in df.columns:
            continue
        top_cats = df[col].value_counts(dropna=True).head(top_n).index.tolist()
        keep_mask = df[col].isin(top_cats) | df[col].isna()
        df.loc[~keep_mask, col] = "Other"
    return df


def binarize_sdoh(df, config=None):
    df = df.copy()
    if config is None:
        config = _SDOH_BINARIZE
    for orig_col, spec in config.items():
        if orig_col not in df.columns:
            continue
        new_col = spec["new_col"]
        method = spec["method"]
        series = df[orig_col]

        if method == "exact":
            pos_set = set(spec.get("positive", []))
            neg_set = set(spec.get("negative", []))
            df[new_col] = np.where(series.isin(pos_set), 1,
                                   np.where(series.isin(neg_set), 0, np.nan))
        elif method == "keyword":
            active_kw = [k.lower() for k in spec.get("active_keywords", [])]
            inactive_kw = [k.lower() for k in spec.get("inactive_keywords", [])]
            default = spec.get("default", np.nan)
            result = pd.Series(default, index=df.index, dtype=float)
            for idx, val in series.items():
                if pd.isna(val):
                    continue
                val_lower = str(val).lower()
                if any(kw in val_lower for kw in inactive_kw):
                    result.at[idx] = 0
                    continue
                if any(kw in val_lower for kw in active_kw):
                    result.at[idx] = 1
                    continue
            df[new_col] = result
        elif method == "exact_categorical":
            mapping = spec.get("mapping", {})
            df[new_col] = series.map(mapping)

        # Log
        n_valid = df[new_col].notna().sum()
        if method == "exact_categorical":
            dist = df[new_col].value_counts(dropna=True)
            dist_str = ", ".join(f"{v}={c}" for v, c in dist.items())
            print(f"    {new_col}: {dist_str}, unmatched={df[orig_col].notna().sum()-n_valid}")
        else:
            n_pos = (df[new_col] == 1).sum()
            n_neg = (df[new_col] == 0).sum()
            print(f"    {new_col}: 1={n_pos}, 0={n_neg}, unmatched={df[orig_col].notna().sum()-n_valid}")
    return df


def compute_age_at_diagnosis(df):
    """Calculate age at diagnosis in years from DiagnosisDate and BirthDTS."""
    df = df.copy()
    if "DiagnosisDate" in df.columns and "BirthDTS" in df.columns:
        dx = pd.to_datetime(df["DiagnosisDate"], errors="coerce", utc=True).dt.tz_localize(None)
        dob = pd.to_datetime(df["BirthDTS"], errors="coerce", utc=True).dt.tz_localize(None)
        df["age_at_diagnosis"] = ((dx - dob).dt.days / 365.25).round(2)
        n_valid = df["age_at_diagnosis"].notna().sum()
        print(f"  [preprocess] age_at_diagnosis computed from DiagnosisDate − BirthDTS  ({n_valid:,} valid)")
    else:
        missing = [c for c in ["DiagnosisDate", "BirthDTS"] if c not in df.columns]
        print(f"  [preprocess] ⚠ Cannot compute age_at_diagnosis — missing: {missing}")
    return df


def clean_sentinel_values(df):
    """
    Replace sentinel/placeholder values (e.g., 9999999) with NaN.
    These appear in some lab results as indicators for unmeasured or
    positive-but-unquantified results.

    Applies clinically reasonable range caps to known columns.
    Values outside these ranges are set to NaN.
    """
    df = df.copy()

    # (column_pattern, min_valid, max_valid, description)
    range_checks = [
        ("urine_microalbumin_creatinine_ratio", 0, 3000, "UACR mg/g"),
        ("urine_microalbumin",                  0, 5000, "Urine microalbumin mg/dL"),
        ("urine_creatinine",                    0, 1000, "Urine creatinine mg/dL"),
    ]

    n_cleaned_total = 0
    for pattern, min_val, max_val, desc in range_checks:
        matching_cols = [c for c in df.columns if pattern in c.lower()]
        for col in matching_cols:
            numeric = pd.to_numeric(df[col], errors="coerce")
            out_of_range = (numeric < min_val) | (numeric > max_val)
            n_bad = out_of_range.sum()
            if n_bad > 0:
                df.loc[out_of_range, col] = np.nan
                n_cleaned_total += n_bad
                print(f"    {col}: {n_bad:,} sentinel/out-of-range values → NaN  (valid range: {min_val}–{max_val} {desc})")

    if n_cleaned_total == 0:
        print(f"  [preprocess] No sentinel values found")
    else:
        print(f"  [preprocess] Cleaned {n_cleaned_total:,} total sentinel values across all columns")

    return df


def convert_binary_string_columns(df):
    """
    Convert columns with Yes/No (or yes/no/YES/NO) string values to 1/0 numeric.
    Specifically handles the severe hypoglycemia event column.
    """
    df = df.copy()

    # Columns known to have Yes/No string encoding
    yes_no_cols = ["Sev_Hypogly_Event"]

    for col in yes_no_cols:
        if col not in df.columns:
            continue
        original_notna = df[col].notna().sum()
        df[col] = (
            df[col].astype(str).str.strip().str.lower()
            .map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
        )
        n_valid = df[col].notna().sum()
        n_pos = int((df[col] == 1).sum())
        n_neg = int((df[col] == 0).sum())
        n_unmatched = original_notna - n_valid
        print(f"  [preprocess] {col}: Yes/No → 1/0  (pos={n_pos:,}, neg={n_neg:,}, unmatched={n_unmatched:,})")

    return df


def preprocess(df):
    n_before = len(df.columns)
    df = compute_age_at_diagnosis(df)
    df = convert_bmi_to_standard(df)
    print(f"  [preprocess] BMI converted to kg/m² (÷16, raw was oz-based imperial)")
    df = clean_sentinel_values(df)
    df = convert_binary_string_columns(df)
    df = binarize_sdoh(df)
    print(f"  [preprocess] SDOH categoricals binarized")
    df = collapse_sdoh_categories(df, top_n=DEFAULT_TOP_N)
    print(f"  [preprocess] Remaining SDOH categoricals collapsed to top-{DEFAULT_TOP_N} + Other")
    print(f"  [preprocess] Done. Columns: {n_before} → {len(df.columns)}")
    return df


# ============================================================================
# LOAD & PREPROCESS
# ============================================================================
DATASET_NAME = "t1d_outcomes"  # ← Update with actual dataset name

print("Loading dataset...", end="")
df_full = Dataset.get(DATASET_NAME).read_table(format="pandas")
print(f" ✓  ({len(df_full):,} patients, {len(df_full.columns)} columns)")

print("\nRunning preprocessing...")
df_full = preprocess(df_full)

print(f"\n✓ df_full ready: {len(df_full):,} rows × {len(df_full.columns)} columns")


# %% [markdown]
# ## Cell 2 — Characteristic Tables & Outcome Comparison

# %%
"""
T1D Characteristic Tables
==========================
1. Overall characteristics (full cohort, no A1C filter)
2. Severe hypoglycemia outcome comparison (positive vs negative)
"""

# ============================================================================
# CONFIGURATION
# ============================================================================
CHAR_OUTPUT_DIR = "analysis_t1d/characteristic_tables"
OUTCOME_OUTPUT_DIR = "analysis_t1d/outcome_comparison"
for d in [CHAR_OUTPUT_DIR, OUTCOME_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

OUTCOME_COL = "Sev_Hypogly_Event"  # Binary: Yes/No → converted to 1/0 in preprocessing

# ============================================================================
# TIMEPOINT MAPS
# ============================================================================
TP_MEAS = {"diagnosis": "_at_diagnosis", "2yr": "_at_2_years", "5yr": "_at_5_years"}
TIMEPOINTS = ["diagnosis", "2yr", "5yr"]
TP_DISPLAY = {"diagnosis": "At Diagnosis", "2yr": "At 2 Years", "5yr": "At 5 Years"}

def meas_col(base, tp):
    return f"{base}{TP_MEAS[tp]}"

# ============================================================================
# VARIABLE DEFINITIONS (T1D column naming)
# ============================================================================
DEMOGRAPHICS = [
    ("Age at Diagnosis",   "age_at_diagnosis",    "continuous"),
    ("Gender",             "gender",              "categorical"),
    ("Race",               "Race",                "categorical"),
    ("Ethnicity",          "Ethnicity",           "categorical"),
    ("Language",           "Preferred_Language",   "categorical"),
    ("Diabetes Duration",  "diabetes_duration",   "continuous"),
]

GLYCEMIC = [
    ("HbA1c (%)",          lambda tp: meas_col("hba1c", tp),      "continuous"),
    ("Glucose (mg/dL)",    lambda tp: meas_col("glucose", tp),     "continuous"),
]

ANTHROPOMETRICS = [
    ("BMI (kg/m²)",        lambda tp: meas_col("bmi", tp),                "continuous"),
    ("BMI Z-score",        lambda tp: meas_col("bmi_zscore", tp),         "continuous"),
    ("BMI Percentile",     lambda tp: meas_col("bmi_percentile", tp),     "continuous"),
    ("Height Z-score",     lambda tp: meas_col("height_zscore", tp),      "continuous"),
    ("Height Percentile",  lambda tp: meas_col("height_percentile", tp),  "continuous"),
    ("Weight Z-score",     lambda tp: meas_col("weight_zscore", tp),      "continuous"),
    ("Weight Percentile",  lambda tp: meas_col("weight_percentile", tp),  "continuous"),
]

LIPIDS = [
    ("Total Cholesterol (mg/dL)",  lambda tp: meas_col("total_cholesterol", tp),  "continuous"),
    ("HDL Cholesterol (mg/dL)",    lambda tp: meas_col("hdl_cholesterol", tp),    "continuous"),
    ("LDL Cholesterol (mg/dL)",    lambda tp: meas_col("ldl_cholesterol", tp),    "continuous"),
    ("Triglycerides (mg/dL)",      lambda tp: meas_col("triglycerides", tp),      "continuous"),
]

VITALS = [
    ("Systolic BP (mmHg)",     lambda tp: meas_col("systolic_blood_pressure", tp),   "continuous"),
    ("Diastolic BP (mmHg)",    lambda tp: meas_col("diastolic_blood_pressure", tp),  "continuous"),
]

RENAL = [
    ("Serum Creatinine (mg/dL)",            lambda tp: meas_col("serum_creatinine", tp),                        "continuous"),
    ("BUN (mg/dL)",                         lambda tp: meas_col("bun", tp),                                     "continuous"),
    ("eGFR (mL/min/1.73m²)",               lambda tp: meas_col("egfr", tp),                                    "continuous"),
    ("UACR (mg/g)",                         lambda tp: meas_col("urine_microalbumin_creatinine_ratio", tp),     "continuous"),
]

LIVER = [
    ("ALT (U/L)",  lambda tp: meas_col("alt", tp),  "continuous"),
    ("AST (U/L)",  lambda tp: meas_col("ast", tp),  "continuous"),
]

OTHER_LABS = [
    ("Serum C-peptide (ng/mL)",  lambda tp: meas_col("serum_c_peptide", tp),  "continuous"),
    ("Blood pH",                 lambda tp: meas_col("blood_ph", tp),          "continuous"),
    ("Bicarbonate (mmol/L)",     lambda tp: meas_col("bicarbonate", tp),       "continuous"),
    ("pCO2 (mmHg)",              lambda tp: meas_col("pco2", tp),              "continuous"),
]

AUTOANTIBODIES = [
    ("GAD65 Ab (U/mL)",      lambda tp: meas_col("gad65_antibody", tp),    "continuous"),
    ("ICA512 Ab (U/mL)",     lambda tp: meas_col("ica512_antibody", tp),   "continuous"),
    ("Insulin Ab (U/mL)",    lambda tp: meas_col("insulin_antibody", tp),  "continuous"),
    ("ZnT8 Ab (U/mL)",       lambda tp: meas_col("znt8_antibody", tp),     "continuous"),
]

CGM_METRICS = [
    ("CGM Duration (days)",         "DurationDays",              "continuous"),
    ("CGM Num Readings",            "NumReadings",               "continuous"),
    ("CGM Mean Glucose (mg/dL)",    "MeanGlucose_mgdL",         "continuous"),
    ("CGM SD (mg/dL)",              "SD_mgdL",                   "continuous"),
    ("CGM CV (%)",                  "CV_percent",                "continuous"),
    ("CGM GMI",                     "GMI",                       "continuous"),
    ("Time Above 250 (%)",          "TimeAbove250_percent",      "continuous"),
    ("Time Above 180 (%)",          "TimeAbove180_percent",      "continuous"),
    ("Time In Range 70-180 (%)",    "TimeInRange70_180_percent", "continuous"),
    ("Time 181-250 (%)",            "Time181_250_percent",       "continuous"),
    ("Time 54-69 (%)",              "Time54_69_percent",         "continuous"),
    ("Time Below 70 (%)",           "TimeBelow70_percent",       "continuous"),
    ("Time Below 54 (%)",           "TimeBelow54_percent",       "continuous"),
    ("Hypo Episodes (Total)",       "HypoEpisodes_Total",       "continuous"),
    ("Severe Hypo Episodes (Total)","SevereHypoEpisodes_Total",  "continuous"),
    ("Severe Hypo Episodes/Day",    "SevereHypoEpisodes_PerDay", "continuous"),
]

SOCIOECONOMIC_BINARY = [
    ("Adverse Childhood Exp.",           "socio_adverse_childhood_experience",   "binary"),
    ("Alcohol Abuse",                    "socio_alcohol_abuse",                  "binary"),
    ("Drug/Substance Abuse",             "socio_drug_substance_abuse",           "binary"),
    ("Food Insecurity",                  "socio_food_insecurity",                "binary"),
    ("Housing Instability",              "socio_housing_instability",            "binary"),
    ("Physical/Sexual Abuse",            "socio_physical_sexual_abuse",          "binary"),
    ("Smoking",                          "socio_smoking",                        "binary"),
    ("Transportation Barrier",           "socio_transportation_barrier",         "binary"),
    ("Physically Active",                "socio_physical_activity_binary",       "binary"),
    ("Social/Family Support (Adequate+)","socio_social_family_support_binary",   "binary"),
    ("Financial Strain (At Risk)",       "socio_financial_strain_binary",        "binary"),
    ("Parental Employment (Employed)",   "socio_parental_employment_binary",     "binary"),
    ("Parental Education (HS+)",         "socio_parental_education_binary",      "binary"),
    ("Insurance Status",                 "socio_insurance_category",             "categorical"),
]

SOCIOECONOMIC_CATEGORICAL = [
    ("Parental Education (raw)",   "socio_education_level_parents_guardian",     "categorical"),
    ("Parental Employment (raw)",  "socio_employment_status_parents_guardian",   "categorical"),
    ("Financial Strain (raw)",     "socio_financial_strain",                     "categorical"),
    ("Insurance Status (raw)",     "socio_insurance_status",                     "categorical"),
    ("Physical Activity (raw)",    "socio_physical_activity",                    "categorical"),
    ("Social/Family Support (raw)","socio_social_family_support",                "categorical"),
]

VARIABLE_SECTIONS = [
    ("DEMOGRAPHICS",                DEMOGRAPHICS,               False),
    ("GLYCEMIC PARAMETERS",         GLYCEMIC,                   True),
    ("ANTHROPOMETRICS",             ANTHROPOMETRICS,             True),
    ("LIPID PANEL",                 LIPIDS,                     True),
    ("VITAL SIGNS",                 VITALS,                     True),
    ("RENAL FUNCTION",              RENAL,                      True),
    ("LIVER FUNCTION",              LIVER,                      True),
    ("OTHER LABORATORY",            OTHER_LABS,                 True),
    ("AUTOANTIBODIES",              AUTOANTIBODIES,             True),
    ("CGM METRICS",                 CGM_METRICS,                False),
    ("SOCIOECONOMIC",               SOCIOECONOMIC_BINARY,       False),
    ("SOCIOECONOMIC (Raw Categorical)", SOCIOECONOMIC_CATEGORICAL, False),
]


# ============================================================================
# STATISTICS FUNCTIONS
# ============================================================================
def resolve_column(var_def, tp):
    col_spec = var_def[1]
    return col_spec(tp) if callable(col_spec) else col_spec


def compute_continuous_stats(series, total_n):
    numeric = pd.to_numeric(series, errors="coerce")
    n_valid = int(numeric.notna().sum())
    n_missing = total_n - n_valid
    pct_missing = n_missing / total_n * 100 if total_n > 0 else 0.0
    if n_valid == 0:
        return {"N (valid)": 0, "Missing": f"{n_missing} ({pct_missing:.1f}%)",
                "Mean ± SD": "—", "Median [Q1–Q3]": "—"}
    mean, sd = numeric.mean(), numeric.std()
    median = numeric.median()
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    return {"N (valid)": n_valid, "Missing": f"{n_missing} ({pct_missing:.1f}%)",
            "Mean ± SD": f"{mean:.2f} ± {sd:.2f}",
            "Median [Q1–Q3]": f"{median:.2f} [{q1:.2f}–{q3:.2f}]"}


def compute_binary_stats(series, total_n):
    binary = pd.to_numeric(series, errors="coerce")
    n_valid = int(binary.notna().sum())
    n_missing = total_n - n_valid
    pct_missing = n_missing / total_n * 100 if total_n > 0 else 0.0
    n_pos = int(binary.sum()) if n_valid > 0 else 0
    pct_pos = n_pos / n_valid * 100 if n_valid > 0 else 0.0
    return {"N (valid)": n_valid, "Missing": f"{n_missing} ({pct_missing:.1f}%)",
            "n (%)": f"{n_pos} / {n_valid} ({pct_pos:.1f}%)"}


def compute_categorical_stats(series, total_n):
    n_valid = int(series.notna().sum())
    n_missing = total_n - n_valid
    pct_missing = n_missing / total_n * 100 if total_n > 0 else 0.0
    if n_valid == 0:
        return {"N (valid)": 0, "Missing": f"{n_missing} ({pct_missing:.1f}%)", "Distribution": "—"}
    counts = series.value_counts(dropna=True)
    parts = [f"{val}: {cnt} ({cnt/total_n*100:.1f}%)" for val, cnt in counts.items()]
    return {"N (valid)": n_valid, "Missing": f"{n_missing} ({pct_missing:.1f}%)",
            "Distribution": "; ".join(parts)}


# ============================================================================
# TABLE BUILDERS
# ============================================================================
def build_characteristic_table(df):
    total_n = len(df)
    rows = []
    for section_name, var_list, tp_varying in VARIABLE_SECTIONS:
        rows.append({k: "" for k in ["Section","Variable","Timepoint","Type",
                     "Cohort N","N (valid)","Missing","Mean ± SD",
                     "Median [Q1–Q3]","n (%)","Distribution"]})
        rows[-1]["Section"] = section_name

        for display_name, col_spec, var_type in var_list:
            for tp in (TIMEPOINTS if tp_varying else ["—"]):
                if tp == "—":
                    col_name = col_spec if isinstance(col_spec, str) else col_spec("diagnosis")
                    tp_label = "All"
                else:
                    col_name = resolve_column((display_name, col_spec, var_type), tp)
                    tp_label = TP_DISPLAY[tp]

                row = {"Section": "", "Variable": display_name,
                       "Timepoint": tp_label, "Type": var_type, "Cohort N": total_n,
                       "N (valid)": "", "Missing": "", "Mean ± SD": "",
                       "Median [Q1–Q3]": "", "n (%)": "", "Distribution": ""}

                if col_name not in df.columns:
                    row.update({"N (valid)": "—", "Missing": f"{total_n} (100.0%)",
                                "Mean ± SD": "—", "Median [Q1–Q3]": "—",
                                "n (%)": "—", "Distribution": "column not found"})
                    rows.append(row)
                    continue

                series = df[col_name]
                if var_type == "continuous":
                    s = compute_continuous_stats(series, total_n)
                    row.update({"N (valid)": s["N (valid)"], "Missing": s["Missing"],
                                "Mean ± SD": s["Mean ± SD"], "Median [Q1–Q3]": s["Median [Q1–Q3]"]})
                elif var_type == "binary":
                    s = compute_binary_stats(series, total_n)
                    row.update({"N (valid)": s["N (valid)"], "Missing": s["Missing"],
                                "n (%)": s["n (%)"]})
                elif var_type == "categorical":
                    s = compute_categorical_stats(series, total_n)
                    row.update({"N (valid)": s["N (valid)"], "Missing": s["Missing"],
                                "Distribution": s["Distribution"]})
                rows.append(row)
    return rows


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2: return np.nan
    v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled if pooled != 0 else 0.0

def cramers_v(ct):
    try: chi2, _, _, _ = chi2_contingency(ct)
    except ValueError: return np.nan
    n = ct.values.sum()
    min_dim = min(ct.shape[0], ct.shape[1]) - 1
    return np.sqrt(chi2 / (n * min_dim)) if (min_dim > 0 and n > 0) else np.nan

def sig_stars(p):
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""

def effect_label(val):
    if pd.isna(val): return ""
    v = abs(val)
    if v < 0.2: return "negligible"
    if v < 0.5: return "small"
    if v < 0.8: return "medium"
    return "large"


def build_outcome_comparison(df, outcome_col):
    df_assessed = df[df[outcome_col].notna()].copy()
    df_pos = df_assessed[df_assessed[outcome_col] == 1]
    df_neg = df_assessed[df_assessed[outcome_col] == 0]
    n_pos, n_neg = len(df_pos), len(df_neg)
    if n_pos == 0 or n_neg == 0:
        return [], n_pos, n_neg

    rows = []
    for section_name, var_list, tp_varying in VARIABLE_SECTIONS:
        rows.append({k: "" for k in [
            "Section","Variable","Feature Timepoint","Type",
            "Positive: N","Positive: Mean ± SD","Positive: Median [Q1–Q3]",
            "Positive: n (%)","Positive: Distribution",
            "Negative: N","Negative: Mean ± SD","Negative: Median [Q1–Q3]",
            "Negative: n (%)","Negative: Distribution",
            "Test","Statistic","P-value","P-value (fmt)","Significance",
            "Effect Size","Effect Size Type","Effect Magnitude"]})
        rows[-1]["Section"] = section_name
        rows[-1]["P-value"] = np.nan

        for display_name, col_spec, var_type in var_list:
            for tp in (TIMEPOINTS if tp_varying else ["—"]):
                if tp == "—":
                    col_name = col_spec if isinstance(col_spec, str) else col_spec("diagnosis")
                    tp_label = "All"
                else:
                    col_name = resolve_column((display_name, col_spec, var_type), tp)
                    tp_label = TP_DISPLAY[tp]

                row = {"Section": "", "Variable": display_name,
                       "Feature Timepoint": tp_label, "Type": var_type, "P-value": np.nan}

                if col_name not in df_assessed.columns:
                    row.update({k: "—" for k in ["Positive: N","Positive: Mean ± SD",
                        "Positive: Median [Q1–Q3]","Positive: n (%)","Positive: Distribution",
                        "Negative: N","Negative: Mean ± SD","Negative: Median [Q1–Q3]",
                        "Negative: n (%)","Negative: Distribution",
                        "Test","Statistic","P-value (fmt)","Effect Size","Effect Size Type"]})
                    row["Significance"] = ""
                    row["Effect Magnitude"] = "column not found"
                    rows.append(row)
                    continue

                pos_data, neg_data = df_pos[col_name], df_neg[col_name]

                if var_type == "continuous":
                    pos = pd.to_numeric(pos_data, errors="coerce").dropna()
                    neg = pd.to_numeric(neg_data, errors="coerce").dropna()
                    row["Positive: N"], row["Negative: N"] = len(pos), len(neg)
                    row["Positive: Mean ± SD"] = f"{pos.mean():.2f} ± {pos.std():.2f}" if len(pos) > 0 else "—"
                    row["Negative: Mean ± SD"] = f"{neg.mean():.2f} ± {neg.std():.2f}" if len(neg) > 0 else "—"
                    row["Positive: Median [Q1–Q3]"] = f"{pos.median():.2f} [{pos.quantile(0.25):.2f}–{pos.quantile(0.75):.2f}]" if len(pos) > 0 else "—"
                    row["Negative: Median [Q1–Q3]"] = f"{neg.median():.2f} [{neg.quantile(0.25):.2f}–{neg.quantile(0.75):.2f}]" if len(neg) > 0 else "—"
                    p, stat = np.nan, np.nan
                    if len(pos) >= 2 and len(neg) >= 2:
                        try: stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
                        except: pass
                    d = cohens_d(pos, neg) if (len(pos) >= 2 and len(neg) >= 2) else np.nan
                    row.update({"Test": "Mann-Whitney U",
                                "Statistic": f"{stat:.1f}" if not pd.isna(stat) else "—",
                                "P-value": p, "P-value (fmt)": f"{p:.4f}" if not pd.isna(p) else "—",
                                "Significance": sig_stars(p),
                                "Effect Size": f"{d:.3f}" if not pd.isna(d) else "—",
                                "Effect Size Type": "Cohen's d", "Effect Magnitude": effect_label(d)})

                elif var_type == "binary":
                    pos_b = pd.to_numeric(pos_data, errors="coerce").dropna()
                    neg_b = pd.to_numeric(neg_data, errors="coerce").dropna()
                    n_pv, n_nv = len(pos_b), len(neg_b)
                    n_p1 = int(pos_b.sum()) if n_pv > 0 else 0
                    n_n1 = int(neg_b.sum()) if n_nv > 0 else 0
                    row["Positive: N"], row["Negative: N"] = n_pv, n_nv
                    row["Positive: n (%)"] = f"{n_p1} / {n_pv} ({n_p1/n_pv*100:.1f}%)" if n_pv > 0 else "—"
                    row["Negative: n (%)"] = f"{n_n1} / {n_nv} ({n_n1/n_nv*100:.1f}%)" if n_nv > 0 else "—"
                    p, stat, v = np.nan, np.nan, np.nan
                    if n_pv > 0 and n_nv > 0:
                        ct = pd.DataFrame({"Pos": [n_p1, n_pv-n_p1], "Neg": [n_n1, n_nv-n_n1]}, index=["Yes","No"])
                        try: stat, p, _, _ = chi2_contingency(ct); v = cramers_v(ct)
                        except: pass
                    row.update({"Test": "Chi-squared",
                                "Statistic": f"{stat:.2f}" if not pd.isna(stat) else "—",
                                "P-value": p, "P-value (fmt)": f"{p:.4f}" if not pd.isna(p) else "—",
                                "Significance": sig_stars(p),
                                "Effect Size": f"{v:.3f}" if not pd.isna(v) else "—",
                                "Effect Size Type": "Cramér's V", "Effect Magnitude": effect_label(v)})

                elif var_type == "categorical":
                    pos_c, neg_c = pos_data.dropna(), neg_data.dropna()
                    n_pv, n_nv = len(pos_c), len(neg_c)
                    def dist_str(s, n):
                        if n == 0: return "—"
                        return "; ".join(f"{v}: {c} ({c/n*100:.1f}%)" for v, c in s.value_counts().items())
                    row["Positive: N"], row["Negative: N"] = n_pv, n_nv
                    row["Positive: Distribution"] = dist_str(pos_c, n_pv)
                    row["Negative: Distribution"] = dist_str(neg_c, n_nv)
                    p, stat, v = np.nan, np.nan, np.nan
                    if n_pv > 0 and n_nv > 0:
                        combined = pd.concat([pos_c.to_frame("val").assign(g="pos"), neg_c.to_frame("val").assign(g="neg")])
                        ct = pd.crosstab(combined["val"], combined["g"])
                        if ct.shape[0] > 1 and ct.shape[1] > 1:
                            try: stat, p, _, _ = chi2_contingency(ct); v = cramers_v(ct)
                            except: pass
                    row.update({"Test": "Chi-squared",
                                "Statistic": f"{stat:.2f}" if not pd.isna(stat) else "—",
                                "P-value": p, "P-value (fmt)": f"{p:.4f}" if not pd.isna(p) else "—",
                                "Significance": sig_stars(p),
                                "Effect Size": f"{v:.3f}" if not pd.isna(v) else "—",
                                "Effect Size Type": "Cramér's V", "Effect Magnitude": effect_label(v)})

                for k in ["Positive: n (%)","Positive: Distribution","Positive: Mean ± SD",
                           "Positive: Median [Q1–Q3]","Negative: n (%)","Negative: Distribution",
                           "Negative: Mean ± SD","Negative: Median [Q1–Q3]"]:
                    if k not in row: row[k] = ""
                rows.append(row)
    return rows, n_pos, n_neg


# ============================================================================
# BUILD & SAVE
# ============================================================================
total_n = len(df_full)

# ---- Table 1: Overall ----
print(f"\n{'='*80}")
print(f"TABLE 1: OVERALL CHARACTERISTICS (N = {total_n:,})")
print(f"{'='*80}")

rows = build_characteristic_table(df_full)
df_table = pd.DataFrame(rows)
col_order = ["Section","Variable","Timepoint","Type","Cohort N",
             "N (valid)","Missing","Mean ± SD","Median [Q1–Q3]","n (%)","Distribution"]
df_table = df_table[[c for c in col_order if c in df_table.columns]]

char_path = os.path.join(CHAR_OUTPUT_DIR, "overall_characteristics.csv")
df_table.to_csv(char_path, index=False)
print(f"  ✓ Saved: {char_path}  ({len(df_table)} rows)")

# Cohort summary
cohort_rows = [{"Cohort": "Full (all T1D)", "N": total_n, "Description": "All patients"}]
if OUTCOME_COL in df_full.columns:
    n_ov = int(df_full[OUTCOME_COL].notna().sum())
    n_p = int((df_full[OUTCOME_COL] == 1).sum())
    n_n = int((df_full[OUTCOME_COL] == 0).sum())
    cohort_rows += [
        {"Cohort": "Outcome assessed", "N": n_ov, "Description": f"Non-null {OUTCOME_COL}"},
        {"Cohort": "Severe hypo (+)", "N": n_p, "Description": "Outcome = 1"},
        {"Cohort": "No severe hypo (−)", "N": n_n, "Description": "Outcome = 0"},
    ]
cohort_path = os.path.join(CHAR_OUTPUT_DIR, "cohort_summary.csv")
pd.DataFrame(cohort_rows).to_csv(cohort_path, index=False)
print(f"  ✓ Cohort summary: {cohort_path}")

# Availability matrix
avail_rows = []
for section_name, var_list, tp_varying in VARIABLE_SECTIONS:
    for display_name, col_spec, var_type in var_list:
        row = {"Section": section_name, "Variable": display_name}
        if tp_varying:
            for tp in TIMEPOINTS:
                col_name = resolve_column((display_name, col_spec, var_type), tp)
                if col_name in df_full.columns:
                    n_v = pd.to_numeric(df_full[col_name], errors="coerce").notna().sum() if var_type == "continuous" else df_full[col_name].notna().sum()
                    row[TP_DISPLAY[tp]] = f"{n_v:,} ({n_v/total_n*100:.0f}%)"
                else:
                    row[TP_DISPLAY[tp]] = "N/A"
        else:
            col_name = col_spec if isinstance(col_spec, str) else col_spec("diagnosis")
            if col_name in df_full.columns:
                n_v = df_full[col_name].notna().sum()
                row["All"] = f"{n_v:,} ({n_v/total_n*100:.0f}%)"
            else:
                row["All"] = "N/A"
        avail_rows.append(row)
avail_path = os.path.join(CHAR_OUTPUT_DIR, "variable_availability_matrix.csv")
pd.DataFrame(avail_rows).to_csv(avail_path, index=False)
print(f"  ✓ Availability matrix: {avail_path}")

# ---- Table 2: Outcome comparison ----
print(f"\n{'='*80}")
print(f"TABLE 2: SEVERE HYPOGLYCEMIA — POSITIVE vs NEGATIVE")
print(f"{'='*80}")

if OUTCOME_COL not in df_full.columns:
    print(f"  ⚠ Column '{OUTCOME_COL}' not found — skipping")
else:
    ov = df_full[OUTCOME_COL].dropna()
    n_p, n_n = int((ov == 1).sum()), int((ov == 0).sum())
    print(f"  Assessed: {len(ov):,}  |  Positive: {n_p:,}  |  Negative: {n_n:,}")

    if n_p == 0 or n_n == 0:
        print(f"  ⚠ Only one class — skipping")
    else:
        comp_rows, _, _ = build_outcome_comparison(df_full, OUTCOME_COL)
        if comp_rows:
            df_comp = pd.DataFrame(comp_rows)
            comp_col_order = [
                "Section","Variable","Feature Timepoint","Type",
                "Positive: N","Positive: Mean ± SD","Positive: Median [Q1–Q3]",
                "Positive: n (%)","Positive: Distribution",
                "Negative: N","Negative: Mean ± SD","Negative: Median [Q1–Q3]",
                "Negative: n (%)","Negative: Distribution",
                "Test","Statistic","P-value (fmt)","Significance",
                "Effect Size","Effect Size Type","Effect Magnitude"]
            df_comp = df_comp[[c for c in comp_col_order if c in df_comp.columns]]
            comp_path = os.path.join(OUTCOME_OUTPUT_DIR, "severe_hypoglycemia_comparison.csv")
            df_comp.to_csv(comp_path, index=False)
            print(f"  ✓ Comparison: {comp_path}  ({len(df_comp)} rows)")

            sig_rows = [{"Variable": r.get("Variable",""), "Feature Timepoint": r.get("Feature Timepoint",""),
                         "Type": r.get("Type",""), "P-value": r.get("P-value (fmt)",""),
                         "Significance": r.get("Significance",""), "Effect Size": r.get("Effect Size",""),
                         "Effect Size Type": r.get("Effect Size Type",""), "Effect Magnitude": r.get("Effect Magnitude","")}
                        for _, r in df_comp.iterrows() if r.get("Significance","") in ("*","**","***")]
            if sig_rows:
                sig_path = os.path.join(OUTCOME_OUTPUT_DIR, "significant_variables_summary.csv")
                pd.DataFrame(sig_rows).sort_values("P-value").to_csv(sig_path, index=False)
                print(f"  ✓ Significant variables: {sig_path}  ({len(sig_rows)} vars)")
            else:
                print(f"  No significant variables at p < 0.05")

# ---- Done ----
del df_full
gc.collect()

print(f"\n{'='*80}")
print(f"✓ T1D Analysis complete.")
print(f"{'='*80}")