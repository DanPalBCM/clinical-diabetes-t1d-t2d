
# %% [markdown]
# # Step 1: Overall Characteristic Tables for T2D Cohort
# 
# **Cell 1** — Preprocessing (reusable: run before any analysis)
# **Cell 2** — Characteristic table generation & export

# %% [markdown]
# ## Cell 1 — Load Data & Preprocess

# %%
"""
Preprocessing for T2D Cohort
=============================
Reusable preprocessing steps:
  1. BMI unit conversion: raw oz_av/in² → standard kg/m²
  2. Drop ICD-only condition columns (keep OUTCOME versions)
  3. Collapse high-cardinality SDOH categoricals to top-N + Other

For any future analysis, just run this cell to get a clean df_full.
"""

import pandas as pd
import numpy as np
from foundry.transforms import Dataset

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

_BMI_COLS = ["bmi_at_diagnosis", "bmi_at_2_years", "bmi_at_5_years"]

# oz_av/in² → kg/m²  =  (oz→kg) / (in→m)²
# Raw BMI = (weight_oz / height_in²) × 703
# This is the imperial BMI formula but with oz instead of lbs.
# Since 1 lb = 16 oz, the raw values are 16× too large.
# Divide by 16 to get standard kg/m².
BMI_RAW_TO_KG_M2 = 1.0 / 16.0

_ICD_ONLY_CONDITIONS = ["Hypertension", "Dyslipidemia", "Microalbuminuria"]
_COND_SUFFIXES = ["_diagnosis", "_2yr", "_5yr"]

_SDOH_COLLAPSE_TOP_N = [
    "socio_education_level_parents_guardian",
    "socio_employment_status_parents_guardian",
    "socio_financial_strain",
    "socio_insurance_status",
    "socio_social_family_support",
]

# SDOH columns to convert from categorical → binary (positive/negative)
_SDOH_BINARIZE = {
    # ---- Physical Activity → 1 = Active, 0 = Inactive ----
    "socio_physical_activity": {
        "new_col": "socio_physical_activity_binary",
        "method": "keyword",
        # If ANY inactive keyword matches (case-insensitive), classify as 0.
        # Checked FIRST — takes priority over active keywords.
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
        # If ANY active keyword matches AND no inactive keyword matched → 1
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
        # Default if no keyword matches
        "default": np.nan,
    },

    # ---- Social/Family Support → 1 = Adequate+, 0 = Limited/None ----
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

    # ---- Financial Strain → 1 = At Risk, 0 = Low Risk ----
    "socio_financial_strain": {
        "new_col": "socio_financial_strain_binary",
        "method": "exact",
        "positive": ["Moderate Risk", "High Risk", "Severe", "Medium Risk"],
        "negative": ["Low Risk"],
    },

    # ---- Parental Employment → 1 = Employed, 0 = Not Employed ----
    "socio_employment_status_parents_guardian": {
        "new_col": "socio_parental_employment_binary",
        "method": "keyword",
        "active_keywords": ["employed", "both employed", "student"],
        "inactive_keywords": ["unemployed", "disabled", "retired",
                              "mixed"],
        "default": np.nan,
    },

    # ---- Parental Education → 1 = HS or higher, 0 = Below HS ----
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
                     "Mother: High School; Father: Some College; Primary caregiver (grandmother): Some College",
                     ],
        "negative": ["Elementary",
                     "Some High School",
                     "Mother: Elementary, Father: High School",
                     "Mother: Some High School (9th Grade); Father: High School (12th Grade)",
                     "Mother: High School; Father: Elementary",
                     "Mother: High School (incomplete, grade 9); Father: Some College",
                     "Some College (Mother), Elementary (Father)",
                     "Mother: High School, Father: Elementary",
                     ],
        # "Unknown" → NaN
    },

    # ---- Insurance Status → categorical: Private, Government, Uninsured ----
    "socio_insurance_status": {
        "new_col": "socio_insurance_category",
        "method": "exact_categorical",
        "mapping": {
            "Private": "Private",
            "Multiple": "Private",
            "Medicaid": "Government",
            "CHIP": "Government",
            "TRICARE WEST": "Government",
            "Medicare": "Government",
            "Uninsured": "Uninsured",
            "Self-Pay": "Uninsured",
        },
    },
}

DEFAULT_TOP_N = 5


def convert_bmi_to_standard(df, inplace=False):
    """Convert BMI columns from oz_av/in² to kg/m²."""
    if not inplace:
        df = df.copy()
    for col in _BMI_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * BMI_RAW_TO_KG_M2
    return df


def drop_icd_only_conditions(df, inplace=False):
    """Remove ICD-code-only condition columns; OUTCOME_* versions are kept."""
    if not inplace:
        df = df.copy()
    cols_to_drop = [
        f"{cond}{suffix}"
        for cond in _ICD_ONLY_CONDITIONS
        for suffix in _COND_SUFFIXES
        if f"{cond}{suffix}" in df.columns
    ]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
    return df


def collapse_sdoh_categories(df, top_n=DEFAULT_TOP_N, columns=None, inplace=False):
    """Keep top-N most frequent categories per SDOH column; rest → 'Other'."""
    if not inplace:
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


def _binarize_sdoh(df, config=None, inplace=False):
    """
    Convert high-cardinality SDOH categoricals to binary (1/0) or
    mapped categorical columns.

    For 'exact' method: maps specific values to 1 (positive) or 0 (negative).
    For 'keyword' method: scans lowercase text for keyword matches.
      - inactive_keywords checked first → 0
      - active_keywords checked second → 1
      - no match → NaN
    For 'exact_categorical' method: maps values to new category labels.

    The original column is KEPT; a new column is added.
    """
    if not inplace:
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
            df[new_col] = np.where(
                series.isin(pos_set), 1,
                np.where(series.isin(neg_set), 0, np.nan)
            )

        elif method == "keyword":
            active_kw = [k.lower() for k in spec.get("active_keywords", [])]
            inactive_kw = [k.lower() for k in spec.get("inactive_keywords", [])]
            default = spec.get("default", np.nan)

            result = pd.Series(default, index=df.index, dtype=float)

            for idx, val in series.items():
                if pd.isna(val):
                    continue
                val_lower = str(val).lower()

                # Check inactive FIRST
                if any(kw in val_lower for kw in inactive_kw):
                    result.at[idx] = 0
                    continue

                # Then check active
                if any(kw in val_lower for kw in active_kw):
                    result.at[idx] = 1
                    continue

            df[new_col] = result

        elif method == "exact_categorical":
            mapping = spec.get("mapping", {})
            df[new_col] = series.map(mapping)

        # Log distribution
        n_valid = df[new_col].notna().sum()
        if method == "exact_categorical":
            dist = df[new_col].value_counts(dropna=True)
            dist_str = ", ".join(f"{v}={c}" for v, c in dist.items())
            n_unk = df[orig_col].notna().sum() - n_valid
            print(f"    {new_col}: {dist_str}, unmatched={n_unk}, missing={len(df)-df[orig_col].notna().sum()}")
        else:
            n_pos = (df[new_col] == 1).sum()
            n_neg = (df[new_col] == 0).sum()
            n_unk = df[orig_col].notna().sum() - n_valid
            print(f"    {new_col}: 1={n_pos}, 0={n_neg}, unmatched={n_unk}, missing={len(df)-df[orig_col].notna().sum()}")

    return df


def preprocess(df, convert_bmi=True, drop_icd_conditions=True,
               collapse_sdoh=True, binarize_sdoh_cats=True,
               top_n_cats=DEFAULT_TOP_N,
               sdoh_columns=None, inplace=False):
    """
    Run all preprocessing steps.

    Parameters
    ----------
    df : DataFrame
    convert_bmi : bool       — convert BMI raw → kg/m²
    drop_icd_conditions : bool — drop ICD-only condition columns
    collapse_sdoh : bool     — collapse remaining SDOH categoricals to top-N
    binarize_sdoh_cats : bool — convert SDOH categoricals to binary columns
    top_n_cats : int         — categories to keep in collapse (default 5)
    sdoh_columns : list|None — override SDOH columns to collapse
    inplace : bool
    """
    if not inplace:
        df = df.copy()

    n_cols_before = len(df.columns)

    if convert_bmi:
        df = convert_bmi_to_standard(df, inplace=True)
        print(f"  [preprocess] BMI converted to kg/m²  (÷ 16, raw was oz-based imperial)")

    if drop_icd_conditions:
        n_before = len(df.columns)
        df = drop_icd_only_conditions(df, inplace=True)
        print(f"  [preprocess] Dropped {n_before - len(df.columns)} ICD-only condition columns")

    if binarize_sdoh_cats:
        df = _binarize_sdoh(df, inplace=True)
        print(f"  [preprocess] SDOH categoricals binarized (new binary columns added)")

    if collapse_sdoh:
        df = collapse_sdoh_categories(df, top_n=top_n_cats, columns=sdoh_columns, inplace=True)
        print(f"  [preprocess] Remaining SDOH categoricals collapsed to top-{top_n_cats} + Other")

    print(f"  [preprocess] Done. Columns: {n_cols_before} → {len(df.columns)}")
    return df


# ============================================================================
# LOAD & PREPROCESS
# ============================================================================

print("Loading dataset...", end="")
df_full = Dataset.get("t2d_outcomes").read_table(format="pandas")
print(f" ✓  ({len(df_full):,} patients, {len(df_full.columns)} columns)")

print("\nRunning preprocessing...")
df_full = preprocess(df_full, top_n_cats=5)

print(f"\n✓ df_full ready: {len(df_full):,} rows × {len(df_full.columns)} columns")


# %%
import os
import warnings
import gc

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "analysis/characteristic_tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_COHORT_FILTER_COL = "a1c_diagnosis"

# ============================================================================
# TIMEPOINT SUFFIX MAPS
# ============================================================================
TP_MEAS = {"diagnosis": "_at_diagnosis", "2yr": "_at_2_years", "5yr": "_at_5_years"}
TP_MED  = {"diagnosis": "_diagnosis",    "2yr": "_2yr",        "5yr": "_5yr"}
TP_COND = TP_MED
TP_A1C  = {"diagnosis": "a1c_diagnosis", "2yr": "a1c_2yr",     "5yr": "a1c_5yr"}

TIMEPOINTS = ["diagnosis", "2yr", "5yr"]
TP_DISPLAY = {"diagnosis": "At Diagnosis", "2yr": "At 2 Years", "5yr": "At 5 Years"}

def meas_col(base, tp): return f"{base}{TP_MEAS[tp]}"
def med_col(base, tp):  return f"{base}{TP_MED[tp]}"
def cond_col(base, tp): return f"{base}{TP_COND[tp]}"

# ============================================================================
# VARIABLE DEFINITIONS
# ============================================================================
DEMOGRAPHICS = [
    ("Age at Diagnosis",       "age_at_diagnosis",  "continuous"),
    ("Sex",                    "sex",               "categorical"),
    ("Race",                   "patient_race",      "categorical"),
    ("Ethnicity",              "ethnic_group",       "categorical"),
    ("Language",               "language",           "categorical"),
]

GLYCEMIC = [
    ("HbA1c (%)",              lambda tp: TP_A1C[tp],                          "continuous"),
    ("Glucose (mg/dL)",        lambda tp: meas_col("glucose", tp),             "continuous"),
]

ANTHROPOMETRICS = [
    ("BMI (kg/m²)",            lambda tp: meas_col("bmi", tp),                 "continuous"),
    ("BMI Z-score",            lambda tp: meas_col("bmi_zscore", tp),          "continuous"),
    ("BMI Percentile",         lambda tp: meas_col("bmi_percentile", tp),      "continuous"),
    ("Height Z-score",         lambda tp: meas_col("height_zscore", tp),       "continuous"),
    ("Height Percentile",      lambda tp: meas_col("height_percentile", tp),   "continuous"),
    ("Weight Z-score",         lambda tp: meas_col("weight_zscore", tp),       "continuous"),
    ("Weight Percentile",      lambda tp: meas_col("weight_percentile", tp),   "continuous"),
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
    ("Serum Creatinine (mg/dL)",            lambda tp: meas_col("serum_creatinine", tp),                    "continuous"),
    ("BUN (mg/dL)",                         lambda tp: meas_col("bun", tp),                                 "continuous"),
    ("eGFR (mL/min/1.73m²)",               lambda tp: meas_col("egfr", tp),                                "continuous"),
    ("Urine Microalbumin (mg/dL)",          lambda tp: meas_col("urine_microalbumin", tp),                  "continuous"),
    ("Urine Microalbumin/Creatinine Ratio", lambda tp: meas_col("urine_microalbumin_creatinine_ratio", tp), "continuous"),
]

LIVER = [
    ("ALT (U/L)",              lambda tp: meas_col("alt", tp),                 "continuous"),
    ("AST (U/L)",              lambda tp: meas_col("ast", tp),                 "continuous"),
]

OTHER_LABS = [
    ("Serum C-peptide (ng/mL)",    lambda tp: meas_col("serum_c_peptide", tp),     "continuous"),
    ("Blood pH",                   lambda tp: meas_col("blood_ph", tp),            "continuous"),
    ("Bicarbonate (mmol/L)",       lambda tp: meas_col("bicarbonate", tp),         "continuous"),
    ("pCO2 (mmHg)",                lambda tp: meas_col("pco2", tp),                "continuous"),
]

MEDICATIONS = [
    ("Insulin",                    lambda tp: med_col("Insulins", tp),                        "binary"),
    ("Metformin (Biguanide)",      lambda tp: med_col("Biguanide", tp),                       "binary"),
    ("GLP-1 Agonists",            lambda tp: med_col("GLP1_agonists", tp),                    "binary"),
]

# Removed Hypertension, Dyslipidemia, Microalbuminuria (now only in OUTCOMES)
CONDITIONS = [
    ("DKA",                        lambda tp: cond_col("DKA", tp),                  "binary"),
    ("Ketosis",                    lambda tp: cond_col("Ketosis", tp),               "binary"),
    ("Diabetic Retinopathy",       lambda tp: cond_col("Diabetic_Retinopathy", tp),  "binary"),
    ("Neuropathy",                 lambda tp: cond_col("Neuropathy", tp),            "binary"),
    ("Hypoglycemia",               lambda tp: cond_col("Hypoglycemia", tp),          "binary"),
]

SOCIOECONOMIC_BINARY = [
    ("Adverse Childhood Exp.",          "socio_adverse_childhood_experience",        "binary"),
    ("Alcohol Abuse",                   "socio_alcohol_abuse",                       "binary"),
    ("Drug/Substance Abuse",            "socio_drug_substance_abuse",                "binary"),
    ("Food Insecurity",                 "socio_food_insecurity",                     "binary"),
    ("Housing Instability",             "socio_housing_instability",                 "binary"),
    ("Physical/Sexual Abuse",           "socio_physical_sexual_abuse",               "binary"),
    ("Smoking",                         "socio_smoking",                             "binary"),
    ("Transportation Barrier",          "socio_transportation_barrier",              "binary"),
    ("Physically Active",               "socio_physical_activity_binary",            "binary"),
    ("Social/Family Support (Adequate+)", "socio_social_family_support_binary",      "binary"),
    ("Financial Strain (At Risk)",      "socio_financial_strain_binary",             "binary"),
    ("Parental Employment (Employed)",  "socio_parental_employment_binary",          "binary"),
    ("Parental Education (HS+)",        "socio_parental_education_binary",           "binary"),
    ("Insurance Status",                "socio_insurance_category",                  "categorical"),
]

OUTCOMES = [
    ("Hypertension",             lambda tp: f"OUTCOME_Hypertension{TP_MEAS[tp]}",            "binary"),
    ("Dyslipidemia",             lambda tp: f"OUTCOME_Dyslipidemia{TP_MEAS[tp]}",            "binary"),
    ("Microalbuminuria",         lambda tp: f"OUTCOME_Microalbuminuria{TP_MEAS[tp]}",         "binary"),
    ("Optimal Glycemic Control", lambda tp: f"OUTCOME_Optimal_Glycemic_Control{TP_MEAS[tp]}", "binary"),
    ("Insulin Independence",     lambda tp: f"OUTCOME_Insulin_Independence{TP_MEAS[tp]}",    "binary"),
    ("Metformin Response",       lambda tp: f"OUTCOME_Metformin_Response{TP_MEAS[tp]}",      "binary"),
    ("GLP-1RA Response",         lambda tp: f"OUTCOME_GLP1RA_Response{TP_MEAS[tp]}",         "binary"),
]

SOCIOECONOMIC_CATEGORICAL = [
    ("Parental Education",     "socio_education_level_parents_guardian",       "categorical"),
    ("Parental Employment",    "socio_employment_status_parents_guardian",     "categorical"),
    ("Financial Strain",       "socio_financial_strain",                       "categorical"),
    ("Insurance Status",       "socio_insurance_status",                       "categorical"),
    ("Physical Activity",      "socio_physical_activity",                      "categorical"),
    ("Social/Family Support",  "socio_social_family_support",                  "categorical"),
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
    ("MEDICATIONS",                 MEDICATIONS,                True),
    ("CONDITIONS / COMPLICATIONS",  CONDITIONS,                 True),
    ("SOCIOECONOMIC",               SOCIOECONOMIC_BINARY,       False),
    ("OUTCOMES",                    OUTCOMES,                   True),
]

# ============================================================================
# STATISTICS FUNCTIONS
# ============================================================================

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
        return {"N (valid)": 0, "Missing": f"{n_missing} ({pct_missing:.1f}%)",
                "Distribution": "—"}
    counts = series.value_counts(dropna=True)
    parts = [f"{val}: {cnt} ({cnt/total_n*100:.1f}%)" for val, cnt in counts.items()]
    return {"N (valid)": n_valid, "Missing": f"{n_missing} ({pct_missing:.1f}%)",
            "Distribution": "; ".join(parts)}


def resolve_column(var_def, tp):
    col_spec = var_def[1]
    return col_spec(tp) if callable(col_spec) else col_spec


# ============================================================================
# TABLE BUILDERS
# ============================================================================

def build_characteristic_table(df, cohort_label, per_timepoint_dfs=None):
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
                    tp_label, df_tp = "All", df
                else:
                    col_name = resolve_column((display_name, col_spec, var_type), tp)
                    tp_label = TP_DISPLAY[tp]
                    df_tp = (per_timepoint_dfs[tp]
                             if per_timepoint_dfs and tp in per_timepoint_dfs
                             else df)

                total_n = len(df_tp)
                row = {"Section": "", "Variable": display_name,
                       "Timepoint": tp_label, "Type": var_type, "Cohort N": total_n,
                       "N (valid)": "", "Missing": "", "Mean ± SD": "",
                       "Median [Q1–Q3]": "", "n (%)": "", "Distribution": ""}

                if col_name not in df_tp.columns:
                    row.update({"N (valid)": "—", "Missing": f"{total_n} (100.0%)",
                                "Mean ± SD": "—", "Median [Q1–Q3]": "—",
                                "n (%)": "—", "Distribution": "column not found"})
                    rows.append(row)
                    continue

                series = df_tp[col_name]
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


def build_availability_matrix(df):
    total_n = len(df)
    rows = []
    for section_name, var_list, tp_varying in VARIABLE_SECTIONS:
        for display_name, col_spec, var_type in var_list:
            row = {"Section": section_name, "Variable": display_name}
            if tp_varying:
                for tp in TIMEPOINTS:
                    col_name = resolve_column((display_name, col_spec, var_type), tp)
                    if col_name in df.columns:
                        n_v = (pd.to_numeric(df[col_name], errors="coerce").notna().sum()
                               if var_type == "continuous" else df[col_name].notna().sum())
                        row[TP_DISPLAY[tp]] = f"{n_v:,} ({n_v/total_n*100:.0f}%)"
                    else:
                        row[TP_DISPLAY[tp]] = "N/A"
            else:
                col_name = col_spec if isinstance(col_spec, str) else col_spec("diagnosis")
                if col_name in df.columns:
                    n_v = df[col_name].notna().sum()
                    row["All"] = f"{n_v:,} ({n_v/total_n*100:.0f}%)"
                else:
                    row["All"] = "N/A"
            rows.append(row)
    return rows


# ============================================================================
# BUILD & SAVE TABLES
# ============================================================================

n_full = len(df_full)

if VALID_COHORT_FILTER_COL not in df_full.columns:
    raise ValueError(f"Filter column '{VALID_COHORT_FILTER_COL}' not found.")

n_valid_dx = int(df_full[VALID_COHORT_FILTER_COL].notna().sum())
n_dropped_dx = n_full - n_valid_dx

print(f"{'='*80}")
print(f"COHORT DEFINITIONS")
print(f"{'='*80}")
print(f"  Full cohort             : {n_full:,} patients")
print(f"  Valid at diagnosis      : {n_valid_dx:,} patients  (A1C not null)")
print(f"  Excluded (no dx A1C)    : {n_dropped_dx:,} ({n_dropped_dx/n_full*100:.1f}%)")

# ---- Per-timepoint valid DataFrames ----
valid_tp_dfs = {}
cohort_rows = [
    {"Cohort": "Full",             "N": n_full,     "Description": "All patients in dataset"},
    {"Cohort": "Valid (diagnosis)", "N": n_valid_dx, "Description": "Non-null a1c_diagnosis"},
]

print(f"\n  Per-timepoint valid cohort sizes:")
for tp in TIMEPOINTS:
    a1c_col = TP_A1C[tp]
    if a1c_col in df_full.columns:
        tp_df = df_full[df_full[a1c_col].notna()].copy()
        valid_tp_dfs[tp] = tp_df
        n_tp = len(tp_df)
        print(f"    {TP_DISPLAY[tp]:15s}: {n_tp:,} patients")
        if tp != "diagnosis":
            cohort_rows.append({"Cohort": f"Valid ({tp})", "N": n_tp,
                                "Description": f"Non-null {a1c_col}"})
    else:
        print(f"    {TP_DISPLAY[tp]:15s}: column '{a1c_col}' not found — using full cohort")
        valid_tp_dfs[tp] = df_full

df_valid_base = valid_tp_dfs.get("diagnosis", df_full)

cohort_summary = pd.DataFrame(cohort_rows)
cohort_summary_path = os.path.join(OUTPUT_DIR, "cohort_summary.csv")
cohort_summary.to_csv(cohort_summary_path, index=False)
print(f"\n  Cohort summary → {cohort_summary_path}")

col_order = ["Section", "Variable", "Timepoint", "Type", "Cohort N",
             "N (valid)", "Missing", "Mean ± SD", "Median [Q1–Q3]",
             "n (%)", "Distribution"]

# ---- Table 1: Full Cohort ----
print(f"\n{'='*80}")
print(f"TABLE 1: FULL COHORT (N = {n_full:,})")
print(f"{'='*80}")

rows_full = build_characteristic_table(df_full, "Full Cohort")
df_table_full = pd.DataFrame(rows_full)[[c for c in col_order]]
path_full = os.path.join(OUTPUT_DIR, "overall_characteristics_full_cohort.csv")
df_table_full.to_csv(path_full, index=False)
print(f"  ✓ Saved → {path_full}  ({len(df_table_full)} rows)")

# ---- Table 2: Valid Cohort ----
print(f"\n{'='*80}")
print(f"TABLE 2: VALID COHORT (per-timepoint A1C denominators)")
print(f"{'='*80}")

rows_valid = build_characteristic_table(df_valid_base, "Valid Cohort",
                                        per_timepoint_dfs=valid_tp_dfs)
df_table_valid = pd.DataFrame(rows_valid)[[c for c in col_order]]
path_valid = os.path.join(OUTPUT_DIR, "overall_characteristics_valid_cohort.csv")
df_table_valid.to_csv(path_valid, index=False)
print(f"  ✓ Saved → {path_valid}  ({len(df_table_valid)} rows)")

# ---- Availability matrix ----
print(f"\n{'='*80}")
print(f"VARIABLE AVAILABILITY")
print(f"{'='*80}")

avail_full = pd.DataFrame(build_availability_matrix(df_full))
avail_valid_rows = []
for section_name, var_list, tp_varying in VARIABLE_SECTIONS:
    for display_name, col_spec, var_type in var_list:
        row = {"Section": section_name, "Variable": display_name}
        if tp_varying:
            for tp in TIMEPOINTS:
                df_tp = valid_tp_dfs.get(tp, df_full)
                n_tp = len(df_tp)
                col_name = resolve_column((display_name, col_spec, var_type), tp)
                if col_name in df_tp.columns:
                    n_v = (pd.to_numeric(df_tp[col_name], errors="coerce").notna().sum()
                           if var_type == "continuous" else df_tp[col_name].notna().sum())
                    row[TP_DISPLAY[tp]] = f"{n_v:,} / {n_tp:,} ({n_v/n_tp*100:.0f}%)"
                else:
                    row[TP_DISPLAY[tp]] = "N/A"
        else:
            df_base = valid_tp_dfs.get("diagnosis", df_full)
            n_base = len(df_base)
            col_name = col_spec if isinstance(col_spec, str) else col_spec("diagnosis")
            if col_name in df_base.columns:
                n_v = df_base[col_name].notna().sum()
                row["All"] = f"{n_v:,} / {n_base:,} ({n_v/n_base*100:.0f}%)"
            else:
                row["All"] = "N/A"
        avail_valid_rows.append(row)

avail_valid = pd.DataFrame(avail_valid_rows)

tp_cols_f = [c for c in avail_full.columns if c not in ("Section", "Variable")]
tp_cols_v = [c for c in avail_valid.columns if c not in ("Section", "Variable")]
avail_full  = avail_full.rename(columns={c: f"Full — {c}" for c in tp_cols_f})
avail_valid = avail_valid.rename(columns={c: f"Valid — {c}" for c in tp_cols_v})

df_avail = avail_full.merge(avail_valid, on=["Section", "Variable"], how="outer")
avail_path = os.path.join(OUTPUT_DIR, "variable_availability_matrix.csv")
df_avail.to_csv(avail_path, index=False)
print(f"  ✓ Availability matrix → {avail_path}")

# ---- HbA1c summary ----
print(f"\n{'='*80}")
print(f"HbA1c AVAILABILITY BY TIMEPOINT")
print(f"{'='*80}")
for tp in TIMEPOINTS:
    a1c_col = TP_A1C[tp]
    if a1c_col in df_full.columns:
        n_a = df_full[a1c_col].notna().sum()
        print(f"  {TP_DISPLAY[tp]:15s}:  {n_a:,} / {n_full:,} ({n_a/n_full*100:.1f}%)")

# ---- Done ----
print(f"\n{'='*80}")
print(f"✓ Step 1 complete.")
print(f"{'='*80}")
print(f"\nOutput files:")
print(f"  1. {path_full}")
print(f"  2. {path_valid}")
print(f"  3. {avail_path}")
print(f"  4. {cohort_summary_path}")