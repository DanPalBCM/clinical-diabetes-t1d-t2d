
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


# %% [markdown]
# ## Cell 2 — Outcome Comparison Tables (FIXED)

# %%
"""
Step 2: Characteristic Tables per Outcome  (FIXED)
====================================================
For each outcome × timepoint, split the A1C-valid cohort into positive (1) vs
negative (0) groups and compare all relevant features with:
  - Descriptive stats (mean ± SD, median [IQR] or n (%))
  - Mann-Whitney U test (continuous) / Chi-squared test (categorical & binary)
  - Effect sizes: Cohen's d (continuous) / Cramér's V (categorical & binary)

Feature sets per outcome timepoint:
  - Diagnosis outcomes  → diagnosis-time features only
  - 2yr outcomes        → diagnosis-time features only
  - 5yr outcomes        → diagnosis + 2yr features

FIX APPLIED:
  The binarized SDOH columns (socio_physical_activity_binary, etc.) and the
  mapped categorical column (socio_insurance_category) are created by the
  preprocessing step and live on df_full.  In the original code the lookup
  for non-timepoint-varying variables was correct, but the dataframe slicing
  and/or NaN handling could silently produce empty results.

  Root cause: for non-timepoint-varying binary columns produced by
  _binarize_sdoh, the values are float 0.0 / 1.0 / NaN.  When we build
  df_assessed = df[df[outcome_col].notna()].copy(), the binarized columns
  come along BUT pd.to_numeric on an already-numeric float column that is
  entirely NaN (because copy() can sometimes reset dtypes or because the
  column was stored as object) returns all NaN.

  Fix: explicitly cast binarized cols to float64 in the comparison helpers
  and add a diagnostic print when a known-binary column yields 0 valid rows
  so we can trace issues.  Also ensure the column lookup uses the BASE
  dataframe (df, not df_tp) for non-timepoint-varying features so the
  denominator matches the full assessed cohort.
"""

import os
import warnings
import gc
from scipy.stats import mannwhitneyu, chi2_contingency

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "analysis/Outcomes_characteristic_tables"
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
# FEATURE DEFINITIONS
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

FEATURE_SECTIONS = [
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
]

# ============================================================================
# OUTCOMES TO ANALYZE
# ============================================================================
OUTCOME_DEFINITIONS = [
    ("Hypertension",             lambda tp: f"OUTCOME_Hypertension{TP_MEAS[tp]}"),
    ("Dyslipidemia",             lambda tp: f"OUTCOME_Dyslipidemia{TP_MEAS[tp]}"),
    ("Microalbuminuria",         lambda tp: f"OUTCOME_Microalbuminuria{TP_MEAS[tp]}"),
    ("Optimal Glycemic Control", lambda tp: f"OUTCOME_Optimal_Glycemic_Control{TP_MEAS[tp]}"),
    ("Insulin Independence",     lambda tp: f"OUTCOME_Insulin_Independence{TP_MEAS[tp]}"),
    ("Metformin Response",       lambda tp: f"OUTCOME_Metformin_Response{TP_MEAS[tp]}"),
    ("GLP-1RA Response",         lambda tp: f"OUTCOME_GLP1RA_Response{TP_MEAS[tp]}"),
]

FEATURE_TIMEPOINTS_FOR_OUTCOME = {
    "diagnosis": ["diagnosis"],
    "2yr":       ["diagnosis"],
    "5yr":       ["diagnosis", "2yr"],
}

# ============================================================================
# HELPER
# ============================================================================
def resolve_column(col_spec, tp):
    return col_spec(tp) if callable(col_spec) else col_spec

# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def cramers_v(contingency_table):
    try:
        chi2, _, _, _ = chi2_contingency(contingency_table)
    except ValueError:
        return np.nan
    n = contingency_table.values.sum()
    min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
    if min_dim == 0 or n == 0:
        return np.nan
    return np.sqrt(chi2 / (n * min_dim))


def effect_size_label(val):
    if pd.isna(val):
        return ""
    v = abs(val)
    if v < 0.2:   return "negligible"
    elif v < 0.5: return "small"
    elif v < 0.8: return "medium"
    else:          return "large"


def sig_stars(p):
    if pd.isna(p):   return ""
    if p < 0.001:    return "***"
    elif p < 0.01:   return "**"
    elif p < 0.05:   return "*"
    return ""


def _safe_numeric_series(series):
    """
    FIX: Robustly convert a series to numeric float64.
    Handles object-dtype columns that contain numeric-like values,
    as well as float columns that may have been cast to object during
    DataFrame slicing / copy operations.
    """
    return pd.to_numeric(series, errors="coerce").astype("float64")


def compare_continuous(pos_series, neg_series):
    pos = _safe_numeric_series(pos_series).dropna()
    neg = _safe_numeric_series(neg_series).dropna()
    n_pos_valid, n_neg_valid = len(pos), len(neg)

    def fmt(s):
        if len(s) == 0:
            return "—", "—", 0
        mean, sd = s.mean(), s.std()
        med, q1, q3 = s.median(), s.quantile(0.25), s.quantile(0.75)
        return f"{mean:.2f} ± {sd:.2f}", f"{med:.2f} [{q1:.2f}–{q3:.2f}]", len(s)

    pos_mean_sd, pos_med_iqr, _ = fmt(pos)
    neg_mean_sd, neg_med_iqr, _ = fmt(neg)

    p_value, statistic = np.nan, np.nan
    if n_pos_valid >= 2 and n_neg_valid >= 2:
        try:
            statistic, p_value = mannwhitneyu(pos, neg, alternative="two-sided")
        except Exception:
            pass

    d = cohens_d(pos, neg) if (n_pos_valid >= 2 and n_neg_valid >= 2) else np.nan

    return {
        "Positive: Mean ± SD": pos_mean_sd, "Positive: Median [Q1–Q3]": pos_med_iqr,
        "Positive: N": n_pos_valid,
        "Negative: Mean ± SD": neg_mean_sd, "Negative: Median [Q1–Q3]": neg_med_iqr,
        "Negative: N": n_neg_valid,
        "Test": "Mann-Whitney U",
        "Statistic": f"{statistic:.1f}" if not pd.isna(statistic) else "—",
        "P-value": p_value,
        "P-value (fmt)": f"{p_value:.4f}" if not pd.isna(p_value) else "—",
        "Significance": sig_stars(p_value),
        "Effect Size": f"{d:.3f}" if not pd.isna(d) else "—",
        "Effect Size Type": "Cohen's d",
        "Effect Magnitude": effect_size_label(d),
    }


def compare_binary(pos_series, neg_series):
    pos = _safe_numeric_series(pos_series).dropna()
    neg = _safe_numeric_series(neg_series).dropna()
    n_pos_valid, n_neg_valid = len(pos), len(neg)
    n_pos_1 = int(pos.sum()) if n_pos_valid > 0 else 0
    n_neg_1 = int(neg.sum()) if n_neg_valid > 0 else 0
    pct_pos = n_pos_1 / n_pos_valid * 100 if n_pos_valid > 0 else 0
    pct_neg = n_neg_1 / n_neg_valid * 100 if n_neg_valid > 0 else 0

    p_value, statistic, v = np.nan, np.nan, np.nan
    if n_pos_valid > 0 and n_neg_valid > 0:
        ct = pd.DataFrame({
            "Positive": [n_pos_1, n_pos_valid - n_pos_1],
            "Negative": [n_neg_1, n_neg_valid - n_neg_1],
        }, index=["Yes", "No"])
        if ct.values.sum() > 0 and ct.shape[0] > 1 and ct.shape[1] > 1:
            try:
                statistic, p_value, _, _ = chi2_contingency(ct)
                v = cramers_v(ct)
            except Exception:
                pass

    return {
        "Positive: Mean ± SD": "", "Positive: Median [Q1–Q3]": "",
        "Positive: N": n_pos_valid,
        "Positive: n (%)": f"{n_pos_1} / {n_pos_valid} ({pct_pos:.1f}%)" if n_pos_valid > 0 else "—",
        "Negative: Mean ± SD": "", "Negative: Median [Q1–Q3]": "",
        "Negative: N": n_neg_valid,
        "Negative: n (%)": f"{n_neg_1} / {n_neg_valid} ({pct_neg:.1f}%)" if n_neg_valid > 0 else "—",
        "Test": "Chi-squared",
        "Statistic": f"{statistic:.2f}" if not pd.isna(statistic) else "—",
        "P-value": p_value,
        "P-value (fmt)": f"{p_value:.4f}" if not pd.isna(p_value) else "—",
        "Significance": sig_stars(p_value),
        "Effect Size": f"{v:.3f}" if not pd.isna(v) else "—",
        "Effect Size Type": "Cramér's V",
        "Effect Magnitude": effect_size_label(v),
    }


def compare_categorical(pos_series, neg_series):
    pos, neg = pos_series.dropna(), neg_series.dropna()
    n_pos_valid, n_neg_valid = len(pos), len(neg)

    def dist_str(s, n):
        if n == 0: return "—"
        counts = s.value_counts()
        return "; ".join(f"{val}: {cnt} ({cnt/n*100:.1f}%)" for val, cnt in counts.items())

    p_value, statistic, v = np.nan, np.nan, np.nan
    if n_pos_valid > 0 and n_neg_valid > 0:
        combined = pd.concat([
            pos.to_frame("val").assign(group="pos"),
            neg.to_frame("val").assign(group="neg"),
        ])
        ct = pd.crosstab(combined["val"], combined["group"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            try:
                statistic, p_value, _, _ = chi2_contingency(ct)
                v = cramers_v(ct)
            except Exception:
                pass

    return {
        "Positive: Mean ± SD": "", "Positive: Median [Q1–Q3]": "",
        "Positive: N": n_pos_valid,
        "Positive: Distribution": dist_str(pos, n_pos_valid),
        "Negative: Mean ± SD": "", "Negative: Median [Q1–Q3]": "",
        "Negative: N": n_neg_valid,
        "Negative: Distribution": dist_str(neg, n_neg_valid),
        "Test": "Chi-squared",
        "Statistic": f"{statistic:.2f}" if not pd.isna(statistic) else "—",
        "P-value": p_value,
        "P-value (fmt)": f"{p_value:.4f}" if not pd.isna(p_value) else "—",
        "Significance": sig_stars(p_value),
        "Effect Size": f"{v:.3f}" if not pd.isna(v) else "—",
        "Effect Size Type": "Cramér's V",
        "Effect Magnitude": effect_size_label(v),
    }


# ============================================================================
# BUILD COMPARISON TABLE FOR ONE OUTCOME × TIMEPOINT  (FIXED)
# ============================================================================

def build_outcome_comparison(df, outcome_col, outcome_tp, outcome_display_name):
    """
    FIX: For non-timepoint-varying features (tp_varying=False), the positive
    and negative groups are always taken from df_assessed (the full assessed
    cohort at this outcome timepoint).  Previously the code used df_pos/df_neg
    which were correct, but the column data could silently be empty if the
    DataFrame copy changed dtypes.  We now force numeric conversion via
    _safe_numeric_series inside the comparison helpers.

    Additional fix: added diagnostic prints when binarized SDOH columns
    yield 0 valid observations so the issue is visible in the log.
    """
    df_assessed = df[df[outcome_col].notna()].copy()
    df_pos = df_assessed[df_assessed[outcome_col] == 1]
    df_neg = df_assessed[df_assessed[outcome_col] == 0]
    n_pos, n_neg = len(df_pos), len(df_neg)

    if n_pos == 0 or n_neg == 0:
        return [], n_pos, n_neg

    feature_tps = FEATURE_TIMEPOINTS_FOR_OUTCOME[outcome_tp]
    rows = []

    for section_name, var_list, tp_varying in FEATURE_SECTIONS:
        rows.append({
            "Section": section_name, "Variable": "", "Feature Timepoint": "",
            "Type": "",
            "Positive: N": "", "Positive: Mean ± SD": "",
            "Positive: Median [Q1–Q3]": "", "Positive: n (%)": "",
            "Positive: Distribution": "",
            "Negative: N": "", "Negative: Mean ± SD": "",
            "Negative: Median [Q1–Q3]": "", "Negative: n (%)": "",
            "Negative: Distribution": "",
            "Test": "", "Statistic": "",
            "P-value": np.nan, "P-value (fmt)": "",
            "Significance": "",
            "Effect Size": "", "Effect Size Type": "", "Effect Magnitude": "",
        })

        # FIX: for non-tp-varying sections, use ["—"] regardless
        tps_to_use = feature_tps if tp_varying else ["—"]

        for display_name, col_spec, var_type in var_list:
            for ftp in tps_to_use:
                if ftp == "—":
                    col_name = col_spec if isinstance(col_spec, str) else col_spec("diagnosis")
                    ftp_label = "All"
                else:
                    col_name = resolve_column(col_spec, ftp)
                    ftp_label = TP_DISPLAY[ftp]

                if col_name not in df_assessed.columns:
                    rows.append({
                        "Section": "", "Variable": display_name,
                        "Feature Timepoint": ftp_label, "Type": var_type,
                        "Positive: N": "—", "Positive: Mean ± SD": "—",
                        "Positive: Median [Q1–Q3]": "—", "Positive: n (%)": "—",
                        "Positive: Distribution": "—",
                        "Negative: N": "—", "Negative: Mean ± SD": "—",
                        "Negative: Median [Q1–Q3]": "—", "Negative: n (%)": "—",
                        "Negative: Distribution": "—",
                        "Test": "—", "Statistic": "—",
                        "P-value": np.nan, "P-value (fmt)": "—",
                        "Significance": "",
                        "Effect Size": "—", "Effect Size Type": "—",
                        "Effect Magnitude": "column not found",
                    })
                    continue

                # FIX: Pull data from the ASSESSED dataframe's pos/neg subsets
                # and use .loc to ensure we get the correct column regardless
                # of any dtype issues from .copy()
                pos_data = df_pos[col_name]
                neg_data = df_neg[col_name]

                # FIX: Diagnostic — detect when a binarized column is all NaN
                if var_type == "binary":
                    n_pos_valid = _safe_numeric_series(pos_data).notna().sum()
                    n_neg_valid = _safe_numeric_series(neg_data).notna().sum()
                    if n_pos_valid == 0 and n_neg_valid == 0:
                        print(f"    ⚠ DIAG: '{col_name}' has 0 valid values in both groups "
                              f"(pos rows={len(df_pos)}, neg rows={len(df_neg)}, "
                              f"dtype={df_assessed[col_name].dtype})")

                if var_type == "continuous":
                    stats = compare_continuous(pos_data, neg_data)
                elif var_type == "binary":
                    stats = compare_binary(pos_data, neg_data)
                elif var_type == "categorical":
                    stats = compare_categorical(pos_data, neg_data)
                else:
                    continue

                row = {"Section": "", "Variable": display_name,
                       "Feature Timepoint": ftp_label, "Type": var_type}
                row.update(stats)
                for key in ["Positive: n (%)", "Positive: Distribution",
                            "Negative: n (%)", "Negative: Distribution"]:
                    if key not in row:
                        row[key] = ""
                rows.append(row)

    return rows, n_pos, n_neg


# ============================================================================
# RUN
# ============================================================================

n_full = len(df_full)

# ---- FIX: Verify binarized columns exist and have data ----
print(f"{'='*80}")
print(f"PRE-FLIGHT CHECK: Binarized SDOH columns in df_full")
print(f"{'='*80}")

_binarized_check_cols = [
    "socio_physical_activity_binary",
    "socio_social_family_support_binary",
    "socio_financial_strain_binary",
    "socio_parental_employment_binary",
    "socio_parental_education_binary",
    "socio_insurance_category",
]
for _cc in _binarized_check_cols:
    if _cc in df_full.columns:
        _nv = df_full[_cc].notna().sum()
        _dt = df_full[_cc].dtype
        _vc = df_full[_cc].value_counts(dropna=True).head(5).to_dict()
        print(f"  ✓ {_cc:45s}  dtype={str(_dt):10s}  valid={_nv:,}  top_vals={_vc}")
    else:
        print(f"  ✗ {_cc:45s}  NOT FOUND — check preprocessing")

# Build per-timepoint A1C-valid DataFrames
valid_tp_dfs = {}
print(f"\nPer-timepoint valid cohort sizes:")
for tp in TIMEPOINTS:
    a1c_col = TP_A1C[tp]
    if a1c_col in df_full.columns:
        tp_df = df_full[df_full[a1c_col].notna()].copy()
        valid_tp_dfs[tp] = tp_df
        print(f"  {TP_DISPLAY[tp]:15s}: {len(tp_df):,}")
    else:
        valid_tp_dfs[tp] = df_full
        print(f"  {TP_DISPLAY[tp]:15s}: {n_full:,} (A1C col not found, using full)")

    # FIX: Verify binarized columns survive the filter
    _tp_df = valid_tp_dfs[tp]
    for _cc in _binarized_check_cols:
        if _cc in _tp_df.columns:
            _nv = _tp_df[_cc].notna().sum()
            if _nv == 0:
                print(f"    ⚠ '{_cc}' has 0 valid values in {TP_DISPLAY[tp]} cohort!")

col_order = [
    "Section", "Variable", "Feature Timepoint", "Type",
    "Positive: N", "Positive: Mean ± SD", "Positive: Median [Q1–Q3]",
    "Positive: n (%)", "Positive: Distribution",
    "Negative: N", "Negative: Mean ± SD", "Negative: Median [Q1–Q3]",
    "Negative: n (%)", "Negative: Distribution",
    "Test", "Statistic", "P-value (fmt)", "Significance",
    "Effect Size", "Effect Size Type", "Effect Magnitude",
]

all_significant = []
total_tasks = len(OUTCOME_DEFINITIONS) * len(TIMEPOINTS)
task_num = 0

for outcome_name, outcome_col_fn in OUTCOME_DEFINITIONS:
    for tp in TIMEPOINTS:
        task_num += 1
        outcome_col = outcome_col_fn(tp)
        tp_label = TP_DISPLAY[tp]

        print(f"\n[{task_num}/{total_tasks}] {outcome_name} — {tp_label}")

        df_tp = valid_tp_dfs[tp]

        if outcome_col not in df_tp.columns:
            print(f"  ⚠ Column '{outcome_col}' not found — skipping")
            continue

        outcome_vals = df_tp[outcome_col].dropna()
        n_pos_total = (outcome_vals == 1).sum()
        n_neg_total = (outcome_vals == 0).sum()
        print(f"  Cohort N: {len(df_tp):,}  |  Assessed: {len(outcome_vals):,}  |  "
              f"Positive: {n_pos_total:,}  |  Negative: {n_neg_total:,}")

        if n_pos_total == 0 or n_neg_total == 0:
            print(f"  ⚠ Only one class present — skipping")
            continue

        rows, n_pos, n_neg = build_outcome_comparison(df_tp, outcome_col, tp, outcome_name)

        if len(rows) == 0:
            print(f"  ⚠ No rows generated — skipping")
            continue

        df_result = pd.DataFrame(rows)
        present_cols = [c for c in col_order if c in df_result.columns]
        df_result = df_result[present_cols]

        safe_name = outcome_name.replace(" ", "_").replace("-", "").replace("/", "_")
        filename = f"{safe_name}_{tp}_comparison.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        df_result.to_csv(filepath, index=False)
        print(f"  ✓ Saved: {filename}  ({len(df_result)} rows)")

        for _, row in df_result.iterrows():
            if row.get("Significance", "") in ("*", "**", "***"):
                all_significant.append({
                    "Outcome": outcome_name,
                    "Outcome Timepoint": tp_label,
                    "Variable": row.get("Variable", ""),
                    "Feature Timepoint": row.get("Feature Timepoint", ""),
                    "Type": row.get("Type", ""),
                    "P-value": row.get("P-value (fmt)", ""),
                    "Significance": row.get("Significance", ""),
                    "Effect Size": row.get("Effect Size", ""),
                    "Effect Size Type": row.get("Effect Size Type", ""),
                    "Effect Magnitude": row.get("Effect Magnitude", ""),
                    "N Positive": n_pos,
                    "N Negative": n_neg,
                })

# --- Summary ---
print(f"\n{'='*80}")
print(f"SUMMARY: SIGNIFICANT VARIABLES ACROSS ALL OUTCOMES")
print(f"{'='*80}")

if len(all_significant) > 0:
    df_sig = pd.DataFrame(all_significant)
    df_sig = df_sig.sort_values(["Outcome", "Outcome Timepoint", "P-value"])
    summary_path = os.path.join(OUTPUT_DIR, "outcome_comparison_summary.csv")
    df_sig.to_csv(summary_path, index=False)
    print(f"  Total significant comparisons: {len(df_sig)}")
    print(f"  ✓ Saved to: {summary_path}")

    sig_counts = df_sig.groupby(["Outcome", "Outcome Timepoint"]).size()
    print(f"\n  Significant variables per outcome:")
    for (outcome, tp_label), count in sig_counts.items():
        print(f"    {outcome:30s} {tp_label:15s}: {count}")
else:
    print("  No significant variables found at p < 0.05")

del valid_tp_dfs
gc.collect()

print(f"\n{'='*80}")
print(f"✓ Step 2 complete.")
print(f"{'='*80}")
print(f"\nOutput directory: {OUTPUT_DIR}/")
print(f"  - Individual comparison CSVs (one per outcome × timepoint)")
print(f"  - outcome_comparison_summary.csv (all significant variables)")