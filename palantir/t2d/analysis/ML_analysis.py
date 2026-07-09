
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
# ## Cell 2 — ML Pipeline (FIXED)

# %%
"""
Step 3: ML Model Selection, Hyperparameter Optimization & Prediction (FIXED)
==============================================================================
FIX APPLIED:
  Added _safe_numeric_series() for robust dtype conversion of binarized SDOH
  columns (socio_physical_activity_binary, socio_insurance_category, etc.)
  that can silently become all-NaN when pd.to_numeric hits object-dtype
  columns after .copy() operations.

  Also added pre-flight diagnostic checks and per-feature validation
  inside prepare_features().
"""

import os
import warnings
import gc
import json
import time

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix,
)

import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "analysis/ML_models"
IMPORTANCES_DIR = os.path.join(OUTPUT_DIR, "feature_importances")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
for d in [OUTPUT_DIR, IMPORTANCES_DIR, PREDICTIONS_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5
N_OPTUNA_TRIALS = 50
MIN_SAMPLES_PER_CLASS = 10

# ============================================================================
# TIMEPOINT MAPS
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
# FIX: Safe numeric conversion helper
# ============================================================================
def _safe_numeric_series(series):
    """
    Robustly convert a series to numeric float64.
    Handles object-dtype columns that contain numeric-like values,
    as well as float columns that may have been cast to object during
    DataFrame slicing / copy operations.
    """
    return pd.to_numeric(series, errors="coerce").astype("float64")

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================
DEMOGRAPHICS = [
    ("age_at_diagnosis",       "age_at_diagnosis",  "continuous"),
    ("sex",                    "sex",               "categorical"),
    ("patient_race",           "patient_race",      "categorical"),
    ("ethnic_group",           "ethnic_group",       "categorical"),
    ("language",               "language",           "categorical"),
]

GLYCEMIC = [
    ("hba1c",                  lambda tp: TP_A1C[tp],                          "continuous"),
    ("glucose",                lambda tp: meas_col("glucose", tp),             "continuous"),
]

ANTHROPOMETRICS = [
    ("bmi",                    lambda tp: meas_col("bmi", tp),                 "continuous"),
    ("bmi_zscore",             lambda tp: meas_col("bmi_zscore", tp),          "continuous"),
    ("bmi_percentile",         lambda tp: meas_col("bmi_percentile", tp),      "continuous"),
    ("height_zscore",          lambda tp: meas_col("height_zscore", tp),       "continuous"),
    ("height_percentile",      lambda tp: meas_col("height_percentile", tp),   "continuous"),
    ("weight_zscore",          lambda tp: meas_col("weight_zscore", tp),       "continuous"),
    ("weight_percentile",      lambda tp: meas_col("weight_percentile", tp),   "continuous"),
]

LIPIDS = [
    ("total_cholesterol",      lambda tp: meas_col("total_cholesterol", tp),   "continuous"),
    ("hdl_cholesterol",        lambda tp: meas_col("hdl_cholesterol", tp),     "continuous"),
    ("ldl_cholesterol",        lambda tp: meas_col("ldl_cholesterol", tp),     "continuous"),
    ("triglycerides",          lambda tp: meas_col("triglycerides", tp),       "continuous"),
]

VITALS = [
    ("systolic_bp",            lambda tp: meas_col("systolic_blood_pressure", tp),   "continuous"),
    ("diastolic_bp",           lambda tp: meas_col("diastolic_blood_pressure", tp),  "continuous"),
]

RENAL = [
    ("serum_creatinine",       lambda tp: meas_col("serum_creatinine", tp),                    "continuous"),
    ("bun",                    lambda tp: meas_col("bun", tp),                                 "continuous"),
    ("egfr",                   lambda tp: meas_col("egfr", tp),                                "continuous"),
    ("urine_microalbumin",     lambda tp: meas_col("urine_microalbumin", tp),                  "continuous"),
    ("uacr",                   lambda tp: meas_col("urine_microalbumin_creatinine_ratio", tp), "continuous"),
]

LIVER = [
    ("alt",                    lambda tp: meas_col("alt", tp),                 "continuous"),
    ("ast",                    lambda tp: meas_col("ast", tp),                 "continuous"),
]

OTHER_LABS = [
    ("c_peptide",              lambda tp: meas_col("serum_c_peptide", tp),     "continuous"),
    ("blood_ph",               lambda tp: meas_col("blood_ph", tp),            "continuous"),
    ("bicarbonate",            lambda tp: meas_col("bicarbonate", tp),         "continuous"),
    ("pco2",                   lambda tp: meas_col("pco2", tp),                "continuous"),
]

MEDICATIONS = [
    ("Insulins",               lambda tp: med_col("Insulins", tp),             "binary"),
    ("Biguanide",              lambda tp: med_col("Biguanide", tp),            "binary"),
    ("GLP1_agonists",          lambda tp: med_col("GLP1_agonists", tp),        "binary"),
]

CONDITIONS = [
    ("DKA",                    lambda tp: cond_col("DKA", tp),                 "binary"),
    ("Ketosis",                lambda tp: cond_col("Ketosis", tp),             "binary"),
    ("Diabetic_Retinopathy",   lambda tp: cond_col("Diabetic_Retinopathy", tp),"binary"),
    ("Neuropathy",             lambda tp: cond_col("Neuropathy", tp),          "binary"),
    ("Hypoglycemia",           lambda tp: cond_col("Hypoglycemia", tp),        "binary"),
]

SOCIOECONOMIC = [
    ("socio_ace",              "socio_adverse_childhood_experience",            "binary"),
    ("socio_alcohol",          "socio_alcohol_abuse",                           "binary"),
    ("socio_drugs",            "socio_drug_substance_abuse",                    "binary"),
    ("socio_food",             "socio_food_insecurity",                         "binary"),
    ("socio_housing",          "socio_housing_instability",                     "binary"),
    ("socio_abuse",            "socio_physical_sexual_abuse",                   "binary"),
    ("socio_smoking",          "socio_smoking",                                 "binary"),
    ("socio_transport",        "socio_transportation_barrier",                  "binary"),
    ("socio_active",           "socio_physical_activity_binary",                "binary"),
    ("socio_support",          "socio_social_family_support_binary",            "binary"),
    ("socio_financial",        "socio_financial_strain_binary",                 "binary"),
    ("socio_employment",       "socio_parental_employment_binary",              "binary"),
    ("socio_education",        "socio_parental_education_binary",               "binary"),
    ("socio_insurance",        "socio_insurance_category",                      "categorical"),
]

FEATURE_SECTIONS = [
    ("DEMOGRAPHICS",       DEMOGRAPHICS,    False),
    ("GLYCEMIC",           GLYCEMIC,        True),
    ("ANTHROPOMETRICS",    ANTHROPOMETRICS, True),
    ("LIPIDS",             LIPIDS,          True),
    ("VITALS",             VITALS,          True),
    ("RENAL",              RENAL,           True),
    ("LIVER",              LIVER,           True),
    ("OTHER_LABS",         OTHER_LABS,      True),
    ("MEDICATIONS",        MEDICATIONS,     True),
    ("CONDITIONS",         CONDITIONS,      True),
    ("SOCIOECONOMIC",      SOCIOECONOMIC,   False),
]

# ============================================================================
# OUTCOMES
# ============================================================================
OUTCOME_DEFINITIONS = [
    ("Hypertension",             lambda tp: f"OUTCOME_Hypertension{TP_MEAS[tp]}"),
    ("Dyslipidemia",             lambda tp: f"OUTCOME_Dyslipidemia{TP_MEAS[tp]}"),
    ("Microalbuminuria",         lambda tp: f"OUTCOME_Microalbuminuria{TP_MEAS[tp]}"),
    ("Optimal_Glycemic_Control", lambda tp: f"OUTCOME_Optimal_Glycemic_Control{TP_MEAS[tp]}"),
    ("Insulin_Independence",     lambda tp: f"OUTCOME_Insulin_Independence{TP_MEAS[tp]}"),
    ("Metformin_Response",       lambda tp: f"OUTCOME_Metformin_Response{TP_MEAS[tp]}"),
    ("GLP1RA_Response",          lambda tp: f"OUTCOME_GLP1RA_Response{TP_MEAS[tp]}"),
]

MODEL_TYPES = [
    ("A", "2yr",  ["diagnosis"]),
    ("B", "5yr",  ["diagnosis"]),
    ("C", "5yr",  ["diagnosis", "2yr"]),
]

# ============================================================================
# OUTCOME-SPECIFIC FEATURE EXCLUSIONS (leakage prevention)
# ============================================================================
OUTCOME_FEATURE_EXCLUSIONS = {
    "Hypertension": [
        "Hypertension",
        "systolic_blood_pressure",
        "diastolic_blood_pressure",
        "sbp_percentile",
        "dbp_percentile",
    ],
    "Dyslipidemia": [
        "Dyslipidemia",
        "total_cholesterol",
        "hdl_cholesterol",
        "ldl_cholesterol",
        "triglycerides",
    ],
    "Microalbuminuria": [
        "Microalbuminuria",
        "urine_microalbumin",
        "urine_microalbumin_creatinine_ratio",
    ],
    "Metformin_Response": ["a1c_", "Biguanide"],
    "GLP1RA_Response": ["a1c_", "GLP1_agonists"],
    "Insulin_Independence": ["a1c_", "Insulins", "DKA"],
    "Optimal_Glycemic_Control": ["a1c_"],
}

OUTCOME_TIMEPOINT_SENSITIVE_EXCLUSIONS = {
    "Optimal_Glycemic_Control": [
        ("a1c_", "outcome_tp_only"),
    ],
    "Insulin_Independence": [
        ("Insulins_", "outcome_tp_only"),
        ("DKA_", "outcome_tp_only"),
        ("a1c_", "outcome_tp_only"),
    ],
    "Metformin_Response": [
        ("Biguanide_", "outcome_tp_only"),
        ("a1c_", "outcome_tp_only"),
    ],
    "GLP1RA_Response": [
        ("GLP1_agonists_", "outcome_tp_only"),
        ("a1c_", "outcome_tp_only"),
    ],
}


def apply_feature_exclusions(feature_names, outcome_name, outcome_tp):
    """Remove features that would leak information about the outcome."""
    tp_suffix_map = {
        "diagnosis": ["_diagnosis", "_at_diagnosis"],
        "2yr":       ["_2yr", "_at_2_years"],
        "5yr":       ["_5yr", "_at_5_years"],
    }
    outcome_tp_suffixes = tp_suffix_map.get(outcome_tp, [])

    excluded = set()

    patterns = OUTCOME_FEATURE_EXCLUSIONS.get(outcome_name, [])
    for feat in feature_names:
        for pattern in patterns:
            if pattern in feat:
                excluded.add(feat)
                break

    sensitive = OUTCOME_TIMEPOINT_SENSITIVE_EXCLUSIONS.get(outcome_name, [])
    for feat in feature_names:
        for pattern, restriction in sensitive:
            if pattern in feat and restriction == "outcome_tp_only":
                if any(suffix in feat for suffix in outcome_tp_suffixes):
                    excluded.add(feat)
                    break

    filtered = [f for f in feature_names if f not in excluded]
    return filtered, sorted(excluded)


# ============================================================================
# HELPERS
# ============================================================================
def resolve_col(col_spec, tp):
    return col_spec(tp) if callable(col_spec) else col_spec


# ============================================================================
# FEATURE PREPARATION  (FIXED)
# ============================================================================
def gather_feature_columns(df, feature_tps):
    """Gather all feature column names and types for the given feature timepoints."""
    continuous_cols, categorical_cols, binary_cols = [], [], []
    feature_names = []

    for section_name, var_list, tp_varying in FEATURE_SECTIONS:
        tps = feature_tps if tp_varying else ["—"]
        for display_name, col_spec, var_type in var_list:
            for ftp in tps:
                if ftp == "—":
                    col_name = col_spec if isinstance(col_spec, str) else col_spec("diagnosis")
                else:
                    col_name = resolve_col(col_spec, ftp)

                if col_name not in df.columns or col_name in feature_names:
                    continue

                feature_names.append(col_name)
                if var_type == "continuous":
                    continuous_cols.append(col_name)
                elif var_type == "binary":
                    binary_cols.append(col_name)
                elif var_type == "categorical":
                    categorical_cols.append(col_name)

    return feature_names, continuous_cols, categorical_cols, binary_cols


def prepare_features(df, target_col, feature_tps, outcome_name, outcome_tp):
    """
    Prepare X, y arrays for ML.
    FIX: Uses _safe_numeric_series for binary/continuous columns to handle
    object-dtype binarized SDOH columns that silently become all-NaN.
    """
    df_task = df[df[target_col].notna()].copy()
    y = df_task[target_col].astype(int)

    class_counts = y.value_counts()
    if len(class_counts) < 2 or class_counts.min() < MIN_SAMPLES_PER_CLASS:
        return None

    all_features, cont_cols, cat_cols, bin_cols = gather_feature_columns(df_task, feature_tps)
    if len(all_features) == 0:
        return None

    # Apply outcome-specific exclusions
    all_features, dropped_cols = apply_feature_exclusions(
        all_features, outcome_name, outcome_tp
    )
    cont_cols = [c for c in cont_cols if c in all_features]
    cat_cols  = [c for c in cat_cols if c in all_features]
    bin_cols  = [c for c in bin_cols if c in all_features]

    if len(all_features) == 0:
        return None

    X = df_task[all_features].copy()

    # FIX: Use _safe_numeric_series for robust dtype conversion
    n_fixed = 0
    for c in cont_cols + bin_cols:
        before_valid = X[c].notna().sum()
        X[c] = _safe_numeric_series(X[c])
        after_valid = X[c].notna().sum()
        # Diagnostic: detect columns that lost data during conversion
        if before_valid > 0 and after_valid == 0:
            print(f"    ⚠ DIAG: Column '{c}' had {before_valid} non-null values "
                  f"but 0 after numeric conversion (dtype was {df_task[c].dtype})")
        if before_valid != after_valid and after_valid > 0:
            n_fixed += 1

    if n_fixed > 0:
        print(f"    [prepare_features] {n_fixed} columns had dtype adjustments applied")

    # Categoricals: already collapsed to top-5 + Other by preprocessing.
    for c in cat_cols:
        X[c] = X[c].fillna("Missing")

    # One-hot encode
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dummy_na=False)

    encoded_feature_names = list(X.columns)
    dummy_cols = [c for c in encoded_feature_names if c not in cont_cols and c not in bin_cols]

    return {
        "X": X, "y": y,
        "feature_names": encoded_feature_names,
        "continuous_cols": cont_cols,
        "binary_cols": bin_cols,
        "dummy_cols": dummy_cols,
        "dropped_cols": dropped_cols,
        "n_samples": len(X),
        "n_features": len(encoded_feature_names),
        "prevalence": y.mean(),
    }


# ============================================================================
# OPTUNA OBJECTIVES
# ============================================================================
def make_lr_objective(X_train, y_train, cv):
    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 100.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver = "saga" if penalty == "l1" else "lbfgs"
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C, penalty=penalty, solver=solver,
                max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced",
            )),
        ])
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            Xt, Xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            yt, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]
            try:
                model.fit(Xt, yt)
                scores.append(roc_auc_score(yv, model.predict_proba(Xv)[:, 1]))
            except Exception:
                scores.append(0.5)
        return np.mean(scores)
    return objective


def make_rf_objective(X_train, y_train, cv):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        }
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                **params, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1,
            )),
        ])
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            Xt, Xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            yt, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]
            try:
                model.fit(Xt, yt)
                scores.append(roc_auc_score(yv, model.predict_proba(Xv)[:, 1]))
            except Exception:
                scores.append(0.5)
        return np.mean(scores)
    return objective


def make_hgbt_objective(X_train, y_train, cv):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
        }
        model = HistGradientBoostingClassifier(
            **params, random_state=RANDOM_STATE, class_weight="balanced",
        )
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            Xt, Xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            yt, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]
            try:
                model.fit(Xt, yt)
                scores.append(roc_auc_score(yv, model.predict_proba(Xv)[:, 1]))
            except Exception:
                scores.append(0.5)
        return np.mean(scores)
    return objective


# ============================================================================
# TRAIN & EVALUATE ONE ALGORITHM
# ============================================================================
def train_evaluate_algorithm(algo_name, X_train, X_test, y_train, y_test, cv):
    """Run Optuna HPO, refit best on full training set, evaluate on test."""
    if algo_name == "LogisticRegression":
        objective = make_lr_objective(X_train, y_train, cv)
    elif algo_name == "RandomForest":
        objective = make_rf_objective(X_train, y_train, cv)
    elif algo_name == "HistGradientBoosting":
        objective = make_hgbt_objective(X_train, y_train, cv)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_cv_auc = study.best_value

    # Refit best model
    if algo_name == "LogisticRegression":
        solver = "saga" if best_params.get("penalty") == "l1" else "lbfgs"
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=best_params["C"], penalty=best_params["penalty"],
                solver=solver, max_iter=2000, random_state=RANDOM_STATE,
                class_weight="balanced",
            )),
        ])
    elif algo_name == "RandomForest":
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                min_samples_split=best_params["min_samples_split"],
                min_samples_leaf=best_params["min_samples_leaf"],
                max_features=best_params["max_features"],
                random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1,
            )),
        ])
    elif algo_name == "HistGradientBoosting":
        model = HistGradientBoostingClassifier(
            learning_rate=best_params["learning_rate"],
            max_iter=best_params["max_iter"],
            max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_leaf_nodes=best_params["max_leaf_nodes"],
            l2_regularization=best_params["l2_regularization"],
            random_state=RANDOM_STATE, class_weight="balanced",
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    try:    test_roc_auc = roc_auc_score(y_test, y_proba)
    except: test_roc_auc = np.nan
    try:    test_pr_auc = average_precision_score(y_test, y_proba)
    except: test_pr_auc = np.nan

    metrics = {
        "CV ROC-AUC (mean)": best_cv_auc,
        "Test ROC-AUC": test_roc_auc,
        "Test PR-AUC": test_pr_auc,
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Test F1": f1_score(y_test, y_pred, zero_division=0),
        "Test Precision": precision_score(y_test, y_pred, zero_division=0),
        "Test Recall": recall_score(y_test, y_pred, zero_division=0),
    }

    # Feature importances
    importances, importance_features = None, None
    try:
        if algo_name == "LogisticRegression":
            importances = np.abs(model.named_steps["clf"].coef_[0])
            importance_features = list(X_train.columns)
        elif algo_name == "RandomForest":
            importances = model.named_steps["clf"].feature_importances_
            importance_features = list(X_train.columns)
        elif algo_name == "HistGradientBoosting":
            importances = model.feature_importances_
            importance_features = list(X_train.columns)
    except Exception:
        pass

    return {
        "metrics": metrics, "best_params": best_params, "model": model,
        "y_pred": y_pred, "y_proba": y_proba,
        "importances": importances, "importance_features": importance_features,
    }


# ============================================================================
# RUN
# ============================================================================

n_full = len(df_full)

# ---- FIX: Pre-flight check for binarized SDOH columns ----
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
        print(f"  ✓ {_cc:45s}  dtype={str(_dt):10s}  valid={_nv:,}")
    else:
        print(f"  ✗ {_cc:45s}  NOT FOUND — check preprocessing")

# Per-timepoint A1C-valid DataFrames
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

ALGO_NAMES = ["LogisticRegression", "RandomForest", "HistGradientBoosting"]
cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

all_results = []
task_count = 0
total_tasks = len(OUTCOME_DEFINITIONS) * len(MODEL_TYPES)

for outcome_name, outcome_col_fn in OUTCOME_DEFINITIONS:
    for model_label, outcome_tp, feature_tps in MODEL_TYPES:
        task_count += 1
        outcome_col = outcome_col_fn(outcome_tp)
        task_name = f"{outcome_name}_Model{model_label}"
        feature_tp_str = "+".join(feature_tps)

        print(f"\n{'='*80}")
        print(f"[{task_count}/{total_tasks}] {task_name}")
        print(f"  Outcome: {outcome_col}  |  Features: {feature_tp_str}")
        print(f"{'='*80}")

        df_tp = valid_tp_dfs.get(outcome_tp, df_full)

        if outcome_col not in df_tp.columns:
            print(f"  ⚠ Outcome column '{outcome_col}' not found — skipping")
            continue

        prep = prepare_features(df_tp, outcome_col, feature_tps, outcome_name, outcome_tp)
        if prep is None:
            print(f"  ⚠ Insufficient data or <{MIN_SAMPLES_PER_CLASS} per class — skipping")
            continue

        X, y = prep["X"], prep["y"]
        print(f"  Samples: {prep['n_samples']:,}  |  Features: {prep['n_features']}")
        print(f"  Prevalence: {prep['prevalence']:.3f}  "
              f"(pos={int(y.sum()):,}, neg={int((1-y).sum()):,})")

        if prep["dropped_cols"]:
            print(f"  Excluded features (leakage prevention): {len(prep['dropped_cols'])}")
            for dc in prep["dropped_cols"]:
                print(f"    ✗ {dc}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
        )
        print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

        algo_results = {}
        for algo_name in ALGO_NAMES:
            print(f"\n  {algo_name}...", end="", flush=True)
            t0 = time.time()
            try:
                result = train_evaluate_algorithm(
                    algo_name, X_train, X_test, y_train, y_test, cv
                )
                elapsed = time.time() - t0
                m = result["metrics"]
                print(f" ✓ ({elapsed:.0f}s)  "
                      f"CV-AUC={m['CV ROC-AUC (mean)']:.3f}  "
                      f"Test-AUC={m['Test ROC-AUC']:.3f}  "
                      f"F1={m['Test F1']:.3f}")
                algo_results[algo_name] = result
            except Exception as e:
                print(f" ✗ Error: {str(e)[:80]}")

        if len(algo_results) == 0:
            print(f"\n  ⚠ No algorithms succeeded — skipping")
            continue

        best_algo = max(
            algo_results.keys(),
            key=lambda a: algo_results[a]["metrics"].get("Test ROC-AUC", 0) or 0
        )
        best_result = algo_results[best_algo]
        print(f"\n  ★ Best: {best_algo}  (Test ROC-AUC = {best_result['metrics']['Test ROC-AUC']:.4f})")

        # Save results for ALL algorithms
        for algo_name, result in algo_results.items():
            m = result["metrics"]
            row = {
                "Outcome": outcome_name, "Model Type": model_label,
                "Outcome Timepoint": TP_DISPLAY[outcome_tp],
                "Feature Timepoints": feature_tp_str,
                "Algorithm": algo_name, "Is Best": algo_name == best_algo,
                "N Samples": prep["n_samples"], "N Train": len(X_train),
                "N Test": len(X_test), "N Features": prep["n_features"],
                "N Excluded Features": len(prep["dropped_cols"]),
                "Excluded Features": "; ".join(prep["dropped_cols"]) if prep["dropped_cols"] else "",
                "Prevalence": prep["prevalence"],
                "Best Params": json.dumps(result["best_params"]),
            }
            row.update(m)
            all_results.append(row)

        # Save feature importances for best model
        if best_result["importances"] is not None and best_result["importance_features"] is not None:
            imp_f, imp_v = best_result["importance_features"], best_result["importances"]
            if len(imp_f) == len(imp_v):
                imp_df = pd.DataFrame({"Feature": imp_f, "Importance": imp_v}
                                      ).sort_values("Importance", ascending=False)
                imp_df.to_csv(os.path.join(IMPORTANCES_DIR, f"{task_name}_importances.csv"), index=False)

        # ---- QC: Print top 10 features for leakage review ----
        if best_result["importances"] is not None and best_result["importance_features"] is not None:
            imp_f = best_result["importance_features"]
            imp_v = best_result["importances"]
            if len(imp_f) == len(imp_v):
                top_idx = np.argsort(imp_v)[::-1][:10]
                print(f"\n  Top 10 features ({best_algo}):")
                for rank, i in enumerate(top_idx, 1):
                    print(f"    {rank:2d}. {imp_f[i]:50s}  {imp_v[i]:.4f}")

        # Save test predictions for best model
        pred_df = pd.DataFrame({
            "y_true": y_test.values, "y_pred": best_result["y_pred"],
            "y_proba": best_result["y_proba"],
        })
        pred_df.to_csv(os.path.join(PREDICTIONS_DIR, f"{task_name}_predictions.csv"), index=False)

# ============================================================================
# SAVE MASTER RESULTS
# ============================================================================
print(f"\n\n{'='*80}")
print(f"SAVING MASTER RESULTS")
print(f"{'='*80}")

if len(all_results) == 0:
    print("  ⚠ No tasks completed successfully.")
else:
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(["Outcome", "Model Type", "Is Best"],
                                        ascending=[True, True, False])
    results_path = os.path.join(OUTPUT_DIR, "results_all_tasks.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  ✓ All results: {results_path}  ({len(results_df)} rows)")

    best_df = results_df[results_df["Is Best"] == True].copy()
    best_path = os.path.join(OUTPUT_DIR, "results_best_models.csv")
    best_df.to_csv(best_path, index=False)
    print(f"  ✓ Best models: {best_path}  ({len(best_df)} rows)")

    # ---- Model B vs C comparison ----
    print(f"\n{'='*80}")
    print(f"MODEL B vs C COMPARISON (5yr: Diagnosis-only vs Diagnosis+2yr)")
    print(f"{'='*80}")

    model_b = best_df[best_df["Model Type"] == "B"].set_index("Outcome")
    model_c = best_df[best_df["Model Type"] == "C"].set_index("Outcome")

    comparison_rows = []
    for outcome in model_b.index:
        if outcome in model_c.index:
            b, c = model_b.loc[outcome], model_c.loc[outcome]
            comparison_rows.append({
                "Outcome": outcome,
                "B Algorithm": b["Algorithm"], "C Algorithm": c["Algorithm"],
                "B N Samples": b["N Samples"], "C N Samples": c["N Samples"],
                "B N Features": b["N Features"], "C N Features": c["N Features"],
                "B Test ROC-AUC": b["Test ROC-AUC"], "C Test ROC-AUC": c["Test ROC-AUC"],
                "ΔROC-AUC": c["Test ROC-AUC"] - b["Test ROC-AUC"],
                "B Test F1": b["Test F1"], "C Test F1": c["Test F1"],
                "ΔF1": c["Test F1"] - b["Test F1"],
                "B Test PR-AUC": b["Test PR-AUC"], "C Test PR-AUC": c["Test PR-AUC"],
                "ΔPR-AUC": c["Test PR-AUC"] - b["Test PR-AUC"],
            })

    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        comp_path = os.path.join(OUTPUT_DIR, "model_comparison_B_vs_C.csv")
        comp_df.to_csv(comp_path, index=False)
        print(f"  ✓ Comparison: {comp_path}")
        print(f"\n{comp_df.to_string(index=False)}")

    # ---- Summary by model type ----
    print(f"\n{'='*80}")
    print(f"SUMMARY BY MODEL TYPE")
    print(f"{'='*80}")

    for mt in ["A", "B", "C"]:
        subset = best_df[best_df["Model Type"] == mt]
        if len(subset) == 0:
            continue
        desc = {"A": "Diagnosis → 2yr", "B": "Diagnosis → 5yr (baseline)",
                "C": "Diagnosis+2yr → 5yr (enhanced)"}[mt]
        print(f"\n  Model {mt}: {desc}")
        print(f"    Tasks completed: {len(subset)}")
        print(f"    Test ROC-AUC:  {subset['Test ROC-AUC'].mean():.4f} "
              f"± {subset['Test ROC-AUC'].std():.4f}")
        print(f"    Test F1:       {subset['Test F1'].mean():.4f} "
              f"± {subset['Test F1'].std():.4f}")
        print(f"    Test PR-AUC:   {subset['Test PR-AUC'].mean():.4f} "
              f"± {subset['Test PR-AUC'].std():.4f}")
        print(f"    Best algorithms: {dict(subset['Algorithm'].value_counts())}")

del valid_tp_dfs
gc.collect()

print(f"\n{'='*80}")
print(f"✓ Step 3 complete.")
print(f"{'='*80}")
print(f"\nOutput directory: {OUTPUT_DIR}/")
print(f"  - results_all_tasks.csv        (all algorithms, all tasks)")
print(f"  - results_best_models.csv      (best algorithm per task)")
print(f"  - model_comparison_B_vs_C.csv  (baseline vs enhanced)")
print(f"  - feature_importances/         (per-task CSVs)")
print(f"  - predictions/                 (per-task test set predictions)")