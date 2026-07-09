
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
# ## Cell 2 — Dimensionality Reduction & Clustering (FIXED)

# %%
"""
Step 4: Unsupervised Clustering & Dimensionality Reduction (FIXED)
===================================================================
FIX APPLIED:
  Added _safe_numeric_series() for robust dtype conversion of binarized SDOH
  columns.  Same root cause as Steps 2/3: object-dtype binary columns from
  _binarize_sdoh silently become all-NaN during pd.to_numeric after .copy().
"""

import os
import warnings
import gc

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "analysis/clustering"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
for d in [OUTPUT_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 1000
PCA_PREREDUCTION_DIM = 50
MIN_SAMPLES = 30

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
# OUTCOMES & MODEL TYPES
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
# OUTCOME-SPECIFIC FEATURE EXCLUSIONS (same as Step 3)
# ============================================================================
OUTCOME_FEATURE_EXCLUSIONS = {
    "Hypertension": [
        "Hypertension", "systolic_blood_pressure", "diastolic_blood_pressure",
        "sbp_percentile", "dbp_percentile",
    ],
    "Dyslipidemia": [
        "Dyslipidemia", "total_cholesterol", "hdl_cholesterol",
        "ldl_cholesterol", "triglycerides",
    ],
    "Microalbuminuria": [
        "Microalbuminuria", "urine_microalbumin",
        "urine_microalbumin_creatinine_ratio",
    ],
    "Optimal_Glycemic_Control": [],
    "Insulin_Independence": [],
    "Metformin_Response": [],
    "GLP1RA_Response": [],
}

OUTCOME_TIMEPOINT_SENSITIVE_EXCLUSIONS = {
    "Optimal_Glycemic_Control": [("a1c_", "outcome_tp_only")],
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

# ============================================================================
# HELPERS
# ============================================================================
def resolve_col(col_spec, tp):
    return col_spec(tp) if callable(col_spec) else col_spec


def apply_feature_exclusions(feature_names, outcome_name, outcome_tp):
    tp_suffix_map = {
        "diagnosis": ["_diagnosis", "_at_diagnosis"],
        "2yr":       ["_2yr", "_at_2_years"],
        "5yr":       ["_5yr", "_at_5_years"],
    }
    outcome_tp_suffixes = tp_suffix_map.get(outcome_tp, [])
    excluded = set()

    for feat in feature_names:
        for pattern in OUTCOME_FEATURE_EXCLUSIONS.get(outcome_name, []):
            if pattern in feat:
                excluded.add(feat)
                break

    for feat in feature_names:
        for pattern, restriction in OUTCOME_TIMEPOINT_SENSITIVE_EXCLUSIONS.get(outcome_name, []):
            if pattern in feat and restriction == "outcome_tp_only":
                if any(suffix in feat for suffix in outcome_tp_suffixes):
                    excluded.add(feat)
                    break

    return [f for f in feature_names if f not in excluded], sorted(excluded)


def gather_feature_columns(df, feature_tps):
    continuous_cols, categorical_cols, binary_cols = [], [], []
    feature_names = []
    for section_name, var_list, tp_varying in FEATURE_SECTIONS:
        tps = feature_tps if tp_varying else ["—"]
        for display_name, col_spec, var_type in var_list:
            for ftp in tps:
                col_name = (col_spec if isinstance(col_spec, str) else col_spec("diagnosis")) if ftp == "—" else resolve_col(col_spec, ftp)
                if col_name not in df.columns or col_name in feature_names:
                    continue
                feature_names.append(col_name)
                if var_type == "continuous":    continuous_cols.append(col_name)
                elif var_type == "binary":      binary_cols.append(col_name)
                elif var_type == "categorical": categorical_cols.append(col_name)
    return feature_names, continuous_cols, categorical_cols, binary_cols


# ============================================================================
# DATA PREPARATION  (FIXED)
# ============================================================================
def prepare_for_dimreduction(df, target_col, feature_tps, outcome_name, outcome_tp):
    """
    Prepare scaled feature matrix X and labels y for dim reduction.
    FIX: Uses _safe_numeric_series for binary/continuous columns.
    """
    df_task = df[df[target_col].notna()].copy()
    y = df_task[target_col].astype(int)

    if len(df_task) < MIN_SAMPLES or len(y.unique()) < 2:
        return None

    all_features, cont_cols, cat_cols, bin_cols = gather_feature_columns(df_task, feature_tps)
    all_features, dropped = apply_feature_exclusions(all_features, outcome_name, outcome_tp)
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
        if before_valid > 0 and after_valid == 0:
            print(f"    ⚠ DIAG: Column '{c}' had {before_valid} non-null values "
                  f"but 0 after numeric conversion (dtype was {df_task[c].dtype})")
        if before_valid != after_valid and after_valid > 0:
            n_fixed += 1

    if n_fixed > 0:
        print(f"    [prepare_for_dimreduction] {n_fixed} columns had dtype adjustments applied")

    # Categoricals: already collapsed to top-5 + Other by preprocessing.
    for c in cat_cols:
        X[c] = X[c].fillna("Missing")

    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dummy_na=False)

    # Remove constant features
    feature_var = X.var(numeric_only=True)
    non_constant = feature_var[feature_var > 0].index.tolist()
    X = X[non_constant]

    if X.shape[1] == 0:
        return None

    # Impute + scale
    X_imputed = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imputed)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "X": X_scaled, "y": y,
        "n_samples": len(X_scaled), "n_features": X_scaled.shape[1],
        "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
        "dropped_cols": dropped,
    }


# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================
def run_pca(X):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    return pca.fit_transform(X), pca.explained_variance_ratio_


def run_tsne(X):
    if X.shape[1] > PCA_PREREDUCTION_DIM:
        X = PCA(n_components=PCA_PREREDUCTION_DIM, random_state=RANDOM_STATE).fit_transform(X)
    perplexity = max(5, min(TSNE_PERPLEXITY, len(X) // 4, len(X) - 1))
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE,
                perplexity=perplexity, max_iter=TSNE_MAX_ITER)
    return tsne.fit_transform(X), perplexity


def silhouette_like_separation(X_2d, y):
    pos_mask, neg_mask = y.values == 1, y.values == 0
    if pos_mask.sum() < 2 or neg_mask.sum() < 2:
        return np.nan
    c_pos = X_2d[pos_mask].mean(axis=0)
    c_neg = X_2d[neg_mask].mean(axis=0)
    between = np.linalg.norm(c_pos - c_neg)
    spread = (np.mean(np.linalg.norm(X_2d[pos_mask] - c_pos, axis=1)) +
              np.mean(np.linalg.norm(X_2d[neg_mask] - c_neg, axis=1))) / 2
    return between / spread if spread > 0 else np.nan


# ============================================================================
# PLOTTING
# ============================================================================
def plot_scatter(X_2d, y, title, xlabel, ylabel, subtitle, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    neg_mask, pos_mask = y.values == 0, y.values == 1

    ax.scatter(X_2d[neg_mask, 0], X_2d[neg_mask, 1],
               c="#4A90D9", label=f"Negative (n={neg_mask.sum()})",
               alpha=0.45, s=18, edgecolors="none", rasterized=True)
    ax.scatter(X_2d[pos_mask, 0], X_2d[pos_mask, 1],
               c="#D94A4A", label=f"Positive (n={pos_mask.sum()})",
               alpha=0.65, s=22, edgecolors="black", linewidth=0.3, rasterized=True)

    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                ha="center", fontsize=10, color="#666666", style="italic")
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    ax.grid(alpha=0.2, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def make_grid_plot(grid_data, method_name, save_path):
    n = len(grid_data)
    if n == 0:
        return
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    fig.suptitle(f"{method_name} Visualization — All Outcomes",
                 fontsize=16, fontweight="bold", y=1.01)

    for idx, (title, X_2d, y, _) in enumerate(grid_data):
        ax = axes[idx]
        neg_mask, pos_mask = y.values == 0, y.values == 1
        ax.scatter(X_2d[neg_mask, 0], X_2d[neg_mask, 1], c="#4A90D9", alpha=0.4,
                   s=10, edgecolors="none", label=f"Neg (n={neg_mask.sum()})", rasterized=True)
        ax.scatter(X_2d[pos_mask, 0], X_2d[pos_mask, 1], c="#D94A4A", alpha=0.6,
                   s=12, edgecolors="black", linewidth=0.2,
                   label=f"Pos (n={pos_mask.sum()})", rasterized=True)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, loc="best", framealpha=0.8)
        ax.grid(alpha=0.15, linestyle="--")
        ax.tick_params(labelsize=7)

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Grid plot: {save_path}")


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

summary_rows = []
grid_data_pca = []
grid_data_tsne = []

total_tasks = len(OUTCOME_DEFINITIONS) * len(MODEL_TYPES)
task_count = 0

for outcome_name, outcome_col_fn in OUTCOME_DEFINITIONS:
    for model_label, outcome_tp, feature_tps in MODEL_TYPES:
        task_count += 1
        outcome_col = outcome_col_fn(outcome_tp)
        task_name = f"{outcome_name}_Model{model_label}"
        feature_tp_str = "+".join(feature_tps)

        print(f"\n[{task_count}/{total_tasks}] {task_name}")

        df_tp = valid_tp_dfs.get(outcome_tp, df_full)

        if outcome_col not in df_tp.columns:
            print(f"  ⚠ Column '{outcome_col}' not found — skipping")
            continue

        prep = prepare_for_dimreduction(df_tp, outcome_col, feature_tps, outcome_name, outcome_tp)
        if prep is None:
            print(f"  ⚠ Insufficient data — skipping")
            continue

        X, y = prep["X"], prep["y"]
        print(f"  Samples: {prep['n_samples']}  |  Features: {prep['n_features']}  "
              f"|  Pos: {prep['n_pos']}  Neg: {prep['n_neg']}")
        if prep["dropped_cols"]:
            print(f"  Excluded: {len(prep['dropped_cols'])} features")

        # ---- PCA ----
        print(f"  PCA...", end="", flush=True)
        explained_var = [np.nan, np.nan]
        sep_pca = np.nan
        try:
            X_pca, explained_var = run_pca(X)
            sep_pca = silhouette_like_separation(X_pca, y)
            title_pca = f"{outcome_name.replace('_', ' ')} — Model {model_label}"
            subtitle_pca = (f"PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%}  |  "
                            f"Separation: {sep_pca:.3f}")
            plot_scatter(X_pca, y, title_pca,
                         f"PC1 ({explained_var[0]:.1%})", f"PC2 ({explained_var[1]:.1%})",
                         subtitle_pca, os.path.join(PLOTS_DIR, f"{task_name}_pca.png"))
            print(f" ✓  Var={explained_var[0]+explained_var[1]:.1%}  Sep={sep_pca:.3f}")
            grid_data_pca.append((title_pca, X_pca, y, explained_var))
        except Exception as e:
            print(f" ✗ {str(e)[:60]}")

        # ---- t-SNE ----
        print(f"  t-SNE...", end="", flush=True)
        sep_tsne = np.nan
        perp_used = np.nan
        try:
            X_tsne, perp_used = run_tsne(X)
            sep_tsne = silhouette_like_separation(X_tsne, y)
            title_tsne = f"{outcome_name.replace('_', ' ')} — Model {model_label}"
            subtitle_tsne = f"Perplexity: {perp_used}  |  Separation: {sep_tsne:.3f}"
            plot_scatter(X_tsne, y, title_tsne, "t-SNE Dim 1", "t-SNE Dim 2",
                         subtitle_tsne, os.path.join(PLOTS_DIR, f"{task_name}_tsne.png"))
            print(f" ✓  Perp={perp_used}  Sep={sep_tsne:.3f}")
            grid_data_tsne.append((title_tsne, X_tsne, y, perp_used))
        except Exception as e:
            print(f" ✗ {str(e)[:60]}")

        summary_rows.append({
            "Outcome": outcome_name, "Model Type": model_label,
            "Outcome Timepoint": TP_DISPLAY[outcome_tp],
            "Feature Timepoints": feature_tp_str,
            "N Samples": prep["n_samples"], "N Features": prep["n_features"],
            "N Positive": prep["n_pos"], "N Negative": prep["n_neg"],
            "N Excluded Features": len(prep["dropped_cols"]),
            "PCA Var Explained (PC1+PC2)": (
                f"{(explained_var[0]+explained_var[1])*100:.1f}%"
                if not np.isnan(explained_var[0]) else "—"),
            "PCA Separation": f"{sep_pca:.3f}" if not np.isnan(sep_pca) else "—",
            "t-SNE Perplexity": perp_used if not np.isnan(perp_used) else "—",
            "t-SNE Separation": f"{sep_tsne:.3f}" if not np.isnan(sep_tsne) else "—",
        })

# ---- Save summary ----
print(f"\n{'='*80}")
print(f"SAVING SUMMARY")
print(f"{'='*80}")

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "dimensionality_reduction_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ Summary: {summary_path}")
    print(f"\n{summary_df.to_string(index=False)}")

# ---- Grid plots ----
if grid_data_pca:
    make_grid_plot(grid_data_pca, "PCA",
                   os.path.join(OUTPUT_DIR, "pca_grid_all_outcomes.png"))

if grid_data_tsne:
    make_grid_plot(grid_data_tsne, "t-SNE",
                   os.path.join(OUTPUT_DIR, "tsne_grid_all_outcomes.png"))

del valid_tp_dfs
gc.collect()

print(f"\n{'='*80}")
print(f"✓ Step 4 complete.")
print(f"{'='*80}")
print(f"\nOutput directory: {OUTPUT_DIR}/")
print(f"  - plots/                              (individual PCA & t-SNE PNGs)")
print(f"  - pca_grid_all_outcomes.png            (all outcomes on one figure)")
print(f"  - tsne_grid_all_outcomes.png           (all outcomes on one figure)")
print(f"  - dimensionality_reduction_summary.csv (separation metrics)")