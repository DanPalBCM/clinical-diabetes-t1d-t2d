
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
# ## Step 5 — COVID-Era Analysis: Pre- vs Post-COVID Characteristic Comparison,
# ## Statistical Testing, ML Prediction & Unsupervised Clustering
#
# **Runs after preprocessing cell (df_full must exist)**
#
# Sections:
#   5A. Cohort definition (pre/post March 15, 2020)
#   5B. Characteristic tables with group comparison (mirroring Step 2 structure)
#   5C. ML prediction: COVID era from baseline features (LR, RF, HGBT + Optuna)
#   5D. Dimensionality reduction & clustering (PCA + t-SNE colored by COVID era)

# %%
"""
Step 5: Pre-COVID vs Post-COVID Cohort Analysis
=================================================
Split by diagnosis date relative to March 15, 2020.
  5A — Define cohorts, descriptive overview
  5B — Characteristic comparison tables (continuous: Mann-Whitney + Cohen's d;
        binary: Chi-squared + Cramér's V; categorical: Chi-squared + Cramér's V)
  5C — ML models predicting COVID era from diagnosis-time features
  5D — PCA & t-SNE visualization colored by COVID era
"""

import os
import warnings
import gc
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import optuna
from optuna.samplers import TPESampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "analysis/covid_era_analysis"
TABLES_DIR  = os.path.join(OUTPUT_DIR, "characteristic_tables")
ML_DIR      = os.path.join(OUTPUT_DIR, "ml_models")
IMP_DIR     = os.path.join(ML_DIR, "feature_importances")
PRED_DIR    = os.path.join(ML_DIR, "predictions")
CLUSTER_DIR = os.path.join(OUTPUT_DIR, "clustering")
PLOTS_DIR   = os.path.join(CLUSTER_DIR, "plots")
for d in [OUTPUT_DIR, TABLES_DIR, ML_DIR, IMP_DIR, PRED_DIR, CLUSTER_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

COVID_CUTOFF = pd.to_datetime("2020-03-15")
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5
N_OPTUNA_TRIALS = 50
MIN_SAMPLES_PER_CLASS = 10
TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 1000
PCA_PREREDUCTION_DIM = 50

# ============================================================================
# TIMEPOINT SUFFIX MAPS (reused from earlier steps)
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
# VARIABLE DEFINITIONS — display-name versions (for characteristic tables)
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

OUTCOMES = [
    ("Hypertension",             lambda tp: f"OUTCOME_Hypertension{TP_MEAS[tp]}",            "binary"),
    ("Dyslipidemia",             lambda tp: f"OUTCOME_Dyslipidemia{TP_MEAS[tp]}",            "binary"),
    ("Microalbuminuria",         lambda tp: f"OUTCOME_Microalbuminuria{TP_MEAS[tp]}",         "binary"),
    ("Optimal Glycemic Control", lambda tp: f"OUTCOME_Optimal_Glycemic_Control{TP_MEAS[tp]}", "binary"),
    ("Insulin Independence",     lambda tp: f"OUTCOME_Insulin_Independence{TP_MEAS[tp]}",    "binary"),
    ("Metformin Response",       lambda tp: f"OUTCOME_Metformin_Response{TP_MEAS[tp]}",      "binary"),
    ("GLP-1RA Response",         lambda tp: f"OUTCOME_GLP1RA_Response{TP_MEAS[tp]}",         "binary"),
]

FEATURE_SECTIONS_DISPLAY = [
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
# VARIABLE DEFINITIONS — short-name versions (for ML & clustering)
# ============================================================================
DEMOGRAPHICS_ML = [
    ("age_at_diagnosis",       "age_at_diagnosis",  "continuous"),
    ("sex",                    "sex",               "categorical"),
    ("patient_race",           "patient_race",      "categorical"),
    ("ethnic_group",           "ethnic_group",       "categorical"),
    ("language",               "language",           "categorical"),
]

GLYCEMIC_ML = [
    ("hba1c",                  lambda tp: TP_A1C[tp],                          "continuous"),
    ("glucose",                lambda tp: meas_col("glucose", tp),             "continuous"),
]

ANTHROPOMETRICS_ML = [
    ("bmi",                    lambda tp: meas_col("bmi", tp),                 "continuous"),
    ("bmi_zscore",             lambda tp: meas_col("bmi_zscore", tp),          "continuous"),
    ("bmi_percentile",         lambda tp: meas_col("bmi_percentile", tp),      "continuous"),
    ("height_zscore",          lambda tp: meas_col("height_zscore", tp),       "continuous"),
    ("height_percentile",      lambda tp: meas_col("height_percentile", tp),   "continuous"),
    ("weight_zscore",          lambda tp: meas_col("weight_zscore", tp),       "continuous"),
    ("weight_percentile",      lambda tp: meas_col("weight_percentile", tp),   "continuous"),
]

LIPIDS_ML = [
    ("total_cholesterol",      lambda tp: meas_col("total_cholesterol", tp),   "continuous"),
    ("hdl_cholesterol",        lambda tp: meas_col("hdl_cholesterol", tp),     "continuous"),
    ("ldl_cholesterol",        lambda tp: meas_col("ldl_cholesterol", tp),     "continuous"),
    ("triglycerides",          lambda tp: meas_col("triglycerides", tp),       "continuous"),
]

VITALS_ML = [
    ("systolic_bp",            lambda tp: meas_col("systolic_blood_pressure", tp),   "continuous"),
    ("diastolic_bp",           lambda tp: meas_col("diastolic_blood_pressure", tp),  "continuous"),
]

RENAL_ML = [
    ("serum_creatinine",       lambda tp: meas_col("serum_creatinine", tp),                    "continuous"),
    ("bun",                    lambda tp: meas_col("bun", tp),                                 "continuous"),
    ("egfr",                   lambda tp: meas_col("egfr", tp),                                "continuous"),
    ("urine_microalbumin",     lambda tp: meas_col("urine_microalbumin", tp),                  "continuous"),
    ("uacr",                   lambda tp: meas_col("urine_microalbumin_creatinine_ratio", tp), "continuous"),
]

LIVER_ML = [
    ("alt",                    lambda tp: meas_col("alt", tp),                 "continuous"),
    ("ast",                    lambda tp: meas_col("ast", tp),                 "continuous"),
]

OTHER_LABS_ML = [
    ("c_peptide",              lambda tp: meas_col("serum_c_peptide", tp),     "continuous"),
    ("blood_ph",               lambda tp: meas_col("blood_ph", tp),            "continuous"),
    ("bicarbonate",            lambda tp: meas_col("bicarbonate", tp),         "continuous"),
    ("pco2",                   lambda tp: meas_col("pco2", tp),                "continuous"),
]

MEDICATIONS_ML = [
    ("Insulins",               lambda tp: med_col("Insulins", tp),             "binary"),
    ("Biguanide",              lambda tp: med_col("Biguanide", tp),            "binary"),
    ("GLP1_agonists",          lambda tp: med_col("GLP1_agonists", tp),        "binary"),
    ("DPP4_inhibitors",        lambda tp: med_col("DPP4_inhibitors", tp),      "binary"),
    ("SGLT2_inhibitors",       lambda tp: med_col("SGLT2_inhibitors", tp),     "binary"),
    ("Sulfonylureas",          lambda tp: med_col("Sulfonylureas", tp),        "binary"),
    ("Meglitinides",           lambda tp: med_col("Meglitinides", tp),         "binary"),
    ("Thiazolidinediones",     lambda tp: med_col("Thiazolidinediones", tp),   "binary"),
    ("Alpha_gluc_inh",         lambda tp: med_col("Alpha_glucosidase_inhibitors", tp), "binary"),
    ("Amylin_analogue",        lambda tp: med_col("Amylin_analogue", tp),      "binary"),
]

CONDITIONS_ML = [
    ("DKA",                    lambda tp: cond_col("DKA", tp),                 "binary"),
    ("Ketosis",                lambda tp: cond_col("Ketosis", tp),             "binary"),
    ("Diabetic_Retinopathy",   lambda tp: cond_col("Diabetic_Retinopathy", tp),"binary"),
    ("Neuropathy",             lambda tp: cond_col("Neuropathy", tp),          "binary"),
    ("Hypoglycemia",           lambda tp: cond_col("Hypoglycemia", tp),        "binary"),
]

SOCIOECONOMIC_ML = [
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

OUTCOMES_ML = [
    ("OUTCOME_Hypertension",             lambda tp: f"OUTCOME_Hypertension{TP_MEAS[tp]}",            "binary"),
    ("OUTCOME_Dyslipidemia",             lambda tp: f"OUTCOME_Dyslipidemia{TP_MEAS[tp]}",            "binary"),
    ("OUTCOME_Microalbuminuria",         lambda tp: f"OUTCOME_Microalbuminuria{TP_MEAS[tp]}",         "binary"),
    ("OUTCOME_Optimal_Glycemic_Control", lambda tp: f"OUTCOME_Optimal_Glycemic_Control{TP_MEAS[tp]}", "binary"),
    ("OUTCOME_Insulin_Independence",     lambda tp: f"OUTCOME_Insulin_Independence{TP_MEAS[tp]}",    "binary"),
    ("OUTCOME_Metformin_Response",       lambda tp: f"OUTCOME_Metformin_Response{TP_MEAS[tp]}",      "binary"),
    ("OUTCOME_GLP1RA_Response",          lambda tp: f"OUTCOME_GLP1RA_Response{TP_MEAS[tp]}",         "binary"),
]

FEATURE_SECTIONS_ML = [
    ("DEMOGRAPHICS",       DEMOGRAPHICS_ML,    False),
    ("GLYCEMIC",           GLYCEMIC_ML,        True),
    ("ANTHROPOMETRICS",    ANTHROPOMETRICS_ML,  True),
    ("LIPIDS",             LIPIDS_ML,          True),
    ("VITALS",             VITALS_ML,          True),
    ("RENAL",              RENAL_ML,           True),
    ("LIVER",              LIVER_ML,           True),
    ("OTHER_LABS",         OTHER_LABS_ML,      True),
    ("MEDICATIONS",        MEDICATIONS_ML,     True),
    ("CONDITIONS",         CONDITIONS_ML,      True),
    ("SOCIOECONOMIC",      SOCIOECONOMIC_ML,   False),
    ("OUTCOMES",           OUTCOMES_ML,        True),
]


# ============================================================================
# HELPER: resolve column spec
# ============================================================================
def resolve_column(col_spec, tp):
    return col_spec(tp) if callable(col_spec) else col_spec


# ############################################################################
#                         5A — COHORT DEFINITION
# ############################################################################

print("=" * 80)
print("STEP 5A: COVID-ERA COHORT DEFINITION")
print("=" * 80)

# Detect date column
DATE_COL = None
for candidate in ["date_of_diagnosis", "diagnosis_date", "dx_date",
                   "date_of_dx", "diagnosis_dt"]:
    if candidate in df_full.columns:
        DATE_COL = candidate
        break

if DATE_COL is None:
    raise ValueError(
        "No diagnosis-date column found. Expected one of: "
        "date_of_diagnosis, diagnosis_date, dx_date, date_of_dx, diagnosis_dt"
    )

df_full[DATE_COL] = pd.to_datetime(df_full[DATE_COL], errors="coerce")
n_no_date = df_full[DATE_COL].isna().sum()
print(f"  Date column: '{DATE_COL}'")
print(f"  Missing dates: {n_no_date:,}")

# Create COVID-era binary label
df_full["covid_era"] = np.where(
    df_full[DATE_COL] >= COVID_CUTOFF, "Post-COVID", "Pre-COVID"
)
df_full["covid_post"] = (df_full["covid_era"] == "Post-COVID").astype(int)

# Exclude rows with no date
df_covid = df_full[df_full[DATE_COL].notna()].copy()

n_pre  = (df_covid["covid_post"] == 0).sum()
n_post = (df_covid["covid_post"] == 1).sum()

print(f"\n  COVID cutoff: {COVID_CUTOFF.date()}")
print(f"  Pre-COVID  : {n_pre:,}")
print(f"  Post-COVID : {n_post:,}")
print(f"  Total      : {len(df_covid):,}")

df_pre  = df_covid[df_covid["covid_post"] == 0]
df_post = df_covid[df_covid["covid_post"] == 1]

# ---- FIX: Verify binarized columns exist and have data ----
print(f"\n  PRE-FLIGHT CHECK: Binarized SDOH columns in df_covid")
_binarized_check_cols = [
    "socio_physical_activity_binary",
    "socio_social_family_support_binary",
    "socio_financial_strain_binary",
    "socio_parental_employment_binary",
    "socio_parental_education_binary",
    "socio_insurance_category",
]
for _cc in _binarized_check_cols:
    if _cc in df_covid.columns:
        _nv = df_covid[_cc].notna().sum()
        _dt = df_covid[_cc].dtype
        _vc = df_covid[_cc].value_counts(dropna=True).head(5).to_dict()
        print(f"    ✓ {_cc:45s}  dtype={str(_dt):10s}  valid={_nv:,}  top_vals={_vc}")
        # Check per group
        _nv_pre  = df_pre[_cc].notna().sum()
        _nv_post = df_post[_cc].notna().sum()
        if _nv_pre == 0:
            print(f"      ⚠ Pre-COVID group has 0 valid values for '{_cc}'!")
        if _nv_post == 0:
            print(f"      ⚠ Post-COVID group has 0 valid values for '{_cc}'!")
    else:
        print(f"    ✗ {_cc:45s}  NOT FOUND — check preprocessing")


# ############################################################################
#                5B — CHARACTERISTIC COMPARISON TABLES
# ############################################################################

print(f"\n{'=' * 80}")
print("STEP 5B: CHARACTERISTIC COMPARISON — PRE-COVID vs POST-COVID")
print("=" * 80)

# ---------- statistical helpers (same as Step 2) ----------

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / sp if sp > 0 else 0.0


def cramers_v(ct):
    try:
        chi2, _, _, _ = chi2_contingency(ct)
    except ValueError:
        return np.nan
    n = ct.values.sum()
    md = min(ct.shape[0], ct.shape[1]) - 1
    if md == 0 or n == 0:
        return np.nan
    return np.sqrt(chi2 / (n * md))


def effect_label(val):
    if pd.isna(val): return ""
    v = abs(val)
    if v < 0.2:   return "negligible"
    elif v < 0.5: return "small"
    elif v < 0.8: return "medium"
    return "large"


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


def compare_continuous(pre_s, post_s):
    pre  = _safe_numeric_series(pre_s).dropna()
    post = _safe_numeric_series(post_s).dropna()
    n_pre, n_post = len(pre), len(post)

    def fmt(s):
        if len(s) == 0:
            return "—", "—"
        m, sd = s.mean(), s.std()
        med, q1, q3 = s.median(), s.quantile(0.25), s.quantile(0.75)
        return f"{m:.2f} ± {sd:.2f}", f"{med:.2f} [{q1:.2f}–{q3:.2f}]"

    pre_ms, pre_mq = fmt(pre)
    post_ms, post_mq = fmt(post)

    p_val, stat = np.nan, np.nan
    if n_pre >= 2 and n_post >= 2:
        try:
            stat, p_val = mannwhitneyu(pre, post, alternative="two-sided")
        except Exception:
            pass

    d = cohens_d(pre, post) if (n_pre >= 2 and n_post >= 2) else np.nan

    return {
        "Pre-COVID: N": n_pre,
        "Pre-COVID: Mean ± SD": pre_ms,
        "Pre-COVID: Median [Q1–Q3]": pre_mq,
        "Post-COVID: N": n_post,
        "Post-COVID: Mean ± SD": post_ms,
        "Post-COVID: Median [Q1–Q3]": post_mq,
        "Test": "Mann-Whitney U",
        "Statistic": f"{stat:.1f}" if not pd.isna(stat) else "—",
        "P-value": p_val,
        "P-value (fmt)": f"{p_val:.4f}" if not pd.isna(p_val) else "—",
        "Significance": sig_stars(p_val),
        "Effect Size": f"{d:.3f}" if not pd.isna(d) else "—",
        "Effect Size Type": "Cohen's d",
        "Effect Magnitude": effect_label(d),
    }


def compare_binary(pre_s, post_s):
    pre  = _safe_numeric_series(pre_s).dropna()
    post = _safe_numeric_series(post_s).dropna()
    n_pre, n_post = len(pre), len(post)
    n_pre_1  = int(pre.sum())  if n_pre  > 0 else 0
    n_post_1 = int(post.sum()) if n_post > 0 else 0
    pct_pre  = n_pre_1  / n_pre  * 100 if n_pre  > 0 else 0
    pct_post = n_post_1 / n_post * 100 if n_post > 0 else 0

    p_val, stat, v = np.nan, np.nan, np.nan
    if n_pre > 0 and n_post > 0:
        ct = pd.DataFrame({
            "Pre-COVID":  [n_pre_1,  n_pre  - n_pre_1],
            "Post-COVID": [n_post_1, n_post - n_post_1],
        }, index=["Yes", "No"])
        if ct.values.sum() > 0 and ct.shape[0] > 1 and ct.shape[1] > 1:
            try:
                stat, p_val, _, _ = chi2_contingency(ct)
                v = cramers_v(ct)
            except Exception:
                pass

    return {
        "Pre-COVID: N": n_pre,
        "Pre-COVID: n (%)": f"{n_pre_1} / {n_pre} ({pct_pre:.1f}%)" if n_pre > 0 else "—",
        "Post-COVID: N": n_post,
        "Post-COVID: n (%)": f"{n_post_1} / {n_post} ({pct_post:.1f}%)" if n_post > 0 else "—",
        "Test": "Chi-squared",
        "Statistic": f"{stat:.2f}" if not pd.isna(stat) else "—",
        "P-value": p_val,
        "P-value (fmt)": f"{p_val:.4f}" if not pd.isna(p_val) else "—",
        "Significance": sig_stars(p_val),
        "Effect Size": f"{v:.3f}" if not pd.isna(v) else "—",
        "Effect Size Type": "Cramér's V",
        "Effect Magnitude": effect_label(v),
    }


def compare_categorical(pre_s, post_s):
    pre, post = pre_s.dropna(), post_s.dropna()
    n_pre, n_post = len(pre), len(post)

    def dist_str(s, n):
        if n == 0: return "—"
        counts = s.value_counts()
        return "; ".join(f"{val}: {cnt} ({cnt/n*100:.1f}%)" for val, cnt in counts.items())

    p_val, stat, v = np.nan, np.nan, np.nan
    if n_pre > 0 and n_post > 0:
        combined = pd.concat([
            pre.to_frame("val").assign(group="pre"),
            post.to_frame("val").assign(group="post"),
        ])
        ct = pd.crosstab(combined["val"], combined["group"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            try:
                stat, p_val, _, _ = chi2_contingency(ct)
                v = cramers_v(ct)
            except Exception:
                pass

    return {
        "Pre-COVID: N": n_pre,
        "Pre-COVID: Distribution": dist_str(pre, n_pre),
        "Post-COVID: N": n_post,
        "Post-COVID: Distribution": dist_str(post, n_post),
        "Test": "Chi-squared",
        "Statistic": f"{stat:.2f}" if not pd.isna(stat) else "—",
        "P-value": p_val,
        "P-value (fmt)": f"{p_val:.4f}" if not pd.isna(p_val) else "—",
        "Significance": sig_stars(p_val),
        "Effect Size": f"{v:.3f}" if not pd.isna(v) else "—",
        "Effect Size Type": "Cramér's V",
        "Effect Magnitude": effect_label(v),
    }


# ---------- Build comparison table across all feature sections ----------

def build_covid_comparison(df_pre, df_post):
    """Compare Pre-COVID vs Post-COVID across all sections & timepoints."""
    rows = []

    for section_name, var_list, tp_varying in FEATURE_SECTIONS_DISPLAY:
        # Section header row
        rows.append({
            "Section": section_name, "Variable": "", "Feature Timepoint": "",
            "Type": "",
            "Pre-COVID: N": "", "Pre-COVID: Mean ± SD": "",
            "Pre-COVID: Median [Q1–Q3]": "", "Pre-COVID: n (%)": "",
            "Pre-COVID: Distribution": "",
            "Post-COVID: N": "", "Post-COVID: Mean ± SD": "",
            "Post-COVID: Median [Q1–Q3]": "", "Post-COVID: n (%)": "",
            "Post-COVID: Distribution": "",
            "Test": "", "Statistic": "",
            "P-value": np.nan, "P-value (fmt)": "",
            "Significance": "",
            "Effect Size": "", "Effect Size Type": "", "Effect Magnitude": "",
        })

        tps_to_use = TIMEPOINTS if tp_varying else ["—"]

        for display_name, col_spec, var_type in var_list:
            for ftp in tps_to_use:
                if ftp == "—":
                    col_name = col_spec if isinstance(col_spec, str) else col_spec("diagnosis")
                    ftp_label = "All"
                else:
                    col_name = resolve_column(col_spec, ftp)
                    ftp_label = TP_DISPLAY[ftp]

                row = {"Section": "", "Variable": display_name,
                       "Feature Timepoint": ftp_label, "Type": var_type}

                if col_name not in df_pre.columns and col_name not in df_post.columns:
                    row.update({
                        "Pre-COVID: N": "—", "Pre-COVID: Mean ± SD": "—",
                        "Pre-COVID: Median [Q1–Q3]": "—", "Pre-COVID: n (%)": "—",
                        "Pre-COVID: Distribution": "—",
                        "Post-COVID: N": "—", "Post-COVID: Mean ± SD": "—",
                        "Post-COVID: Median [Q1–Q3]": "—", "Post-COVID: n (%)": "—",
                        "Post-COVID: Distribution": "—",
                        "Test": "—", "Statistic": "—",
                        "P-value": np.nan, "P-value (fmt)": "—",
                        "Significance": "",
                        "Effect Size": "—", "Effect Size Type": "—",
                        "Effect Magnitude": "column not found",
                    })
                    rows.append(row)
                    continue

                pre_data  = df_pre[col_name]  if col_name in df_pre.columns  else pd.Series(dtype=float)
                post_data = df_post[col_name] if col_name in df_post.columns else pd.Series(dtype=float)

                # FIX: Diagnostic — detect when a binarized column is all NaN
                if var_type == "binary":
                    n_pre_valid  = _safe_numeric_series(pre_data).notna().sum()
                    n_post_valid = _safe_numeric_series(post_data).notna().sum()
                    if n_pre_valid == 0 and n_post_valid == 0:
                        print(f"    ⚠ DIAG: '{col_name}' has 0 valid values in both groups "
                              f"(pre rows={len(df_pre)}, post rows={len(df_post)}, "
                              f"dtype={df_covid[col_name].dtype if col_name in df_covid.columns else 'N/A'})")

                if var_type == "continuous":
                    stats = compare_continuous(pre_data, post_data)
                elif var_type == "binary":
                    stats = compare_binary(pre_data, post_data)
                elif var_type == "categorical":
                    stats = compare_categorical(pre_data, post_data)
                else:
                    rows.append(row)
                    continue

                row.update(stats)
                # Fill missing keys for uniform columns
                for key in ["Pre-COVID: Mean ± SD", "Pre-COVID: Median [Q1–Q3]",
                            "Pre-COVID: n (%)", "Pre-COVID: Distribution",
                            "Post-COVID: Mean ± SD", "Post-COVID: Median [Q1–Q3]",
                            "Post-COVID: n (%)", "Post-COVID: Distribution"]:
                    if key not in row:
                        row[key] = ""
                rows.append(row)

    return rows


rows_covid = build_covid_comparison(df_pre, df_post)
df_covid_table = pd.DataFrame(rows_covid)

col_order_5b = [
    "Section", "Variable", "Feature Timepoint", "Type",
    "Pre-COVID: N", "Pre-COVID: Mean ± SD", "Pre-COVID: Median [Q1–Q3]",
    "Pre-COVID: n (%)", "Pre-COVID: Distribution",
    "Post-COVID: N", "Post-COVID: Mean ± SD", "Post-COVID: Median [Q1–Q3]",
    "Post-COVID: n (%)", "Post-COVID: Distribution",
    "Test", "Statistic", "P-value (fmt)", "Significance",
    "Effect Size", "Effect Size Type", "Effect Magnitude",
]
present_cols = [c for c in col_order_5b if c in df_covid_table.columns]
df_covid_table = df_covid_table[present_cols]

path_5b = os.path.join(TABLES_DIR, "covid_era_characteristic_comparison.csv")
df_covid_table.to_csv(path_5b, index=False)
print(f"  ✓ Characteristic comparison table → {path_5b}  ({len(df_covid_table)} rows)")

# ---- Significant-variable summary ----
all_sig_5b = []
for _, row in df_covid_table.iterrows():
    if row.get("Significance", "") in ("*", "**", "***"):
        all_sig_5b.append({
            "Variable": row.get("Variable", ""),
            "Feature Timepoint": row.get("Feature Timepoint", ""),
            "Type": row.get("Type", ""),
            "P-value": row.get("P-value (fmt)", ""),
            "Significance": row.get("Significance", ""),
            "Effect Size": row.get("Effect Size", ""),
            "Effect Size Type": row.get("Effect Size Type", ""),
            "Effect Magnitude": row.get("Effect Magnitude", ""),
            "Pre-COVID N": row.get("Pre-COVID: N", ""),
            "Post-COVID N": row.get("Post-COVID: N", ""),
        })

if all_sig_5b:
    df_sig_5b = pd.DataFrame(all_sig_5b).sort_values("P-value")
    sig_path = os.path.join(TABLES_DIR, "covid_era_significant_variables.csv")
    df_sig_5b.to_csv(sig_path, index=False)
    print(f"  ✓ Significant variables → {sig_path}  ({len(df_sig_5b)} variables)")
else:
    print("  No significant differences found at p < 0.05")


# ############################################################################
#           5C — ML: PREDICT COVID ERA FROM DIAGNOSIS FEATURES
# ############################################################################

print(f"\n{'=' * 80}")
print("STEP 5C: ML — PREDICTING COVID ERA FROM DIAGNOSIS-TIME FEATURES")
print("=" * 80)

# ---- Feature gathering (diagnosis timepoint only) ----

def gather_ml_features(df, feature_tps):
    continuous, categorical, binary = [], [], []
    names = []
    for _, var_list, tp_varying in FEATURE_SECTIONS_ML:
        tps = feature_tps if tp_varying else ["—"]
        for _, col_spec, var_type in var_list:
            for ftp in tps:
                col_name = (col_spec if isinstance(col_spec, str) else col_spec("diagnosis")) \
                           if ftp == "—" else resolve_column(col_spec, ftp)
                if col_name not in df.columns or col_name in names:
                    continue
                names.append(col_name)
                if var_type == "continuous":    continuous.append(col_name)
                elif var_type == "binary":      binary.append(col_name)
                elif var_type == "categorical": categorical.append(col_name)
    return names, continuous, categorical, binary


def prepare_ml_covid(df, feature_tps):
    y = df["covid_post"].astype(int)
    cc = y.value_counts()
    if len(cc) < 2 or cc.min() < MIN_SAMPLES_PER_CLASS:
        return None

    names, cont, cat, binr = gather_ml_features(df, feature_tps)
    if len(names) == 0:
        return None

    X = df[names].copy()
    for c in cont + binr:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in cat:
        X[c] = X[c].fillna("Missing")
    if cat:
        X = pd.get_dummies(X, columns=cat, drop_first=True, dummy_na=False)

    encoded = list(X.columns)
    dummy = [c for c in encoded if c not in cont and c not in binr]

    return {
        "X": X, "y": y,
        "feature_names": encoded,
        "continuous_cols": cont,
        "binary_cols": binr,
        "dummy_cols": dummy,
        "n_samples": len(X),
        "n_features": len(encoded),
        "prevalence": y.mean(),
    }


# ---- Optuna objectives (same as Step 3) ----

def make_lr_obj(Xtr, ytr, cv):
    def obj(trial):
        C = trial.suggest_float("C", 1e-4, 100.0, log=True)
        pen = trial.suggest_categorical("penalty", ["l1", "l2"])
        m = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(
                C=C, penalty=pen,
                solver="saga" if pen == "l1" else "lbfgs",
                max_iter=2000, random_state=RANDOM_STATE,
                class_weight="balanced")),
        ])
        scores = []
        for ti, vi in cv.split(Xtr, ytr):
            try:
                m.fit(Xtr.iloc[ti], ytr.iloc[ti])
                scores.append(roc_auc_score(ytr.iloc[vi], m.predict_proba(Xtr.iloc[vi])[:, 1]))
            except Exception:
                scores.append(0.5)
        return np.mean(scores)
    return obj


def make_rf_obj(Xtr, ytr, cv):
    def obj(trial):
        p = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth":        trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        }
        m = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(**p, random_state=RANDOM_STATE,
                                           class_weight="balanced", n_jobs=-1)),
        ])
        scores = []
        for ti, vi in cv.split(Xtr, ytr):
            try:
                m.fit(Xtr.iloc[ti], ytr.iloc[ti])
                scores.append(roc_auc_score(ytr.iloc[vi], m.predict_proba(Xtr.iloc[vi])[:, 1]))
            except Exception:
                scores.append(0.5)
        return np.mean(scores)
    return obj


def make_hgbt_obj(Xtr, ytr, cv):
    def obj(trial):
        p = {
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter":         trial.suggest_int("max_iter", 50, 500, step=50),
            "max_depth":        trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "max_leaf_nodes":   trial.suggest_int("max_leaf_nodes", 15, 127),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
        }
        m = HistGradientBoostingClassifier(**p, random_state=RANDOM_STATE,
                                            class_weight="balanced")
        scores = []
        for ti, vi in cv.split(Xtr, ytr):
            try:
                m.fit(Xtr.iloc[ti], ytr.iloc[ti])
                scores.append(roc_auc_score(ytr.iloc[vi], m.predict_proba(Xtr.iloc[vi])[:, 1]))
            except Exception:
                scores.append(0.5)
        return np.mean(scores)
    return obj


def train_evaluate(algo, Xtr, Xte, ytr, yte, cv):
    if algo == "LogisticRegression":
        objective = make_lr_obj(Xtr, ytr, cv)
    elif algo == "RandomForest":
        objective = make_rf_obj(Xtr, ytr, cv)
    elif algo == "HistGradientBoosting":
        objective = make_hgbt_obj(Xtr, ytr, cv)
    else:
        raise ValueError(algo)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    bp = study.best_params
    cv_auc = study.best_value

    # Refit best
    if algo == "LogisticRegression":
        model = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(
                C=bp["C"], penalty=bp["penalty"],
                solver="saga" if bp["penalty"] == "l1" else "lbfgs",
                max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced")),
        ])
    elif algo == "RandomForest":
        model = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=bp["n_estimators"], max_depth=bp["max_depth"],
                min_samples_split=bp["min_samples_split"],
                min_samples_leaf=bp["min_samples_leaf"],
                max_features=bp["max_features"],
                random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)),
        ])
    elif algo == "HistGradientBoosting":
        model = HistGradientBoostingClassifier(
            learning_rate=bp["learning_rate"], max_iter=bp["max_iter"],
            max_depth=bp["max_depth"], min_samples_leaf=bp["min_samples_leaf"],
            max_leaf_nodes=bp["max_leaf_nodes"],
            l2_regularization=bp["l2_regularization"],
            random_state=RANDOM_STATE, class_weight="balanced")

    model.fit(Xtr, ytr)
    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    try:    auc  = roc_auc_score(yte, y_proba)
    except: auc  = np.nan
    try:    prauc = average_precision_score(yte, y_proba)
    except: prauc = np.nan

    metrics = {
        "CV ROC-AUC (mean)": cv_auc,
        "Test ROC-AUC":      auc,
        "Test PR-AUC":       prauc,
        "Test Accuracy":     accuracy_score(yte, y_pred),
        "Test F1":           f1_score(yte, y_pred, zero_division=0),
        "Test Precision":    precision_score(yte, y_pred, zero_division=0),
        "Test Recall":       recall_score(yte, y_pred, zero_division=0),
    }

    imps, imp_feats = None, None
    try:
        if algo == "LogisticRegression":
            imps = np.abs(model.named_steps["clf"].coef_[0])
            imp_feats = list(Xtr.columns)
        elif algo == "RandomForest":
            imps = model.named_steps["clf"].feature_importances_
            imp_feats = list(Xtr.columns)
        elif algo == "HistGradientBoosting":
            imps = model.feature_importances_
            imp_feats = list(Xtr.columns)
    except Exception:
        pass

    return {
        "metrics": metrics, "best_params": bp, "model": model,
        "y_pred": y_pred, "y_proba": y_proba,
        "importances": imps, "importance_features": imp_feats,
    }


# ---- Run ML for multiple feature-timepoint configurations ----

ML_CONFIGS = [
    ("Diagnosis_only",      ["diagnosis"]),
    ("Diagnosis_plus_2yr",  ["diagnosis", "2yr"]),
]

ALGO_NAMES = ["LogisticRegression", "RandomForest", "HistGradientBoosting"]
cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

all_ml_results = []

for config_label, feature_tps in ML_CONFIGS:
    print(f"\n  --- Config: {config_label} (features from {'+'.join(feature_tps)}) ---")

    prep = prepare_ml_covid(df_covid, feature_tps)
    if prep is None:
        print("  ⚠ Insufficient data — skipping")
        continue

    X, y = prep["X"], prep["y"]
    print(f"  Samples: {prep['n_samples']:,}  |  Features: {prep['n_features']}")
    print(f"  Prevalence (Post-COVID): {prep['prevalence']:.3f}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(Xtr):,}  |  Test: {len(Xte):,}")

    algo_results = {}
    for algo in ALGO_NAMES:
        print(f"    {algo}...", end="", flush=True)
        t0 = time.time()
        try:
            res = train_evaluate(algo, Xtr, Xte, ytr, yte, cv)
            elapsed = time.time() - t0
            m = res["metrics"]
            print(f" ✓ ({elapsed:.0f}s)  CV-AUC={m['CV ROC-AUC (mean)']:.3f}  "
                  f"Test-AUC={m['Test ROC-AUC']:.3f}  F1={m['Test F1']:.3f}")
            algo_results[algo] = res
        except Exception as e:
            print(f" ✗ {str(e)[:80]}")

    if not algo_results:
        continue

    best_algo = max(algo_results, key=lambda a: algo_results[a]["metrics"].get("Test ROC-AUC", 0) or 0)
    best_res  = algo_results[best_algo]
    print(f"  ★ Best: {best_algo}  (Test ROC-AUC = {best_res['metrics']['Test ROC-AUC']:.4f})")

    for algo, res in algo_results.items():
        row = {
            "Config": config_label,
            "Feature Timepoints": "+".join(feature_tps),
            "Algorithm": algo,
            "Is Best": algo == best_algo,
            "N Samples": prep["n_samples"],
            "N Train": len(Xtr), "N Test": len(Xte),
            "N Features": prep["n_features"],
            "Prevalence": prep["prevalence"],
            "Best Params": json.dumps(res["best_params"]),
        }
        row.update(res["metrics"])
        all_ml_results.append(row)

    # Save feature importances for best
    if best_res["importances"] is not None and best_res["importance_features"] is not None:
        imp_f, imp_v = best_res["importance_features"], best_res["importances"]
        if len(imp_f) == len(imp_v):
            imp_df = pd.DataFrame({"Feature": imp_f, "Importance": imp_v}) \
                       .sort_values("Importance", ascending=False)
            imp_df.to_csv(os.path.join(IMP_DIR, f"covid_era_{config_label}_importances.csv"),
                          index=False)
            print(f"\n  Top 10 features ({best_algo}):")
            for rank, (_, r) in enumerate(imp_df.head(10).iterrows(), 1):
                print(f"    {rank:2d}. {r['Feature']:50s}  {r['Importance']:.4f}")

    # Save predictions
    pred_df = pd.DataFrame({
        "y_true": yte.values, "y_pred": best_res["y_pred"],
        "y_proba": best_res["y_proba"],
    })
    pred_df.to_csv(os.path.join(PRED_DIR, f"covid_era_{config_label}_predictions.csv"), index=False)

# Save master results
if all_ml_results:
    ml_df = pd.DataFrame(all_ml_results).sort_values(["Config", "Is Best"], ascending=[True, False])
    ml_path = os.path.join(ML_DIR, "covid_era_ml_results.csv")
    ml_df.to_csv(ml_path, index=False)
    print(f"\n  ✓ All ML results → {ml_path}  ({len(ml_df)} rows)")

    best_ml = ml_df[ml_df["Is Best"] == True]
    best_ml_path = os.path.join(ML_DIR, "covid_era_ml_best_models.csv")
    best_ml.to_csv(best_ml_path, index=False)
    print(f"  ✓ Best models → {best_ml_path}")


# ############################################################################
#           5D — DIMENSIONALITY REDUCTION & CLUSTERING
# ############################################################################

print(f"\n{'=' * 80}")
print("STEP 5D: DIMENSIONALITY REDUCTION & CLUSTERING (PCA + t-SNE)")
print("=" * 80)


def prepare_for_dimred(df, feature_tps):
    y = df["covid_post"].astype(int)
    if len(df) < 30 or len(y.unique()) < 2:
        return None

    names, cont, cat, binr = gather_ml_features(df, feature_tps)
    if len(names) == 0:
        return None

    X = df[names].copy()
    for c in cont + binr:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in cat:
        X[c] = X[c].fillna("Missing")
    if cat:
        X = pd.get_dummies(X, columns=cat, drop_first=True, dummy_na=False)

    # Drop constant
    fvar = X.var(numeric_only=True)
    X = X[fvar[fvar > 0].index.tolist()]
    if X.shape[1] == 0:
        return None

    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_sc  = StandardScaler().fit_transform(X_imp)
    X_sc  = np.nan_to_num(X_sc, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "X": X_sc, "y": y,
        "n_samples": len(X_sc), "n_features": X_sc.shape[1],
        "n_pre": int((y == 0).sum()), "n_post": int((y == 1).sum()),
    }


def run_pca_2d(X):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    return pca.fit_transform(X), pca.explained_variance_ratio_


def run_tsne_2d(X):
    if X.shape[1] > PCA_PREREDUCTION_DIM:
        X = PCA(n_components=PCA_PREREDUCTION_DIM, random_state=RANDOM_STATE).fit_transform(X)
    perp = max(5, min(TSNE_PERPLEXITY, len(X) // 4, len(X) - 1))
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE,
                perplexity=perp, max_iter=TSNE_MAX_ITER)
    return tsne.fit_transform(X), perp


def centroid_separation(X2, y):
    m0, m1 = y.values == 0, y.values == 1
    if m0.sum() < 2 or m1.sum() < 2:
        return np.nan
    c0 = X2[m0].mean(axis=0)
    c1 = X2[m1].mean(axis=0)
    between = np.linalg.norm(c0 - c1)
    spread = (np.mean(np.linalg.norm(X2[m0] - c0, axis=1)) +
              np.mean(np.linalg.norm(X2[m1] - c1, axis=1))) / 2
    return between / spread if spread > 0 else np.nan


def plot_covid_scatter(X2, y, title, xlabel, ylabel, subtitle, path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    pre_m, post_m = y.values == 0, y.values == 1

    ax.scatter(X2[pre_m, 0], X2[pre_m, 1],
               c="#4A90D9", label=f"Pre-COVID (n={pre_m.sum()})",
               alpha=0.45, s=18, edgecolors="none", rasterized=True)
    ax.scatter(X2[post_m, 0], X2[post_m, 1],
               c="#D94A4A", label=f"Post-COVID (n={post_m.sum()})",
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
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


cluster_summary_rows = []

for config_label, feature_tps in ML_CONFIGS:
    task_name = f"covid_era_{config_label}"
    print(f"\n  --- {config_label} ---")

    prep = prepare_for_dimred(df_covid, feature_tps)
    if prep is None:
        print("  ⚠ Insufficient data — skipping")
        continue

    X, y = prep["X"], prep["y"]
    print(f"  Samples: {prep['n_samples']}  |  Features: {prep['n_features']}  "
          f"|  Pre: {prep['n_pre']}  Post: {prep['n_post']}")

    # PCA
    print("  PCA...", end="", flush=True)
    ev = [np.nan, np.nan]
    sep_pca = np.nan
    try:
        X_pca, ev = run_pca_2d(X)
        sep_pca = centroid_separation(X_pca, y)
        title = f"COVID Era — {config_label.replace('_', ' ')}"
        subtitle = f"PC1: {ev[0]:.1%}, PC2: {ev[1]:.1%}  |  Separation: {sep_pca:.3f}"
        plot_covid_scatter(
            X_pca, y, title,
            f"PC1 ({ev[0]:.1%})", f"PC2 ({ev[1]:.1%})",
            subtitle, os.path.join(PLOTS_DIR, f"{task_name}_pca.png"))
        print(f" ✓  Var={ev[0]+ev[1]:.1%}  Sep={sep_pca:.3f}")
    except Exception as e:
        print(f" ✗ {str(e)[:60]}")

    # t-SNE
    print("  t-SNE...", end="", flush=True)
    sep_tsne = np.nan
    perp_used = np.nan
    try:
        X_tsne, perp_used = run_tsne_2d(X)
        sep_tsne = centroid_separation(X_tsne, y)
        title = f"COVID Era — {config_label.replace('_', ' ')}"
        subtitle = f"Perplexity: {perp_used}  |  Separation: {sep_tsne:.3f}"
        plot_covid_scatter(
            X_tsne, y, title, "t-SNE Dim 1", "t-SNE Dim 2",
            subtitle, os.path.join(PLOTS_DIR, f"{task_name}_tsne.png"))
        print(f" ✓  Perp={perp_used}  Sep={sep_tsne:.3f}")
    except Exception as e:
        print(f" ✗ {str(e)[:60]}")

    cluster_summary_rows.append({
        "Config": config_label,
        "Feature Timepoints": "+".join(feature_tps),
        "N Samples": prep["n_samples"],
        "N Features": prep["n_features"],
        "N Pre-COVID": prep["n_pre"],
        "N Post-COVID": prep["n_post"],
        "PCA Var Explained (PC1+PC2)": f"{(ev[0]+ev[1])*100:.1f}%" if not np.isnan(ev[0]) else "—",
        "PCA Separation": f"{sep_pca:.3f}" if not np.isnan(sep_pca) else "—",
        "t-SNE Perplexity": perp_used if not np.isnan(perp_used) else "—",
        "t-SNE Separation": f"{sep_tsne:.3f}" if not np.isnan(sep_tsne) else "—",
    })

if cluster_summary_rows:
    cs_df = pd.DataFrame(cluster_summary_rows)
    cs_path = os.path.join(CLUSTER_DIR, "covid_era_clustering_summary.csv")
    cs_df.to_csv(cs_path, index=False)
    print(f"\n  ✓ Clustering summary → {cs_path}")


# ############################################################################
#                          FINAL SUMMARY
# ############################################################################

gc.collect()

print(f"\n{'=' * 80}")
print("✓ STEP 5 COMPLETE — COVID-ERA ANALYSIS")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}/")
print(f"  characteristic_tables/")
print(f"    - covid_era_characteristic_comparison.csv")
print(f"    - covid_era_significant_variables.csv")
print(f"  ml_models/")
print(f"    - covid_era_ml_results.csv")
print(f"    - covid_era_ml_best_models.csv")
print(f"    - feature_importances/  (per-config CSVs)")
print(f"    - predictions/          (per-config test-set predictions)")
print(f"  clustering/")
print(f"    - covid_era_clustering_summary.csv")
print(f"    - plots/                (PCA & t-SNE PNGs per config)")