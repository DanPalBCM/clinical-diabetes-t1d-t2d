"""
comparison_harness.py
======================
SINGLE SOURCE OF TRUTH for everything that MUST be identical between the
classical / deep-sequence baselines (NB4), the agentic LLM workflows (NB5),
and the new Model D deliberative ensemble (NB6).

Rev 2 changes (reviewer fixes)
-------------------------------
1. Outcome-specific collinear feature exclusions (COLLINEAR_FEATURES_BY_OUTCOME).
   Clinical variables that DEFINE an outcome in our preprocessing are removed
   for that outcome's prediction task — applied identically to ML, LLM, and
   Model D so all models see the same input for fair comparison.
2. Prevalence-adaptive sampling: replaces fixed N_TEST_PER_TASK=100 with a
   function that scales test-set size with the eligible population.
3. Horizons restricted to year_2 only (year_5 exploration deferred).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Visit windows ────────────────────────────────────────────────────────────
VISIT_COLS = [f"v{i}" for i in range(1, 11)]

# ── Prediction horizons ──────────────────────────────────────────────────────
# Rev 2: year_2 only.  Year_5 exploration deferred to a future run.
HORIZONS = {
    "year_2": {"target_visit": "v4", "input_visits": ["v1", "v2", "v3"]},
}

# ── Outcomes ──────────────────────────────────────────────────────────────────
ALL_OUTCOMES = [
    "OUTCOME_Optimal_Glycemic_Control", "OUTCOME_Hypertension",
    "OUTCOME_Microalbuminuria", "OUTCOME_Dyslipidemia",
    "OUTCOME_Insulin_Independence", "OUTCOME_Metformin_Response",
    "OUTCOME_GLP1RA_Response",
]
LLM_OUTCOMES = [
    "OUTCOME_Optimal_Glycemic_Control", "OUTCOME_Hypertension",
    "OUTCOME_Microalbuminuria", "OUTCOME_Dyslipidemia",
]

# ── Feature groups ────────────────────────────────────────────────────────────
LABS = ["HBA1C", "GLUCOSE", "BMI_ZSCORE", "TOTAL_CHOLESTEROL", "HDL_CHOLESTEROL",
        "LDL_CHOLESTEROL", "TRIGLYCERIDES", "ALT", "AST", "BUN", "SERUM_CREATININE",
        "SERUM_C_PEPTIDE", "UACR_RATIO", "BETA_HYDROXYBUTYRATE",
        "SBP_OUTPATIENT", "SBP_INPATIENT", "DBP_OUTPATIENT", "DBP_INPATIENT"]
MEDS = ["Insulins", "Biguanide", "GLP1_agonists"]
CONDITIONS = ["DKA", "Ketosis", "Diabetic_Retinopathy", "Neuropathy"]
DEMOGRAPHICS = ["age_at_diagnosis", "sex", "ethnicity_hispanic", "race_white",
                "race_black", "race_asian", "diabetes_duration"]
SDOH = ["socio_food_insecurity", "socio_housing_instability", "socio_financial_strain_binary",
        "socio_insurance_category", "socio_parental_education_binary",
        "socio_social_family_support_binary", "socio_adverse_childhood_experience",
        "socio_transportation_barrier"]
CGM = ["CGM_MEAN_GLUCOSE", "CGM_SD", "CGM_CV", "CGM_GMI", "CGM_TIR_70_180",
       "CGM_TAR_ABOVE_180", "CGM_TAR_ABOVE_250", "CGM_TAR_181_250",
       "CGM_TBR_BELOW_70", "CGM_TBR_54_69", "CGM_TBR_BELOW_54",
       "CGM_HYPO_EPISODES", "CGM_HYPO_EPISODES_PER_DAY",
       "CGM_SEVERE_HYPO_EPISODES", "CGM_SEVERE_HYPO_EPISODES_PER_DAY",
       "CGM_NUM_READINGS", "CGM_NUM_HIGH_READINGS", "CGM_NUM_LOW_READINGS",
       "CGM_DURATION_DAYS", "CGM_PERCENT_MISSING"]

MODALITY_CONFIGS = {
    "EHR_only": set(LABS + MEDS + CONDITIONS + DEMOGRAPHICS),
    "EHR_SDOH": set(LABS + MEDS + CONDITIONS + DEMOGRAPHICS + SDOH),
    "EHR_SDOH_CGM": None,
}

# ── Outcome-specific collinear feature exclusions (Rev 2) ────────────────────
# These clinical variables DEFINE the outcome in our preprocessing — including
# them would be circular.  Applied identically to ALL model families.
COLLINEAR_FEATURES_BY_OUTCOME = {
    "OUTCOME_Hypertension": {
        "SBP_OUTPATIENT", "SBP_INPATIENT", "DBP_OUTPATIENT", "DBP_INPATIENT",
    },
    "OUTCOME_Dyslipidemia": {
        "TOTAL_CHOLESTEROL", "HDL_CHOLESTEROL", "LDL_CHOLESTEROL", "TRIGLYCERIDES",
    },
    "OUTCOME_Optimal_Glycemic_Control": {
        "HBA1C",
    },
    "OUTCOME_Microalbuminuria": {
        "UACR_RATIO",
    },
    "OUTCOME_Insulin_Independence": {
        "Insulins",
    },
    "OUTCOME_Metformin_Response": {
        "Biguanide", "HBA1C",
    },
    "OUTCOME_GLP1RA_Response": {
        "GLP1_agonists", "HBA1C",
    },
}


def get_outcome_exclusions(outcome):
    """Return the set of feature names to EXCLUDE for the given outcome.
    These are collinear by definition (they define the outcome in preprocessing).
    All model families (ML, LLM, Model D) must call this so they see the same
    feature set for fair comparison."""
    return COLLINEAR_FEATURES_BY_OUTCOME.get(outcome, set())


# ── Prevalence-adaptive sampling (Rev 2) ──────────────────────────────────────
N_TEST_MIN = 50
N_TEST_MAX = 250       # Rev 3: scaled up to 250 (forecast ~18h T2D, under 24h budget)
N_TEST_BASE = 150

RANDOM_STATE = 42


def get_adaptive_test_size(n_total):
    """Prevalence-adaptive test size: scales with sqrt of eligible population.
    Returns more test samples for outcomes with larger patient pools, while
    keeping the size reasonable for smaller ones."""
    adaptive = int(N_TEST_BASE * np.sqrt(max(n_total, 40) / 200))
    adaptive = int(np.clip(adaptive, N_TEST_MIN, N_TEST_MAX))
    return min(adaptive, n_total // 2)


# ── Shared evaluation split ───────────────────────────────────────────────────

def get_labeled_labels(df, outcome, target_visit):
    """Series indexed by mrn of int labels for patients with a non-missing
    outcome at target_visit."""
    t = (df[df["feature"] == outcome][["mrn", target_visit]]
         .dropna(subset=[target_visit]))
    t = t.groupby("mrn")[target_visit].first()
    return t.astype(int).sort_index()


def make_comparison_split(df, outcome, target_visit, seed=RANDOM_STATE):
    """Deterministic, stratified train/test split shared by NB4, NB5, and NB6.

    Uses prevalence-adaptive test sizing: outcomes with larger eligible
    populations get proportionally larger test sets.

    Returns
    -------
    (train_mrns, test_mrns, y) : sorted lists of mrns + the full label Series,
                                 or (None, None, y) if the task is too small.
    """
    y = get_labeled_labels(df, outcome, target_visit)
    n_total = len(y)
    if n_total < 40 or y.nunique() < 2:
        return None, None, y
    test_size = get_adaptive_test_size(n_total)
    try:
        train_mrns, test_mrns = train_test_split(
            y.index.to_numpy(),
            test_size=test_size,
            random_state=seed,
            stratify=y.values,
        )
    except ValueError:
        return None, None, y
    return sorted(train_mrns), sorted(test_mrns), y


"""
NB4 — T1D Temporal Model Training (Rev 2 — reviewer fixes)
==================================================================
Trains LSTM, GRU, 1D-CNN, Transformer + classical baselines.

Rev 2 changes
--------------
1. Outcome-specific collinear feature removal via get_outcome_exclusions().
2. Wall-clock timing per model written to nb4_model_timing_t1d for cost comparison.
3. Horizons now year_2 only (via harness).
4. Prevalence-adaptive sampling (via harness make_comparison_split).
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
from transforms.api import Input, Output, lightweight, transform

from .comparison_harness import (
    HORIZONS, ALL_OUTCOMES,
    LABS, MEDS, CONDITIONS, DEMOGRAPHICS, SDOH, CGM, MODALITY_CONFIGS,
    RANDOM_STATE, make_comparison_split, get_outcome_exclusions,
)

LSTM_DATASET = "ri.foundry.main.dataset.242259a4-7302-407a-b468-cc37bb534411"
OUTPUT_BASE = "/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t1d"

N_FOLDS = 5
N_BOOTSTRAP = 200


# ── PyTorch Models ──

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)


class GRUClassifier(nn.Module):
    def __init__(self, n_features, hidden=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1]).squeeze(-1)


class TemporalCNN(nn.Module):
    def __init__(self, n_features, n_steps, hidden=64, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=min(3, n_steps), padding=1),
            nn.ReLU(), nn.Dropout(dropout), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(self, n_features, n_steps, d_model=64, nhead=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_steps, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        x = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]
        x = self.encoder(x)
        return self.fc(x.mean(dim=1)).squeeze(-1)


def train_torch_model(model, X_train, y_train, X_test, epochs=100, lr=1e-3, batch_size=32):
    device = torch.device("cpu")
    model = model.to(device)
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test).to(device)

    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X_te)).cpu().numpy()


# ── Model registry ──
MODELS_CFG = {
    "Logistic_Regression": "flat",
    "Random_Forest": "flat",
    "XGBoost": "flat",
    "XGBoost_Temporal": "temp",
    "LSTM": "torch",
    "GRU": "torch",
    "Temporal_CNN": "torch",
    "Transformer": "torch",
}
ABLATION_MODELS_CFG = {
    "Logistic_Regression": "flat",
    "Random_Forest": "flat",
    "XGBoost": "flat",
    "XGBoost_Temporal": "temp",
}

ALGORITHM_CLASS = {
    "Logistic_Regression": "classical_ml",
    "Random_Forest": "classical_ml",
    "XGBoost": "classical_ml",
    "XGBoost_Temporal": "temporal_ml",
    "LSTM": "deep_learning",
    "GRU": "deep_learning",
    "Temporal_CNN": "deep_learning",
    "Transformer": "deep_learning",
}


# ── Data Prep ──

def prepare_data(df, outcome, input_visits, target_visit, allowed_features=None,
                 exclude_features=None):
    """Prepare data for model training.

    Args:
        allowed_features: If not None, filter non-outcome features to only those in this set.
        exclude_features: If not None, remove these features (collinear exclusions).
    """
    target_df = df[df["feature"] == outcome][["mrn", target_visit]].dropna(subset=[target_visit])
    target_df = target_df.rename(columns={target_visit: "target"}).set_index("mrn")
    valid_mrns = sorted(set(target_df.index))

    non_outcome = df[~df["feature"].str.startswith("OUTCOME_")]

    if allowed_features is not None:
        non_outcome = non_outcome[non_outcome["feature"].isin(allowed_features)]

    # Rev 2: remove collinear features for this outcome
    if exclude_features:
        non_outcome = non_outcome[~non_outcome["feature"].isin(exclude_features)]

    feat_names = sorted(non_outcome["feature"].unique())
    n_p, n_t, n_f = len(valid_mrns), len(input_visits), len(feat_names)

    tensor_3d = np.full((n_p, n_t, n_f), np.nan)
    for fi, feat in enumerate(feat_names):
        fd = non_outcome[non_outcome["feature"] == feat].set_index("mrn")
        for pi, mrn in enumerate(valid_mrns):
            if mrn in fd.index:
                row = fd.loc[mrn]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                for ti, v in enumerate(input_visits):
                    tensor_3d[pi, ti, fi] = row[v]

    y = target_df.loc[valid_mrns, "target"].astype(int).values
    X_flat = tensor_3d.reshape(n_p, -1)
    flat_names = [f"{f}__{v}" for v in input_visits for f in feat_names]
    return X_flat, tensor_3d, y, flat_names, feat_names, valid_mrns


def add_temporal_features(X_3d):
    n_p, n_t, n_f = X_3d.shape
    feats = []
    t_vals = np.arange(n_t, dtype=float)
    for f in range(n_f):
        s = X_3d[:, :, f]
        feats.append(s[:, -1:])
        feats.append(np.nanmean(s, axis=1, keepdims=True))
        feats.append(np.nanstd(s, axis=1, keepdims=True))
        feats.append(s[:, -1:] - s[:, 0:1])
        slopes = np.full((n_p, 1), np.nan)
        for i in range(n_p):
            mask = ~np.isnan(s[i])
            if mask.sum() >= 2:
                slopes[i, 0] = np.polyfit(t_vals[mask], s[i][mask], 1)[0]
        feats.append(slopes)
    return np.hstack(feats)


def evaluate(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    r = {}
    try:
        r["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        r["roc_auc"] = np.nan
    try:
        r["pr_auc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        r["pr_auc"] = np.nan
    r["f1"] = f1_score(y_true, y_pred, zero_division=0)
    r["brier"] = brier_score_loss(y_true, y_prob)
    return r


def bootstrap_ci(y_true, y_scores, fn, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(fn(y_true[idx], y_scores[idx]))
        except (ValueError, ZeroDivisionError):
            pass
    if not scores:
        return np.nan, np.nan, np.nan
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def train_eval_one(model_name, mtype, X_flat, X_temp, X_3d, y, tr_idx, te_idx,
                   n_t, n_f, flat_names=None, feat_names=None, collect_importance=False):
    y_tr = y[tr_idx]
    importance_rows = []

    if mtype == "flat":
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X_flat[tr_idx])
        Xte = imp.transform(X_flat[te_idx])
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        if model_name == "Logistic_Regression":
            m = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=1.0,
                                   max_iter=2000, solver="saga", random_state=RANDOM_STATE)
        elif model_name == "Random_Forest":
            m = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                       random_state=RANDOM_STATE, n_jobs=-1)
        else:
            m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, eval_metric="logloss")
        m.fit(Xtr, y_tr)
        yp = m.predict_proba(Xte)[:, 1]

        if collect_importance:
            imp_vals = getattr(m, "feature_importances_", None)
            if imp_vals is None and hasattr(m, "coef_"):
                imp_vals = np.abs(m.coef_[0])
            if imp_vals is not None:
                for j, fn in enumerate(flat_names):
                    importance_rows.append({"model": model_name, "feature": fn,
                                            "importance": float(imp_vals[j])})

    elif mtype == "temp":
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X_temp[tr_idx])
        Xte = imp.transform(X_temp[te_idx])
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=RANDOM_STATE, eval_metric="logloss")
        m.fit(Xtr, y_tr)
        yp = m.predict_proba(Xte)[:, 1]

        if collect_importance:
            imp_vals = m.feature_importances_
            temp_names = []
            for fname in feat_names:
                temp_names.extend([f"{fname}__last", f"{fname}__mean", f"{fname}__std",
                                   f"{fname}__delta", f"{fname}__slope"])
            for j, fn in enumerate(temp_names):
                importance_rows.append({"model": model_name, "feature": fn,
                                        "importance": float(imp_vals[j])})

    elif mtype == "torch":
        X3_tr = X_3d[tr_idx].copy()
        X3_te = X_3d[te_idx].copy()
        for f in range(n_f):
            vals = X3_tr[:, :, f].flatten()
            med = np.nanmedian(vals) if np.any(~np.isnan(vals)) else 0.0
            X3_tr[:, :, f] = np.where(np.isnan(X3_tr[:, :, f]), med, X3_tr[:, :, f])
            X3_te[:, :, f] = np.where(np.isnan(X3_te[:, :, f]), med, X3_te[:, :, f])
            mu, sd = X3_tr[:, :, f].mean(), X3_tr[:, :, f].std() + 1e-8
            X3_tr[:, :, f] = (X3_tr[:, :, f] - mu) / sd
            X3_te[:, :, f] = (X3_te[:, :, f] - mu) / sd

        if model_name == "LSTM":
            mo = LSTMClassifier(n_f, hidden=64, n_layers=2)
        elif model_name == "GRU":
            mo = GRUClassifier(n_f, hidden=64, n_layers=2)
        elif model_name == "Temporal_CNN":
            mo = TemporalCNN(n_f, n_t, hidden=64)
        else:
            mo = TransformerClassifier(n_f, n_t, d_model=64, nhead=4, n_layers=2)

        yp = train_torch_model(mo, X3_tr, y_tr, X3_te, epochs=100)

        if collect_importance:
            base_auc = evaluate(y[te_idx], yp).get("roc_auc", np.nan)
            if not np.isnan(base_auc):
                mo.eval()
                rng = np.random.RandomState(RANDOM_STATE)
                for feat_idx, feat_name in enumerate(feat_names):
                    X3_perm = X3_te.copy()
                    perm_order = rng.permutation(X3_perm.shape[0])
                    X3_perm[:, :, feat_idx] = X3_perm[perm_order, :, feat_idx]
                    with torch.no_grad():
                        yp_perm = torch.sigmoid(mo(torch.FloatTensor(X3_perm))).cpu().numpy()
                    perm_auc = evaluate(y[te_idx], yp_perm).get("roc_auc", np.nan)
                    delta = (base_auc - perm_auc) if not np.isnan(perm_auc) else 0.0
                    importance_rows.append({"model": model_name, "feature": feat_name,
                                            "importance": float(delta)})
    else:
        raise ValueError(f"unknown mtype {mtype}")

    return yp, importance_rows


@lightweight(cpu_cores=4, memory_gb=16)
@transform(
    results_out=Output(f"{OUTPUT_BASE}/nb4_all_model_results_t1d"),
    importance_out=Output(f"{OUTPUT_BASE}/nb4_all_feature_importance_t1d"),
    ablation_out=Output(f"{OUTPUT_BASE}/nb4_modality_ablation_t1d"),
    matched_out=Output(f"{OUTPUT_BASE}/nb4_matched_comparison_t1d"),
    timing_out=Output(f"{OUTPUT_BASE}/nb4_model_timing_t1d"),
    lstm_data=Input(LSTM_DATASET),
)
def compute(lstm_data, results_out, importance_out, ablation_out, matched_out, timing_out):
    df = lstm_data.pandas()

    all_results = []
    all_importance = []
    ablation_results = []
    matched_results = []
    timing_rows = []

    # ════════════════════════════════════════════════════════════════════════
    # PART 1 — Reference baselines: 5-fold CV (with collinear exclusions)
    # ════════════════════════════════════════════════════════════════════════
    for hz_name, hz_cfg in HORIZONS.items():
        tv = hz_cfg["target_visit"]
        iv = hz_cfg["input_visits"]

        for outcome in ALL_OUTCOMES:
            print(f"\n{'='*60}\n  [CV] {hz_name} | {outcome}\n{'='*60}")
            exclusions = get_outcome_exclusions(outcome)
            if exclusions:
                print(f"  Excluding collinear features: {exclusions}")
            try:
                X_flat, X_3d, y, flat_names, feat_names, mrns = prepare_data(
                    df, outcome, iv, tv, exclude_features=exclusions)
            except Exception as e:
                print(f"  SKIP: {e}")
                continue

            if len(y) < 30 or np.unique(y).shape[0] < 2:
                print(f"  SKIP: n={len(y)}, n_pos={y.sum()}")
                continue

            n_t, n_f = len(iv), len(feat_names)
            X_temp = add_temporal_features(X_3d)
            print(f"  n={len(y)}, n_pos={y.sum()} ({y.mean():.1%}), 3d={X_3d.shape}")

            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

            for model_name, mtype in MODELS_CFG.items():
                fold_metrics = []
                model_start = time.time()
                for fi, (tr_idx, te_idx) in enumerate(skf.split(X_flat, y)):
                    try:
                        collect = (fi == N_FOLDS - 1)
                        yp, imp_rows = train_eval_one(
                            model_name, mtype, X_flat, X_temp, X_3d, y, tr_idx, te_idx,
                            n_t, n_f, flat_names, feat_names, collect_importance=collect)
                        for r in imp_rows:
                            all_importance.append({"outcome": outcome, "horizon": hz_name, **r})
                        metrics = evaluate(y[te_idx], yp)
                        metrics["fold"] = fi
                        fold_metrics.append(metrics)
                    except Exception as e:
                        print(f"    {model_name} fold {fi}: ERROR {e}")
                model_elapsed = time.time() - model_start

                if fold_metrics:
                    fdf = pd.DataFrame(fold_metrics)
                    agg = {
                        "outcome": outcome, "horizon": hz_name, "model": model_name,
                        "roc_auc_mean": fdf["roc_auc"].mean(), "roc_auc_std": fdf["roc_auc"].std(),
                        "pr_auc_mean": fdf["pr_auc"].mean(), "pr_auc_std": fdf["pr_auc"].std(),
                        "f1_mean": fdf["f1"].mean(), "f1_std": fdf["f1"].std(),
                        "brier_mean": fdf["brier"].mean(), "brier_std": fdf["brier"].std(),
                        "n_samples": len(y), "n_positive": int(y.sum()),
                        "prevalence": float(y.mean()), "n_folds": len(fold_metrics),
                    }
                    all_results.append(agg)
                    print(f"    {model_name:25s} AUC={agg['roc_auc_mean']:.3f}+-{agg['roc_auc_std']:.3f}  "
                          f"PR={agg['pr_auc_mean']:.3f}  F1={agg['f1_mean']:.3f}  ({model_elapsed:.1f}s)")

                    timing_rows.append({
                        "model": model_name, "outcome": outcome, "horizon": hz_name,
                        "phase": "cv_5fold", "wall_clock_seconds": round(model_elapsed, 2),
                        "n_train": len(y) - len(y) // N_FOLDS,
                        "n_test": len(y) // N_FOLDS,
                        "algorithm_class": ALGORITHM_CLASS[model_name],
                        "n_features": n_f,
                    })

    # ════════════════════════════════════════════════════════════════════════
    # PART 2 — Matched held-out comparison (with collinear exclusions)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  MATCHED HELD-OUT COMPARISON\n{'='*60}")
    for hz_name, hz_cfg in HORIZONS.items():
        tv = hz_cfg["target_visit"]
        iv = hz_cfg["input_visits"]

        for outcome in ALL_OUTCOMES:
            exclusions = get_outcome_exclusions(outcome)
            train_mrns, test_mrns, _ = make_comparison_split(df, outcome, tv)
            if test_mrns is None:
                print(f"  [matched] SKIP {hz_name}|{outcome}: too small / single-class")
                continue
            try:
                X_flat, X_3d, y, flat_names, feat_names, mrns = prepare_data(
                    df, outcome, iv, tv, exclude_features=exclusions)
            except Exception as e:
                print(f"  [matched] SKIP {hz_name}|{outcome}: {e}")
                continue

            pos = {m: i for i, m in enumerate(mrns)}
            tr_idx = np.array([pos[m] for m in train_mrns if m in pos])
            te_idx = np.array([pos[m] for m in test_mrns if m in pos])
            if len(te_idx) < 10 or np.unique(y[tr_idx]).shape[0] < 2 \
                    or np.unique(y[te_idx]).shape[0] < 2:
                print(f"  [matched] SKIP {hz_name}|{outcome}: degenerate split")
                continue

            n_t, n_f = len(iv), len(feat_names)
            X_temp = add_temporal_features(X_3d)
            y_te = y[te_idx]
            print(f"  [matched] {hz_name}|{outcome}: n_train={len(tr_idx)} n_test={len(te_idx)} "
                  f"test_pos={int(y_te.sum())}")

            for model_name, mtype in MODELS_CFG.items():
                try:
                    t0 = time.time()
                    yp, _ = train_eval_one(model_name, mtype, X_flat, X_temp, X_3d, y,
                                           tr_idx, te_idx, n_t, n_f, collect_importance=False)
                    t_elapsed = time.time() - t0
                    auc_m, auc_lo, auc_hi = bootstrap_ci(y_te, yp, roc_auc_score)
                    pr_m, pr_lo, pr_hi = bootstrap_ci(y_te, yp, average_precision_score)
                    brier_m, brier_lo, brier_hi = bootstrap_ci(y_te, yp, brier_score_loss)
                    f1 = f1_score(y_te, (yp >= 0.5).astype(int), zero_division=0)
                    matched_results.append({
                        "outcome": outcome, "horizon": hz_name, "model": model_name,
                        "roc_auc_mean": auc_m, "roc_auc_ci_low": auc_lo, "roc_auc_ci_high": auc_hi,
                        "pr_auc_mean": pr_m, "pr_auc_ci_low": pr_lo, "pr_auc_ci_high": pr_hi,
                        "brier_mean": brier_m, "brier_ci_low": brier_lo, "brier_ci_high": brier_hi,
                        "f1": float(f1),
                        "n_test": int(len(te_idx)), "n_test_positive": int(y_te.sum()),
                        "test_prevalence": float(y_te.mean()), "n_train": int(len(tr_idx)),
                    })
                    timing_rows.append({
                        "model": model_name, "outcome": outcome, "horizon": hz_name,
                        "phase": "matched_holdout", "wall_clock_seconds": round(t_elapsed, 2),
                        "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
                        "algorithm_class": ALGORITHM_CLASS[model_name],
                        "n_features": n_f,
                    })
                    print(f"    {model_name:25s} AUC={auc_m:.3f} [{auc_lo:.3f},{auc_hi:.3f}] ({t_elapsed:.1f}s)")
                except Exception as e:
                    print(f"    [matched] {model_name}: ERROR {e}")

    # ════════════════════════════════════════════════════════════════════════
    # PART 3 — Modality ablation (with collinear exclusions)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  MODALITY ABLATION STUDY\n{'='*60}")
    for modality_name, allowed_features in MODALITY_CONFIGS.items():
        print(f"\n── Modality: {modality_name} ──")
        for hz_name, hz_cfg in HORIZONS.items():
            tv = hz_cfg["target_visit"]
            iv = hz_cfg["input_visits"]
            for outcome in ALL_OUTCOMES:
                exclusions = get_outcome_exclusions(outcome)
                try:
                    X_flat, X_3d, y, flat_names, feat_names, mrns = prepare_data(
                        df, outcome, iv, tv, allowed_features=allowed_features,
                        exclude_features=exclusions)
                except Exception:
                    continue
                if len(y) < 30 or np.unique(y).shape[0] < 2:
                    continue

                n_t, n_f = len(iv), len(feat_names)
                X_temp = add_temporal_features(X_3d)
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

                for model_name, mtype in ABLATION_MODELS_CFG.items():
                    fold_metrics = []
                    for fi, (tr_idx, te_idx) in enumerate(skf.split(X_flat, y)):
                        try:
                            yp, _ = train_eval_one(model_name, mtype, X_flat, X_temp, X_3d, y,
                                                   tr_idx, te_idx, n_t, n_f, collect_importance=False)
                            metrics = evaluate(y[te_idx], yp)
                            metrics["fold"] = fi
                            fold_metrics.append(metrics)
                        except Exception as e:
                            print(f"    {modality_name}/{model_name} fold {fi}: ERROR {e}")

                    if fold_metrics:
                        fdf = pd.DataFrame(fold_metrics)
                        ablation_results.append({
                            "modality": modality_name, "outcome": outcome, "horizon": hz_name,
                            "model": model_name,
                            "roc_auc_mean": fdf["roc_auc"].mean(), "roc_auc_std": fdf["roc_auc"].std(),
                            "pr_auc_mean": fdf["pr_auc"].mean(), "pr_auc_std": fdf["pr_auc"].std(),
                            "f1_mean": fdf["f1"].mean(), "f1_std": fdf["f1"].std(),
                            "brier_mean": fdf["brier"].mean(), "brier_std": fdf["brier"].std(),
                            "n_samples": len(y), "n_positive": int(y.sum()),
                            "prevalence": float(y.mean()), "n_folds": len(fold_metrics),
                            "n_features": n_f,
                        })
                        print(f"    {modality_name:12s} {model_name:25s} {outcome:45s} "
                              f"AUC={fdf['roc_auc'].mean():.3f}")

    # ════════════════════════════════════════════════════════════════════════
    # Write outputs
    # ════════════════════════════════════════════════════════════════════════
    results_df = pd.DataFrame(all_results)
    if len(results_df) > 0:
        for c in ["outcome", "horizon", "model"]:
            results_df[c] = results_df[c].astype(str)
        for c in ["roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std",
                  "f1_mean", "f1_std", "brier_mean", "brier_std", "prevalence"]:
            results_df[c] = results_df[c].astype("float64")
        for c in ["n_samples", "n_positive", "n_folds"]:
            results_df[c] = results_df[c].astype("int64")
    results_out.write_table(results_df)

    imp_df = pd.DataFrame(all_importance) if all_importance else pd.DataFrame(
        columns=["outcome", "horizon", "model", "feature", "importance"])
    if len(imp_df) > 0:
        for c in ["outcome", "horizon", "model", "feature"]:
            imp_df[c] = imp_df[c].astype(str)
        imp_df["importance"] = imp_df["importance"].astype("float64")
    importance_out.write_table(imp_df)

    ablation_df = pd.DataFrame(ablation_results) if ablation_results else pd.DataFrame(
        columns=["modality", "outcome", "horizon", "model", "roc_auc_mean", "roc_auc_std",
                 "pr_auc_mean", "pr_auc_std", "f1_mean", "f1_std", "brier_mean", "brier_std",
                 "n_samples", "n_positive", "prevalence", "n_folds", "n_features"])
    if len(ablation_df) > 0:
        for c in ["modality", "outcome", "horizon", "model"]:
            ablation_df[c] = ablation_df[c].astype(str)
        for c in ["roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std",
                  "f1_mean", "f1_std", "brier_mean", "brier_std", "prevalence"]:
            ablation_df[c] = ablation_df[c].astype("float64")
        for c in ["n_samples", "n_positive", "n_folds", "n_features"]:
            ablation_df[c] = ablation_df[c].astype("int64")
    ablation_out.write_table(ablation_df)

    matched_df = pd.DataFrame(matched_results) if matched_results else pd.DataFrame(
        columns=["outcome", "horizon", "model", "roc_auc_mean", "roc_auc_ci_low",
                 "roc_auc_ci_high", "pr_auc_mean", "pr_auc_ci_low", "pr_auc_ci_high",
                 "brier_mean", "brier_ci_low", "brier_ci_high", "f1",
                 "n_test", "n_test_positive", "test_prevalence", "n_train"])
    if len(matched_df) > 0:
        for c in ["outcome", "horizon", "model"]:
            matched_df[c] = matched_df[c].astype(str)
        for c in ["roc_auc_mean", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc_mean",
                  "pr_auc_ci_low", "pr_auc_ci_high", "brier_mean", "brier_ci_low",
                  "brier_ci_high", "f1", "test_prevalence"]:
            matched_df[c] = matched_df[c].astype("float64")
        for c in ["n_test", "n_test_positive", "n_train"]:
            matched_df[c] = matched_df[c].astype("int64")
    matched_out.write_table(matched_df)

    # Rev 2: timing output for cost comparison
    timing_df = pd.DataFrame(timing_rows) if timing_rows else pd.DataFrame(
        columns=["model", "outcome", "horizon", "phase", "wall_clock_seconds",
                 "n_train", "n_test", "algorithm_class", "n_features"])
    if len(timing_df) > 0:
        for c in ["model", "outcome", "horizon", "phase", "algorithm_class"]:
            timing_df[c] = timing_df[c].astype(str)
        timing_df["wall_clock_seconds"] = timing_df["wall_clock_seconds"].astype("float64")
        for c in ["n_train", "n_test", "n_features"]:
            timing_df[c] = timing_df[c].astype("int64")
    timing_out.write_table(timing_df)

    print(f"\n{'='*60}\nNB4 COMPLETE (T1D)\n{'='*60}")
    print(f"CV reference: {len(all_results)} | matched: {len(matched_results)} | "
          f"ablation: {len(ablation_results)} | timing: {len(timing_rows)}")


"""
NB5 — Agentic Architecture for Clinical Outcome Prediction (T1D)
   (Rev 2 — collinear exclusions + year_2 only)
==================================================================
LLM-based clinical prediction (Models A/B/C + 2 ablation models).

Rev 2 changes
--------------
1. Outcome-specific collinear feature exclusions via get_outcome_exclusions().
2. Horizons now year_2 only (via harness).
3. Prevalence-adaptive sampling (via harness make_comparison_split).
"""

import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, matthews_corrcoef
from transforms.api import Input, Output, transform, configure
from palantir_models.transforms import GenericCompletionLanguageModelInput
from palantir_models.models import GenericCompletionLanguageModel
from language_model_service_api.languagemodelservice_api_completion_v3 import GenericCompletionRequest

from .comparison_harness import (
    HORIZONS, LLM_OUTCOMES,
    LABS, MEDS, CONDITIONS, DEMOGRAPHICS, SDOH, CGM,
    RANDOM_STATE, make_comparison_split, get_outcome_exclusions,
)

logger = logging.getLogger(__name__)

LSTM_DATASET = "ri.foundry.main.dataset.242259a4-7302-407a-b468-cc37bb534411"
NB4_RESULTS = "ri.foundry.main.dataset.c59e1e68-14cd-44b5-a5ac-caab1998140f"
OUTPUT_BASE = "/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t1d"
SONNET_RID = "ri.language-model-service..language-model.anthropic-claude-4-6-sonnet"

ALL_OUTCOMES = LLM_OUTCOMES

OUTCOME_DESCRIPTIONS = {
    "OUTCOME_Optimal_Glycemic_Control": "Will this patient maintain optimal glycemic control (HbA1c < 7%)?",
    "OUTCOME_Hypertension": "Will this patient develop hypertension?",
    "OUTCOME_Microalbuminuria": "Will this patient develop microalbuminuria (early kidney disease)?",
    "OUTCOME_Dyslipidemia": "Will this patient have dyslipidemia?",
    "OUTCOME_Insulin_Independence": "Will this patient maintain insulin independence (off insulin with A1c<7%, no DKA)?",
    "OUTCOME_Metformin_Response": "Will this patient respond to metformin monotherapy (A1c<7% on metformin alone)?",
    "OUTCOME_GLP1RA_Response": "Will this patient respond to GLP-1 receptor agonist therapy (A1c<7%)?",
}

N_BOOTSTRAP = 200
MAX_PARALLEL_PATIENTS = 8       # concurrent patient processing (I/O-bound LLM calls)


# ── LLM Wrapper (thread-safe) ──

class LLMWrapper:
    """Thread-safe LLM wrapper with cost tracking."""
    def __init__(self, llm_instance):
        self.llm = llm_instance
        self._lock = threading.Lock()
        self.total_calls = 0
        self.total_latency = 0.0
        self.est_input_tokens = 0
        self.est_output_tokens = 0

    def call(self, system_prompt, user_prompt, max_tokens=600, temperature=0.0):
        prompt = system_prompt + "\n\n" + user_prompt
        request = GenericCompletionRequest(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
        start = time.time()
        try:
            response = self.llm.create_completion(request)
            output = response.completion or ""
            elapsed = time.time() - start
            with self._lock:
                self.total_calls += 1
                self.total_latency += elapsed
                self.est_input_tokens += len(prompt) // 4
                self.est_output_tokens += len(output) // 4
            return output
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            with self._lock:
                self.total_latency += time.time() - start
            return None

    def reset(self):
        with self._lock:
            self.total_calls = 0
            self.total_latency = 0.0
            self.est_input_tokens = 0
            self.est_output_tokens = 0


# ── Prompt Builders ──

def format_features(patient_feats, visits, subset=None, exclude=None):
    """Format patient features for prompt. Respects both subset (include-only)
    and exclude (collinear removal) filters."""
    visit_labels = {f"v{i}": f"{i*6}mo" for i in range(1, 11)}
    lines = []
    for feat, vals in sorted(patient_feats.items()):
        if subset and feat not in subset:
            continue
        if exclude and feat in exclude:
            continue
        parts = []
        for v in visits:
            val = vals.get(v)
            if val is not None and not np.isnan(val):
                parts.append(f"{visit_labels[v]}={val:.2f}")
        if parts:
            lines.append(f"  {feat}: {', '.join(parts)}")
    return "\n".join(lines) if lines else "  No data"


def specialist_prompt(agent_type, patient_feats, visits, outcome_desc, exclude=None):
    roles = {
        "lab": ("Expert pediatric endocrinologist analyzing labs/vitals.", set(LABS)),
        "medication": ("Clinical pharmacologist analyzing diabetes medications.", set(MEDS)),
        "sdoh": ("Social determinants of health expert.", set(DEMOGRAPHICS + SDOH)),
        "severity": ("Disease severity specialist.", set(CONDITIONS)),
        "cgm": ("CGM data analyst specializing in continuous glucose monitoring patterns.", set(CGM)),
        "ehr": ("Expert pediatric endocrinologist analyzing EHR clinical data (labs, medications, conditions).",
                set(LABS + MEDS + CONDITIONS)),
        "sdoh_full": ("Social determinants of health and demographics expert.",
                      set(DEMOGRAPHICS + SDOH)),
    }
    role, feats = roles[agent_type]
    # Apply collinear exclusions within the modality subset
    if exclude:
        feats = feats - exclude
    data = format_features(patient_feats, visits, feats, exclude=exclude)
    sys = f"You are a {role} Provide a brief assessment relevant to: {outcome_desc}"
    usr = (f"Patient data:\n{data}\n\n"
           f"Respond: 1) Key findings 2) Risk: LOW/MODERATE/HIGH 3) Confidence: 0.0-1.0 4) Brief rationale")
    return sys, usr


def synthesis_prompt(sub_assessments, outcome_desc):
    parts = "\n\n".join(f"--- {k} ---\n{v}" for k, v in sub_assessments.items() if v)
    sys = "You are a senior pediatric endocrinologist synthesizing specialist assessments for a final prediction."
    usr = (f"Clinical question: {outcome_desc}\n\nSpecialist assessments:\n{parts}\n\n"
           f'Respond in JSON: {{"prediction": "YES" or "NO", "confidence": 0.0-1.0, "rationale": "brief"}}')
    return sys, usr


def single_prompt(patient_feats, visits, outcome_desc, cot=False, exclude=None):
    data = format_features(patient_feats, visits, exclude=exclude)
    sys = "You are an expert pediatric endocrinologist specializing in Type 1 Diabetes prediction."
    cot_text = ("Think step by step:\n1. Metabolic indicators (labs, vitals)?\n"
                "2. Treatment trajectory (medications)?\n"
                "3. Social risk factors (SDOH, demographics)?\n"
                "4. CGM patterns (glucose variability, time in range)?\n"
                "5. Disease severity (complications)?\n\n") if cot else ""
    usr = (f"Predict: {outcome_desc}\n\nPatient data:\n{data}\n\n{cot_text}"
           f'Respond in JSON: {{"prediction": "YES" or "NO", "confidence": 0.0-1.0, "rationale": "brief"}}')
    return sys, usr


def parse_response(text):
    if not text:
        return None, 0.5
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            d = json.loads(text[start:end])
            pred = 1 if str(d.get("prediction", "")).upper() in ("YES", "1", "TRUE") else 0
            return pred, float(d.get("confidence", 0.5))
    except (json.JSONDecodeError, ValueError):
        pass
    upper = text.upper()
    if '"YES"' in upper:
        return 1, 0.6
    if '"NO"' in upper:
        return 0, 0.6
    return None, 0.5


# ── Agent Pipelines ──

def run_multi_agent(llm, patient_feats, visits, outcome_desc, agents, exclude=None):
    subs = {}
    for a in agents:
        s, u = specialist_prompt(a, patient_feats, visits, outcome_desc, exclude=exclude)
        subs[a] = llm.call(s, u, max_tokens=400)
    s, u = synthesis_prompt(subs, outcome_desc)
    resp = llm.call(s, u, max_tokens=300)
    return parse_response(resp)


def run_vote(llm, patient_feats, visits, outcome_desc, agents, weighted=False, exclude=None):
    preds, confs = [], []
    for a in agents:
        s, u = specialist_prompt(a, patient_feats, visits, outcome_desc, exclude=exclude)
        u += f'\n\nAlso predict directly in JSON: {{"prediction": "YES"/"NO", "confidence": 0.0-1.0}}'
        resp = llm.call(s, u, max_tokens=500)
        p, c = parse_response(resp)
        if p is not None:
            preds.append(p)
            confs.append(c)
    if not preds:
        return None, 0.5
    if weighted:
        tc = sum(confs)
        return (1 if sum(p * c for p, c in zip(preds, confs)) / max(tc, 1e-8) >= 0.5 else 0,
                sum(p * c for p, c in zip(preds, confs)) / max(tc, 1e-8))
    return (1 if np.mean(preds) >= 0.5 else 0, np.mean(preds))


def get_patient_feats(df, mrn, visits):
    pdf = df[df["mrn"] == mrn]
    feats = {}
    for _, row in pdf.iterrows():
        feats[row["feature"]] = {v: row[v] for v in visits}
    return feats


def bootstrap_ci(y_true, y_scores, fn, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(fn(y_true[idx], y_scores[idx]))
        except (ValueError, ZeroDivisionError):
            pass
    if not scores:
        return np.nan, np.nan, np.nan
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


# ── Config Definitions (3 architectures + 2 ablation models) ──

CONFIGS = {
    "model_a": {"family": "single_llm", "llm": True, "cot": False,
                "description": "Single-pass LLM — sees all features, direct prediction"},
    "model_b": {"family": "multi_agent", "llm": True, "spec": "by_modality",
                "topo": "parallel_synthesis", "synth": "llm_synthesis", "know": "domain_specific",
                "agents": ["ehr", "sdoh_full", "cgm"],
                "description": "Multi-agent — EHR + SDOH + CGM specialists → synthesizer"},
    "model_c": {"family": "single_llm_cot", "llm": True, "cot": True,
                "description": "Single-pass Chain-of-Thought — structured step-by-step reasoning"},
    "model_a_no_sdoh": {"family": "single_llm", "llm": True, "cot": False,
                        "ablation_exclude": set(DEMOGRAPHICS + SDOH),
                        "description": "Ablation: Single-pass LLM without SDOH features"},
    "model_a_no_cgm": {"family": "single_llm", "llm": True, "cot": False,
                       "ablation_exclude": set(CGM),
                       "description": "Ablation: Single-pass LLM without CGM features"},
}


@configure(["DYNAMIC_ALLOCATION_ENABLED_8_16"])
@transform(
    results_out=Output(f"{OUTPUT_BASE}/nb5_agentic_results_t1d"),
    ablation_out=Output(f"{OUTPUT_BASE}/nb5_ablation_index_t1d"),
    cost_out=Output(f"{OUTPUT_BASE}/nb5_cost_latency_t1d"),
    interp_scores_out=Output(f"{OUTPUT_BASE}/nb5_interpretability_scores_t1d"),
    interp_agree_out=Output(f"{OUTPUT_BASE}/nb5_interpretability_agreement_t1d"),
    lstm_data=Input(LSTM_DATASET),
    nb4_results=Input(NB4_RESULTS),
    llm_sonnet=GenericCompletionLanguageModelInput(SONNET_RID),
)
def compute(ctx, lstm_data, nb4_results, llm_sonnet, results_out, ablation_out,
            cost_out, interp_scores_out, interp_agree_out):

    df = lstm_data.dataframe().toPandas()
    llm = LLMWrapper(llm_sonnet)

    all_results = []
    cost_data = []

    for hz_name, hz_cfg in HORIZONS.items():
        tv = hz_cfg["target_visit"]
        iv = hz_cfg["input_visits"]

        for outcome in ALL_OUTCOMES:
            logger.info(f"{'='*50}\n  {hz_name} | {outcome}")

            # Rev 2: outcome-specific collinear exclusions
            collinear_excl = get_outcome_exclusions(outcome)
            if collinear_excl:
                logger.info(f"  Excluding collinear features: {collinear_excl}")

            train_mrns, test_mrns, y_all = make_comparison_split(df, outcome, tv)
            if test_mrns is None:
                logger.info(f"  SKIP: too small / single-class")
                continue
            mrns = test_mrns
            y = y_all.loc[mrns].values
            logger.info(f"  n_test={len(y)}, n_pos={int(y.sum())} ({y.mean():.1%})")

            outcome_desc = OUTCOME_DESCRIPTIONS[outcome]

            for cid, cfg in CONFIGS.items():
                llm.reset()

                # Combine collinear exclusions with any ablation exclusions
                combined_exclude = set(collinear_excl)
                if "ablation_exclude" in cfg:
                    combined_exclude = combined_exclude | cfg["ablation_exclude"]

                # ── Parallel patient processing (I/O-bound LLM calls) ──
                def process_patient(mrn):
                    pf = get_patient_feats(df, mrn, iv)
                    if "agents" in cfg:
                        if cfg.get("vote"):
                            pred, conf = run_vote(llm, pf, iv, outcome_desc, cfg["agents"],
                                                  weighted=(cfg["vote"] == "weighted"),
                                                  exclude=combined_exclude)
                        else:
                            pred, conf = run_multi_agent(llm, pf, iv, outcome_desc, cfg["agents"],
                                                        exclude=combined_exclude)
                    else:
                        s, u = single_prompt(pf, iv, outcome_desc,
                                             cot=cfg.get("cot", False),
                                             exclude=combined_exclude)
                        resp = llm.call(s, u)
                        pred, conf = parse_response(resp)
                    return mrn, (pred if pred is not None else 0), conf

                results_map = {}
                with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PATIENTS) as executor:
                    futures = {executor.submit(process_patient, mrn): mrn for mrn in mrns}
                    done_count = 0
                    for future in as_completed(futures):
                        mrn, pred, conf = future.result()
                        results_map[mrn] = (pred, conf)
                        done_count += 1
                        if done_count % 25 == 0:
                            logger.info(f"    {cid}: {done_count}/{len(mrns)} patients done")

                # Preserve original order for metrics
                predictions = [results_map[mrn][0] for mrn in mrns]
                confidences = [results_map[mrn][1] for mrn in mrns]

                y_pred = np.array(predictions)
                y_conf = np.array(confidences)

                auc_m, auc_lo, auc_hi = bootstrap_ci(y, y_conf, roc_auc_score)
                pr_m, pr_lo, pr_hi = bootstrap_ci(y, y_conf, average_precision_score)
                brier_m, brier_lo, brier_hi = bootstrap_ci(y, y_conf, brier_score_loss)
                try:
                    mcc = matthews_corrcoef(y, y_pred)
                except Exception:
                    mcc = None

                all_results.append({
                    "config_id": cid, "config_label": cid.replace("_", " ").title(),
                    "method_family": cfg["family"],
                    "specialization": cfg.get("spec"), "topology": cfg.get("topo"),
                    "synthesis": cfg.get("synth"), "knowledge": cfg.get("know"),
                    "agents_active": str(cfg.get("agents", [])),
                    "is_default": cid == "model_b",
                    "outcome": outcome, "horizon": hz_name,
                    "roc_auc_mean": auc_m, "roc_auc_ci_low": auc_lo, "roc_auc_ci_high": auc_hi,
                    "pr_auc_mean": pr_m, "pr_auc_ci_low": pr_lo, "pr_auc_ci_high": pr_hi,
                    "brier_mean": brier_m, "brier_ci_low": brier_lo, "brier_ci_high": brier_hi,
                    "mcc_mean": mcc, "mcc_ci_low": None, "mcc_ci_high": None,
                    "n_samples": len(y), "n_positive": int(y.sum()),
                    "prevalence": float(y.mean()), "n_folds": 1,
                })

                cost_data.append({
                    "config_id": cid, "config_label": cid.replace("_", " ").title(),
                    "method_family": cfg["family"],
                    "cost_usd_per_patient": round(
                        (llm.est_input_tokens * 3 / 1e6 + llm.est_output_tokens * 15 / 1e6) / max(len(mrns), 1), 4),
                    "latency_sec_per_patient": round(llm.total_latency / max(len(mrns), 1), 2),
                    "total_tokens_per_patient": (llm.est_input_tokens + llm.est_output_tokens) // max(len(mrns), 1),
                    "input_tokens": llm.est_input_tokens,
                    "output_tokens": llm.est_output_tokens,
                    "n_llm_calls": llm.total_calls,
                    "outcome": outcome, "horizon": hz_name,
                })

                logger.info(f"    {cid:25s} AUC={auc_m:.3f} calls={llm.total_calls}")

    spark = ctx.spark_session
    res_df = pd.DataFrame(all_results)
    str_cols = ["config_id", "config_label", "method_family", "specialization", "topology",
                "synthesis", "knowledge", "agents_active", "outcome", "horizon"]
    float_cols = ["roc_auc_mean", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc_mean",
                  "pr_auc_ci_low", "pr_auc_ci_high", "brier_mean", "brier_ci_low", "brier_ci_high",
                  "mcc_mean", "mcc_ci_low", "mcc_ci_high", "prevalence"]
    int_cols = ["n_samples", "n_positive", "n_folds"]
    bool_cols = ["is_default"]
    for c in str_cols:
        if c in res_df.columns:
            res_df[c] = res_df[c].astype(str).replace("None", "")
    for c in float_cols:
        if c in res_df.columns:
            res_df[c] = pd.to_numeric(res_df[c], errors="coerce")
    for c in int_cols:
        if c in res_df.columns:
            res_df[c] = pd.to_numeric(res_df[c], errors="coerce").fillna(0).astype(int)
    for c in bool_cols:
        if c in res_df.columns:
            res_df[c] = res_df[c].fillna(False).astype(bool)
    results_out.write_dataframe(spark.createDataFrame(res_df))

    abl = []
    for dim in ["architecture", "modality_ablation"]:
        if dim == "architecture":
            configs = ["model_a", "model_b", "model_c"]
        elif dim == "modality_ablation":
            configs = ["model_a", "model_a_no_sdoh", "model_a_no_cgm"]
        for i, c in enumerate(configs):
            abl.append({"ablation_dimension": dim, "config_id": c,
                        "variant_label": c, "variant_order": i, "is_reference": c == "model_a"})
    ablation_out.write_dataframe(spark.createDataFrame(pd.DataFrame(abl)))

    cost_out.write_dataframe(spark.createDataFrame(pd.DataFrame(cost_data)) if cost_data
                             else spark.createDataFrame(pd.DataFrame(columns=["config_id"])))

    interp_scores_out.write_dataframe(spark.createDataFrame(pd.DataFrame({
        "config_id": ["model_b", "model_c"] * 4,
        "criterion": ["factual_accuracy"] * 2 + ["clinical_plausibility"] * 2 +
                     ["completeness"] * 2 + ["actionability"] * 2,
        "rater_id": ["placeholder"] * 8,
        "mean_score": [float("nan")] * 8, "sd_score": [float("nan")] * 8, "n_items": [0] * 8,
    })))

    interp_agree_out.write_dataframe(spark.createDataFrame(pd.DataFrame({
        "criterion": ["factual_accuracy", "clinical_plausibility", "completeness", "actionability", "overall"],
        "kappa_type": ["fleiss"] * 5,
        "kappa": [float("nan")] * 5, "kappa_ci_low": [float("nan")] * 5,
        "kappa_ci_high": [float("nan")] * 5, "n_items": [0] * 5, "n_raters": [0] * 5,
    })))

    logger.info(f"\nNB5 COMPLETE (T1D) — {len(all_results)} experiments, {len(cost_data)} cost entries")


"""
NB6 — Model D: Evidence-Grounded Deliberative Ensemble (T1D)
=============================================================
Builds on Model C (the top-performing full-context CoT predictor) and adds
three mechanisms credited in the agentic-AI literature:

  Stage 0 — Knowledge injection (ADA/ISPAD guideline snippets)
  Stage 1 — Full-context CoT predictor emitting probability + structured
             per-modality cited evidence tuples (not YES/NO)
  Stage 2 — Verifier/critic agent (Reflexion-style): checks each cited
             evidence point against actual patient data, max 1 revision
  Stage 3 — Self-consistency calibration: K reasoning paths at T>0,
             average probabilities for smooth, calibrated scores

Additionally runs faithfulness analysis on a per-patient subsample:
  - Extractive grounding: verify cited (feature, visit, value) tuples
  - Counterfactual/redaction: mask cited features, check prediction shift
  - Reports sufficiency and comprehensiveness scores

Architecture ablation: C → C+verify → C+self-consistency → full D
"""

import json
import re
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, matthews_corrcoef
from transforms.api import Input, Output, transform, configure
from palantir_models.transforms import GenericCompletionLanguageModelInput
from palantir_models.models import GenericCompletionLanguageModel
from language_model_service_api.languagemodelservice_api_completion_v3 import GenericCompletionRequest

from .comparison_harness import (
    HORIZONS, LLM_OUTCOMES,
    LABS, MEDS, CONDITIONS, DEMOGRAPHICS, SDOH, CGM,
    RANDOM_STATE, make_comparison_split, get_outcome_exclusions,
)

logger = logging.getLogger(__name__)

LSTM_DATASET = "ri.foundry.main.dataset.242259a4-7302-407a-b468-cc37bb534411"
OUTPUT_BASE = "/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t1d"
SONNET_RID = "ri.language-model-service..language-model.anthropic-claude-4-6-sonnet"

ALL_OUTCOMES = LLM_OUTCOMES
N_BOOTSTRAP = 200
K_SELF_CONSISTENCY = 3          # number of reasoning paths for self-consistency (reduced from 5)
SC_TEMPERATURE = 0.7            # temperature for diversity in self-consistency
FAITHFULNESS_SUBSAMPLE = 10     # patients per outcome for faithfulness analysis (reduced from 20)
PRED_CHANGE_THRESHOLD = 0.10    # probability shift to count as "prediction moved"
MAX_PARALLEL_PATIENTS = 8       # concurrent patient processing (I/O-bound LLM calls)
GROUP_MASK_K_VALUES = [1, 3, 5, 10]  # cumulative top-K group-masking curve points

OUTCOME_DESCRIPTIONS = {
    "OUTCOME_Optimal_Glycemic_Control": "Will this patient maintain optimal glycemic control (HbA1c < 7%)?",
    "OUTCOME_Hypertension": "Will this patient develop hypertension?",
    "OUTCOME_Microalbuminuria": "Will this patient develop microalbuminuria (early kidney disease)?",
    "OUTCOME_Dyslipidemia": "Will this patient have dyslipidemia?",
    "OUTCOME_Insulin_Independence": "Will this patient maintain insulin independence (off insulin with A1c<7%, no DKA)?",
    "OUTCOME_Metformin_Response": "Will this patient respond to metformin monotherapy (A1c<7% on metformin alone)?",
    "OUTCOME_GLP1RA_Response": "Will this patient respond to GLP-1 receptor agonist therapy (A1c<7%)?",
}

# ── ADA / ISPAD Guideline Snippets for Knowledge Injection ──

GUIDELINE_SNIPPETS = {
    "OUTCOME_Optimal_Glycemic_Control": [
        "ADA 2024: Target HbA1c <7% for most pediatric T1D patients; individualize based on hypoglycemia risk.",
        "ISPAD 2022: CGM time-in-range >70% (70-180 mg/dL) correlates with HbA1c <7% and improved outcomes.",
        "Younger age at diagnosis and longer diabetes duration are associated with higher HbA1c trajectories.",
    ],
    "OUTCOME_Hypertension": [
        "ADA 2024: Screen BP at every visit; hypertension defined as BP ≥95th percentile for age/height/sex on ≥3 occasions.",
        "ISPAD 2022: Elevated BMI z-score is the strongest modifiable risk factor for hypertension in pediatric diabetes.",
        "Microalbuminuria and hypertension frequently co-occur; both should be screened simultaneously.",
    ],
    "OUTCOME_Microalbuminuria": [
        "ADA 2024: Screen UACR annually starting 5 years after T1D diagnosis or at puberty onset.",
        "ISPAD 2022: Persistent microalbuminuria (UACR 30-300 mg/g on ≥2 of 3 samples) warrants ACEi/ARB initiation.",
        "Poor glycemic control (HbA1c >9%) and longer diabetes duration increase microalbuminuria risk.",
    ],
    "OUTCOME_Dyslipidemia": [
        "ADA 2024: Screen fasting lipid panel at diagnosis, then every 5 years if normal; annually if abnormal.",
        "ISPAD 2022: LDL >100 mg/dL warrants lifestyle intervention; >130 mg/dL consider statin in children >10 yr.",
        "Dyslipidemia prevalence increases with poor glycemic control and obesity in pediatric T1D.",
    ],
    "OUTCOME_Insulin_Independence": [
        "Insulin independence in T1D is rare and typically reflects honeymoon phase or misclassification.",
        "Preserved C-peptide >0.6 ng/mL at diagnosis predicts longer partial remission.",
    ],
    "OUTCOME_Metformin_Response": [
        "ADA 2024: Metformin is first-line for T2D; in T1D it is adjunctive. Response defined as A1c <7% on monotherapy.",
        "BMI z-score reduction and insulin resistance markers predict metformin responsiveness.",
    ],
    "OUTCOME_GLP1RA_Response": [
        "ADA 2024: GLP-1 RAs approved as adjunct in T2D youth; limited T1D evidence.",
        "Weight reduction and improved postprandial glucose are primary GLP-1 RA benefits.",
    ],
}


# ── LLM Wrapper ──

class LLMWrapper:
    """Thread-safe LLM wrapper with cost tracking."""
    def __init__(self, llm_instance):
        self.llm = llm_instance
        self._lock = threading.Lock()
        self.total_calls = 0
        self.total_latency = 0.0
        self.est_input_tokens = 0
        self.est_output_tokens = 0

    def call(self, system_prompt, user_prompt, max_tokens=800, temperature=0.0):
        prompt = system_prompt + "\n\n" + user_prompt
        request = GenericCompletionRequest(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
        start = time.time()
        try:
            response = self.llm.create_completion(request)
            output = response.completion or ""
            elapsed = time.time() - start
            with self._lock:
                self.total_calls += 1
                self.total_latency += elapsed
                self.est_input_tokens += len(prompt) // 4
                self.est_output_tokens += len(output) // 4
            return output
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            with self._lock:
                self.total_latency += time.time() - start
            return None

    def reset(self):
        with self._lock:
            self.total_calls = 0
            self.total_latency = 0.0
            self.est_input_tokens = 0
            self.est_output_tokens = 0


# ── Feature Formatting ──

def format_features(patient_feats, visits, exclude=None):
    visit_labels = {f"v{i}": f"{i*6}mo" for i in range(1, 11)}
    lines = []
    for feat, vals in sorted(patient_feats.items()):
        if exclude and feat in exclude:
            continue
        parts = []
        for v in visits:
            val = vals.get(v)
            if val is not None and not np.isnan(val):
                # Include the visit token (v1/v2/...) AND the time label so the model
                # can cite the exact visit id used for extractive grounding.
                parts.append(f"{v}({visit_labels[v]})={val:.2f}")
        if parts:
            lines.append(f"  {feat}: {', '.join(parts)}")
    return "\n".join(lines) if lines else "  No data"


def get_patient_feats(df, mrn, visits):
    pdf = df[df["mrn"] == mrn]
    feats = {}
    for _, row in pdf.iterrows():
        feats[row["feature"]] = {v: row[v] for v in visits}
    return feats


# ── Stage 0: Knowledge Injection ──

def get_guideline_context(outcome):
    snippets = GUIDELINE_SNIPPETS.get(outcome, [])
    if not snippets:
        return ""
    text = "\n".join(f"  - {s}" for s in snippets)
    return f"\nRelevant clinical guidelines:\n{text}\n"


# ── Stage 1: Evidence-Grounded CoT Predictor ──

def stage1_prompt(patient_feats, visits, outcome_desc, guideline_ctx, exclude=None):
    data = format_features(patient_feats, visits, exclude=exclude)
    sys = (
        "You are an expert pediatric endocrinologist specializing in Type 1 Diabetes prediction. "
        "You must provide evidence-grounded predictions with specific data citations."
    )
    usr = (
        f"Clinical question: {outcome_desc}\n"
        f"{guideline_ctx}\n"
        f"Patient data (each feature shows value at each visit window, e.g. v3(18mo)=8.50):\n{data}\n\n"
        "Think step by step through each modality (labs/vitals, medications, conditions, SDOH, CGM).\n\n"
        "Then respond in EXACTLY this plain-text structured format (do NOT use JSON, do NOT use "
        "markdown tables). Put PROBABILITY and EVIDENCE first so they are never cut off:\n\n"
        "PROBABILITY: <a single number between 0.0 and 1.0>\n\n"
        "EVIDENCE:\n"
        "(one line per data point that drives your prediction, pipe-delimited. Copy the FEATURE "
        "name and the VISIT token and the numeric VALUE EXACTLY as they appear in the patient data "
        "above. DIRECTION is 'increases_risk' or 'decreases_risk'. IMPORTANCE is 0.0-1.0.)\n"
        "FEATURE | VISIT | VALUE | DIRECTION | IMPORTANCE\n"
        "HBA1C | v3 | 8.50 | increases_risk | 0.40\n"
        "<additional evidence lines...>\n\n"
        "REASONING: <your concise clinical chain of thought>\n"
    )
    return sys, usr


def parse_stage1_response(text):
    """Parse Stage 1 response returning (probability, evidence_list, reasoning).
    Primary format is plain-text delimited; JSON and regex are kept as fallbacks.
    Each evidence item is a dict: feature, visit, value, direction, importance."""
    if not text:
        return 0.5, [], ""

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    # ── Probability (try PROBABILITY: line first, then JSON key) ──
    prob = 0.5
    pm = re.search(r'PROBABILITY\s*[:=]\s*\*{0,2}\s*([01]?\.?[0-9]+)', cleaned, re.IGNORECASE)
    if not pm:
        pm = re.search(r'"?probability"?\s*[:=]\s*([01]?\.?[0-9]+)', cleaned, re.IGNORECASE)
    if pm:
        try:
            prob = max(0.0, min(1.0, float(pm.group(1))))
        except ValueError:
            pass

    # ── Evidence: pipe-delimited lines (primary format) ──
    evidence = []
    for line in cleaned.splitlines():
        if line.count("|") < 2:
            continue
        parts = [p.strip() for p in line.split("|")]
        feat = parts[0].lstrip("-*• ").strip()
        if not feat or feat.upper() in ("FEATURE", "FEATURE NAME"):
            continue  # skip header / non-evidence rows
        visit_m = re.search(r'v(\d+)', parts[1], re.IGNORECASE)
        val_m = re.search(r'(-?[0-9]*\.?[0-9]+)', parts[2])
        if not visit_m or not val_m:
            continue
        try:
            value = float(val_m.group(1))
        except ValueError:
            continue
        direction = parts[3].strip().lower() if len(parts) > 3 else ""
        importance = 0.0
        if len(parts) > 4:
            imp_m = re.search(r'([0-9]*\.?[0-9]+)', parts[4])
            if imp_m:
                try:
                    importance = max(0.0, min(1.0, float(imp_m.group(1))))
                except ValueError:
                    pass
        evidence.append({
            "feature": feat, "visit": f"v{visit_m.group(1)}", "value": value,
            "direction": direction, "importance": importance, "interpretation": "",
        })

    # ── Fallback: JSON-style evidence (if model ignored the delimited format) ──
    if not evidence:
        for m in re.finditer(
            r'"feature"\s*:\s*"([^"]+)"[^}]*?"visit"\s*:\s*"?(v?\d+)"?[^}]*?"value"\s*:\s*(-?[0-9.eE+]+)',
            cleaned):
            try:
                vis = m.group(2)
                vis = vis if vis.lower().startswith("v") else f"v{vis}"
                evidence.append({
                    "feature": m.group(1), "visit": vis, "value": float(m.group(3)),
                    "direction": "", "importance": 0.0, "interpretation": "",
                })
            except (ValueError, IndexError):
                pass

    # ── Reasoning ──
    reasoning = ""
    rm = re.search(r'REASONING\s*[:=]\s*(.+)', cleaned, re.IGNORECASE | re.DOTALL)
    if rm:
        reasoning = rm.group(1).strip()[:2000]

    return prob, evidence, reasoning


# ── Stage 2: Verifier/Critic Agent ──

def stage2_prompt(stage1_output, patient_feats, visits, outcome_desc, exclude=None):
    data = format_features(patient_feats, visits, exclude=exclude)
    sys = (
        "You are a clinical evidence auditor. Your job is to verify the accuracy of "
        "cited evidence in a clinical prediction. Check each cited data point against "
        "the actual patient data. Challenge any weak or incorrect citations."
    )
    usr = (
        f"Clinical question: {outcome_desc}\n\n"
        f"PREDICTOR'S OUTPUT:\n{stage1_output}\n\n"
        f"ACTUAL PATIENT DATA:\n{data}\n\n"
        "For each cited evidence point:\n"
        "1. Verify if the cited (feature, visit, value) matches the actual data\n"
        "2. Flag any hallucinated or incorrect citations\n"
        "3. If evidence is weak or incorrect, provide a revised assessment\n\n"
        "Respond in JSON:\n"
        '{\n'
        '  "verified_evidence": [{"feature": "<name>", "visit": "<vN>", "cited_value": <n>, '
        '"actual_value": <n or null>, "is_correct": <bool>}, ...],\n'
        '  "revision_needed": <bool>,\n'
        '  "revised_probability": <float 0.0-1.0 or null if no revision needed>,\n'
        '  "revised_reasoning": "<brief or null>"\n'
        '}'
    )
    return sys, usr


def parse_stage2_response(text):
    """Parse verifier response with regex fallback."""
    if not text:
        return [], False, None, ""

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    # Strategy 1: Full JSON parse
    try:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            d = json.loads(cleaned[start:end])
            verified = d.get("verified_evidence", [])
            if not isinstance(verified, list):
                verified = []
            revision_needed = bool(d.get("revision_needed", False))
            revised_prob = d.get("revised_probability")
            if revised_prob is not None:
                revised_prob = max(0.0, min(1.0, float(revised_prob)))
            revised_reasoning = str(d.get("revised_reasoning", ""))
            return verified, revision_needed, revised_prob, revised_reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Strategy 2: Regex fallback
    revision_match = re.search(r'"revision_needed"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
    revision_needed = revision_match and revision_match.group(1).lower() == "true"
    revised_prob = None
    prob_match = re.search(r'"revised_probability"\s*:\s*([0-9]*\.?[0-9]+)', cleaned)
    if prob_match and revision_needed:
        try:
            revised_prob = max(0.0, min(1.0, float(prob_match.group(1))))
        except ValueError:
            pass
    return [], revision_needed, revised_prob, ""


# ── Stage 3: Self-Consistency Calibration ──

def run_self_consistency(llm, sys_prompt, usr_prompt, k=K_SELF_CONSISTENCY, temp=SC_TEMPERATURE):
    """Run K reasoning paths at temperature>0, extract probabilities, average."""
    probs = []
    all_evidence = []
    for _ in range(k):
        resp = llm.call(sys_prompt, usr_prompt, max_tokens=2000, temperature=temp)
        prob, evidence, _ = parse_stage1_response(resp)
        probs.append(prob)
        all_evidence.append(evidence)
    if not probs:
        return 0.5, []
    return float(np.mean(probs)), all_evidence


# ── Full Model D Pipeline ──

def run_model_d_full(llm, patient_feats, visits, outcome_desc, outcome, exclude=None):
    """Full 4-stage Model D pipeline.
    Returns (probability, evidence, reasoning, verified, stage_costs)."""
    guideline_ctx = get_guideline_context(outcome)
    stage_costs = {}

    # Stage 0: knowledge injection is embedded in the Stage 1 prompt
    # Stage 1: Evidence-grounded CoT
    llm_before = (llm.total_calls, llm.est_input_tokens, llm.est_output_tokens, llm.total_latency)
    sys1, usr1 = stage1_prompt(patient_feats, visits, outcome_desc, guideline_ctx, exclude=exclude)
    resp1 = llm.call(sys1, usr1, max_tokens=2000, temperature=0.0)
    prob1, evidence1, reasoning1 = parse_stage1_response(resp1)
    stage_costs["stage1"] = {
        "calls": llm.total_calls - llm_before[0],
        "input_tokens": llm.est_input_tokens - llm_before[1],
        "output_tokens": llm.est_output_tokens - llm_before[2],
        "latency": llm.total_latency - llm_before[3],
    }

    # Stage 2: Verifier/Critic
    llm_before = (llm.total_calls, llm.est_input_tokens, llm.est_output_tokens, llm.total_latency)
    sys2, usr2 = stage2_prompt(resp1 or "", patient_feats, visits, outcome_desc, exclude=exclude)
    resp2 = llm.call(sys2, usr2, max_tokens=600, temperature=0.0)
    verified, revision_needed, revised_prob, _ = parse_stage2_response(resp2)
    prob_after_verify = revised_prob if (revision_needed and revised_prob is not None) else prob1
    stage_costs["stage2"] = {
        "calls": llm.total_calls - llm_before[0],
        "input_tokens": llm.est_input_tokens - llm_before[1],
        "output_tokens": llm.est_output_tokens - llm_before[2],
        "latency": llm.total_latency - llm_before[3],
    }

    # Stage 3: Self-consistency calibration
    llm_before = (llm.total_calls, llm.est_input_tokens, llm.est_output_tokens, llm.total_latency)
    sc_prob, sc_evidence = run_self_consistency(llm, sys1, usr1, k=K_SELF_CONSISTENCY, temp=SC_TEMPERATURE)
    # Final probability: average of verified probability and self-consistency mean
    final_prob = (prob_after_verify + sc_prob) / 2.0
    stage_costs["stage3"] = {
        "calls": llm.total_calls - llm_before[0],
        "input_tokens": llm.est_input_tokens - llm_before[1],
        "output_tokens": llm.est_output_tokens - llm_before[2],
        "latency": llm.total_latency - llm_before[3],
    }

    return final_prob, evidence1, reasoning1, verified, stage_costs


# ── Ablation Variants ──

def run_model_c_plus_verify(llm, patient_feats, visits, outcome_desc, outcome, exclude=None):
    """Model C + verification only (no self-consistency)."""
    guideline_ctx = get_guideline_context(outcome)
    sys1, usr1 = stage1_prompt(patient_feats, visits, outcome_desc, guideline_ctx, exclude=exclude)
    resp1 = llm.call(sys1, usr1, max_tokens=2000, temperature=0.0)
    prob1, evidence1, _ = parse_stage1_response(resp1)
    sys2, usr2 = stage2_prompt(resp1 or "", patient_feats, visits, outcome_desc, exclude=exclude)
    resp2 = llm.call(sys2, usr2, max_tokens=600, temperature=0.0)
    _, revision_needed, revised_prob, _ = parse_stage2_response(resp2)
    return revised_prob if (revision_needed and revised_prob is not None) else prob1


def run_model_c_plus_sc(llm, patient_feats, visits, outcome_desc, outcome, exclude=None):
    """Model C + self-consistency only (no verification)."""
    guideline_ctx = get_guideline_context(outcome)
    sys1, usr1 = stage1_prompt(patient_feats, visits, outcome_desc, guideline_ctx, exclude=exclude)
    sc_prob, _ = run_self_consistency(llm, sys1, usr1, k=K_SELF_CONSISTENCY, temp=SC_TEMPERATURE)
    return sc_prob


# ── Faithfulness Analysis ──

def verify_extractive_grounding(evidence_list, patient_feats, rel_tol=0.05, abs_tol=0.5):
    """Check each cited (feature, visit, value) against actual data using a
    relative-or-absolute tolerance (whichever is larger). This catches
    hallucinated evidence outright.
    Returns (n_correct, n_total, accuracy, detail_list)."""
    if not evidence_list:
        return 0, 0, 0.0, []
    n_correct = 0
    n_total = 0
    detail = []
    for ev in evidence_list:
        if not isinstance(ev, dict):
            continue
        feat = ev.get("feature", "")
        visit = ev.get("visit", "")
        cited_val = ev.get("value")
        if not feat or not visit or cited_val is None:
            continue
        n_total += 1
        actual_vals = patient_feats.get(feat, {})
        actual_val = actual_vals.get(visit)
        verified = False
        actual_out = None
        if actual_val is not None and not (isinstance(actual_val, float) and np.isnan(actual_val)):
            actual_out = float(actual_val)
            try:
                if abs(float(cited_val) - actual_out) <= max(abs_tol, rel_tol * abs(actual_out)):
                    verified = True
                    n_correct += 1
            except (ValueError, TypeError):
                pass
        detail.append({
            "feature": feat, "visit": visit,
            "cited_value": float(cited_val) if cited_val is not None else None,
            "actual_value": actual_out, "verified": verified,
            "importance": float(ev.get("importance", 0.0) or 0.0),
            "direction": ev.get("direction", ""),
        })
    accuracy = n_correct / n_total if n_total > 0 else 0.0
    return n_correct, n_total, accuracy, detail


def run_counterfactual_faithfulness(llm, patient_feats, visits, outcome_desc, outcome,
                                   evidence_list, baseline_prob, exclude=None):
    """For each cited important feature, mask it and re-run prediction.
    Returns (sufficiency_score, comprehensiveness_score, per_feature_deltas)."""
    if not evidence_list:
        return 0.0, 0.0, []

    cited_features = {}
    for ev in evidence_list:
        if isinstance(ev, dict) and ev.get("feature"):
            # keep the max importance if a feature is cited more than once
            f = ev["feature"]
            cited_features[f] = max(cited_features.get(f, 0.0), float(ev.get("importance", 0.0) or 0.0))

    if not cited_features:
        return 0.0, 0.0, []

    guideline_ctx = get_guideline_context(outcome)
    deltas = []

    # Comprehensiveness: mask each cited feature individually
    for feat_name, importance in cited_features.items():
        masked_exclude = set(exclude or set()) | {feat_name}
        sys_p, usr_p = stage1_prompt(patient_feats, visits, outcome_desc, guideline_ctx, exclude=masked_exclude)
        resp = llm.call(sys_p, usr_p, max_tokens=2000, temperature=0.0)
        prob_masked, _, _ = parse_stage1_response(resp)
        delta = abs(baseline_prob - prob_masked)
        deltas.append({"feature": feat_name, "importance": importance,
                       "baseline_prob": baseline_prob,
                       "masked_prob": prob_masked, "delta": delta,
                       "prediction_moved": delta >= PRED_CHANGE_THRESHOLD})

    # Comprehensiveness: fraction of cited features whose removal changes prediction
    n_moved = sum(1 for d in deltas if d["prediction_moved"])
    comprehensiveness = n_moved / len(deltas) if deltas else 0.0

    # Sufficiency: mask ALL non-cited features, keep only cited ones
    all_feats = set(patient_feats.keys())
    non_cited = all_feats - set(cited_features.keys()) - set(exclude or set())
    suff_exclude = set(exclude or set()) | non_cited
    sys_s, usr_s = stage1_prompt(patient_feats, visits, outcome_desc, guideline_ctx, exclude=suff_exclude)
    resp_s = llm.call(sys_s, usr_s, max_tokens=2000, temperature=0.0)
    prob_suff, _, _ = parse_stage1_response(resp_s)
    sufficiency = 1.0 - abs(baseline_prob - prob_suff)  # 1.0 = perfect sufficiency

    return sufficiency, comprehensiveness, deltas


def run_group_masking_curve(llm, patient_feats, visits, outcome_desc, outcome,
                            evidence_list, baseline_prob, exclude=None,
                            k_values=GROUP_MASK_K_VALUES):
    """Cumulative top-K group masking: rank cited features by importance, then mask
    the top-K most-important features TOGETHER and measure how far the prediction
    moves. Produces a comprehensiveness-vs-K curve. Returns list of dicts."""
    # Rank unique cited features by their (max) self-reported importance
    feat_importance = {}
    for ev in evidence_list:
        if isinstance(ev, dict) and ev.get("feature"):
            f = ev["feature"]
            feat_importance[f] = max(feat_importance.get(f, 0.0),
                                     float(ev.get("importance", 0.0) or 0.0))
    if not feat_importance:
        return []
    ranked = [f for f, _ in sorted(feat_importance.items(), key=lambda kv: kv[1], reverse=True)]

    guideline_ctx = get_guideline_context(outcome)
    rows = []
    seen_sizes = set()
    for k in k_values:
        n_mask = min(k, len(ranked))
        if n_mask in seen_sizes:
            continue  # avoid duplicate calls when k exceeds available features
        seen_sizes.add(n_mask)
        top_k_feats = set(ranked[:n_mask])
        masked_exclude = set(exclude or set()) | top_k_feats
        sys_p, usr_p = stage1_prompt(patient_feats, visits, outcome_desc, guideline_ctx,
                                     exclude=masked_exclude)
        resp = llm.call(sys_p, usr_p, max_tokens=2000, temperature=0.0)
        prob_masked, _, _ = parse_stage1_response(resp)
        delta = abs(baseline_prob - prob_masked)
        rows.append({
            "k": k, "n_masked": n_mask,
            "baseline_prob": baseline_prob, "masked_prob": prob_masked,
            "delta": delta, "prediction_moved": delta >= PRED_CHANGE_THRESHOLD,
        })
    return rows


# ── Bootstrap CI ──

def bootstrap_ci(y_true, y_scores, fn, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(fn(y_true[idx], y_scores[idx]))
        except (ValueError, ZeroDivisionError):
            pass
    if not scores:
        return np.nan, np.nan, np.nan
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ── Model D Configs (architecture ablation) ──

MODEL_D_CONFIGS = {
    "model_d_full": {
        "description": "Full Model D: knowledge + CoT evidence + verify + self-consistency",
        "runner": "full",
    },
    "model_c_plus_verify": {
        "description": "Ablation: Model C + verification only",
        "runner": "verify_only",
    },
    "model_c_plus_sc": {
        "description": "Ablation: Model C + self-consistency only",
        "runner": "sc_only",
    },
}


@configure(["DYNAMIC_ALLOCATION_ENABLED_8_16"])
@transform(
    results_out=Output(f"{OUTPUT_BASE}/nb6_model_d_results_t1d"),
    cost_out=Output(f"{OUTPUT_BASE}/nb6_model_d_cost_t1d"),
    faithfulness_out=Output(f"{OUTPUT_BASE}/nb6_faithfulness_results_t1d"),
    evidence_detail_out=Output(f"{OUTPUT_BASE}/nb6_evidence_detail_t1d"),
    group_masking_out=Output(f"{OUTPUT_BASE}/nb6_group_masking_curve_t1d"),
    ablation_out=Output(f"{OUTPUT_BASE}/nb6_ablation_index_t1d"),
    lstm_data=Input(LSTM_DATASET),
    llm_sonnet=GenericCompletionLanguageModelInput(SONNET_RID),
)
def compute(ctx, lstm_data, llm_sonnet, results_out, cost_out, faithfulness_out,
            evidence_detail_out, group_masking_out, ablation_out):

    df = lstm_data.dataframe().toPandas()
    llm = LLMWrapper(llm_sonnet)

    all_results = []
    cost_data = []
    faithfulness_data = []
    evidence_detail_data = []
    group_masking_data = []

    for hz_name, hz_cfg in HORIZONS.items():
        tv = hz_cfg["target_visit"]
        iv = hz_cfg["input_visits"]

        for outcome in ALL_OUTCOMES:
            logger.info(f"{'='*50}\n  NB6 {hz_name} | {outcome}")

            collinear_excl = get_outcome_exclusions(outcome)
            if collinear_excl:
                logger.info(f"  Excluding collinear features: {collinear_excl}")

            train_mrns, test_mrns, y_all = make_comparison_split(df, outcome, tv)
            if test_mrns is None:
                logger.info("  SKIP: too small / single-class")
                continue
            mrns = test_mrns
            y = y_all.loc[mrns].values
            logger.info(f"  n_test={len(y)}, n_pos={int(y.sum())} ({y.mean():.1%})")

            outcome_desc = OUTCOME_DESCRIPTIONS.get(outcome, outcome)

            # Subsample for faithfulness analysis
            rng = np.random.RandomState(RANDOM_STATE)
            faith_indices = rng.choice(len(mrns), min(FAITHFULNESS_SUBSAMPLE, len(mrns)), replace=False)
            faith_mrns = set(mrns[i] for i in faith_indices)

            for cid, cfg in MODEL_D_CONFIGS.items():
                llm.reset()
                per_patient_evidence = {}

                # ── Parallel patient processing (I/O-bound LLM calls) ──
                def process_patient(mrn):
                    pf = get_patient_feats(df, mrn, iv)
                    if cfg["runner"] == "full":
                        prob, evidence, reasoning, verified, _ = run_model_d_full(
                            llm, pf, iv, outcome_desc, outcome, exclude=collinear_excl)
                        return mrn, prob, evidence, reasoning, pf
                    elif cfg["runner"] == "verify_only":
                        prob = run_model_c_plus_verify(
                            llm, pf, iv, outcome_desc, outcome, exclude=collinear_excl)
                    elif cfg["runner"] == "sc_only":
                        prob = run_model_c_plus_sc(
                            llm, pf, iv, outcome_desc, outcome, exclude=collinear_excl)
                    else:
                        prob = 0.5
                    return mrn, prob, None, None, None

                results_map = {}
                with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PATIENTS) as executor:
                    futures = {executor.submit(process_patient, mrn): mrn for mrn in mrns}
                    done_count = 0
                    for future in as_completed(futures):
                        mrn, prob, evidence, reasoning, pf = future.result()
                        results_map[mrn] = prob
                        if evidence is not None and mrn in faith_mrns:
                            per_patient_evidence[mrn] = (prob, evidence, reasoning, pf)
                        done_count += 1
                        if done_count % 25 == 0:
                            logger.info(f"    {cid}: {done_count}/{len(mrns)} patients done")

                # Preserve original order for metrics
                probabilities = [results_map[mrn] for mrn in mrns]

                y_prob = np.array(probabilities)
                y_pred = (y_prob >= 0.5).astype(int)

                auc_m, auc_lo, auc_hi = bootstrap_ci(y, y_prob, roc_auc_score)
                pr_m, pr_lo, pr_hi = bootstrap_ci(y, y_prob, average_precision_score)
                brier_m, brier_lo, brier_hi = bootstrap_ci(y, y_prob, brier_score_loss)
                try:
                    mcc = matthews_corrcoef(y, y_pred)
                except Exception:
                    mcc = None

                all_results.append({
                    "config_id": cid,
                    "config_label": cfg["description"],
                    "method_family": "deliberative_ensemble",
                    "outcome": outcome, "horizon": hz_name,
                    "roc_auc_mean": auc_m, "roc_auc_ci_low": auc_lo, "roc_auc_ci_high": auc_hi,
                    "pr_auc_mean": pr_m, "pr_auc_ci_low": pr_lo, "pr_auc_ci_high": pr_hi,
                    "brier_mean": brier_m, "brier_ci_low": brier_lo, "brier_ci_high": brier_hi,
                    "mcc_mean": mcc,
                    "n_samples": len(y), "n_positive": int(y.sum()),
                    "prevalence": float(y.mean()),
                })

                cost_data.append({
                    "config_id": cid,
                    "method_family": "deliberative_ensemble",
                    "cost_usd_per_patient": round(
                        (llm.est_input_tokens * 3 / 1e6 + llm.est_output_tokens * 15 / 1e6) / max(len(mrns), 1), 4),
                    "latency_sec_per_patient": round(llm.total_latency / max(len(mrns), 1), 2),
                    "total_tokens_per_patient": (llm.est_input_tokens + llm.est_output_tokens) // max(len(mrns), 1),
                    "input_tokens": llm.est_input_tokens,
                    "output_tokens": llm.est_output_tokens,
                    "n_llm_calls": llm.total_calls,
                    "outcome": outcome, "horizon": hz_name,
                    "k_self_consistency": K_SELF_CONSISTENCY if cfg["runner"] in ("full", "sc_only") else 0,
                })

                logger.info(f"    {cid:30s} AUC={auc_m:.3f} calls={llm.total_calls}")

                # Faithfulness analysis for full Model D only (parallel across patients)
                if cid == "model_d_full" and per_patient_evidence:
                    logger.info(f"  Running faithfulness analysis on {len(per_patient_evidence)} patients...")

                    def analyze_patient(item):
                        mrn, (base_prob, evidence, reasoning, pf) = item
                        n_correct, n_total, eg_accuracy, eg_detail = verify_extractive_grounding(evidence, pf)
                        sufficiency, comprehensiveness, feat_deltas = run_counterfactual_faithfulness(
                            llm, pf, iv, outcome_desc, outcome, evidence, base_prob,
                            exclude=collinear_excl)
                        group_curve = run_group_masking_curve(
                            llm, pf, iv, outcome_desc, outcome, evidence, base_prob,
                            exclude=collinear_excl)
                        return (mrn, base_prob, reasoning, n_correct, n_total, eg_accuracy,
                                sufficiency, comprehensiveness, eg_detail, feat_deltas, group_curve)

                    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PATIENTS) as fexec:
                        ffutures = [fexec.submit(analyze_patient, it)
                                    for it in per_patient_evidence.items()]
                        for fut in as_completed(ffutures):
                            (mrn, base_prob, reasoning, n_correct, n_total, eg_accuracy,
                             sufficiency, comprehensiveness, eg_detail, feat_deltas,
                             group_curve) = fut.result()

                            delta_by_feat = {d["feature"]: d for d in feat_deltas}

                            # Cumulative top-K group-masking curve rows
                            for gc in group_curve:
                                group_masking_data.append({
                                    "outcome": outcome, "horizon": hz_name, "mrn": str(mrn),
                                    "k": gc["k"], "n_masked": gc["n_masked"],
                                    "baseline_probability": gc["baseline_prob"],
                                    "masked_probability": gc["masked_prob"],
                                    "delta": gc["delta"],
                                    "prediction_moved": bool(gc["prediction_moved"]),
                                })

                            faithfulness_data.append({
                                "outcome": outcome, "horizon": hz_name,
                                "mrn": str(mrn),
                                "extractive_n_correct": n_correct,
                                "extractive_n_total": n_total,
                                "extractive_accuracy": eg_accuracy,
                                "sufficiency_score": sufficiency,
                                "comprehensiveness_score": comprehensiveness,
                                "n_cited_features": len(eg_detail),
                                "n_verified_features": sum(1 for e in eg_detail if e["verified"]),
                                "n_features_moved": sum(
                                    1 for d in feat_deltas if d.get("prediction_moved", False)),
                                "baseline_probability": base_prob,
                                "reasoning_narrative": (reasoning or "")[:1000],
                            })

                            # Per-cited-feature interpretability surface
                            for ed in eg_detail:
                                dd = delta_by_feat.get(ed["feature"], {})
                                evidence_detail_data.append({
                                    "outcome": outcome, "horizon": hz_name, "mrn": str(mrn),
                                    "feature": ed["feature"], "visit": ed["visit"],
                                    "cited_value": ed["cited_value"],
                                    "actual_value": ed["actual_value"],
                                    "verified": bool(ed["verified"]),
                                    "importance": ed["importance"],
                                    "direction": ed["direction"],
                                    "baseline_probability": base_prob,
                                    "masked_probability": dd.get("masked_prob"),
                                    "counterfactual_delta": dd.get("delta"),
                                    "prediction_moved": bool(dd.get("prediction_moved", False)),
                                })

    # ════════════════════════════════════════════════════════════════════════
    # Write outputs
    # ════════════════════════════════════════════════════════════════════════
    spark = ctx.spark_session

    res_df = pd.DataFrame(all_results)
    if len(res_df) > 0:
        for c in ["config_id", "config_label", "method_family", "outcome", "horizon"]:
            res_df[c] = res_df[c].astype(str)
        for c in ["roc_auc_mean", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc_mean",
                  "pr_auc_ci_low", "pr_auc_ci_high", "brier_mean", "brier_ci_low", "brier_ci_high",
                  "mcc_mean", "prevalence"]:
            if c in res_df.columns:
                res_df[c] = pd.to_numeric(res_df[c], errors="coerce")
        for c in ["n_samples", "n_positive"]:
            res_df[c] = res_df[c].astype("int64")
    results_out.write_dataframe(spark.createDataFrame(res_df) if len(res_df) > 0
                                else spark.createDataFrame(pd.DataFrame(columns=["config_id"])))

    cost_df = pd.DataFrame(cost_data)
    if len(cost_df) > 0:
        for c in ["config_id", "method_family", "outcome", "horizon"]:
            cost_df[c] = cost_df[c].astype(str)
        for c in ["cost_usd_per_patient", "latency_sec_per_patient"]:
            cost_df[c] = cost_df[c].astype("float64")
        for c in ["total_tokens_per_patient", "input_tokens", "output_tokens", "n_llm_calls", "k_self_consistency"]:
            cost_df[c] = cost_df[c].astype("int64")
    cost_out.write_dataframe(spark.createDataFrame(cost_df) if len(cost_df) > 0
                             else spark.createDataFrame(pd.DataFrame(columns=["config_id"])))

    faith_df = pd.DataFrame(faithfulness_data)
    if len(faith_df) > 0:
        for c in ["outcome", "horizon", "mrn", "reasoning_narrative"]:
            faith_df[c] = faith_df[c].astype(str)
        for c in ["extractive_accuracy", "sufficiency_score", "comprehensiveness_score", "baseline_probability"]:
            faith_df[c] = faith_df[c].astype("float64")
        for c in ["extractive_n_correct", "extractive_n_total", "n_cited_features",
                  "n_verified_features", "n_features_moved"]:
            faith_df[c] = faith_df[c].astype("int64")
    faithfulness_out.write_dataframe(spark.createDataFrame(faith_df) if len(faith_df) > 0
                                     else spark.createDataFrame(pd.DataFrame(columns=["outcome"])))

    # Per-cited-feature interpretability surface (the "extra value" of the agent)
    ev_df = pd.DataFrame(evidence_detail_data)
    if len(ev_df) > 0:
        for c in ["outcome", "horizon", "mrn", "feature", "visit", "direction"]:
            ev_df[c] = ev_df[c].astype(str)
        for c in ["cited_value", "actual_value", "importance", "baseline_probability",
                  "masked_probability", "counterfactual_delta"]:
            ev_df[c] = pd.to_numeric(ev_df[c], errors="coerce")
        for c in ["verified", "prediction_moved"]:
            ev_df[c] = ev_df[c].astype(bool)
    evidence_detail_out.write_dataframe(spark.createDataFrame(ev_df) if len(ev_df) > 0
                                        else spark.createDataFrame(pd.DataFrame(columns=["outcome"])))

    # Cumulative top-K group-masking curve (comprehensiveness vs # top features removed)
    gm_df = pd.DataFrame(group_masking_data)
    if len(gm_df) > 0:
        for c in ["outcome", "horizon", "mrn"]:
            gm_df[c] = gm_df[c].astype(str)
        for c in ["baseline_probability", "masked_probability", "delta"]:
            gm_df[c] = pd.to_numeric(gm_df[c], errors="coerce")
        for c in ["k", "n_masked"]:
            gm_df[c] = gm_df[c].astype("int64")
        gm_df["prediction_moved"] = gm_df["prediction_moved"].astype(bool)
    group_masking_out.write_dataframe(spark.createDataFrame(gm_df) if len(gm_df) > 0
                                      else spark.createDataFrame(pd.DataFrame(columns=["outcome"])))

    # Ablation index: C → C+verify → C+SC → full D
    abl = [
        {"ablation_dimension": "model_d_components", "config_id": "model_c_plus_verify",
         "variant_label": "C + Verification", "variant_order": 0, "is_reference": False},
        {"ablation_dimension": "model_d_components", "config_id": "model_c_plus_sc",
         "variant_label": "C + Self-Consistency", "variant_order": 1, "is_reference": False},
        {"ablation_dimension": "model_d_components", "config_id": "model_d_full",
         "variant_label": "Full Model D", "variant_order": 2, "is_reference": True},
    ]
    ablation_out.write_dataframe(spark.createDataFrame(pd.DataFrame(abl)))

    logger.info(f"\nNB6 COMPLETE (T1D) — {len(all_results)} experiments, "
                f"{len(faithfulness_data)} faithfulness analyses")


"""
NB7 — Algorithm Cost & Class Comparison (T1D)
===============================================
Compares computational cost across algorithm classes to quantify the
value of agentic workflows in clinical informatics:

  1. Classical ML (LR, RF, XGBoost)        — CPU seconds
  2. Temporal ML (XGBoost + temporal feats) — CPU seconds
  3. Deep Learning (LSTM, GRU, CNN, Transformer) — CPU seconds
  4. Single-agent LLM (Models A, C)        — tokens, USD, latency
  5. Multi-agent LLM (Model B)             — tokens, USD, latency
  6. Deliberative Ensemble (Model D)       — tokens, USD, latency

Analyses produced:
  - Per-algorithm-class cost summary
  - Cost-performance Pareto frontier
  - Cascade analysis: when cheap model suffices vs expensive model adds value
  - Marginal value: AUC gain per dollar
"""

import numpy as np
import pandas as pd
from transforms.api import Input, Output, lightweight, transform

OUTPUT_BASE = "/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t1d"

# NB4 outputs
NB4_MATCHED = f"{OUTPUT_BASE}/nb4_matched_comparison_t1d"
NB4_TIMING = f"{OUTPUT_BASE}/nb4_model_timing_t1d"

# NB5 outputs
NB5_RESULTS = f"{OUTPUT_BASE}/nb5_agentic_results_t1d"
NB5_COST = f"{OUTPUT_BASE}/nb5_cost_latency_t1d"

# NB6 outputs
NB6_RESULTS = f"{OUTPUT_BASE}/nb6_model_d_results_t1d"
NB6_COST = f"{OUTPUT_BASE}/nb6_model_d_cost_t1d"

# Algorithm class mapping for ML models
ML_ALGORITHM_CLASS = {
    "Logistic_Regression": "classical_ml",
    "Random_Forest": "classical_ml",
    "XGBoost": "classical_ml",
    "XGBoost_Temporal": "temporal_ml",
    "LSTM": "deep_learning",
    "GRU": "deep_learning",
    "Temporal_CNN": "deep_learning",
    "Transformer": "deep_learning",
}

# LLM algorithm class mapping
LLM_ALGORITHM_CLASS = {
    "model_a": "single_agent_llm",
    "model_c": "single_agent_llm",
    "model_b": "multi_agent_llm",
    "model_a_no_sdoh": "single_agent_llm",
    "model_a_no_cgm": "single_agent_llm",
}

MODEL_D_ALGORITHM_CLASS = {
    "model_d_full": "deliberative_ensemble",
    "model_c_plus_verify": "deliberative_ensemble",
    "model_c_plus_sc": "deliberative_ensemble",
}

# Approximate cost per CPU-second for ML models (cloud compute estimate)
ML_COST_PER_SEC_USD = 0.00005  # ~$0.18/hr for a 4-core instance


@lightweight(cpu_cores=2, memory_gb=4)
@transform(
    cost_comparison_out=Output(f"{OUTPUT_BASE}/nb7_cost_comparison_t1d"),
    pareto_out=Output(f"{OUTPUT_BASE}/nb7_pareto_frontier_t1d"),
    cascade_out=Output(f"{OUTPUT_BASE}/nb7_cascade_analysis_t1d"),
    nb4_matched=Input(NB4_MATCHED),
    nb4_timing=Input(NB4_TIMING),
    nb5_results=Input(NB5_RESULTS),
    nb5_cost=Input(NB5_COST),
    nb6_results=Input(NB6_RESULTS),
    nb6_cost=Input(NB6_COST),
)
def compute(nb4_matched, nb4_timing, nb5_results, nb5_cost,
            nb6_results, nb6_cost,
            cost_comparison_out, pareto_out, cascade_out):

    # Load inputs
    df_nb4_matched = nb4_matched.pandas()
    df_nb4_timing = nb4_timing.pandas()
    df_nb5_results = nb5_results.pandas()
    df_nb5_cost = nb5_cost.pandas()
    df_nb6_results = nb6_results.pandas()
    df_nb6_cost = nb6_cost.pandas()

    comparison_rows = []

    # ════════════════════════════════════════════════════════════════════════
    # 1. ML Models (from NB4)
    # ════════════════════════════════════════════════════════════════════════
    for _, row in df_nb4_matched.iterrows():
        model = row["model"]
        outcome = row["outcome"]
        horizon = row["horizon"]

        # Get timing for this model+outcome+horizon (matched holdout phase)
        timing_match = df_nb4_timing[
            (df_nb4_timing["model"] == model) &
            (df_nb4_timing["outcome"] == outcome) &
            (df_nb4_timing["horizon"] == horizon) &
            (df_nb4_timing["phase"] == "matched_holdout")
        ]
        wall_clock = timing_match["wall_clock_seconds"].values[0] if len(timing_match) > 0 else np.nan
        n_test = int(row.get("n_test", 0))
        cost_per_patient = (wall_clock * ML_COST_PER_SEC_USD / max(n_test, 1)) if not np.isnan(wall_clock) else np.nan

        comparison_rows.append({
            "model_id": model, "model_label": model.replace("_", " "),
            "algorithm_class": ML_ALGORITHM_CLASS.get(model, "unknown"),
            "paradigm": "ML",
            "outcome": outcome, "horizon": horizon,
            "roc_auc": row.get("roc_auc_mean", np.nan),
            "roc_auc_ci_low": row.get("roc_auc_ci_low", np.nan),
            "roc_auc_ci_high": row.get("roc_auc_ci_high", np.nan),
            "pr_auc": row.get("pr_auc_mean", np.nan),
            "brier": row.get("brier_mean", np.nan),
            "wall_clock_seconds": wall_clock,
            "cost_usd_per_patient": cost_per_patient,
            "latency_sec_per_patient": (wall_clock / max(n_test, 1)) if not np.isnan(wall_clock) else np.nan,
            "total_tokens_per_patient": 0,
            "n_llm_calls_per_patient": 0,
            "n_test": n_test,
        })

    # ════════════════════════════════════════════════════════════════════════
    # 2. LLM Models A/B/C (from NB5)
    # ════════════════════════════════════════════════════════════════════════
    for _, row in df_nb5_results.iterrows():
        cid = row["config_id"]
        outcome = row["outcome"]
        horizon = row["horizon"]

        cost_match = df_nb5_cost[
            (df_nb5_cost["config_id"] == cid) &
            (df_nb5_cost["outcome"] == outcome) &
            (df_nb5_cost["horizon"] == horizon)
        ]
        cost_usd = cost_match["cost_usd_per_patient"].values[0] if len(cost_match) > 0 else np.nan
        latency = cost_match["latency_sec_per_patient"].values[0] if len(cost_match) > 0 else np.nan
        tokens = int(cost_match["total_tokens_per_patient"].values[0]) if len(cost_match) > 0 else 0
        n_calls = int(cost_match["n_llm_calls"].values[0]) if len(cost_match) > 0 else 0
        n_test = int(row.get("n_samples", 0))

        comparison_rows.append({
            "model_id": cid, "model_label": row.get("config_label", cid),
            "algorithm_class": LLM_ALGORITHM_CLASS.get(cid, "single_agent_llm"),
            "paradigm": "LLM",
            "outcome": outcome, "horizon": horizon,
            "roc_auc": row.get("roc_auc_mean", np.nan),
            "roc_auc_ci_low": row.get("roc_auc_ci_low", np.nan),
            "roc_auc_ci_high": row.get("roc_auc_ci_high", np.nan),
            "pr_auc": row.get("pr_auc_mean", np.nan),
            "brier": row.get("brier_mean", np.nan),
            "wall_clock_seconds": latency * n_test if not np.isnan(latency) else np.nan,
            "cost_usd_per_patient": cost_usd,
            "latency_sec_per_patient": latency,
            "total_tokens_per_patient": tokens,
            "n_llm_calls_per_patient": n_calls // max(n_test, 1),
            "n_test": n_test,
        })

    # ════════════════════════════════════════════════════════════════════════
    # 3. Model D variants (from NB6)
    # ════════════════════════════════════════════════════════════════════════
    for _, row in df_nb6_results.iterrows():
        cid = row["config_id"]
        outcome = row["outcome"]
        horizon = row["horizon"]

        cost_match = df_nb6_cost[
            (df_nb6_cost["config_id"] == cid) &
            (df_nb6_cost["outcome"] == outcome) &
            (df_nb6_cost["horizon"] == horizon)
        ]
        cost_usd = cost_match["cost_usd_per_patient"].values[0] if len(cost_match) > 0 else np.nan
        latency = cost_match["latency_sec_per_patient"].values[0] if len(cost_match) > 0 else np.nan
        tokens = int(cost_match["total_tokens_per_patient"].values[0]) if len(cost_match) > 0 else 0
        n_calls = int(cost_match["n_llm_calls"].values[0]) if len(cost_match) > 0 else 0
        n_test = int(row.get("n_samples", 0))

        comparison_rows.append({
            "model_id": cid, "model_label": row.get("config_label", cid),
            "algorithm_class": MODEL_D_ALGORITHM_CLASS.get(cid, "deliberative_ensemble"),
            "paradigm": "LLM_Ensemble",
            "outcome": outcome, "horizon": horizon,
            "roc_auc": row.get("roc_auc_mean", np.nan),
            "roc_auc_ci_low": row.get("roc_auc_ci_low", np.nan),
            "roc_auc_ci_high": row.get("roc_auc_ci_high", np.nan),
            "pr_auc": row.get("pr_auc_mean", np.nan),
            "brier": row.get("brier_mean", np.nan),
            "wall_clock_seconds": latency * n_test if not np.isnan(latency) else np.nan,
            "cost_usd_per_patient": cost_usd,
            "latency_sec_per_patient": latency,
            "total_tokens_per_patient": tokens,
            "n_llm_calls_per_patient": n_calls // max(n_test, 1),
            "n_test": n_test,
        })

    comparison_df = pd.DataFrame(comparison_rows)

    # ════════════════════════════════════════════════════════════════════════
    # 4. Pareto Frontier — per outcome, which models are Pareto-optimal?
    # ════════════════════════════════════════════════════════════════════════
    pareto_rows = []
    if len(comparison_df) > 0:
        for outcome in comparison_df["outcome"].unique():
            odf = comparison_df[comparison_df["outcome"] == outcome].copy()
            odf = odf.dropna(subset=["roc_auc", "cost_usd_per_patient"])
            if len(odf) == 0:
                continue

            # Sort by cost ascending
            odf = odf.sort_values("cost_usd_per_patient")

            # Find Pareto-optimal: no other model is both cheaper AND better
            pareto_mask = []
            for idx, row in odf.iterrows():
                dominated = False
                for idx2, row2 in odf.iterrows():
                    if idx == idx2:
                        continue
                    if (row2["cost_usd_per_patient"] <= row["cost_usd_per_patient"] and
                            row2["roc_auc"] >= row["roc_auc"] and
                            (row2["cost_usd_per_patient"] < row["cost_usd_per_patient"] or
                             row2["roc_auc"] > row["roc_auc"])):
                        dominated = True
                        break
                pareto_mask.append(not dominated)

            for i, (idx, row) in enumerate(odf.iterrows()):
                pareto_rows.append({
                    "outcome": outcome,
                    "model_id": row["model_id"],
                    "algorithm_class": row["algorithm_class"],
                    "roc_auc": row["roc_auc"],
                    "cost_usd_per_patient": row["cost_usd_per_patient"],
                    "is_pareto_optimal": pareto_mask[i],
                })

    # ════════════════════════════════════════════════════════════════════════
    # 5. Cascade Analysis — when does the expensive model add value?
    # ════════════════════════════════════════════════════════════════════════
    cascade_rows = []
    if len(comparison_df) > 0:
        for outcome in comparison_df["outcome"].unique():
            odf = comparison_df[comparison_df["outcome"] == outcome].dropna(subset=["roc_auc", "cost_usd_per_patient"])
            if len(odf) < 2:
                continue

            # Identify cheapest model and most expensive model
            cheapest = odf.loc[odf["cost_usd_per_patient"].idxmin()]
            most_expensive = odf.loc[odf["cost_usd_per_patient"].idxmax()]
            best_auc = odf.loc[odf["roc_auc"].idxmax()]

            auc_gain = best_auc["roc_auc"] - cheapest["roc_auc"]
            cost_ratio = most_expensive["cost_usd_per_patient"] / max(cheapest["cost_usd_per_patient"], 1e-8)

            # Marginal value: AUC gain per dollar
            cost_delta = most_expensive["cost_usd_per_patient"] - cheapest["cost_usd_per_patient"]
            marginal_auc_per_dollar = auc_gain / max(cost_delta, 1e-8) if cost_delta > 0 else 0.0

            # Is the expensive model worth it?
            # Heuristic: if AUC gain > 0.03 (clinically meaningful), recommend cascade
            cascade_recommended = auc_gain > 0.03

            cascade_rows.append({
                "outcome": outcome,
                "cheapest_model": cheapest["model_id"],
                "cheapest_class": cheapest["algorithm_class"],
                "cheapest_auc": cheapest["roc_auc"],
                "cheapest_cost": cheapest["cost_usd_per_patient"],
                "best_model": best_auc["model_id"],
                "best_class": best_auc["algorithm_class"],
                "best_auc": best_auc["roc_auc"],
                "best_cost": best_auc["cost_usd_per_patient"],
                "most_expensive_model": most_expensive["model_id"],
                "most_expensive_cost": most_expensive["cost_usd_per_patient"],
                "auc_gain_cheap_to_best": auc_gain,
                "cost_ratio_expensive_to_cheap": cost_ratio,
                "marginal_auc_per_dollar": marginal_auc_per_dollar,
                "cascade_recommended": cascade_recommended,
                "recommendation": (
                    f"Use {best_auc['model_id']} (AUC +{auc_gain:.3f} over {cheapest['model_id']})"
                    if cascade_recommended else
                    f"Use {cheapest['model_id']} (expensive model gain only +{auc_gain:.3f})"
                ),
            })

    # ════════════════════════════════════════════════════════════════════════
    # Write outputs
    # ════════════════════════════════════════════════════════════════════════
    if len(comparison_df) > 0:
        for c in ["model_id", "model_label", "algorithm_class", "paradigm", "outcome", "horizon"]:
            comparison_df[c] = comparison_df[c].astype(str)
        for c in ["roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc", "brier",
                  "wall_clock_seconds", "cost_usd_per_patient", "latency_sec_per_patient"]:
            if c in comparison_df.columns:
                comparison_df[c] = pd.to_numeric(comparison_df[c], errors="coerce")
        for c in ["total_tokens_per_patient", "n_llm_calls_per_patient", "n_test"]:
            comparison_df[c] = comparison_df[c].fillna(0).astype("int64")
    cost_comparison_out.write_table(comparison_df if len(comparison_df) > 0
                                    else pd.DataFrame(columns=["model_id"]))

    pareto_df = pd.DataFrame(pareto_rows)
    if len(pareto_df) > 0:
        for c in ["outcome", "model_id", "algorithm_class"]:
            pareto_df[c] = pareto_df[c].astype(str)
        for c in ["roc_auc", "cost_usd_per_patient"]:
            pareto_df[c] = pareto_df[c].astype("float64")
        pareto_df["is_pareto_optimal"] = pareto_df["is_pareto_optimal"].astype(bool)
    pareto_out.write_table(pareto_df if len(pareto_df) > 0
                           else pd.DataFrame(columns=["outcome"]))

    cascade_df = pd.DataFrame(cascade_rows)
    if len(cascade_df) > 0:
        for c in ["outcome", "cheapest_model", "cheapest_class", "best_model", "best_class",
                  "most_expensive_model", "recommendation"]:
            cascade_df[c] = cascade_df[c].astype(str)
        for c in ["cheapest_auc", "cheapest_cost", "best_auc", "best_cost", "most_expensive_cost",
                  "auc_gain_cheap_to_best", "cost_ratio_expensive_to_cheap", "marginal_auc_per_dollar"]:
            cascade_df[c] = cascade_df[c].astype("float64")
        cascade_df["cascade_recommended"] = cascade_df["cascade_recommended"].astype(bool)
    cascade_out.write_table(cascade_df if len(cascade_df) > 0
                            else pd.DataFrame(columns=["outcome"]))

    print(f"\n{'='*60}\nNB7 COST ANALYSIS COMPLETE (T1D)\n{'='*60}")
    print(f"Comparison entries: {len(comparison_df)}")
    print(f"Pareto entries: {len(pareto_df)}")
    print(f"Cascade entries: {len(cascade_df)}")
    if len(comparison_df) > 0:
        print("\n── Cost by Algorithm Class ──")
        class_summary = comparison_df.groupby("algorithm_class").agg(
            mean_auc=("roc_auc", "mean"),
            mean_cost=("cost_usd_per_patient", "mean"),
            mean_latency=("latency_sec_per_patient", "mean"),
        ).reset_index()
        for _, r in class_summary.iterrows():
            print(f"  {r['algorithm_class']:25s} AUC={r['mean_auc']:.3f}  "
                  f"Cost=${r['mean_cost']:.4f}/pt  Latency={r['mean_latency']:.2f}s/pt")

### Now for T2D: 
"""
comparison_harness.py — T2D
======================
SINGLE SOURCE OF TRUTH for everything that MUST be identical between the
classical / deep-sequence baselines (NB4), the agentic LLM workflows (NB5),
and the new Model D deliberative ensemble (NB6) — TYPE 2 DIABETES cohort.

Key T2D differences from T1D:
  - No CGM features (CGM not routinely collected for T2D at this center)
  - All 7 outcomes eligible for LLM prediction (LLM_OUTCOMES_T2D)

Rev 2 changes (reviewer fixes)
-------------------------------
1. Outcome-specific collinear feature exclusions (COLLINEAR_FEATURES_BY_OUTCOME).
2. Prevalence-adaptive sampling.
3. Horizons restricted to year_2 only.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Visit windows ────────────────────────────────────────────────────────────
VISIT_COLS = [f"v{i}" for i in range(1, 11)]

# ── Prediction horizons ──────────────────────────────────────────────────────
HORIZONS = {
    "year_2": {"target_visit": "v4", "input_visits": ["v1", "v2", "v3"]},
}

# ── Outcomes ──────────────────────────────────────────────────────────────────
ALL_OUTCOMES = [
    "OUTCOME_Optimal_Glycemic_Control", "OUTCOME_Hypertension",
    "OUTCOME_Microalbuminuria", "OUTCOME_Dyslipidemia",
    "OUTCOME_Insulin_Independence", "OUTCOME_Metformin_Response",
    "OUTCOME_GLP1RA_Response",
]
# T2D: All outcomes eligible for LLM
LLM_OUTCOMES_T2D = ALL_OUTCOMES
# Backward-compat aliases
LLM_OUTCOMES = LLM_OUTCOMES_T2D

# ── Feature groups ────────────────────────────────────────────────────────────
LABS = ["HBA1C", "GLUCOSE", "BMI_ZSCORE", "TOTAL_CHOLESTEROL", "HDL_CHOLESTEROL",
        "LDL_CHOLESTEROL", "TRIGLYCERIDES", "ALT", "AST", "BUN", "SERUM_CREATININE",
        "SERUM_C_PEPTIDE", "UACR_RATIO", "BETA_HYDROXYBUTYRATE",
        "SBP_OUTPATIENT", "SBP_INPATIENT", "DBP_OUTPATIENT", "DBP_INPATIENT"]
MEDS = ["Insulins", "Biguanide", "GLP1_agonists"]
CONDITIONS = ["DKA", "Ketosis", "Diabetic_Retinopathy", "Neuropathy"]
DEMOGRAPHICS = ["age_at_diagnosis", "sex", "ethnicity_hispanic", "race_white",
                "race_black", "race_asian", "diabetes_duration"]
SDOH = ["socio_food_insecurity", "socio_housing_instability", "socio_financial_strain_binary",
        "socio_insurance_category", "socio_parental_education_binary",
        "socio_social_family_support_binary", "socio_adverse_childhood_experience",
        "socio_transportation_barrier"]
# T2D: No CGM features
CGM = []

# T2D modality configs — no CGM tier
MODALITY_CONFIGS = {
    "EHR_only": set(LABS + MEDS + CONDITIONS + DEMOGRAPHICS),
    "EHR_SDOH": set(LABS + MEDS + CONDITIONS + DEMOGRAPHICS + SDOH),
}
# Backward-compat aliases
MODALITY_CONFIGS_T2D = MODALITY_CONFIGS

# ── Outcome-specific collinear feature exclusions (Rev 2) ────────────────────
COLLINEAR_FEATURES_BY_OUTCOME = {
    "OUTCOME_Hypertension": {
        "SBP_OUTPATIENT", "SBP_INPATIENT", "DBP_OUTPATIENT", "DBP_INPATIENT",
    },
    "OUTCOME_Dyslipidemia": {
        "TOTAL_CHOLESTEROL", "HDL_CHOLESTEROL", "LDL_CHOLESTEROL", "TRIGLYCERIDES",
    },
    "OUTCOME_Optimal_Glycemic_Control": {
        "HBA1C",
    },
    "OUTCOME_Microalbuminuria": {
        "UACR_RATIO",
    },
    "OUTCOME_Insulin_Independence": {
        "Insulins",
    },
    "OUTCOME_Metformin_Response": {
        "Biguanide", "HBA1C",
    },
    "OUTCOME_GLP1RA_Response": {
        "GLP1_agonists", "HBA1C",
    },
}


def get_outcome_exclusions(outcome):
    """Return the set of feature names to EXCLUDE for the given outcome."""
    return COLLINEAR_FEATURES_BY_OUTCOME.get(outcome, set())


# ── Prevalence-adaptive sampling (Rev 2) ──────────────────────────────────────
N_TEST_MIN = 50
N_TEST_MAX = 250       # Rev 3: scaled up to 250 (forecast ~18h T2D, under 24h budget)
N_TEST_BASE = 150

RANDOM_STATE = 42


def get_adaptive_test_size(n_total):
    """Prevalence-adaptive test size."""
    adaptive = int(N_TEST_BASE * np.sqrt(max(n_total, 40) / 200))
    adaptive = int(np.clip(adaptive, N_TEST_MIN, N_TEST_MAX))
    return min(adaptive, n_total // 2)


# ── Shared evaluation split ───────────────────────────────────────────────────

def get_labeled_labels(df, outcome, target_visit):
    t = (df[df["feature"] == outcome][["mrn", target_visit]]
         .dropna(subset=[target_visit]))
    t = t.groupby("mrn")[target_visit].first()
    return t.astype(int).sort_index()


def make_comparison_split(df, outcome, target_visit, seed=RANDOM_STATE):
    """Deterministic, stratified train/test split shared by NB4, NB5, and NB6."""
    y = get_labeled_labels(df, outcome, target_visit)
    n_total = len(y)
    if n_total < 40 or y.nunique() < 2:
        return None, None, y
    test_size = get_adaptive_test_size(n_total)
    try:
        train_mrns, test_mrns = train_test_split(
            y.index.to_numpy(),
            test_size=test_size,
            random_state=seed,
            stratify=y.values,
        )
    except ValueError:
        return None, None, y
    return sorted(train_mrns), sorted(test_mrns), y

"""
NB4 — T2D Temporal Model Training (Rev 2 — reviewer fixes)
==================================================================
Trains LSTM, GRU, 1D-CNN, Transformer + classical baselines.

T2D key differences from T1D:
  - No CGM features
  - No EHR_SDOH_CGM modality tier
  - All 7 outcomes eligible

Rev 2 changes
--------------
1. Outcome-specific collinear feature removal via get_outcome_exclusions().
2. Wall-clock timing per model written to nb4_model_timing_t2d for cost comparison.
3. Horizons now year_2 only (via harness).
4. Prevalence-adaptive sampling (via harness make_comparison_split).
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
from transforms.api import Input, Output, lightweight, transform

from .comparison_harness import (
    HORIZONS, ALL_OUTCOMES,
    LABS, MEDS, CONDITIONS, DEMOGRAPHICS, SDOH, MODALITY_CONFIGS,
    RANDOM_STATE, make_comparison_split, get_outcome_exclusions,
)

LSTM_DATASET = "ri.foundry.main.dataset.872815d9-3093-4ccb-ade5-2d44b1c2edd5"
OUTPUT_BASE = "/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d"

N_FOLDS = 5
N_BOOTSTRAP = 200


# ── PyTorch Models ──

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)


class GRUClassifier(nn.Module):
    def __init__(self, n_features, hidden=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1]).squeeze(-1)


class TemporalCNN(nn.Module):
    def __init__(self, n_features, n_steps, hidden=64, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=min(3, n_steps), padding=1),
            nn.ReLU(), nn.Dropout(dropout), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(self, n_features, n_steps, d_model=64, nhead=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_steps, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        x = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]
        x = self.encoder(x)
        return self.fc(x.mean(dim=1)).squeeze(-1)


def train_torch_model(model, X_train, y_train, X_test, epochs=100, lr=1e-3, batch_size=32):
    device = torch.device("cpu")
    model = model.to(device)
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test).to(device)

    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X_te)).cpu().numpy()


# ── Model registry ──
MODELS_CFG = {
    "Logistic_Regression": "flat",
    "Random_Forest": "flat",
    "XGBoost": "flat",
    "XGBoost_Temporal": "temp",
    "LSTM": "torch",
    "GRU": "torch",
    "Temporal_CNN": "torch",
    "Transformer": "torch",
}
ABLATION_MODELS_CFG = {
    "Logistic_Regression": "flat",
    "Random_Forest": "flat",
    "XGBoost": "flat",
    "XGBoost_Temporal": "temp",
}

ALGORITHM_CLASS = {
    "Logistic_Regression": "classical_ml",
    "Random_Forest": "classical_ml",
    "XGBoost": "classical_ml",
    "XGBoost_Temporal": "temporal_ml",
    "LSTM": "deep_learning",
    "GRU": "deep_learning",
    "Temporal_CNN": "deep_learning",
    "Transformer": "deep_learning",
}


# ── Data Prep ──

def prepare_data(df, outcome, input_visits, target_visit, allowed_features=None,
                 exclude_features=None):
    target_df = df[df["feature"] == outcome][["mrn", target_visit]].dropna(subset=[target_visit])
    target_df = target_df.rename(columns={target_visit: "target"}).set_index("mrn")
    valid_mrns = sorted(set(target_df.index))

    non_outcome = df[~df["feature"].str.startswith("OUTCOME_")]

    if allowed_features is not None:
        non_outcome = non_outcome[non_outcome["feature"].isin(allowed_features)]

    # Rev 2: remove collinear features for this outcome
    if exclude_features:
        non_outcome = non_outcome[~non_outcome["feature"].isin(exclude_features)]

    feat_names = sorted(non_outcome["feature"].unique())
    n_p, n_t, n_f = len(valid_mrns), len(input_visits), len(feat_names)

    tensor_3d = np.full((n_p, n_t, n_f), np.nan)
    for fi, feat in enumerate(feat_names):
        fd = non_outcome[non_outcome["feature"] == feat].set_index("mrn")
        for pi, mrn in enumerate(valid_mrns):
            if mrn in fd.index:
                row = fd.loc[mrn]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                for ti, v in enumerate(input_visits):
                    tensor_3d[pi, ti, fi] = row[v]

    y = target_df.loc[valid_mrns, "target"].astype(int).values
    X_flat = tensor_3d.reshape(n_p, -1)
    flat_names = [f"{f}__{v}" for v in input_visits for f in feat_names]
    return X_flat, tensor_3d, y, flat_names, feat_names, valid_mrns


def add_temporal_features(X_3d):
    n_p, n_t, n_f = X_3d.shape
    feats = []
    t_vals = np.arange(n_t, dtype=float)
    for f in range(n_f):
        s = X_3d[:, :, f]
        feats.append(s[:, -1:])
        feats.append(np.nanmean(s, axis=1, keepdims=True))
        feats.append(np.nanstd(s, axis=1, keepdims=True))
        feats.append(s[:, -1:] - s[:, 0:1])
        slopes = np.full((n_p, 1), np.nan)
        for i in range(n_p):
            mask = ~np.isnan(s[i])
            if mask.sum() >= 2:
                slopes[i, 0] = np.polyfit(t_vals[mask], s[i][mask], 1)[0]
        feats.append(slopes)
    return np.hstack(feats)


def evaluate(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    r = {}
    try:
        r["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        r["roc_auc"] = np.nan
    try:
        r["pr_auc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        r["pr_auc"] = np.nan
    r["f1"] = f1_score(y_true, y_pred, zero_division=0)
    r["brier"] = brier_score_loss(y_true, y_prob)
    return r


def bootstrap_ci(y_true, y_scores, fn, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(fn(y_true[idx], y_scores[idx]))
        except (ValueError, ZeroDivisionError):
            pass
    if not scores:
        return np.nan, np.nan, np.nan
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def train_eval_one(model_name, mtype, X_flat, X_temp, X_3d, y, tr_idx, te_idx,
                   n_t, n_f, flat_names=None, feat_names=None, collect_importance=False):
    y_tr = y[tr_idx]
    importance_rows = []

    if mtype == "flat":
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X_flat[tr_idx])
        Xte = imp.transform(X_flat[te_idx])
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        if model_name == "Logistic_Regression":
            m = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=1.0,
                                   max_iter=2000, solver="saga", random_state=RANDOM_STATE)
        elif model_name == "Random_Forest":
            m = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                       random_state=RANDOM_STATE, n_jobs=-1)
        else:
            m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, eval_metric="logloss")
        m.fit(Xtr, y_tr)
        yp = m.predict_proba(Xte)[:, 1]

        if collect_importance:
            imp_vals = getattr(m, "feature_importances_", None)
            if imp_vals is None and hasattr(m, "coef_"):
                imp_vals = np.abs(m.coef_[0])
            if imp_vals is not None:
                for j, fn in enumerate(flat_names):
                    importance_rows.append({"model": model_name, "feature": fn,
                                            "importance": float(imp_vals[j])})

    elif mtype == "temp":
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X_temp[tr_idx])
        Xte = imp.transform(X_temp[te_idx])
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=RANDOM_STATE, eval_metric="logloss")
        m.fit(Xtr, y_tr)
        yp = m.predict_proba(Xte)[:, 1]

        if collect_importance:
            imp_vals = m.feature_importances_
            temp_names = []
            for fname in feat_names:
                temp_names.extend([f"{fname}__last", f"{fname}__mean", f"{fname}__std",
                                   f"{fname}__delta", f"{fname}__slope"])
            for j, fn in enumerate(temp_names):
                importance_rows.append({"model": model_name, "feature": fn,
                                        "importance": float(imp_vals[j])})

    elif mtype == "torch":
        X3_tr = X_3d[tr_idx].copy()
        X3_te = X_3d[te_idx].copy()
        for f in range(n_f):
            vals = X3_tr[:, :, f].flatten()
            med = np.nanmedian(vals) if np.any(~np.isnan(vals)) else 0.0
            X3_tr[:, :, f] = np.where(np.isnan(X3_tr[:, :, f]), med, X3_tr[:, :, f])
            X3_te[:, :, f] = np.where(np.isnan(X3_te[:, :, f]), med, X3_te[:, :, f])
            mu, sd = X3_tr[:, :, f].mean(), X3_tr[:, :, f].std() + 1e-8
            X3_tr[:, :, f] = (X3_tr[:, :, f] - mu) / sd
            X3_te[:, :, f] = (X3_te[:, :, f] - mu) / sd

        if model_name == "LSTM":
            mo = LSTMClassifier(n_f, hidden=64, n_layers=2)
        elif model_name == "GRU":
            mo = GRUClassifier(n_f, hidden=64, n_layers=2)
        elif model_name == "Temporal_CNN":
            mo = TemporalCNN(n_f, n_t, hidden=64)
        else:
            mo = TransformerClassifier(n_f, n_t, d_model=64, nhead=4, n_layers=2)

        yp = train_torch_model(mo, X3_tr, y_tr, X3_te, epochs=100)

        if collect_importance:
            base_auc = evaluate(y[te_idx], yp).get("roc_auc", np.nan)
            if not np.isnan(base_auc):
                mo.eval()
                rng = np.random.RandomState(RANDOM_STATE)
                for feat_idx, feat_name in enumerate(feat_names):
                    X3_perm = X3_te.copy()
                    perm_order = rng.permutation(X3_perm.shape[0])
                    X3_perm[:, :, feat_idx] = X3_perm[perm_order, :, feat_idx]
                    with torch.no_grad():
                        yp_perm = torch.sigmoid(mo(torch.FloatTensor(X3_perm))).cpu().numpy()
                    perm_auc = evaluate(y[te_idx], yp_perm).get("roc_auc", np.nan)
                    delta = (base_auc - perm_auc) if not np.isnan(perm_auc) else 0.0
                    importance_rows.append({"model": model_name, "feature": feat_name,
                                            "importance": float(delta)})
    else:
        raise ValueError(f"unknown mtype {mtype}")

    return yp, importance_rows


@lightweight(cpu_cores=4, memory_gb=16)
@transform(
    results_out=Output(f"{OUTPUT_BASE}/nb4_all_model_results_t2d"),
    importance_out=Output(f"{OUTPUT_BASE}/nb4_all_feature_importance_t2d"),
    ablation_out=Output(f"{OUTPUT_BASE}/nb4_modality_ablation_t2d"),
    matched_out=Output(f"{OUTPUT_BASE}/nb4_matched_comparison_t2d"),
    timing_out=Output(f"{OUTPUT_BASE}/nb4_model_timing_t2d"),
    lstm_data=Input(LSTM_DATASET),
)
def compute(lstm_data, results_out, importance_out, ablation_out, matched_out, timing_out):
    df = lstm_data.pandas()

    all_results = []
    all_importance = []
    ablation_results = []
    matched_results = []
    timing_rows = []

    # ════════════════════════════════════════════════════════════════════════
    # PART 1 — Reference baselines: 5-fold CV (with collinear exclusions)
    # ════════════════════════════════════════════════════════════════════════
    for hz_name, hz_cfg in HORIZONS.items():
        tv = hz_cfg["target_visit"]
        iv = hz_cfg["input_visits"]

        for outcome in ALL_OUTCOMES:
            print(f"\n{'='*60}\n  [CV] {hz_name} | {outcome}\n{'='*60}")
            exclusions = get_outcome_exclusions(outcome)
            if exclusions:
                print(f"  Excluding collinear features: {exclusions}")
            try:
                X_flat, X_3d, y, flat_names, feat_names, mrns = prepare_data(
                    df, outcome, iv, tv, exclude_features=exclusions)
            except Exception as e:
                print(f"  SKIP: {e}")
                continue

            if len(y) < 30 or np.unique(y).shape[0] < 2:
                print(f"  SKIP: n={len(y)}, n_pos={y.sum()}")
                continue

            n_t, n_f = len(iv), len(feat_names)
            X_temp = add_temporal_features(X_3d)
            print(f"  n={len(y)}, n_pos={y.sum()} ({y.mean():.1%}), 3d={X_3d.shape}")

            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

            for model_name, mtype in MODELS_CFG.items():
                fold_metrics = []
                model_start = time.time()
                for fi, (tr_idx, te_idx) in enumerate(skf.split(X_flat, y)):
                    try:
                        collect = (fi == N_FOLDS - 1)
                        yp, imp_rows = train_eval_one(
                            model_name, mtype, X_flat, X_temp, X_3d, y, tr_idx, te_idx,
                            n_t, n_f, flat_names, feat_names, collect_importance=collect)
                        for r in imp_rows:
                            all_importance.append({"outcome": outcome, "horizon": hz_name, **r})
                        metrics = evaluate(y[te_idx], yp)
                        metrics["fold"] = fi
                        fold_metrics.append(metrics)
                    except Exception as e:
                        print(f"    {model_name} fold {fi}: ERROR {e}")
                model_elapsed = time.time() - model_start

                if fold_metrics:
                    fdf = pd.DataFrame(fold_metrics)
                    agg = {
                        "outcome": outcome, "horizon": hz_name, "model": model_name,
                        "roc_auc_mean": fdf["roc_auc"].mean(), "roc_auc_std": fdf["roc_auc"].std(),
                        "pr_auc_mean": fdf["pr_auc"].mean(), "pr_auc_std": fdf["pr_auc"].std(),
                        "f1_mean": fdf["f1"].mean(), "f1_std": fdf["f1"].std(),
                        "brier_mean": fdf["brier"].mean(), "brier_std": fdf["brier"].std(),
                        "n_samples": len(y), "n_positive": int(y.sum()),
                        "prevalence": float(y.mean()), "n_folds": len(fold_metrics),
                    }
                    all_results.append(agg)
                    print(f"    {model_name:25s} AUC={agg['roc_auc_mean']:.3f}+-{agg['roc_auc_std']:.3f}  "
                          f"PR={agg['pr_auc_mean']:.3f}  F1={agg['f1_mean']:.3f}  ({model_elapsed:.1f}s)")

                    timing_rows.append({
                        "model": model_name, "outcome": outcome, "horizon": hz_name,
                        "phase": "cv_5fold", "wall_clock_seconds": round(model_elapsed, 2),
                        "n_train": len(y) - len(y) // N_FOLDS,
                        "n_test": len(y) // N_FOLDS,
                        "algorithm_class": ALGORITHM_CLASS[model_name],
                        "n_features": n_f,
                    })

    # ════════════════════════════════════════════════════════════════════════
    # PART 2 — Matched held-out comparison (with collinear exclusions)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  MATCHED HELD-OUT COMPARISON\n{'='*60}")
    for hz_name, hz_cfg in HORIZONS.items():
        tv = hz_cfg["target_visit"]
        iv = hz_cfg["input_visits"]

        for outcome in ALL_OUTCOMES:
            exclusions = get_outcome_exclusions(outcome)
            train_mrns, test_mrns, _ = make_comparison_split(df, outcome, tv)
            if test_mrns is None:
                print(f"  [matched] SKIP {hz_name}|{outcome}: too small / single-class")
                continue
            try:
                X_flat, X_3d, y, flat_names, feat_names, mrns = prepare_data(
                    df, outcome, iv, tv, exclude_features=exclusions)
            except Exception as e:
                print(f"  [matched] SKIP {hz_name}|{outcome}: {e}")
                continue

            pos = {m: i for i, m in enumerate(mrns)}
            tr_idx = np.array([pos[m] for m in train_mrns if m in pos])
            te_idx = np.array([pos[m] for m in test_mrns if m in pos])
            if len(te_idx) < 10 or np.unique(y[tr_idx]).shape[0] < 2 \
                    or np.unique(y[te_idx]).shape[0] < 2:
                print(f"  [matched] SKIP {hz_name}|{outcome}: degenerate split")
                continue

            n_t, n_f = len(iv), len(feat_names)
            X_temp = add_temporal_features(X_3d)
            y_te = y[te_idx]
            print(f"  [matched] {hz_name}|{outcome}: n_train={len(tr_idx)} n_test={len(te_idx)} "
                  f"test_pos={int(y_te.sum())}")

            for model_name, mtype in MODELS_CFG.items():
                try:
                    t0 = time.time()
                    yp, _ = train_eval_one(model_name, mtype, X_flat, X_temp, X_3d, y,
                                           tr_idx, te_idx, n_t, n_f, collect_importance=False)
                    t_elapsed = time.time() - t0
                    auc_m, auc_lo, auc_hi = bootstrap_ci(y_te, yp, roc_auc_score)
                    pr_m, pr_lo, pr_hi = bootstrap_ci(y_te, yp, average_precision_score)
                    brier_m, brier_lo, brier_hi = bootstrap_ci(y_te, yp, brier_score_loss)
                    f1 = f1_score(y_te, (yp >= 0.5).astype(int), zero_division=0)
                    matched_results.append({
                        "outcome": outcome, "horizon": hz_name, "model": model_name,
                        "roc_auc_mean": auc_m, "roc_auc_ci_low": auc_lo, "roc_auc_ci_high": auc_hi,
                        "pr_auc_mean": pr_m, "pr_auc_ci_low": pr_lo, "pr_auc_ci_high": pr_hi,
                        "brier_mean": brier_m, "brier_ci_low": brier_lo, "brier_ci_high": brier_hi,
                        "f1": float(f1),
                        "n_test": int(len(te_idx)), "n_test_positive": int(y_te.sum()),
                        "test_prevalence": float(y_te.mean()), "n_train": int(len(tr_idx)),
                    })
                    timing_rows.append({
                        "model": model_name, "outcome": outcome, "horizon": hz_name,
                        "phase": "matched_holdout", "wall_clock_seconds": round(t_elapsed, 2),
                        "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
                        "algorithm_class": ALGORITHM_CLASS[model_name],
                        "n_features": n_f,
                    })
                    print(f"    {model_name:25s} AUC={auc_m:.3f} [{auc_lo:.3f},{auc_hi:.3f}] ({t_elapsed:.1f}s)")
                except Exception as e:
                    print(f"    [matched] {model_name}: ERROR {e}")

    # ════════════════════════════════════════════════════════════════════════
    # PART 3 — Modality ablation (with collinear exclusions)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  MODALITY ABLATION STUDY\n{'='*60}")
    for modality_name, allowed_features in MODALITY_CONFIGS.items():
        print(f"\n── Modality: {modality_name} ──")
        for hz_name, hz_cfg in HORIZONS.items():
            tv = hz_cfg["target_visit"]
            iv = hz_cfg["input_visits"]
            for outcome in ALL_OUTCOMES:
                exclusions = get_outcome_exclusions(outcome)
                try:
                    X_flat, X_3d, y, flat_names, feat_names, mrns = prepare_data(
                        df, outcome, iv, tv, allowed_features=allowed_features,
                        exclude_features=exclusions)
                except Exception:
                    continue
                if len(y) < 30 or np.unique(y).shape[0] < 2:
                    continue

                n_t, n_f = len(iv), len(feat_names)
                X_temp = add_temporal_features(X_3d)
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

                for model_name, mtype in ABLATION_MODELS_CFG.items():
                    fold_metrics = []
                    for fi, (tr_idx, te_idx) in enumerate(skf.split(X_flat, y)):
                        try:
                            yp, _ = train_eval_one(model_name, mtype, X_flat, X_temp, X_3d, y,
                                                   tr_idx, te_idx, n_t, n_f, collect_importance=False)
                            metrics = evaluate(y[te_idx], yp)
                            metrics["fold"] = fi
                            fold_metrics.append(metrics)
                        except Exception as e:
                            print(f"    {modality_name}/{model_name} fold {fi}: ERROR {e}")

                    if fold_metrics:
                        fdf = pd.DataFrame(fold_metrics)
                        ablation_results.append({
                            "modality": modality_name, "outcome": outcome, "horizon": hz_name,
                            "model": model_name,
                            "roc_auc_mean": fdf["roc_auc"].mean(), "roc_auc_std": fdf["roc_auc"].std(),
                            "pr_auc_mean": fdf["pr_auc"].mean(), "pr_auc_std": fdf["pr_auc"].std(),
                            "f1_mean": fdf["f1"].mean(), "f1_std": fdf["f1"].std(),
                            "brier_mean": fdf["brier"].mean(), "brier_std": fdf["brier"].std(),
                            "n_samples": len(y), "n_positive": int(y.sum()),
                            "prevalence": float(y.mean()), "n_folds": len(fold_metrics),
                            "n_features": n_f,
                        })
                        print(f"    {modality_name:12s} {model_name:25s} {outcome:45s} "
                              f"AUC={fdf['roc_auc'].mean():.3f}")

    # ════════════════════════════════════════════════════════════════════════
    # Write outputs
    # ════════════════════════════════════════════════════════════════════════
    results_df = pd.DataFrame(all_results)
    if len(results_df) > 0:
        for c in ["outcome", "horizon", "model"]:
            results_df[c] = results_df[c].astype(str)
        for c in ["roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std",
                  "f1_mean", "f1_std", "brier_mean", "brier_std", "prevalence"]:
            results_df[c] = results_df[c].astype("float64")
        for c in ["n_samples", "n_positive", "n_folds"]:
            results_df[c] = results_df[c].astype("int64")
    results_out.write_table(results_df)

    imp_df = pd.DataFrame(all_importance) if all_importance else pd.DataFrame(
        columns=["outcome", "horizon", "model", "feature", "importance"])
    if len(imp_df) > 0:
        for c in ["outcome", "horizon", "model", "feature"]:
            imp_df[c] = imp_df[c].astype(str)
        imp_df["importance"] = imp_df["importance"].astype("float64")
    importance_out.write_table(imp_df)

    ablation_df = pd.DataFrame(ablation_results) if ablation_results else pd.DataFrame(
        columns=["modality", "outcome", "horizon", "model", "roc_auc_mean", "roc_auc_std",
                 "pr_auc_mean", "pr_auc_std", "f1_mean", "f1_std", "brier_mean", "brier_std",
                 "n_samples", "n_positive", "prevalence", "n_folds", "n_features"])
    if len(ablation_df) > 0:
        for c in ["modality", "outcome", "horizon", "model"]:
            ablation_df[c] = ablation_df[c].astype(str)
        for c in ["roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std",
                  "f1_mean", "f1_std", "brier_mean", "brier_std", "prevalence"]:
            ablation_df[c] = ablation_df[c].astype("float64")
        for c in ["n_samples", "n_positive", "n_folds", "n_features"]:
            ablation_df[c] = ablation_df[c].astype("int64")
    ablation_out.write_table(ablation_df)

    matched_df = pd.DataFrame(matched_results) if matched_results else pd.DataFrame(
        columns=["outcome", "horizon", "model", "roc_auc_mean", "roc_auc_ci_low",
                 "roc_auc_ci_high", "pr_auc_mean", "pr_auc_ci_low", "pr_auc_ci_high",
                 "brier_mean", "brier_ci_low", "brier_ci_high", "f1",
                 "n_test", "n_test_positive", "test_prevalence", "n_train"])
    if len(matched_df) > 0:
        for c in ["outcome", "horizon", "model"]:
            matched_df[c] = matched_df[c].astype(str)
        for c in ["roc_auc_mean", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc_mean",
                  "pr_auc_ci_low", "pr_auc_ci_high", "brier_mean", "brier_ci_low",
                  "brier_ci_high", "f1", "test_prevalence"]:
            matched_df[c] = matched_df[c].astype("float64")
        for c in ["n_test", "n_test_positive", "n_train"]:
            matched_df[c] = matched_df[c].astype("int64")
    matched_out.write_table(matched_df)

    # Rev 2: timing output for cost comparison
    timing_df = pd.DataFrame(timing_rows) if timing_rows else pd.DataFrame(
        columns=["model", "outcome", "horizon", "phase", "wall_clock_seconds",
                 "n_train", "n_test", "algorithm_class", "n_features"])
    if len(timing_df) > 0:
        for c in ["model", "outcome", "horizon", "phase", "algorithm_class"]:
            timing_df[c] = timing_df[c].astype(str)
        timing_df["wall_clock_seconds"] = timing_df["wall_clock_seconds"].astype("float64")
        for c in ["n_train", "n_test", "n_features"]:
            timing_df[c] = timing_df[c].astype("int64")
    timing_out.write_table(timing_df)

    print(f"\n{'='*60}\nNB4 COMPLETE (T2D)\n{'='*60}")
    print(f"CV reference: {len(all_results)} | matched: {len(matched_results)} | "
          f"ablation: {len(ablation_results)} | timing: {len(timing_rows)}")

"""
NB5 — Agentic Architecture for Clinical Outcome Prediction (T2D)
   (Rev 2 — collinear exclusions + year_2 only)
==================================================================
LLM-based clinical prediction (Models A/B/C + 2 ablation models).

T2D key differences from T1D:
  - No CGM features/agent
  - All 7 outcomes eligible for LLM
  - System prompts reference "Type 2 Diabetes"

Rev 2 changes
--------------
1. Outcome-specific collinear feature exclusions via get_outcome_exclusions().
2. Horizons now year_2 only (via harness).
3. Prevalence-adaptive sampling (via harness make_comparison_split).
"""

import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, matthews_corrcoef
from transforms.api import Input, Output, transform, configure
from palantir_models.transforms import GenericCompletionLanguageModelInput
from palantir_models.models import GenericCompletionLanguageModel
from language_model_service_api.languagemodelservice_api_completion_v3 import GenericCompletionRequest

from .comparison_harness import (
    HORIZONS, LLM_OUTCOMES_T2D,
    LABS, MEDS, CONDITIONS, DEMOGRAPHICS, SDOH,
    RANDOM_STATE, make_comparison_split, get_outcome_exclusions,
)

logger = logging.getLogger(__name__)

LSTM_DATASET = "ri.foundry.main.dataset.872815d9-3093-4ccb-ade5-2d44b1c2edd5"
NB4_RESULTS = "ri.foundry.main.dataset.fc262d54-6200-4203-b994-7fd1246f477c"
OUTPUT_BASE = "/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d"
SONNET_RID = "ri.language-model-service..language-model.anthropic-claude-4-6-sonnet"

ALL_OUTCOMES = LLM_OUTCOMES_T2D

OUTCOME_DESCRIPTIONS = {
    "OUTCOME_Optimal_Glycemic_Control": "Will this patient maintain optimal glycemic control (HbA1c < 7%)?",
    "OUTCOME_Hypertension": "Will this patient develop hypertension?",
    "OUTCOME_Microalbuminuria": "Will this patient develop microalbuminuria (early kidney disease)?",
    "OUTCOME_Dyslipidemia": "Will this patient have dyslipidemia?",
    "OUTCOME_Insulin_Independence": "Will this patient maintain insulin independence (off insulin with A1c<7%, no DKA)?",
    "OUTCOME_Metformin_Response": "Will this patient respond to metformin monotherapy (A1c<7% on metformin alone)?",
    "OUTCOME_GLP1RA_Response": "Will this patient respond to GLP-1 receptor agonist therapy (A1c<7%)?",
}

N_BOOTSTRAP = 200
MAX_PARALLEL_PATIENTS = 8       # concurrent patient processing (I/O-bound LLM calls)


# ── LLM Wrapper (thread-safe) ──

class LLMWrapper:
    """Thread-safe LLM wrapper with cost tracking."""
    def __init__(self, llm_instance):
        self.llm = llm_instance
        self._lock = threading.Lock()
        self.total_calls = 0
        self.total_latency = 0.0
        self.est_input_tokens = 0
        self.est_output_tokens = 0

    def call(self, system_prompt, user_prompt, max_tokens=600, temperature=0.0):
        prompt = system_prompt + "\n\n" + user_prompt
        request = GenericCompletionRequest(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
        start = time.time()
        try:
            response = self.llm.create_completion(request)
            output = response.completion or ""
            elapsed = time.time() - start
            with self._lock:
                self.total_calls += 1
                self.total_latency += elapsed
                self.est_input_tokens += len(prompt) // 4
                self.est_output_tokens += len(output) // 4
            return output
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            with self._lock:
                self.total_latency += time.time() - start
            return None

    def reset(self):
        with self._lock:
            self.total_calls = 0
            self.total_latency = 0.0
            self.est_input_tokens = 0
            self.est_output_tokens = 0


# ── Prompt Builders (T2D: no CGM) ──

def format_features(patient_feats, visits, subset=None, exclude=None):
    visit_labels = {f"v{i}": f"{i*6}mo" for i in range(1, 11)}
    lines = []
    for feat, vals in sorted(patient_feats.items()):
        if subset and feat not in subset:
            continue
        if exclude and feat in exclude:
            continue
        parts = []
        for v in visits:
            val = vals.get(v)
            if val is not None and not np.isnan(val):
                parts.append(f"{visit_labels[v]}={val:.2f}")
        if parts:
            lines.append(f"  {feat}: {', '.join(parts)}")
    return "\n".join(lines) if lines else "  No data"


def specialist_prompt(agent_type, patient_feats, visits, outcome_desc, exclude=None):
    roles = {
        "lab": ("Expert pediatric endocrinologist analyzing labs/vitals.", set(LABS)),
        "medication": ("Clinical pharmacologist analyzing diabetes medications.", set(MEDS)),
        "sdoh": ("Social determinants of health expert.", set(DEMOGRAPHICS + SDOH)),
        "severity": ("Disease severity specialist.", set(CONDITIONS)),
        "ehr": ("Expert pediatric endocrinologist analyzing EHR clinical data (labs, medications, conditions).",
                set(LABS + MEDS + CONDITIONS)),
        "sdoh_full": ("Social determinants of health and demographics expert.",
                      set(DEMOGRAPHICS + SDOH)),
    }
    role, feats = roles[agent_type]
    if exclude:
        feats = feats - exclude
    data = format_features(patient_feats, visits, feats, exclude=exclude)
    sys = f"You are a {role} Provide a brief assessment relevant to: {outcome_desc}"
    usr = (f"Patient data:\n{data}\n\n"
           f"Respond: 1) Key findings 2) Risk: LOW/MODERATE/HIGH 3) Confidence: 0.0-1.0 4) Brief rationale")
    return sys, usr


def synthesis_prompt(sub_assessments, outcome_desc):
    parts = "\n\n".join(f"--- {k} ---\n{v}" for k, v in sub_assessments.items() if v)
    sys = "You are a senior pediatric endocrinologist synthesizing specialist assessments for a final prediction."
    usr = (f"Clinical question: {outcome_desc}\n\nSpecialist assessments:\n{parts}\n\n"
           f'Respond in JSON: {{"prediction": "YES" or "NO", "confidence": 0.0-1.0, "rationale": "brief"}}')
    return sys, usr


def single_prompt(patient_feats, visits, outcome_desc, cot=False, exclude=None):
    data = format_features(patient_feats, visits, exclude=exclude)
    sys = "You are an expert pediatric endocrinologist specializing in Type 2 Diabetes prediction."
    cot_text = ("Think step by step:\n1. Metabolic indicators (labs, vitals)?\n"
                "2. Treatment trajectory (medications)?\n"
                "3. Social risk factors (SDOH, demographics)?\n"
                "4. Disease severity (complications)?\n\n") if cot else ""
    usr = (f"Predict: {outcome_desc}\n\nPatient data:\n{data}\n\n{cot_text}"
           f'Respond in JSON: {{"prediction": "YES" or "NO", "confidence": 0.0-1.0, "rationale": "brief"}}')
    return sys, usr


def parse_response(text):
    if not text:
        return None, 0.5
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            d = json.loads(text[start:end])
            pred = 1 if str(d.get("prediction", "")).upper() in ("YES", "1", "TRUE") else 0
            return pred, float(d.get("confidence", 0.5))
    except (json.JSONDecodeError, ValueError):
        pass
    upper = text.upper()
    if '"YES"' in upper:
        return 1, 0.6
    if '"NO"' in upper:
        return 0, 0.6
    return None, 0.5


# ── Agent Pipelines ──

def run_multi_agent(llm, patient_feats, visits, outcome_desc, agents, exclude=None):
    subs = {}
    for a in agents:
        s, u = specialist_prompt(a, patient_feats, visits, outcome_desc, exclude=exclude)
        subs[a] = llm.call(s, u, max_tokens=400)
    s, u = synthesis_prompt(subs, outcome_desc)
    resp = llm.call(s, u, max_tokens=300)
    return parse_response(resp)


def get_patient_feats(df, mrn, visits):
    pdf = df[df["mrn"] == mrn]
    feats = {}
    for _, row in pdf.iterrows():
        feats[row["feature"]] = {v: row[v] for v in visits}
    return feats


def bootstrap_ci(y_true, y_scores, fn, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(fn(y_true[idx], y_scores[idx]))
        except (ValueError, ZeroDivisionError):
            pass
    if not scores:
        return np.nan, np.nan, np.nan
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


# ── Config Definitions (T2D: no CGM agent) ──

CONFIGS = {
    "model_a": {"family": "single_llm", "llm": True, "cot": False,
                "description": "Single-pass LLM — sees all features, direct prediction"},
    "model_b": {"family": "multi_agent", "llm": True, "spec": "by_modality",
                "topo": "parallel_synthesis", "synth": "llm_synthesis", "know": "domain_specific",
                "agents": ["ehr", "sdoh_full"],
                "description": "Multi-agent — EHR + SDOH specialists → synthesizer"},
    "model_c": {"family": "single_llm_cot", "llm": True, "cot": True,
                "description": "Single-pass Chain-of-Thought — structured step-by-step reasoning"},
    "model_a_no_sdoh": {"family": "single_llm", "llm": True, "cot": False,
                        "ablation_exclude": set(DEMOGRAPHICS + SDOH),
                        "description": "Ablation: Single-pass LLM without SDOH features"},
}


@configure(["DYNAMIC_ALLOCATION_ENABLED_8_16"])
@transform(
    results_out=Output(f"{OUTPUT_BASE}/nb5_agentic_results_t2d"),
    ablation_out=Output(f"{OUTPUT_BASE}/nb5_ablation_index_t2d"),
    cost_out=Output(f"{OUTPUT_BASE}/nb5_cost_latency_t2d"),
    interp_scores_out=Output(f"{OUTPUT_BASE}/nb5_interpretability_scores_t2d"),
    interp_agree_out=Output(f"{OUTPUT_BASE}/nb5_interpretability_agreement_t2d"),
    lstm_data=Input(LSTM_DATASET),
    nb4_results=Input(NB4_RESULTS),
    llm_sonnet=GenericCompletionLanguageModelInput(SONNET_RID),
)
def compute(ctx, lstm_data, nb4_results, llm_sonnet, results_out, ablation_out,
            cost_out, interp_scores_out, interp_agree_out):

    df = lstm_data.dataframe().toPandas()
    llm = LLMWrapper(llm_sonnet)

    all_results = []
    cost_data = []

    for hz_name, hz_cfg in HORIZONS.items():
        tv = hz_cfg["target_visit"]
        iv = hz_cfg["input_visits"]

        for outcome in ALL_OUTCOMES:
            logger.info(f"{'='*50}\n  {hz_name} | {outcome}")

            collinear_excl = get_outcome_exclusions(outcome)
            if collinear_excl:
                logger.info(f"  Excluding collinear features: {collinear_excl}")

            train_mrns, test_mrns, y_all = make_comparison_split(df, outcome, tv)
            if test_mrns is None:
                logger.info(f"  SKIP: too small / single-class")
                continue
            mrns = test_mrns
            y = y_all.loc[mrns].values
            logger.info(f"  n_test={len(y)}, n_pos={int(y.sum())} ({y.mean():.1%})")

            outcome_desc = OUTCOME_DESCRIPTIONS[outcome]

            for cid, cfg in CONFIGS.items():
                llm.reset()

                combined_exclude = set(collinear_excl)
                if "ablation_exclude" in cfg:
                    combined_exclude = combined_exclude | cfg["ablation_exclude"]

                # ── Parallel patient processing (I/O-bound LLM calls) ──
                def process_patient(mrn):
                    pf = get_patient_feats(df, mrn, iv)
                    if "agents" in cfg:
                        pred, conf = run_multi_agent(llm, pf, iv, outcome_desc, cfg["agents"],
                                                    exclude=combined_exclude)
                    else:
                        s, u = single_prompt(pf, iv, outcome_desc,
                                             cot=cfg.get("cot", False),
                                             exclude=combined_exclude)
                        resp = llm.call(s, u)
                        pred, conf = parse_response(resp)
                    return mrn, (pred if pred is not None else 0), conf

                results_map = {}
                with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PATIENTS) as executor:
                    futures = {executor.submit(process_patient, mrn): mrn for mrn in mrns}
                    done_count = 0
                    for future in as_completed(futures):
                        mrn, pred, conf = future.result()
                        results_map[mrn] = (pred, conf)
                        done_count += 1
                        if done_count % 25 == 0:
                            logger.info(f"    {cid}: {done_count}/{len(mrns)} patients done")

                # Preserve original order for metrics
                predictions = [results_map[mrn][0] for mrn in mrns]
                confidences = [results_map[mrn][1] for mrn in mrns]

                y_pred = np.array(predictions)
                y_conf = np.array(confidences)

                auc_m, auc_lo, auc_hi = bootstrap_ci(y, y_conf, roc_auc_score)
                pr_m, pr_lo, pr_hi = bootstrap_ci(y, y_conf, average_precision_score)
                brier_m, brier_lo, brier_hi = bootstrap_ci(y, y_conf, brier_score_loss)
                try:
                    mcc = matthews_corrcoef(y, y_pred)
                except Exception:
                    mcc = None

                all_results.append({
                    "config_id": cid, "config_label": cid.replace("_", " ").title(),
                    "method_family": cfg["family"],
                    "specialization": cfg.get("spec"), "topology": cfg.get("topo"),
                    "synthesis": cfg.get("synth"), "knowledge": cfg.get("know"),
                    "agents_active": str(cfg.get("agents", [])),
                    "is_default": cid == "model_b",
                    "outcome": outcome, "horizon": hz_name,
                    "roc_auc_mean": auc_m, "roc_auc_ci_low": auc_lo, "roc_auc_ci_high": auc_hi,
                    "pr_auc_mean": pr_m, "pr_auc_ci_low": pr_lo, "pr_auc_ci_high": pr_hi,
                    "brier_mean": brier_m, "brier_ci_low": brier_lo, "brier_ci_high": brier_hi,
                    "mcc_mean": mcc, "mcc_ci_low": None, "mcc_ci_high": None,
                    "n_samples": len(y), "n_positive": int(y.sum()),
                    "prevalence": float(y.mean()), "n_folds": 1,
                })

                cost_data.append({
                    "config_id": cid, "config_label": cid.replace("_", " ").title(),
                    "method_family": cfg["family"],
                    "cost_usd_per_patient": round(
                        (llm.est_input_tokens * 3 / 1e6 + llm.est_output_tokens * 15 / 1e6) / max(len(mrns), 1), 4),
                    "latency_sec_per_patient": round(llm.total_latency / max(len(mrns), 1), 2),
                    "total_tokens_per_patient": (llm.est_input_tokens + llm.est_output_tokens) // max(len(mrns), 1),
                    "input_tokens": llm.est_input_tokens,
                    "output_tokens": llm.est_output_tokens,
                    "n_llm_calls": llm.total_calls,
                    "outcome": outcome, "horizon": hz_name,
                })

                logger.info(f"    {cid:25s} AUC={auc_m:.3f} calls={llm.total_calls}")

    spark = ctx.spark_session
    res_df = pd.DataFrame(all_results)
    str_cols = ["config_id", "config_label", "method_family", "specialization", "topology",
                "synthesis", "knowledge", "agents_active", "outcome", "horizon"]
    float_cols = ["roc_auc_mean", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc_mean",
                  "pr_auc_ci_low", "pr_auc_ci_high", "brier_mean", "brier_ci_low", "brier_ci_high",
                  "mcc_mean", "mcc_ci_low", "mcc_ci_high", "prevalence"]
    int_cols = ["n_samples", "n_positive", "n_folds"]
    bool_cols = ["is_default"]
    for c in str_cols:
        if c in res_df.columns:
            res_df[c] = res_df[c].astype(str).replace("None", "")
    for c in float_cols:
        if c in res_df.columns:
            res_df[c] = pd.to_numeric(res_df[c], errors="coerce")
    for c in int_cols:
        if c in res_df.columns:
            res_df[c] = pd.to_numeric(res_df[c], errors="coerce").fillna(0).astype(int)
    for c in bool_cols:
        if c in res_df.columns:
            res_df[c] = res_df[c].fillna(False).astype(bool)
    results_out.write_dataframe(spark.createDataFrame(res_df))

    abl = []
    for dim in ["architecture", "modality_ablation"]:
        if dim == "architecture":
            configs = ["model_a", "model_b", "model_c"]
        elif dim == "modality_ablation":
            configs = ["model_a", "model_a_no_sdoh"]
        for i, c in enumerate(configs):
            abl.append({"ablation_dimension": dim, "config_id": c,
                        "variant_label": c, "variant_order": i, "is_reference": c == "model_a"})
    ablation_out.write_dataframe(spark.createDataFrame(pd.DataFrame(abl)))

    cost_out.write_dataframe(spark.createDataFrame(pd.DataFrame(cost_data)) if cost_data
                             else spark.createDataFrame(pd.DataFrame(columns=["config_id"])))

    interp_scores_out.write_dataframe(spark.createDataFrame(pd.DataFrame({
        "config_id": ["model_b", "model_c"] * 4,
        "criterion": ["factual_accuracy"] * 2 + ["clinical_plausibility"] * 2 +
                     ["completeness"] * 2 + ["actionability"] * 2,
        "rater_id": ["placeholder"] * 8,
        "mean_score": [float("nan")] * 8, "sd_score": [float("nan")] * 8, "n_items": [0] * 8,
    })))

    interp_agree_out.write_dataframe(spark.createDataFrame(pd.DataFrame({
        "criterion": ["factual_accuracy", "clinical_plausibility", "completeness", "actionability", "overall"],
        "kappa_type": ["fleiss"] * 5,
        "kappa": [float("nan")] * 5, "kappa_ci_low": [float("nan")] * 5,
        "kappa_ci_high": [float("nan")] * 5, "n_items": [0] * 5, "n_raters": [0] * 5,
    })))

    logger.info(f"\nNB5 COMPLETE (T2D) — {len(all_results)} experiments, {len(cost_data)} cost entries")


"""
NB6 — Model D: Evidence-Grounded Deliberative Ensemble (T2D)
=============================================================
Builds on Model C and adds evidence-grounding, verification, and
self-consistency calibration.

T2D key differences from T1D:
  - No CGM features
  - All 7 outcomes eligible for LLM
  - System prompts reference "Type 2 Diabetes"
  - Guideline snippets reflect T2D clinical context

See T1D nb6_model_d.py for full architecture documentation.
"""

import json
import re
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, matthews_corrcoef
from transforms.api import Input, Output, transform, configure
from palantir_models.transforms import GenericCompletionLanguageModelInput
from palantir_models.models import GenericCompletionLanguageModel
from language_model_service_api.languagemodelservice_api_completion_v3 import GenericCompletionRequest

from .comparison_harness import (
    HORIZONS, LLM_OUTCOMES_T2D,
    LABS, MEDS, CONDITIONS, DEMOGRAPHICS, SDOH,
    RANDOM_STATE, make_comparison_split, get_outcome_exclusions,
)

logger = logging.getLogger(__name__)

LSTM_DATASET = "ri.foundry.main.dataset.872815d9-3093-4ccb-ade5-2d44b1c2edd5"
OUTPUT_BASE = "/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d"
SONNET_RID = "ri.language-model-service..language-model.anthropic-claude-4-6-sonnet"

ALL_OUTCOMES = LLM_OUTCOMES_T2D
N_BOOTSTRAP = 200
K_SELF_CONSISTENCY = 3          # reduced from 5 for efficiency
SC_TEMPERATURE = 0.7
FAITHFULNESS_SUBSAMPLE = 10     # reduced from 20 for efficiency
PRED_CHANGE_THRESHOLD = 0.10
MAX_PARALLEL_PATIENTS = 8       # concurrent patient processing
GROUP_MASK_K_VALUES = [1, 3, 5, 10]  # cumulative top-K group-masking curve points

OUTCOME_DESCRIPTIONS = {
    "OUTCOME_Optimal_Glycemic_Control": "Will this patient maintain optimal glycemic control (HbA1c < 7%)?",
    "OUTCOME_Hypertension": "Will this patient develop hypertension?",
    "OUTCOME_Microalbuminuria": "Will this patient develop microalbuminuria (early kidney disease)?",
    "OUTCOME_Dyslipidemia": "Will this patient have dyslipidemia?",
    "OUTCOME_Insulin_Independence": "Will this patient maintain insulin independence (off insulin with A1c<7%, no DKA)?",
    "OUTCOME_Metformin_Response": "Will this patient respond to metformin monotherapy (A1c<7% on metformin alone)?",
    "OUTCOME_GLP1RA_Response": "Will this patient respond to GLP-1 receptor agonist therapy (A1c<7%)?",
}

# ── ADA / ISPAD Guideline Snippets (T2D-specific) ──

GUIDELINE_SNIPPETS = {
    "OUTCOME_Optimal_Glycemic_Control": [
        "ADA 2024: Target HbA1c <7% for youth with T2D; metformin + insulin are first-line therapies.",
        "ADA 2024: Youth-onset T2D is more aggressive than adult-onset; beta-cell decline is faster.",
        "ISPAD 2022: Intensive lifestyle intervention should accompany pharmacotherapy in pediatric T2D.",
    ],
    "OUTCOME_Hypertension": [
        "ADA 2024: Screen BP at every visit; hypertension in T2D youth often co-occurs with obesity.",
        "ADA 2024: Elevated BMI z-score is the strongest modifiable risk factor for hypertension.",
        "Weight management is primary intervention; ACEi/ARB if BP ≥95th percentile persists.",
    ],
    "OUTCOME_Microalbuminuria": [
        "ADA 2024: Screen UACR at T2D diagnosis and annually thereafter (earlier than T1D).",
        "ISPAD 2022: Persistent microalbuminuria warrants ACEi/ARB; glycemic control is protective.",
        "Obesity and hypertension accelerate nephropathy progression in youth T2D.",
    ],
    "OUTCOME_Dyslipidemia": [
        "ADA 2024: Screen fasting lipid panel at diagnosis; T2D youth have high dyslipidemia prevalence.",
        "ISPAD 2022: LDL >130 mg/dL warrants statin consideration in children >10 yr.",
        "Insulin resistance drives the atherogenic lipid triad (high TG, low HDL, small dense LDL).",
    ],
    "OUTCOME_Insulin_Independence": [
        "ADA 2024: Some T2D youth may transition off insulin with improved glycemic control and weight loss.",
        "Preserved beta-cell function (C-peptide) and shorter diabetes duration predict insulin independence.",
    ],
    "OUTCOME_Metformin_Response": [
        "ADA 2024: Metformin is first-line for T2D; response defined as A1c <7% on monotherapy.",
        "TODAY study: ~50% of youth with T2D lose glycemic control on metformin alone within 3 years.",
        "Lower baseline A1c and shorter disease duration predict better metformin response.",
    ],
    "OUTCOME_GLP1RA_Response": [
        "ADA 2024: GLP-1 RAs (liraglutide, exenatide) are approved adjuncts for youth T2D.",
        "ELLIPSE trial: Liraglutide significantly improved A1c vs placebo in youth T2D.",
        "Weight reduction and improved postprandial glucose are primary benefits.",
    ],
}


# ── LLM Wrapper (thread-safe) ──

class LLMWrapper:
    """Thread-safe LLM wrapper with cost tracking."""
    def __init__(self, llm_instance):
        self.llm = llm_instance
        self._lock = threading.Lock()
        self.total_calls = 0
        self.total_latency = 0.0
        self.est_input_tokens = 0
        self.est_output_tokens = 0

    def call(self, system_prompt, user_prompt, max_tokens=800, temperature=0.0):
        prompt = system_prompt + "\n\n" + user_prompt
        request = GenericCompletionRequest(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
        start = time.time()
        try:
            response = self.llm.create_completion(request)
            output = response.completion or ""
            elapsed = time.time() - start
            with self._lock:
                self.total_calls += 1
                self.total_latency += elapsed
                self.est_input_tokens += len(prompt) // 4
                self.est_output_tokens += len(output) // 4
            return output
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            with self._lock:
                self.total_latency += time.time() - start
            return None

    def reset(self):
        with self._lock:
            self.total_calls = 0
            self.total_latency = 0.0
            self.est_input_tokens = 0
            self.est_output_tokens = 0


# ── Feature Formatting ──

def format_features(patient_feats, visits, exclude=None):
    visit_labels = {f"v{i}": f"{i*6}mo" for i in range(1, 11)}
    lines = []
    for feat, vals in sorted(patient_feats.items()):
        if exclude and feat in exclude:
            continue
        parts = []
        for v in visits:
            val = vals.get(v)
            if val is not None and not np.isnan(val):
                # Include the visit token (v1/v2/...) AND the time label so the model
                # can cite the exact visit id used for extractive grounding.
                parts.append(f"{v}({visit_labels[v]})={val:.2f}")
        if parts:
            lines.append(f"  {feat}: {', '.join(parts)}")
    return "\n".join(lines) if lines else "  No data"


def get_patient_feats(df, mrn, visits):
    pdf = df[df["mrn"] == mrn]
    feats = {}
    for _, row in pdf.iterrows():
        feats[row["feature"]] = {v: row[v] for v in visits}
    return feats


def get_guideline_context(outcome):
    snippets = GUIDELINE_SNIPPETS.get(outcome, [])
    if not snippets:
        return ""
    text = "\n".join(f"  - {s}" for s in snippets)
    return f"\nRelevant clinical guidelines:\n{text}\n"


# ── Stage 1: Evidence-Grounded CoT Predictor ──

def stage1_prompt(patient_feats, visits, outcome_desc, guideline_ctx, exclude=None):
    data = format_features(patient_feats, visits, exclude=exclude)
    sys = (
        "You are an expert pediatric endocrinologist specializing in Type 2 Diabetes prediction. "
        "You must provide evidence-grounded predictions with specific data citations."
    )
    usr = (
        f"Clinical question: {outcome_desc}\n"
        f"{guideline_ctx}\n"
        f"Patient data (each feature shows value at each visit window, e.g. v3(18mo)=8.50):\n{data}\n\n"
        "Think step by step through each modality (labs/vitals, medications, conditions, SDOH).\n\n"
        "Then respond in EXACTLY this plain-text structured format (do NOT use JSON, do NOT use "
        "markdown tables). Put PROBABILITY and EVIDENCE first so they are never cut off:\n\n"
        "PROBABILITY: <a single number between 0.0 and 1.0>\n\n"
        "EVIDENCE:\n"
        "(one line per data point that drives your prediction, pipe-delimited. Copy the FEATURE "
        "name and the VISIT token and the numeric VALUE EXACTLY as they appear in the patient data "
        "above. DIRECTION is 'increases_risk' or 'decreases_risk'. IMPORTANCE is 0.0-1.0.)\n"
        "FEATURE | VISIT | VALUE | DIRECTION | IMPORTANCE\n"
        "HBA1C | v3 | 8.50 | increases_risk | 0.40\n"
        "<additional evidence lines...>\n\n"
        "REASONING: <your concise clinical chain of thought>\n"
    )
    return sys, usr


def parse_stage1_response(text):
    """Parse Stage 1 response returning (probability, evidence_list, reasoning).
    Primary format is plain-text delimited; JSON and regex are kept as fallbacks.
    Each evidence item is a dict: feature, visit, value, direction, importance."""
    if not text:
        return 0.5, [], ""

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    # ── Probability (try PROBABILITY: line first, then JSON key) ──
    prob = 0.5
    pm = re.search(r'PROBABILITY\s*[:=]\s*\*{0,2}\s*([01]?\.?[0-9]+)', cleaned, re.IGNORECASE)
    if not pm:
        pm = re.search(r'"?probability"?\s*[:=]\s*([01]?\.?[0-9]+)', cleaned, re.IGNORECASE)
    if pm:
        try:
            prob = max(0.0, min(1.0, float(pm.group(1))))
        except ValueError:
            pass

    # ── Evidence: pipe-delimited lines (primary format) ──
    evidence = []
    for line in cleaned.splitlines():
        if line.count("|") < 2:
            continue
        parts = [p.strip() for p in line.split("|")]
        feat = parts[0].lstrip("-*• ").strip()
        if not feat or feat.upper() in ("FEATURE", "FEATURE NAME"):
            continue  # skip header / non-evidence rows
        visit_m = re.search(r'v(\d+)', parts[1], re.IGNORECASE)
        val_m = re.search(r'(-?[0-9]*\.?[0-9]+)', parts[2])
        if not visit_m or not val_m:
            continue
        try:
            value = float(val_m.group(1))
        except ValueError:
            continue
        direction = parts[3].strip().lower() if len(parts) > 3 else ""
        importance = 0.0
        if len(parts) > 4:
            imp_m = re.search(r'([0-9]*\.?[0-9]+)', parts[4])
            if imp_m:
                try:
                    importance = max(0.0, min(1.0, float(imp_m.group(1))))
                except ValueError:
                    pass
        evidence.append({
            "feature": feat, "visit": f"v{visit_m.group(1)}", "value": value,
            "direction": direction, "importance": importance, "interpretation": "",
        })

    # ── Fallback: JSON-style evidence (if model ignored the delimited format) ──
    if not evidence:
        for m in re.finditer(
            r'"feature"\s*:\s*"([^"]+)"[^}]*?"visit"\s*:\s*"?(v?\d+)"?[^}]*?"value"\s*:\s*(-?[0-9.eE+]+)',
            cleaned):
            try:
                vis = m.group(2)
                vis = vis if vis.lower().startswith("v") else f"v{vis}"
                evidence.append({
                    "feature": m.group(1), "visit": vis, "value": float(m.group(3)),
                    "direction": "", "importance": 0.0, "interpretation": "",
                })
            except (ValueError, IndexError):
                pass

    # ── Reasoning ──
    reasoning = ""
    rm = re.search(r'REASONING\s*[:=]\s*(.+)', cleaned, re.IGNORECASE | re.DOTALL)
    if rm:
        reasoning = rm.group(1).strip()[:2000]

    return prob, evidence, reasoning


# ── Stage 2: Verifier/Critic Agent ──

def stage2_prompt(stage1_output, patient_feats, visits, outcome_desc, exclude=None):
    data = format_features(patient_feats, visits, exclude=exclude)
    sys = (
        "You are a clinical evidence auditor. Your job is to verify the accuracy of "
        "cited evidence in a clinical prediction. Check each cited data point against "
        "the actual patient data. Challenge any weak or incorrect citations."
    )
    usr = (
        f"Clinical question: {outcome_desc}\n\n"
        f"PREDICTOR'S OUTPUT:\n{stage1_output}\n\n"
        f"ACTUAL PATIENT DATA:\n{data}\n\n"
        "For each cited evidence point:\n"
        "1. Verify if the cited (feature, visit, value) matches the actual data\n"
        "2. Flag any hallucinated or incorrect citations\n"
        "3. If evidence is weak or incorrect, provide a revised assessment\n\n"
        "Respond in JSON:\n"
        '{\n'
        '  "verified_evidence": [{"feature": "<name>", "visit": "<vN>", "cited_value": <n>, '
        '"actual_value": <n or null>, "is_correct": <bool>}, ...],\n'
        '  "revision_needed": <bool>,\n'
        '  "revised_probability": <float 0.0-1.0 or null if no revision needed>,\n'
        '  "revised_reasoning": "<brief or null>"\n'
        '}'
    )
    return sys, usr


def parse_stage2_response(text):
    """Parse verifier response with regex fallback."""
    if not text:
        return [], False, None, ""

    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            d = json.loads(cleaned[start:end])
            verified = d.get("verified_evidence", [])
            if not isinstance(verified, list):
                verified = []
            revision_needed = bool(d.get("revision_needed", False))
            revised_prob = d.get("revised_probability")
            if revised_prob is not None:
                revised_prob = max(0.0, min(1.0, float(revised_prob)))
            revised_reasoning = str(d.get("revised_reasoning", ""))
            return verified, revision_needed, revised_prob, revised_reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    revision_match = re.search(r'"revision_needed"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
    revision_needed = revision_match and revision_match.group(1).lower() == "true"
    revised_prob = None
    prob_match = re.search(r'"revised_probability"\s*:\s*([0-9]*\.?[0-9]+)', cleaned)
    if prob_match and revision_needed:
        try:
            revised_prob = max(0.0, min(1.0, float(prob_match.group(1))))
        except ValueError:
            pass
    return [], revision_needed, revised_prob, ""


# ── Stage 3: Self-Consistency Calibration ──

def run_self_consistency(llm, sys_prompt, usr_prompt, k=K_SELF_CONSISTENCY, temp=SC_TEMPERATURE):
    probs = []
    all_evidence = []
    for _ in range(k):
        resp = llm.call(sys_prompt, usr_prompt, max_tokens=2000, temperature=temp)
        prob, evidence, _ = parse_stage1_response(resp)
        probs.append(prob)
        all_evidence.append(evidence)
    if not probs:
        return 0.5, []
    return float(np.mean(probs)), all_evidence


# ── Full Model D Pipeline ──

def run_model_d_full(llm, patient_feats, visits, outcome_desc, outcome, exclude=None):
    guideline_ctx = get_guideline_context(outcome)
    stage_costs = {}

    # Stage 1
    llm_before = (llm.total_calls, llm.est_input_tokens, llm.est_output_tokens, llm.total_latency)
    sys1, usr1 = stage1_prompt(patient_feats, visits, outcome_desc, guideline_ctx, exclude=exclude)
    resp1 = llm.call(sys1, usr1, max_tokens=2000, temperature=0.0)
    prob1, evidence1, reasoning1 = parse_stage1_response(resp1)
    stage_costs["stage1"] = {
        "calls": llm.total_calls - llm_before[0],
        "input_tokens": llm.est_input_tokens - llm_before[1],
        "output_tokens": llm.est_output_tokens - llm_before[2],
        "latency": llm.total_latency - llm_before[3],
    }

    # Stage 2
    llm_before = (llm.total_calls, llm.est_input_tokens, llm.est_output_tokens, llm.total_latency)
    sys2, usr2 = stage2_prompt(resp1 or "", patient_feats, visits, outcome_desc, exclude=exclude)
    resp2 = llm.call(sys2, usr2, max_tokens=600, temperature=0.0)
    verified, revision_needed, revised_prob, _ = parse_stage2_response(resp2)
    prob_after_verify = revised_prob if (revision_needed and revised_prob is not None) else prob1
    stage_costs["stage2"] = {
        "calls": llm.total_calls - llm_before[0],
        "input_tokens": llm.est_input_tokens - llm_before[1],
        "output_tokens": llm.est_output_tokens - llm_before[2],
        "latency": llm.total_latency - llm_before[3],
    }

    # Stage 3
    llm_before = (llm.total_calls, llm.est_input_tokens, llm.est_output_tokens, llm.total_latency)
    sc_prob, sc_evidence = run_self_consistency(llm, sys1, usr1, k=K_SELF_CONSISTENCY, temp=SC_TEMPERATURE)
    final_prob = (prob_after_verify + sc_prob) / 2.0
    stage_costs["stage3"] = {
        "calls": llm.total_calls - llm_before[0],
        "input_tokens": llm.est_input_tokens - llm_before[1],
        "output_tokens": llm.est_output_tokens - llm_before[2],
        "latency": llm.total_latency - llm_before[3],
    }

    return final_prob, evidence1, reasoning1, verified, stage_costs


def run_model_c_plus_verify(llm, patient_feats, visits, outcome_desc, outcome, exclude=None):
    guideline_ctx = get_guideline_context(outcome)
    sys1, usr1 = stage1_prompt(patient_feats, visits, outcome_desc, guideline_ctx, exclude=exclude)
    resp1 = llm.call(sys1, usr1, max_tokens=2000, temperature=0.0)
    prob1, _, _ = parse_stage1_response(resp1)
    sys2, usr2 = stage2_prompt(resp1 or "", patient_feats, visits, outcome_desc, exclude=exclude)
    resp2 = llm.call(sys2, usr2, max_tokens=600, temperature=0.0)
    _, revision_needed, revised_prob, _ = parse_stage2_response(resp2)
    return revised_prob if (revision_needed and revised_prob is not None) else prob1


def run_model_c_plus_sc(llm, patient_feats, visits, outcome_desc, outcome, exclude=None):
    guideline_ctx = get_guideline_context(outcome)
    sys1, usr1 = stage1_prompt(patient_feats, visits, outcome_desc, guideline_ctx, exclude=exclude)
    sc_prob, _ = run_self_consistency(llm, sys1, usr1, k=K_SELF_CONSISTENCY, temp=SC_TEMPERATURE)
    return sc_prob


# ── Faithfulness Analysis ──

def verify_extractive_grounding(evidence_list, patient_feats, rel_tol=0.05, abs_tol=0.5):
    """Check each cited (feature, visit, value) against actual data using a
    relative-or-absolute tolerance (whichever is larger). This catches
    hallucinated evidence outright.
    Returns (n_correct, n_total, accuracy, detail_list)."""
    if not evidence_list:
        return 0, 0, 0.0, []
    n_correct = 0
    n_total = 0
    detail = []
    for ev in evidence_list:
        if not isinstance(ev, dict):
            continue
        feat = ev.get("feature", "")
        visit = ev.get("visit", "")
        cited_val = ev.get("value")
        if not feat or not visit or cited_val is None:
            continue
        n_total += 1
        actual_vals = patient_feats.get(feat, {})
        actual_val = actual_vals.get(visit)
        verified = False
        actual_out = None
        if actual_val is not None and not (isinstance(actual_val, float) and np.isnan(actual_val)):
            actual_out = float(actual_val)
            try:
                if abs(float(cited_val) - actual_out) <= max(abs_tol, rel_tol * abs(actual_out)):
                    verified = True
                    n_correct += 1
            except (ValueError, TypeError):
                pass
        detail.append({
            "feature": feat, "visit": visit,
            "cited_value": float(cited_val) if cited_val is not None else None,
            "actual_value": actual_out, "verified": verified,
            "importance": float(ev.get("importance", 0.0) or 0.0),
            "direction": ev.get("direction", ""),
        })
    accuracy = n_correct / n_total if n_total > 0 else 0.0
    return n_correct, n_total, accuracy, detail


def run_counterfactual_faithfulness(llm, patient_feats, visits, outcome_desc, outcome,
                                   evidence_list, baseline_prob, exclude=None):
    if not evidence_list:
        return 0.0, 0.0, []

    cited_features = {}
    for ev in evidence_list:
        if isinstance(ev, dict) and ev.get("feature"):
            f = ev["feature"]
            cited_features[f] = max(cited_features.get(f, 0.0), float(ev.get("importance", 0.0) or 0.0))

    if not cited_features:
        return 0.0, 0.0, []

    guideline_ctx = get_guideline_context(outcome)
    deltas = []

    for feat_name, importance in cited_features.items():
        masked_exclude = set(exclude or set()) | {feat_name}
        sys_p, usr_p = stage1_prompt(patient_feats, visits, outcome_desc, guideline_ctx, exclude=masked_exclude)
        resp = llm.call(sys_p, usr_p, max_tokens=2000, temperature=0.0)
        prob_masked, _, _ = parse_stage1_response(resp)
        delta = abs(baseline_prob - prob_masked)
        deltas.append({"feature": feat_name, "importance": importance,
                       "baseline_prob": baseline_prob,
                       "masked_prob": prob_masked, "delta": delta,
                       "prediction_moved": delta >= PRED_CHANGE_THRESHOLD})

    n_moved = sum(1 for d in deltas if d["prediction_moved"])
    comprehensiveness = n_moved / len(deltas) if deltas else 0.0

    all_feats = set(patient_feats.keys())
    non_cited = all_feats - set(cited_features.keys()) - set(exclude or set())
    suff_exclude = set(exclude or set()) | non_cited
    sys_s, usr_s = stage1_prompt(patient_feats, visits, outcome_desc, guideline_ctx, exclude=suff_exclude)
    resp_s = llm.call(sys_s, usr_s, max_tokens=2000, temperature=0.0)
    prob_suff, _, _ = parse_stage1_response(resp_s)
    sufficiency = 1.0 - abs(baseline_prob - prob_suff)

    return sufficiency, comprehensiveness, deltas


def run_group_masking_curve(llm, patient_feats, visits, outcome_desc, outcome,
                            evidence_list, baseline_prob, exclude=None,
                            k_values=GROUP_MASK_K_VALUES):
    """Cumulative top-K group masking: rank cited features by importance, then mask
    the top-K most-important features TOGETHER and measure how far the prediction
    moves. Produces a comprehensiveness-vs-K curve. Returns list of dicts."""
    feat_importance = {}
    for ev in evidence_list:
        if isinstance(ev, dict) and ev.get("feature"):
            f = ev["feature"]
            feat_importance[f] = max(feat_importance.get(f, 0.0),
                                     float(ev.get("importance", 0.0) or 0.0))
    if not feat_importance:
        return []
    ranked = [f for f, _ in sorted(feat_importance.items(), key=lambda kv: kv[1], reverse=True)]

    guideline_ctx = get_guideline_context(outcome)
    rows = []
    seen_sizes = set()
    for k in k_values:
        n_mask = min(k, len(ranked))
        if n_mask in seen_sizes:
            continue  # avoid duplicate calls when k exceeds available features
        seen_sizes.add(n_mask)
        top_k_feats = set(ranked[:n_mask])
        masked_exclude = set(exclude or set()) | top_k_feats
        sys_p, usr_p = stage1_prompt(patient_feats, visits, outcome_desc, guideline_ctx,
                                     exclude=masked_exclude)
        resp = llm.call(sys_p, usr_p, max_tokens=2000, temperature=0.0)
        prob_masked, _, _ = parse_stage1_response(resp)
        delta = abs(baseline_prob - prob_masked)
        rows.append({
            "k": k, "n_masked": n_mask,
            "baseline_prob": baseline_prob, "masked_prob": prob_masked,
            "delta": delta, "prediction_moved": delta >= PRED_CHANGE_THRESHOLD,
        })
    return rows


def bootstrap_ci(y_true, y_scores, fn, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(fn(y_true[idx], y_scores[idx]))
        except (ValueError, ZeroDivisionError):
            pass
    if not scores:
        return np.nan, np.nan, np.nan
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


MODEL_D_CONFIGS = {
    "model_d_full": {
        "description": "Full Model D: knowledge + CoT evidence + verify + self-consistency",
        "runner": "full",
    },
    "model_c_plus_verify": {
        "description": "Ablation: Model C + verification only",
        "runner": "verify_only",
    },
    "model_c_plus_sc": {
        "description": "Ablation: Model C + self-consistency only",
        "runner": "sc_only",
    },
}


@configure(["DYNAMIC_ALLOCATION_ENABLED_8_16"])
@transform(
    results_out=Output(f"{OUTPUT_BASE}/nb6_model_d_results_t2d"),
    cost_out=Output(f"{OUTPUT_BASE}/nb6_model_d_cost_t2d"),
    faithfulness_out=Output(f"{OUTPUT_BASE}/nb6_faithfulness_results_t2d"),
    evidence_detail_out=Output(f"{OUTPUT_BASE}/nb6_evidence_detail_t2d"),
    group_masking_out=Output(f"{OUTPUT_BASE}/nb6_group_masking_curve_t2d"),
    ablation_out=Output(f"{OUTPUT_BASE}/nb6_ablation_index_t2d"),
    lstm_data=Input(LSTM_DATASET),
    llm_sonnet=GenericCompletionLanguageModelInput(SONNET_RID),
)
def compute(ctx, lstm_data, llm_sonnet, results_out, cost_out, faithfulness_out,
            evidence_detail_out, group_masking_out, ablation_out):

    df = lstm_data.dataframe().toPandas()
    llm = LLMWrapper(llm_sonnet)

    all_results = []
    cost_data = []
    faithfulness_data = []
    evidence_detail_data = []
    group_masking_data = []

    for hz_name, hz_cfg in HORIZONS.items():
        tv = hz_cfg["target_visit"]
        iv = hz_cfg["input_visits"]

        for outcome in ALL_OUTCOMES:
            logger.info(f"{'='*50}\n  NB6 {hz_name} | {outcome}")

            collinear_excl = get_outcome_exclusions(outcome)
            if collinear_excl:
                logger.info(f"  Excluding collinear features: {collinear_excl}")

            train_mrns, test_mrns, y_all = make_comparison_split(df, outcome, tv)
            if test_mrns is None:
                logger.info("  SKIP: too small / single-class")
                continue
            mrns = test_mrns
            y = y_all.loc[mrns].values
            logger.info(f"  n_test={len(y)}, n_pos={int(y.sum())} ({y.mean():.1%})")

            outcome_desc = OUTCOME_DESCRIPTIONS.get(outcome, outcome)

            rng = np.random.RandomState(RANDOM_STATE)
            faith_indices = rng.choice(len(mrns), min(FAITHFULNESS_SUBSAMPLE, len(mrns)), replace=False)
            faith_mrns = set(mrns[i] for i in faith_indices)

            for cid, cfg in MODEL_D_CONFIGS.items():
                llm.reset()
                per_patient_evidence = {}

                # ── Parallel patient processing (I/O-bound LLM calls) ──
                def process_patient(mrn):
                    pf = get_patient_feats(df, mrn, iv)
                    if cfg["runner"] == "full":
                        prob, evidence, reasoning, verified, _ = run_model_d_full(
                            llm, pf, iv, outcome_desc, outcome, exclude=collinear_excl)
                        return mrn, prob, evidence, reasoning, pf
                    elif cfg["runner"] == "verify_only":
                        prob = run_model_c_plus_verify(
                            llm, pf, iv, outcome_desc, outcome, exclude=collinear_excl)
                    elif cfg["runner"] == "sc_only":
                        prob = run_model_c_plus_sc(
                            llm, pf, iv, outcome_desc, outcome, exclude=collinear_excl)
                    else:
                        prob = 0.5
                    return mrn, prob, None, None, None

                results_map = {}
                with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PATIENTS) as executor:
                    futures = {executor.submit(process_patient, mrn): mrn for mrn in mrns}
                    done_count = 0
                    for future in as_completed(futures):
                        mrn, prob, evidence, reasoning, pf = future.result()
                        results_map[mrn] = prob
                        if evidence is not None and mrn in faith_mrns:
                            per_patient_evidence[mrn] = (prob, evidence, reasoning, pf)
                        done_count += 1
                        if done_count % 25 == 0:
                            logger.info(f"    {cid}: {done_count}/{len(mrns)} patients done")

                # Preserve original order for metrics
                probabilities = [results_map[mrn] for mrn in mrns]

                y_prob = np.array(probabilities)
                y_pred = (y_prob >= 0.5).astype(int)

                auc_m, auc_lo, auc_hi = bootstrap_ci(y, y_prob, roc_auc_score)
                pr_m, pr_lo, pr_hi = bootstrap_ci(y, y_prob, average_precision_score)
                brier_m, brier_lo, brier_hi = bootstrap_ci(y, y_prob, brier_score_loss)
                try:
                    mcc = matthews_corrcoef(y, y_pred)
                except Exception:
                    mcc = None

                all_results.append({
                    "config_id": cid,
                    "config_label": cfg["description"],
                    "method_family": "deliberative_ensemble",
                    "outcome": outcome, "horizon": hz_name,
                    "roc_auc_mean": auc_m, "roc_auc_ci_low": auc_lo, "roc_auc_ci_high": auc_hi,
                    "pr_auc_mean": pr_m, "pr_auc_ci_low": pr_lo, "pr_auc_ci_high": pr_hi,
                    "brier_mean": brier_m, "brier_ci_low": brier_lo, "brier_ci_high": brier_hi,
                    "mcc_mean": mcc,
                    "n_samples": len(y), "n_positive": int(y.sum()),
                    "prevalence": float(y.mean()),
                })

                cost_data.append({
                    "config_id": cid,
                    "method_family": "deliberative_ensemble",
                    "cost_usd_per_patient": round(
                        (llm.est_input_tokens * 3 / 1e6 + llm.est_output_tokens * 15 / 1e6) / max(len(mrns), 1), 4),
                    "latency_sec_per_patient": round(llm.total_latency / max(len(mrns), 1), 2),
                    "total_tokens_per_patient": (llm.est_input_tokens + llm.est_output_tokens) // max(len(mrns), 1),
                    "input_tokens": llm.est_input_tokens,
                    "output_tokens": llm.est_output_tokens,
                    "n_llm_calls": llm.total_calls,
                    "outcome": outcome, "horizon": hz_name,
                    "k_self_consistency": K_SELF_CONSISTENCY if cfg["runner"] in ("full", "sc_only") else 0,
                })

                logger.info(f"    {cid:30s} AUC={auc_m:.3f} calls={llm.total_calls}")

                if cid == "model_d_full" and per_patient_evidence:
                    logger.info(f"  Running faithfulness analysis on {len(per_patient_evidence)} patients...")

                    def analyze_patient(item):
                        mrn, (base_prob, evidence, reasoning, pf) = item
                        n_correct, n_total, eg_accuracy, eg_detail = verify_extractive_grounding(evidence, pf)
                        sufficiency, comprehensiveness, feat_deltas = run_counterfactual_faithfulness(
                            llm, pf, iv, outcome_desc, outcome, evidence, base_prob,
                            exclude=collinear_excl)
                        group_curve = run_group_masking_curve(
                            llm, pf, iv, outcome_desc, outcome, evidence, base_prob,
                            exclude=collinear_excl)
                        return (mrn, base_prob, reasoning, n_correct, n_total, eg_accuracy,
                                sufficiency, comprehensiveness, eg_detail, feat_deltas, group_curve)

                    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PATIENTS) as fexec:
                        ffutures = [fexec.submit(analyze_patient, it)
                                    for it in per_patient_evidence.items()]
                        for fut in as_completed(ffutures):
                            (mrn, base_prob, reasoning, n_correct, n_total, eg_accuracy,
                             sufficiency, comprehensiveness, eg_detail, feat_deltas,
                             group_curve) = fut.result()

                            delta_by_feat = {d["feature"]: d for d in feat_deltas}

                            # Cumulative top-K group-masking curve rows
                            for gc in group_curve:
                                group_masking_data.append({
                                    "outcome": outcome, "horizon": hz_name, "mrn": str(mrn),
                                    "k": gc["k"], "n_masked": gc["n_masked"],
                                    "baseline_probability": gc["baseline_prob"],
                                    "masked_probability": gc["masked_prob"],
                                    "delta": gc["delta"],
                                    "prediction_moved": bool(gc["prediction_moved"]),
                                })

                            faithfulness_data.append({
                                "outcome": outcome, "horizon": hz_name,
                                "mrn": str(mrn),
                                "extractive_n_correct": n_correct,
                                "extractive_n_total": n_total,
                                "extractive_accuracy": eg_accuracy,
                                "sufficiency_score": sufficiency,
                                "comprehensiveness_score": comprehensiveness,
                                "n_cited_features": len(eg_detail),
                                "n_verified_features": sum(1 for e in eg_detail if e["verified"]),
                                "n_features_moved": sum(
                                    1 for d in feat_deltas if d.get("prediction_moved", False)),
                                "baseline_probability": base_prob,
                                "reasoning_narrative": (reasoning or "")[:1000],
                            })

                            for ed in eg_detail:
                                dd = delta_by_feat.get(ed["feature"], {})
                                evidence_detail_data.append({
                                    "outcome": outcome, "horizon": hz_name, "mrn": str(mrn),
                                    "feature": ed["feature"], "visit": ed["visit"],
                                    "cited_value": ed["cited_value"],
                                    "actual_value": ed["actual_value"],
                                    "verified": bool(ed["verified"]),
                                    "importance": ed["importance"],
                                    "direction": ed["direction"],
                                    "baseline_probability": base_prob,
                                    "masked_probability": dd.get("masked_prob"),
                                    "counterfactual_delta": dd.get("delta"),
                                    "prediction_moved": bool(dd.get("prediction_moved", False)),
                                })

    spark = ctx.spark_session

    res_df = pd.DataFrame(all_results)
    if len(res_df) > 0:
        for c in ["config_id", "config_label", "method_family", "outcome", "horizon"]:
            res_df[c] = res_df[c].astype(str)
        for c in ["roc_auc_mean", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc_mean",
                  "pr_auc_ci_low", "pr_auc_ci_high", "brier_mean", "brier_ci_low", "brier_ci_high",
                  "mcc_mean", "prevalence"]:
            if c in res_df.columns:
                res_df[c] = pd.to_numeric(res_df[c], errors="coerce")
        for c in ["n_samples", "n_positive"]:
            res_df[c] = res_df[c].astype("int64")
    results_out.write_dataframe(spark.createDataFrame(res_df) if len(res_df) > 0
                                else spark.createDataFrame(pd.DataFrame(columns=["config_id"])))

    cost_df = pd.DataFrame(cost_data)
    if len(cost_df) > 0:
        for c in ["config_id", "method_family", "outcome", "horizon"]:
            cost_df[c] = cost_df[c].astype(str)
        for c in ["cost_usd_per_patient", "latency_sec_per_patient"]:
            cost_df[c] = cost_df[c].astype("float64")
        for c in ["total_tokens_per_patient", "input_tokens", "output_tokens", "n_llm_calls", "k_self_consistency"]:
            cost_df[c] = cost_df[c].astype("int64")
    cost_out.write_dataframe(spark.createDataFrame(cost_df) if len(cost_df) > 0
                             else spark.createDataFrame(pd.DataFrame(columns=["config_id"])))

    faith_df = pd.DataFrame(faithfulness_data)
    if len(faith_df) > 0:
        for c in ["outcome", "horizon", "mrn", "reasoning_narrative"]:
            faith_df[c] = faith_df[c].astype(str)
        for c in ["extractive_accuracy", "sufficiency_score", "comprehensiveness_score", "baseline_probability"]:
            faith_df[c] = faith_df[c].astype("float64")
        for c in ["extractive_n_correct", "extractive_n_total", "n_cited_features",
                  "n_verified_features", "n_features_moved"]:
            faith_df[c] = faith_df[c].astype("int64")
    faithfulness_out.write_dataframe(spark.createDataFrame(faith_df) if len(faith_df) > 0
                                     else spark.createDataFrame(pd.DataFrame(columns=["outcome"])))

    # Per-cited-feature interpretability surface (the "extra value" of the agent)
    ev_df = pd.DataFrame(evidence_detail_data)
    if len(ev_df) > 0:
        for c in ["outcome", "horizon", "mrn", "feature", "visit", "direction"]:
            ev_df[c] = ev_df[c].astype(str)
        for c in ["cited_value", "actual_value", "importance", "baseline_probability",
                  "masked_probability", "counterfactual_delta"]:
            ev_df[c] = pd.to_numeric(ev_df[c], errors="coerce")
        for c in ["verified", "prediction_moved"]:
            ev_df[c] = ev_df[c].astype(bool)
    evidence_detail_out.write_dataframe(spark.createDataFrame(ev_df) if len(ev_df) > 0
                                        else spark.createDataFrame(pd.DataFrame(columns=["outcome"])))

    # Cumulative top-K group-masking curve (comprehensiveness vs # top features removed)
    gm_df = pd.DataFrame(group_masking_data)
    if len(gm_df) > 0:
        for c in ["outcome", "horizon", "mrn"]:
            gm_df[c] = gm_df[c].astype(str)
        for c in ["baseline_probability", "masked_probability", "delta"]:
            gm_df[c] = pd.to_numeric(gm_df[c], errors="coerce")
        for c in ["k", "n_masked"]:
            gm_df[c] = gm_df[c].astype("int64")
        gm_df["prediction_moved"] = gm_df["prediction_moved"].astype(bool)
    group_masking_out.write_dataframe(spark.createDataFrame(gm_df) if len(gm_df) > 0
                                      else spark.createDataFrame(pd.DataFrame(columns=["outcome"])))

    abl = [
        {"ablation_dimension": "model_d_components", "config_id": "model_c_plus_verify",
         "variant_label": "C + Verification", "variant_order": 0, "is_reference": False},
        {"ablation_dimension": "model_d_components", "config_id": "model_c_plus_sc",
         "variant_label": "C + Self-Consistency", "variant_order": 1, "is_reference": False},
        {"ablation_dimension": "model_d_components", "config_id": "model_d_full",
         "variant_label": "Full Model D", "variant_order": 2, "is_reference": True},
    ]
    ablation_out.write_dataframe(spark.createDataFrame(pd.DataFrame(abl)))

    logger.info(f"\nNB6 COMPLETE (T2D) — {len(all_results)} experiments, "
                f"{len(faithfulness_data)} faithfulness analyses")


"""
NB7 — Algorithm Cost & Class Comparison (T2D)
===============================================
Compares computational cost across algorithm classes.
See T1D nb7_cost_analysis.py for full documentation.
"""

import numpy as np
import pandas as pd
from transforms.api import Input, Output, lightweight, transform

OUTPUT_BASE = "/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d"

NB4_MATCHED = f"{OUTPUT_BASE}/nb4_matched_comparison_t2d"
NB4_TIMING = f"{OUTPUT_BASE}/nb4_model_timing_t2d"
NB5_RESULTS = f"{OUTPUT_BASE}/nb5_agentic_results_t2d"
NB5_COST = f"{OUTPUT_BASE}/nb5_cost_latency_t2d"
NB6_RESULTS = f"{OUTPUT_BASE}/nb6_model_d_results_t2d"
NB6_COST = f"{OUTPUT_BASE}/nb6_model_d_cost_t2d"

ML_ALGORITHM_CLASS = {
    "Logistic_Regression": "classical_ml",
    "Random_Forest": "classical_ml",
    "XGBoost": "classical_ml",
    "XGBoost_Temporal": "temporal_ml",
    "LSTM": "deep_learning",
    "GRU": "deep_learning",
    "Temporal_CNN": "deep_learning",
    "Transformer": "deep_learning",
}

LLM_ALGORITHM_CLASS = {
    "model_a": "single_agent_llm",
    "model_c": "single_agent_llm",
    "model_b": "multi_agent_llm",
    "model_a_no_sdoh": "single_agent_llm",
}

MODEL_D_ALGORITHM_CLASS = {
    "model_d_full": "deliberative_ensemble",
    "model_c_plus_verify": "deliberative_ensemble",
    "model_c_plus_sc": "deliberative_ensemble",
}

ML_COST_PER_SEC_USD = 0.00005


@lightweight(cpu_cores=2, memory_gb=4)
@transform(
    cost_comparison_out=Output(f"{OUTPUT_BASE}/nb7_cost_comparison_t2d"),
    pareto_out=Output(f"{OUTPUT_BASE}/nb7_pareto_frontier_t2d"),
    cascade_out=Output(f"{OUTPUT_BASE}/nb7_cascade_analysis_t2d"),
    nb4_matched=Input(NB4_MATCHED),
    nb4_timing=Input(NB4_TIMING),
    nb5_results=Input(NB5_RESULTS),
    nb5_cost=Input(NB5_COST),
    nb6_results=Input(NB6_RESULTS),
    nb6_cost=Input(NB6_COST),
)
def compute(nb4_matched, nb4_timing, nb5_results, nb5_cost,
            nb6_results, nb6_cost,
            cost_comparison_out, pareto_out, cascade_out):

    df_nb4_matched = nb4_matched.pandas()
    df_nb4_timing = nb4_timing.pandas()
    df_nb5_results = nb5_results.pandas()
    df_nb5_cost = nb5_cost.pandas()
    df_nb6_results = nb6_results.pandas()
    df_nb6_cost = nb6_cost.pandas()

    comparison_rows = []

    # 1. ML Models (NB4)
    for _, row in df_nb4_matched.iterrows():
        model = row["model"]
        outcome = row["outcome"]
        horizon = row["horizon"]
        timing_match = df_nb4_timing[
            (df_nb4_timing["model"] == model) &
            (df_nb4_timing["outcome"] == outcome) &
            (df_nb4_timing["horizon"] == horizon) &
            (df_nb4_timing["phase"] == "matched_holdout")
        ]
        wall_clock = timing_match["wall_clock_seconds"].values[0] if len(timing_match) > 0 else np.nan
        n_test = int(row.get("n_test", 0))
        cost_per_patient = (wall_clock * ML_COST_PER_SEC_USD / max(n_test, 1)) if not np.isnan(wall_clock) else np.nan

        comparison_rows.append({
            "model_id": model, "model_label": model.replace("_", " "),
            "algorithm_class": ML_ALGORITHM_CLASS.get(model, "unknown"),
            "paradigm": "ML",
            "outcome": outcome, "horizon": horizon,
            "roc_auc": row.get("roc_auc_mean", np.nan),
            "roc_auc_ci_low": row.get("roc_auc_ci_low", np.nan),
            "roc_auc_ci_high": row.get("roc_auc_ci_high", np.nan),
            "pr_auc": row.get("pr_auc_mean", np.nan),
            "brier": row.get("brier_mean", np.nan),
            "wall_clock_seconds": wall_clock,
            "cost_usd_per_patient": cost_per_patient,
            "latency_sec_per_patient": (wall_clock / max(n_test, 1)) if not np.isnan(wall_clock) else np.nan,
            "total_tokens_per_patient": 0,
            "n_llm_calls_per_patient": 0,
            "n_test": n_test,
        })

    # 2. LLM Models A/B/C (NB5)
    for _, row in df_nb5_results.iterrows():
        cid = row["config_id"]
        outcome = row["outcome"]
        horizon = row["horizon"]
        cost_match = df_nb5_cost[
            (df_nb5_cost["config_id"] == cid) &
            (df_nb5_cost["outcome"] == outcome) &
            (df_nb5_cost["horizon"] == horizon)
        ]
        cost_usd = cost_match["cost_usd_per_patient"].values[0] if len(cost_match) > 0 else np.nan
        latency = cost_match["latency_sec_per_patient"].values[0] if len(cost_match) > 0 else np.nan
        tokens = int(cost_match["total_tokens_per_patient"].values[0]) if len(cost_match) > 0 else 0
        n_calls = int(cost_match["n_llm_calls"].values[0]) if len(cost_match) > 0 else 0
        n_test = int(row.get("n_samples", 0))

        comparison_rows.append({
            "model_id": cid, "model_label": row.get("config_label", cid),
            "algorithm_class": LLM_ALGORITHM_CLASS.get(cid, "single_agent_llm"),
            "paradigm": "LLM",
            "outcome": outcome, "horizon": horizon,
            "roc_auc": row.get("roc_auc_mean", np.nan),
            "roc_auc_ci_low": row.get("roc_auc_ci_low", np.nan),
            "roc_auc_ci_high": row.get("roc_auc_ci_high", np.nan),
            "pr_auc": row.get("pr_auc_mean", np.nan),
            "brier": row.get("brier_mean", np.nan),
            "wall_clock_seconds": latency * n_test if not np.isnan(latency) else np.nan,
            "cost_usd_per_patient": cost_usd,
            "latency_sec_per_patient": latency,
            "total_tokens_per_patient": tokens,
            "n_llm_calls_per_patient": n_calls // max(n_test, 1),
            "n_test": n_test,
        })

    # 3. Model D variants (NB6)
    for _, row in df_nb6_results.iterrows():
        cid = row["config_id"]
        outcome = row["outcome"]
        horizon = row["horizon"]
        cost_match = df_nb6_cost[
            (df_nb6_cost["config_id"] == cid) &
            (df_nb6_cost["outcome"] == outcome) &
            (df_nb6_cost["horizon"] == horizon)
        ]
        cost_usd = cost_match["cost_usd_per_patient"].values[0] if len(cost_match) > 0 else np.nan
        latency = cost_match["latency_sec_per_patient"].values[0] if len(cost_match) > 0 else np.nan
        tokens = int(cost_match["total_tokens_per_patient"].values[0]) if len(cost_match) > 0 else 0
        n_calls = int(cost_match["n_llm_calls"].values[0]) if len(cost_match) > 0 else 0
        n_test = int(row.get("n_samples", 0))

        comparison_rows.append({
            "model_id": cid, "model_label": row.get("config_label", cid),
            "algorithm_class": MODEL_D_ALGORITHM_CLASS.get(cid, "deliberative_ensemble"),
            "paradigm": "LLM_Ensemble",
            "outcome": outcome, "horizon": horizon,
            "roc_auc": row.get("roc_auc_mean", np.nan),
            "roc_auc_ci_low": row.get("roc_auc_ci_low", np.nan),
            "roc_auc_ci_high": row.get("roc_auc_ci_high", np.nan),
            "pr_auc": row.get("pr_auc_mean", np.nan),
            "brier": row.get("brier_mean", np.nan),
            "wall_clock_seconds": latency * n_test if not np.isnan(latency) else np.nan,
            "cost_usd_per_patient": cost_usd,
            "latency_sec_per_patient": latency,
            "total_tokens_per_patient": tokens,
            "n_llm_calls_per_patient": n_calls // max(n_test, 1),
            "n_test": n_test,
        })

    comparison_df = pd.DataFrame(comparison_rows)

    # 4. Pareto Frontier
    pareto_rows = []
    if len(comparison_df) > 0:
        for outcome in comparison_df["outcome"].unique():
            odf = comparison_df[comparison_df["outcome"] == outcome].copy()
            odf = odf.dropna(subset=["roc_auc", "cost_usd_per_patient"])
            if len(odf) == 0:
                continue
            odf = odf.sort_values("cost_usd_per_patient")
            pareto_mask = []
            for idx, row in odf.iterrows():
                dominated = False
                for idx2, row2 in odf.iterrows():
                    if idx == idx2:
                        continue
                    if (row2["cost_usd_per_patient"] <= row["cost_usd_per_patient"] and
                            row2["roc_auc"] >= row["roc_auc"] and
                            (row2["cost_usd_per_patient"] < row["cost_usd_per_patient"] or
                             row2["roc_auc"] > row["roc_auc"])):
                        dominated = True
                        break
                pareto_mask.append(not dominated)

            for i, (idx, row) in enumerate(odf.iterrows()):
                pareto_rows.append({
                    "outcome": outcome,
                    "model_id": row["model_id"],
                    "algorithm_class": row["algorithm_class"],
                    "roc_auc": row["roc_auc"],
                    "cost_usd_per_patient": row["cost_usd_per_patient"],
                    "is_pareto_optimal": pareto_mask[i],
                })

    # 5. Cascade Analysis
    cascade_rows = []
    if len(comparison_df) > 0:
        for outcome in comparison_df["outcome"].unique():
            odf = comparison_df[comparison_df["outcome"] == outcome].dropna(subset=["roc_auc", "cost_usd_per_patient"])
            if len(odf) < 2:
                continue
            cheapest = odf.loc[odf["cost_usd_per_patient"].idxmin()]
            most_expensive = odf.loc[odf["cost_usd_per_patient"].idxmax()]
            best_auc = odf.loc[odf["roc_auc"].idxmax()]

            auc_gain = best_auc["roc_auc"] - cheapest["roc_auc"]
            cost_ratio = most_expensive["cost_usd_per_patient"] / max(cheapest["cost_usd_per_patient"], 1e-8)
            cost_delta = most_expensive["cost_usd_per_patient"] - cheapest["cost_usd_per_patient"]
            marginal_auc_per_dollar = auc_gain / max(cost_delta, 1e-8) if cost_delta > 0 else 0.0
            cascade_recommended = auc_gain > 0.03

            cascade_rows.append({
                "outcome": outcome,
                "cheapest_model": cheapest["model_id"],
                "cheapest_class": cheapest["algorithm_class"],
                "cheapest_auc": cheapest["roc_auc"],
                "cheapest_cost": cheapest["cost_usd_per_patient"],
                "best_model": best_auc["model_id"],
                "best_class": best_auc["algorithm_class"],
                "best_auc": best_auc["roc_auc"],
                "best_cost": best_auc["cost_usd_per_patient"],
                "most_expensive_model": most_expensive["model_id"],
                "most_expensive_cost": most_expensive["cost_usd_per_patient"],
                "auc_gain_cheap_to_best": auc_gain,
                "cost_ratio_expensive_to_cheap": cost_ratio,
                "marginal_auc_per_dollar": marginal_auc_per_dollar,
                "cascade_recommended": cascade_recommended,
                "recommendation": (
                    f"Use {best_auc['model_id']} (AUC +{auc_gain:.3f} over {cheapest['model_id']})"
                    if cascade_recommended else
                    f"Use {cheapest['model_id']} (expensive model gain only +{auc_gain:.3f})"
                ),
            })

    # Write outputs
    if len(comparison_df) > 0:
        for c in ["model_id", "model_label", "algorithm_class", "paradigm", "outcome", "horizon"]:
            comparison_df[c] = comparison_df[c].astype(str)
        for c in ["roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc", "brier",
                  "wall_clock_seconds", "cost_usd_per_patient", "latency_sec_per_patient"]:
            if c in comparison_df.columns:
                comparison_df[c] = pd.to_numeric(comparison_df[c], errors="coerce")
        for c in ["total_tokens_per_patient", "n_llm_calls_per_patient", "n_test"]:
            comparison_df[c] = comparison_df[c].fillna(0).astype("int64")
    cost_comparison_out.write_table(comparison_df if len(comparison_df) > 0
                                    else pd.DataFrame(columns=["model_id"]))

    pareto_df = pd.DataFrame(pareto_rows)
    if len(pareto_df) > 0:
        for c in ["outcome", "model_id", "algorithm_class"]:
            pareto_df[c] = pareto_df[c].astype(str)
        for c in ["roc_auc", "cost_usd_per_patient"]:
            pareto_df[c] = pareto_df[c].astype("float64")
        pareto_df["is_pareto_optimal"] = pareto_df["is_pareto_optimal"].astype(bool)
    pareto_out.write_table(pareto_df if len(pareto_df) > 0
                           else pd.DataFrame(columns=["outcome"]))

    cascade_df = pd.DataFrame(cascade_rows)
    if len(cascade_df) > 0:
        for c in ["outcome", "cheapest_model", "cheapest_class", "best_model", "best_class",
                  "most_expensive_model", "recommendation"]:
            cascade_df[c] = cascade_df[c].astype(str)
        for c in ["cheapest_auc", "cheapest_cost", "best_auc", "best_cost", "most_expensive_cost",
                  "auc_gain_cheap_to_best", "cost_ratio_expensive_to_cheap", "marginal_auc_per_dollar"]:
            cascade_df[c] = cascade_df[c].astype("float64")
        cascade_df["cascade_recommended"] = cascade_df["cascade_recommended"].astype(bool)
    cascade_out.write_table(cascade_df if len(cascade_df) > 0
                            else pd.DataFrame(columns=["outcome"]))

    print(f"\n{'='*60}\nNB7 COST ANALYSIS COMPLETE (T2D)\n{'='*60}")
    print(f"Comparison entries: {len(comparison_df)}")
    print(f"Pareto entries: {len(pareto_df)}")
    print(f"Cascade entries: {len(cascade_df)}")
    if len(comparison_df) > 0:
        print("\n── Cost by Algorithm Class ──")
        class_summary = comparison_df.groupby("algorithm_class").agg(
            mean_auc=("roc_auc", "mean"),
            mean_cost=("cost_usd_per_patient", "mean"),
            mean_latency=("latency_sec_per_patient", "mean"),
        ).reset_index()
        for _, r in class_summary.iterrows():
            print(f"  {r['algorithm_class']:25s} AUC={r['mean_auc']:.3f}  "
                  f"Cost=${r['mean_cost']:.4f}/pt  Latency={r['mean_latency']:.2f}s/pt")
