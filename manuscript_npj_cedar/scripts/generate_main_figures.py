"""
Generate the MAIN manuscript figure panels (fig1..fig6) LOCALLY.

Adapted from the Palantir Foundry notebook 8 (nb8). The figure-building code is
UNCHANGED; only the data layer is swapped: instead of reading Foundry datasets
via ``foundry.transforms.Dataset``, this script parses the raw dataset dumps
exported from Foundry into ``additional_analysis/figure_data/{t2d,t1d}_data.txt``
(one file per cohort, holding every nb4-nb7 table as a tab-delimited block).

Outputs the main composites (and the manuscript-numbers report) into
``figures/main_figures/`` so they land exactly where main.tex expects them:

    figures/main_figures/fig2.{pdf,svg}
    figures/main_figures/fig3.{pdf,svg}
    ...

(Fig 1 is hand-composed by compose_fig1_v2.py into figures/main_figures/fig1_v2.*;
its data-driven sub-panels are written to additional_analysis/figure_sources/fig1_panels/.)

Usage (from the project root or anywhere; paths are resolved relative to this file):
    python scripts/generate_main_figures.py

Narrative arc (T2D = primary cohort; T1D = regime-contrast + supplement):

    fig1  Study design & clinical landscape
          (A) reasoning-paradigm ladder  ML -> single -> multi -> CEDAR
          (B) 2-year task timeline + modalities
          (C) outcome prevalence landscape
    fig2  The performance landscape
          (A) best model per outcome across ALL families (95% CI)
          (B) config x outcome ROC-AUC heatmap
          (C) ROC-AUC distribution by algorithm class
    fig3  Mechanism: why naive agents fail, why CEDAR works
          (A) collapse — single-agent Direct/CoT vs CEDAR
          (B) architecture progression  Vanilla -> Multi -> CoT -> CEDAR
          (C) mechanism ablation  CEDAR / CoT+Verify / CoT+SC (95% CI)
    fig4  Faithfulness & trust
          (A) extractive / sufficiency / comprehensiveness per outcome
          (B) cumulative top-K group-masking curve
          (C) evidence audit — cited importance vs actual counterfactual impact
    fig5  Clinical economics
          (A) cost per patient by algorithm class (log)
          (B) cost-performance Pareto frontier (all families)
          (C) cascade — marginal AUC per dollar + escalation recommendation
    fig6  Data-regime contrast & modality value (optional / can go to main or supp)
          (A) T1D grand comparison (classical ML wins the well-powered outcomes)
          (B) modality ablation T2D (value of SDoH)
          (C) modality ablation T1D (value of SDoH + CGM)

Dataset suffix conventions (IMPORTANT — they differ by notebook):
    nb1 / nb4 / nb5 :  T2D = ''      T1D = '_t1d'
    nb6 / nb7       :  T2D = '_t2d'  T1D = '_t1d'
"""

import io
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless: no display needed on SageMaker
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch
import seaborn as sns

# Shared publication style + palette + significance helpers (PubliPlots-based).
import figstyle as fs


# ══════════════════════════════════════════════════════════════════
# Paths (resolved relative to this file, so the script runs from anywhere)
# ══════════════════════════════════════════════════════════════════
_HERE     = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.normpath(os.path.join(_HERE, ".."))
DATA_DIR  = os.path.join(_PROJECT, "additional_analysis", "figure_data")
OUTDIR    = os.path.join(_PROJECT, "figures", "main_figures")   # main.tex: figures/main_figures/
REPORT_DIR = os.path.join(_PROJECT, "additional_analysis")

# ══════════════════════════════════════════════════════════════════
# Publication style — one Nature-style palette + typography for the whole
# manuscript, via the shared figstyle module (built on PubliPlots).
# ══════════════════════════════════════════════════════════════════
fs.apply_style()

FORMATS = ["pdf", "svg"]
HORIZON = "year_2"

# Suffix conventions (verified against the registered aliases):
#   nb4 / nb5 / nb6 / nb7 :  T2D = '_t2d'   T1D = '_t1d'
# nb8 reads ONLY nb4-nb7 datasets, so every dataset below uses the _t2d/_t1d map.
SUFFIX = {"t2d": "_t2d", "t1d": "_t1d"}


# ══════════════════════════════════════════════════════════════════
# Local multi-table loader (replaces foundry.transforms.Dataset)
#
# Each cohort file (t2d_data.txt / t1d_data.txt) is a concatenation of Foundry
# tables: a strict dataset-name line (e.g. 'nb7_cost_comparison_t2d'), a blank
# line, then a tab-delimited table with a leading unnamed index column. Some
# tables (faithfulness / evidence) contain quoted free-text cells with embedded
# newlines, so we split on the strict name pattern and try progressively more
# permissive CSV parses.
# ══════════════════════════════════════════════════════════════════
_NAME_RE = re.compile(r"^((?:nb\d*|reviewer)_[a-z0-9_]+_t[12]d)\s*$")
_CACHE = {}   # cohort -> {dataset_name: DataFrame}


def _parse_block(block):
    block = block.strip("\n")
    if not block.strip():
        return None
    for kw in (dict(sep="\t", engine="python"),
               dict(sep="\t", engine="python", on_bad_lines="skip"),
               dict(sep="\t", engine="python", quoting=3)):   # QUOTE_NONE
        try:
            df = pd.read_csv(io.StringIO(block), **kw)
            return df.loc[:, ~df.columns.astype(str).str.match(r"Unnamed")]
        except Exception:
            continue
    return None


def _load_cohort_file(cohort):
    """Parse every table in additional_analysis/figure_data/<cohort>_data.txt."""
    if cohort in _CACHE:
        return _CACHE[cohort]
    path = os.path.join(DATA_DIR, f"{cohort}_data.txt")
    tables = {}
    if not os.path.exists(path):
        print(f"    [missing file] {path}")
        _CACHE[cohort] = tables
        return tables
    lines = open(path).read().split("\n")
    bounds = [(i, m.group(1)) for i, ln in enumerate(lines)
              for m in [_NAME_RE.match(ln)] if m]
    for j, (idx, name) in enumerate(bounds):
        end = bounds[j + 1][0] if j + 1 < len(bounds) else len(lines)
        df = _parse_block("\n".join(lines[idx + 1:end]))
        if df is not None:
            tables[name] = df
    _CACHE[cohort] = tables
    return tables

# ══════════════════════════════════════════════════════════════════
# Naming & color system — DEFINED ONCE, reused in every panel so a name
# and a color always mean the same model everywhere.
#
# METHOD_NAME: the deliberative ensemble (formerly "Model D"). Rename here and
# it propagates to every figure. Suggested: CEDAR = Calibrated Evidence-grounded
# Deliberative Agentic Reasoning.
# ══════════════════════════════════════════════════════════════════
METHOD_NAME = "CEDAR"           # <- our headline model (was model_d_full)
CEDAR_RED   = fs.CEDAR          # headline colour from the shared palette

# Short figure labels keyed by technical id / raw model name (lowercased).
SHORT_NAME = {
    # classical ML
    "logistic regression": "LogReg", "random forest": "RF", "xgboost": "XGB",
    "xgboost_temporal": "XGB-T", "xgboost temporal": "XGB-T",
    # deep learning
    "lstm": "LSTM", "gru": "GRU", "temporal cnn": "TCN", "transformer": "TF",
    # agentic (nb5) — model_a is the plain single-pass LLM baseline.
    # Change "Vanilla" here to "Single"/"Basic" if you prefer; it propagates.
    "model_a": "Vanilla", "model_c": "CoT", "model_b": "Multi",
    "model_a_no_sdoh": "Vanilla\u2212SDoH", "model_a_no_cgm": "Vanilla\u2212CGM",
    # deliberative ensemble family (nb6)
    "model_d_full": METHOD_NAME,
    "model_c_plus_verify": "CoT+Verify", "model_c_plus_sc": "CoT+SC",
}

# Per-model colors (used in per-model panels: fig2A, fig3A/B/C, fig2B labels).
# All drawn from the ONE shared palette; non-CEDAR models get muted family tones
# so CEDAR (headline red) draws the eye.
MODEL_COLORS = {
    "model_a": fs.PALETTE["blue"],        # single-pass / Direct
    "model_c": fs.PALETTE["purple"],      # CoT
    "model_b": fs.PALETTE["orange"],      # Multi
    "model_a_no_sdoh": "#9bb8e0",
    "model_a_no_cgm": "#9bb8e0",
    "model_d_full": CEDAR_RED,            # CEDAR (full) — headline
    "model_c_plus_verify": "#d98a7d",     # CEDAR family, medium
    "model_c_plus_sc": "#eab5ab",         # CEDAR family, light
}

# Algorithm-class colors (class-level panels: fig2C, fig5, fig6) — shared palette.
CLASS_COLORS = fs.CLASS_COLORS
CLASS_PRETTY = {
    "classical_ml": "Classical ML", "temporal_ml": "Temporal ML",
    "deep_learning": "Temporal ML",   # alias (remapped at load); same label
    "single_agent_llm": "Single LLM",
    "multi_agent_llm": "Multi-agent LLM",
    "deliberative_ensemble": f"Multi-step LLM ({METHOD_NAME})",
}
# Explicit display order (excludes the 'deep_learning' alias so it never shows
# as a separate legend/box entry).
CLASS_ORDER = ["classical_ml", "temporal_ml", "single_agent_llm",
               "multi_agent_llm", "deliberative_ensemble"]

# Fixed FIGURE display order for the performance-landscape panels (Fig 2 B/C/D/E):
# CEDAR first, then classical ML, temporal ML, single LLM, multi-agent LLM last.
# Kept identical across cohorts so row/box order never flips between T2D and T1D.
FIG_CLASS_ORDER = ["deliberative_ensemble", "classical_ml", "temporal_ml",
                   "single_agent_llm", "multi_agent_llm"]


def _norm(s):
    """Collapse spaces / underscores / hyphens so 'Logistic Regression',
    'Logistic_Regression' and 'logistic-regression' all match one key."""
    return re.sub(r"[\s_\-]+", "_", str(s).strip().lower())


# Normalized view of SHORT_NAME so lookups work whether we're handed a
# model_id ('Logistic_Regression', 'model_c_plus_verify') or a spaced label.
NORM_SHORT = {_norm(k): v for k, v in SHORT_NAME.items()}


def short_model(name):
    """Map any model_id / model_label / config_id to its short figure label.
    Robust to verbose labels like 'Ablation: Model C + verification only' and
    'Full Model D: knowledge + CoT evidence + verify + self-consistency'."""
    key = _norm(name)
    if key in NORM_SHORT:
        return NORM_SHORT[key]
    # verbose model_label fallbacks. Check the FULL method first, because its
    # label ("Full Model D: ... verify ... self-consistency") contains the
    # verify/consist substrings used to detect the ablation arms.
    if "full" in key and "model_d" in key:
        return METHOD_NAME
    if "verification" in key or "verify" in key:
        return "CoT+Verify"
    if "consist" in key:                       # self-consistency
        return "CoT+SC"
    return str(name)


def model_color(name):
    """Per-model color from technical id (falls back to class-neutral grey)."""
    return MODEL_COLORS.get(_norm(name), "#95a5a6")


OUTCOME_ORDER = ["Optimal Glycemic Control", "Insulin Independence",
                 "Metformin Response", "GLP1RA Response", "Dyslipidemia",
                 "Hypertension", "Microalbuminuria"]


# ══════════════════════════════════════════════════════════════════
# Generic helpers
# ══════════════════════════════════════════════════════════════════
SUPP_OUTDIR = os.path.join(_PROJECT, "figures", "supplementary_figures")


def save_fig(fig, name):
    # figures named figS_* are supplementary and go to the supplement folder;
    # everything else is a main figure.
    outdir = SUPP_OUTDIR if name.startswith("figS_") else OUTDIR
    os.makedirs(outdir, exist_ok=True)
    for fmt in FORMATS:
        path = os.path.join(outdir, f"{name}.{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"    saved -> {path}")
    plt.close(fig)


def panel_label(ax, letter, dx=-0.08, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="right")


# Algorithm-class remap applied at load time so EVERY panel sees the same
# grouping. Classical ML = flat-feature models (LogReg/RF/XGBoost); Temporal ML =
# sequence / temporal-feature models (XGBoost-Temporal + LSTM/GRU/TCN/Transformer,
# i.e. the former 'deep_learning' class folded in).
CLASS_REMAP = {"deep_learning": "temporal_ml"}


def try_ds(name):
    """Look a dataset up by its (suffixed) name from the local cohort dump.
    e.g. 'nb7_cost_comparison_t2d' -> table parsed from t2d_data.txt."""
    cohort = "t1d" if name.endswith("_t1d") else "t2d"
    df = _load_cohort_file(cohort).get(name)
    if df is not None:
        df = df.copy()
        if "algorithm_class" in df.columns:
            df["algorithm_class"] = df["algorithm_class"].replace(CLASS_REMAP)
        print(f"    loaded {name:38s} {df.shape}")
        return df
    print(f"    [missing] {name:38s} (not in {cohort}_data.txt)")
    return None


_MASK_BASELINE_PATH = os.path.join(
    _PROJECT, "additional_analysis", "masking_group_analysis_baseline",
    "data_masking_analysis_with_baseline.txt")
_MASK_BASELINE_CACHE = {}


def load_masking_baseline(cohort):
    """Parse the cited-vs-random masking-baseline dump. Returns a dict with
    'pooled' (arm,k,mean_abs_delta,sd_abs_delta) and 'per_outcome'
    (outcome,k,arm,mean_abs_delta,sd_abs_delta) DataFrames for the cohort, or
    None if the file/blocks are absent. Same concat-of-named-tables format as
    the cohort files, but its own leading index column."""
    if cohort in _MASK_BASELINE_CACHE:
        return _MASK_BASELINE_CACHE[cohort]
    result = None
    if os.path.exists(_MASK_BASELINE_PATH):
        lines = open(_MASK_BASELINE_PATH).read().split("\n")
        name_re = re.compile(r"^(nb6_masking_baseline[a-z0-9_]*_t[12]d)\s*$")
        bounds = [(i, m.group(1)) for i, ln in enumerate(lines)
                  for m in [name_re.match(ln)] if m]
        tables = {}
        for j, (idx, name) in enumerate(bounds):
            end = bounds[j + 1][0] if j + 1 < len(bounds) else len(lines)
            df = _parse_block("\n".join(lines[idx + 1:end]))
            if df is not None:
                tables[name] = df
        pooled = tables.get(f"nb6_masking_baseline_pooled_{cohort}")
        per_out = tables.get(f"nb6_masking_baseline_{cohort}")
        if pooled is not None or per_out is not None:
            result = {"pooled": pooled, "per_outcome": per_out}
    _MASK_BASELINE_CACHE[cohort] = result
    if result is None:
        print(f"    [missing] masking baseline ({cohort}) at {_MASK_BASELINE_PATH}")
    return result


def _first_ds(*names):
    """Return the first dataset that exists among `names` (None-safe;
    avoids DataFrame truthiness pitfalls of `a or b`)."""
    for nm in names:
        df = try_ds(nm)
        if df is not None:
            return df
    return None


def pick(df, cands, required=True, what=""):
    for c in cands:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of {cands} for '{what}'. Have: {list(df.columns)}")
    return None


def pretty_outcome(s):
    return str(s).replace("OUTCOME_", "").replace("_", " ").strip()


# Compact, readable feature labels for the attention heatmap.
_FEATURE_LABELS = {
    "BMI_ZSCORE": "BMI z-score", "HBA1C": "HbA1c", "GLUCOSE": "Glucose",
    "SERUM_C_PEPTIDE": "C-peptide", "diabetes_duration": "Diabetes duration",
    "race_black": "Race: Black", "ethnicity_hispanic": "Ethnicity: Hispanic",
    "SBP_OUTPATIENT": "SBP (outpatient)", "DBP_OUTPATIENT": "DBP (outpatient)",
    "SBP_INPATIENT": "SBP (inpatient)", "DBP_INPATIENT": "DBP (inpatient)",
    "BLOOD_PH": "Blood pH", "BICARBONATE": "Bicarbonate", "Biguanide": "Biguanide",
    "Insulins": "Insulins", "SERUM_CREATININE": "Creatinine",
    "CGM_MEAN_GLUCOSE": "CGM mean glucose", "CGM_GMI": "CGM GMI",
    "CGM_TIR_70_180": "CGM TIR 70-180", "CGM_TAR_ABOVE_180": "CGM TAR >180",
    "CGM_TAR_ABOVE_250": "CGM TAR >250", "CGM_SD": "CGM SD",
    "CGM_SEVERE_HYPO_EPISODES": "CGM severe hypo",
}


def pretty_feature(s):
    s = str(s)
    if s in _FEATURE_LABELS:
        return _FEATURE_LABELS[s]
    if s.startswith("OUTCOME_"):
        return pretty_outcome(s) + " (prior)"
    return s.replace("_", " ").strip()


def order_outcomes(vals):
    vals = list(dict.fromkeys(vals))
    return [o for o in OUTCOME_ORDER if o in vals] + \
           [o for o in vals if o not in OUTCOME_ORDER]


def class_color(c):
    return CLASS_COLORS.get(str(c), "#95a5a6")


def class_order(present):
    present = list(dict.fromkeys(present))
    return [c for c in CLASS_ORDER if c in present] + \
           [c for c in present if c not in CLASS_ORDER]


def fig_class_order(present):
    """Fixed left-to-right order for the Fig-2 landscape panels (CEDAR first,
    multi-agent last), identical across cohorts."""
    present = list(dict.fromkeys(present))
    return [c for c in FIG_CLASS_ORDER if c in present] + \
           [c for c in present if c not in FIG_CLASS_ORDER]


def as_bool(v):
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes", "t"}
    return bool(v)


def filter_h(df, target=HORIZON):
    if df is None:
        return None
    hz = pick(df, ["horizon", "time_horizon", "window"], required=False, what="horizon")
    if hz is not None and target in set(df[hz].astype(str)):
        return df[df[hz].astype(str) == target].copy()
    return df.copy()


def ci_err(mean, lo, hi):
    mean = np.asarray(mean, dtype=float)
    lower = np.clip(mean - np.asarray(lo, dtype=float), 0, None)
    upper = np.clip(np.asarray(hi, dtype=float) - mean, 0, None)
    return np.vstack([np.nan_to_num(lower), np.nan_to_num(upper)])


def empty_panel(ax, msg):
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9,
            color="#7f8c8d", wrap=True)


# Column candidate lists (shared).
AUC   = ["roc_auc", "roc_auc_mean", "auc"]
CILO  = ["roc_auc_ci_low", "auc_ci_low", "ci_low"]
CIHI  = ["roc_auc_ci_high", "auc_ci_high", "ci_high"]
CFG   = ["config_id", "config", "model_id", "model"]
OUT   = ["outcome", "target", "label"]
CLS   = ["algorithm_class", "class", "model_class"]
COST  = ["cost_usd_per_patient", "cost_usd", "cost"]


def load_all(cohort):
    """Load every dataset needed for the manuscript figures for one cohort.
    All nb4-nb7 datasets use the _t2d/_t1d suffix. Missing datasets -> None."""
    s = SUFFIX[cohort]
    print(f"  loading datasets for cohort '{cohort}' (suffix '{s}')")
    d = {
        "matched":  try_ds(f"nb4_matched_comparison{s}"),
        "modality": try_ds(f"nb4_modality_ablation{s}"),
        "modality_all": try_ds(f"nb_modality_ablation_all{s}"),
        "agentic":  try_ds(f"nb5_agentic_results{s}"),
        "model_d":  try_ds(f"nb6_model_d_results{s}"),
        "faith":    try_ds(f"nb6_faithfulness_results{s}"),
        "gmask":    try_ds(f"nb6_group_masking_curve{s}"),
        # Full evidence_detail is huge; fall back to the slim/downsampled export
        # (make_evidence_audit_slim.py) when the full table isn't available locally.
        "evidence": _first_ds(f"nb6_evidence_detail{s}", f"nb6_evidence_audit_slim{s}"),
        "cost":     try_ds(f"nb7_cost_comparison{s}"),
        "cascade":  try_ds(f"nb7_cascade_analysis{s}"),
        # follow-up data-availability over visits (Fig 1 complement); optional.
        "availability": try_ds(f"nb1_data_availability{s}"),
        # CEDAR feature attention (Fig 3C); small pre-aggregated export, optional.
        "feat_attn": try_ds(f"nb6_feature_attention{s}"),
    }
    # Horizon-filter the frames that carry a horizon column.
    for k in ["matched", "modality", "agentic", "model_d", "faith", "gmask",
              "evidence", "cost"]:
        d[k] = filter_h(d[k])
    return d


# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Study design & clinical landscape
# ══════════════════════════════════════════════════════════════════
def _draw_paradigm_ladder(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    stages = [
        ("Classical ML / DL", "pattern matching\nover flattened features", "#2c7fb8"),
        ("Single LLM", "direct YES/NO\n+ confidence", "#8fb0d0"),
        ("Multi-agent LLM", "modality specialists\n+ synthesis", "#e8c07d"),
        (METHOD_NAME, "multi-step LLM\n(evidence-grounded)", CEDAR_RED),
    ]
    n = len(stages); w = 0.205; gap = (1 - n * w) / (n + 1)
    for i, (title, sub, color) in enumerate(stages):
        x = gap + i * (w + gap)
        box = FancyBboxPatch((x, 0.52), w, 0.34, boxstyle="round,pad=0.012,rounding_size=0.02",
                             linewidth=1.4, edgecolor=color, facecolor=color + "22")
        ax.add_patch(box)
        ax.text(x + w / 2, 0.79, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)
        ax.text(x + w / 2, 0.63, sub, ha="center", va="center", fontsize=7.5, color="#2c3e50")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.005, 0.69), (x + w + gap - 0.005, 0.69),
                                         arrowstyle="-|>", mutation_scale=14, color="#7f8c8d", lw=1.4))
    # CEDAR internal stages
    d_x = gap + 3 * (w + gap)
    sub_stages = ["S0 knowledge", "S1 CoT+evidence", "S2 verify", "S3 self-consistency"]
    for j, s in enumerate(sub_stages):
        ax.text(d_x + w / 2, 0.47 - j * 0.055, f"• {s}", ha="center", va="center",
                fontsize=6.8, color=CEDAR_RED)
    # bottom reliability arrow
    ax.add_patch(FancyArrowPatch((gap, 0.14), (1 - gap, 0.14),
                                 arrowstyle="-|>", mutation_scale=18, color="#34495e", lw=2))
    ax.text(0.5, 0.20, "increasing deliberation, calibration & evidence grounding",
            ha="center", va="bottom", fontsize=8.5, style="italic", color="#34495e")
    ax.text(0.19, 0.08, "AUC can collapse below chance", ha="center", fontsize=7,
            color="#c0392b")
    ax.text(0.87, 0.08, "calibrated, robust, auditable", ha="center", fontsize=7,
            color="#27ae60")
    ax.set_title("Four reasoning paradigms compared", fontsize=11, fontweight="bold", pad=6)


def _draw_task_timeline(ax, cohort):
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 3.2); ax.axis("off")
    for v in range(1, 11):
        if v <= 3:
            fc, ec, txt = "#2c7fb8", "#1f5f8b", "input"
        elif v == 4:
            fc, ec, txt = "#d1495b", "#a13545", "target"
        else:
            fc, ec, txt = "#ecf0f1", "#bdc3c7", ""
        box = FancyBboxPatch((v - 0.9, 1.7), 0.8, 0.7, boxstyle="round,pad=0.02,rounding_size=0.06",
                             linewidth=1.1, edgecolor=ec, facecolor=fc)
        ax.add_patch(box)
        ax.text(v - 0.5, 2.05, f"v{v}", ha="center", va="center", fontsize=8,
                color="white" if v <= 4 else "#7f8c8d", fontweight="bold")
    ax.annotate("", xy=(3.2, 1.5), xytext=(0.1, 1.5),
                arrowprops=dict(arrowstyle="-", color="#2c7fb8", lw=2))
    ax.text(1.6, 1.25, "0–18 mo history (v1–v3)", ha="center", fontsize=8, color="#2c7fb8")
    ax.annotate("", xy=(3.6, 2.55), xytext=(2.6, 2.9),
                arrowprops=dict(arrowstyle="-|>", color="#d1495b", lw=1.6))
    ax.text(2.4, 2.95, "predict outcome at 18–24 mo", ha="center", fontsize=8, color="#d1495b")
    # modality chips
    chips = [("EHR labs · meds · conditions", "#34495e", True),
             ("CGM (20 glycemic metrics)", "#16a085", cohort == "t1d"),
             ("SDoH (LLM-extracted)", "#8e44ad", True)]
    x = 0.1
    for name, color, present in chips:
        alpha = 1.0 if present else 0.25
        chip = FancyBboxPatch((x, 0.35), 0.28 * len(name) / 4 + 1.6, 0.55,
                              boxstyle="round,pad=0.03,rounding_size=0.1",
                              linewidth=1.2, edgecolor=color, facecolor=color + "22", alpha=alpha)
        ax.add_patch(chip)
        label = name + ("" if present else "  (n/a for T2D)")
        ax.text(x + 0.15, 0.62, label, ha="left", va="center", fontsize=7.2,
                color=color, alpha=alpha)
        x += 0.28 * len(name) / 4 + 1.9
    ax.set_title("2-year prediction task & data modalities", fontsize=11, fontweight="bold", pad=6)


def _plot_outcome_prevalence(ax, ctx):
    src = ctx.get("model_d") if ctx.get("model_d") is not None else ctx.get("agentic")
    if src is None:
        empty_panel(ax, "no prevalence source (nb6/nb5)"); return
    ocol = pick(src, OUT, what="outcome")
    prev = pick(src, ["prevalence", "test_prevalence"], required=False, what="prevalence")
    npos = pick(src, ["n_positive"], required=False, what="n_positive")
    nsamp = pick(src, ["n_samples", "n_test"], required=False, what="n_samples")
    d = src.copy(); d[ocol] = d[ocol].map(pretty_outcome)
    if prev is not None:
        g = d.groupby(ocol)[prev].mean()
    elif npos and nsamp:
        g = d.groupby(ocol).apply(lambda x: x[npos].sum() / max(x[nsamp].sum(), 1))
    else:
        empty_panel(ax, "no prevalence columns"); return
    # per-outcome event / assessable counts for the bar annotations
    npos_by = d.groupby(ocol)[npos].max() if npos else None
    nsamp_by = d.groupby(ocol)[nsamp].max() if nsamp else None

    order = order_outcomes(g.index)[::-1]
    vals = (g.reindex(order) * (100 if g.max() <= 1.0 else 1)).values
    y = np.arange(len(order))
    ax.barh(y, vals, fs.BAR_W, color=fs.PALETTE["blue"], edgecolor="white", linewidth=0.5)
    for yi, oc, v in zip(y, order, vals):
        if pd.notna(v):
            lab = f"{v:.0f}%"
            if npos_by is not None and nsamp_by is not None:
                lab += f"  ({int(npos_by[oc])}/{int(nsamp_by[oc])})"   # events / assessable
            ax.text(v + 1.5, yi, lab, va="center", fontsize=7, color=fs.INK)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8)
    ax.set_xlim(0, max(np.nanmax(vals) * 1.35, 10))   # room for the count labels
    ax.set(xlabel="Prevalence at 2-year target visit (%)", ylabel="",
           title="Outcome landscape (events / assessable)")


def _plot_data_availability(ax, avail):
    """Follow-up coverage over the 10 visit windows: # (and %) of patients with
    observed data at each visit. Reads nb1_data_availability_* (see
    additional_analysis/DATA_AVAILABILITY_INSTRUCTIONS.md)."""
    if avail is None or getattr(avail, "empty", True):
        empty_panel(ax, "no nb1_data_availability\n(run the Foundry analysis)"); return
    vcol = pick(avail, ["visit", "visit_label"], what="visit")
    ncol = pick(avail, ["n_with_data", "n_patients", "n"], what="n_with_data")
    pcol = pick(avail, ["pct_with_data", "coverage", "fraction"], required=False, what="pct")
    d = avail.copy()
    d["_vx"] = d[vcol].map(lambda s: int(re.search(r"(\d+)", str(s)).group(1)))
    d = d.sort_values("_vx")
    x = d["_vx"].values
    n = d[ncol].values.astype(float)
    # shade the input window (v1-v3) and mark the target (v4)
    ax.axvspan(0.5, 3.5, color=fs.PALETTE["blue"], alpha=0.08, lw=0)
    ax.axvline(4, ls="--", color=fs.CEDAR, lw=1.2, alpha=0.8)
    ax.text(4, ax.get_ylim()[1], "target (v4, 24mo)", color=fs.CEDAR,
            fontsize=7, ha="center", va="bottom")
    ax.plot(x, n, "-o", color=fs.PALETTE["blue"], lw=2, markersize=5,
            markeredgecolor="white", markeredgewidth=0.6)
    for xi, ni, (_, r) in zip(x, n, d.iterrows()):
        lab = f"{int(ni)}"
        if pcol is not None and pd.notna(r[pcol]):
            pv = r[pcol] * (100 if r[pcol] <= 1.0 else 1)
            lab += f"\n{pv:.0f}%"
        ax.text(xi, ni, lab, fontsize=6.5, ha="center", va="bottom", color=fs.INK)
    ax.set_xticks(range(1, int(x.max()) + 1))
    ax.set_xticklabels([f"v{i}" for i in range(1, int(x.max()) + 1)], fontsize=7.5)
    ax.set(xlabel="Visit window (6-month intervals)",
           ylabel="Patients with observed data",
           title="Data availability over follow-up")


def build_fig1(ctx, cohort):
    # NOTE: Figure 1 is a hand-composed schematic (assembled by compose_fig1_v2.py
    # into figures/main_figures/fig1_v2.*). This generator emits only the
    # data-driven sub-panels that feed that composite; they are build INPUTS, not
    # final figures, so they are written to
    # additional_analysis/figure_sources/fig1_panels/ rather than to main_figures/.
    panel_dir = os.path.join(_PROJECT, "additional_analysis", "figure_sources", "fig1_panels")
    os.makedirs(panel_dir, exist_ok=True)

    def _save_panel(fig, name):
        for fmt in FORMATS:
            fig.savefig(os.path.join(panel_dir, f"{name}.{fmt}"),
                        dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"    saved -> {os.path.join(panel_dir, name)}.{{{','.join(FORMATS)}}}")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _plot_outcome_prevalence(ax, ctx)
    _save_panel(fig, "fig1C_prevalence")

    # Data-availability-over-time panel (needs nb1_data_availability_*).
    avail = ctx.get("availability")
    if avail is not None:
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        _plot_data_availability(ax, avail)
        _save_panel(fig, "fig1B_availability")
    else:
        print("  [fig1B_availability] skipped: no nb1_data_availability "
              "(see additional_analysis/DATA_AVAILABILITY_INSTRUCTIONS.md)")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Performance landscape (all families)
# ══════════════════════════════════════════════════════════════════
STAT_CLASSES = {"classical_ml", "temporal_ml"}   # non-LLM baselines (deep_learning folded into temporal_ml)
ENSEMBLE_CLASSES = {"deliberative_ensemble"}                      # CEDAR family


LLM_CLASSES = {"single_agent_llm", "deliberative_ensemble"}  # any LLM (excl. naive multi-agent)


def _plot_paradigm_ladder(ax, cost, title):
    """Dumbbell per outcome: best classical/temporal ML baseline vs best LLM
    (single-LLM or CEDAR family) AUC, connected by a line. Shows the LLM-vs-ML
    gap per outcome. The naive multi-agent system is excluded (it is not a
    serious contender; see the mechanism figure)."""
    o = pick(cost, OUT, what="outcome"); a = pick(cost, AUC, what="auc")
    lab = pick(cost, ["model_id", "model_label", "model"], what="model label")
    cls = pick(cost, CLS, what="class")
    d = cost.copy(); d[o] = d[o].map(pretty_outcome)

    def best_of(sub):
        sub = sub.dropna(subset=[a])
        return (None, np.nan) if sub.empty else \
            (sub.loc[sub[a].idxmax(), lab], float(sub[a].max()))

    rows = []
    for oc in order_outcomes(d[o].unique()):
        s = d[d[o] == oc]
        ml_m, ml_a = best_of(s[s[cls].isin(STAT_CLASSES)])
        llm_m, llm_a = best_of(s[s[cls].isin(LLM_CLASSES)])
        rows.append((oc, ml_m, ml_a, llm_m, llm_a))
    # order by the LLM-over-ML gain (largest LLM advantage on top)
    rows.sort(key=lambda r: (np.nan_to_num(r[4]) - np.nan_to_num(r[2])))
    y = np.arange(len(rows))
    c_ml = class_color("classical_ml"); c_llm = class_color("single_agent_llm")
    for i, (oc, ml_m, ml_a, llm_m, llm_a) in enumerate(rows):
        if not (np.isnan(ml_a) or np.isnan(llm_a)):
            ax.plot([ml_a, llm_a], [i, i], color="#b8b8b8", lw=2.6, zorder=1)
        if not np.isnan(ml_a):
            ax.scatter(ml_a, i, s=95, color=c_ml, zorder=3,
                       edgecolor="white", linewidth=0.8)
        if not np.isnan(llm_a):
            ax.scatter(llm_a, i, s=110, color=c_llm, zorder=3, marker="^",
                       edgecolor="white", linewidth=0.8)
            dlt = llm_a - ml_a
            ax.text(max(ml_a, llm_a) + 0.014, i, f"{dlt:+.02f}", va="center",
                    ha="left", fontsize=11, fontweight="bold",
                    color=("#3a7d3a" if dlt >= 0 else "#888"))
    ax.axvline(0.5, ls="--", color="gray", lw=1, alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlim(0.45, 1.02)
    ax.set(ylabel="")
    ax.set_xlabel("ROC AUC", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", markersize=9, color=c_ml,
                   label="Best classical/temporal ML"),
        plt.Line2D([], [], marker="^", ls="", markersize=11, color=c_llm,
                   label="Best LLM (single or CEDAR)"),
    ], frameon=False, fontsize=11, loc="lower left")


def _plot_best_per_outcome(ax, cost, title):
    o = pick(cost, OUT, what="outcome"); a = pick(cost, AUC, what="auc")
    lab = pick(cost, ["model_id", "model_label", "model"], what="model label")
    cls = pick(cost, CLS, required=False, what="class")
    lo = pick(cost, CILO, required=False, what="ci_low")
    hi = pick(cost, CIHI, required=False, what="ci_high")
    idx = cost.groupby(o)[a].idxmax()
    best = cost.loc[idx].copy()
    best[o] = best[o].map(pretty_outcome)
    order = order_outcomes(best[o].unique())
    best = best.set_index(o).reindex(order).reset_index()
    x = np.arange(len(best))
    colors = [class_color(best[cls].iloc[i]) if cls else fs.INK for i in range(len(best))]
    yerr = ci_err(best[a], best[lo], best[hi]) if (lo and hi) else None
    ax.bar(x, best[a].values, fs.BAR_W, yerr=yerr, capsize=2.5, color=colors,
           edgecolor="white", linewidth=0.6,
           error_kw={"lw": 0.9, "ecolor": fs.INK})
    ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{o_}\n({short_model(m)})" for o_, m in zip(best[o], best[lab])],
                       rotation=25, ha="right", fontsize=8)
    ax.set_xlim(-0.7, len(best) - 0.3)     # breathing room so bars aren't stretched
    ax.set_ylim(0.0, 1.0)
    ax.set(ylabel="ROC AUC (best model, 95% CI)", title=title)
    if cls:
        present = class_order(best[cls].unique())
        ax.legend(handles=[Patch(facecolor=class_color(c), label=CLASS_PRETTY.get(c, c))
                           for c in present], frameon=False, fontsize=7.5, loc="upper right")


def _plot_config_heatmap(ax, cost, show_legend=True, class_rank=None):
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LinearSegmentedColormap
    o = pick(cost, OUT, what="outcome"); a = pick(cost, AUC, what="auc")
    lab = pick(cost, ["model_id", "model_label", "model"], what="model label")
    cls = pick(cost, CLS, required=False, what="class")
    d = cost.copy(); d[o] = d[o].map(pretty_outcome)
    d["_short"] = d[lab].map(short_model)
    mat = d.pivot_table(index="_short", columns=o, values=a, aggfunc="max")
    mat = mat.reindex(columns=[c for c in order_outcomes(mat.columns)])

    # Row order: GROUP by algorithm class (best class first), and within a class
    # by mean AUC. This keeps each class a contiguous block we can separate with
    # horizontal dividers. `class_rank` overrides the default block order (used by
    # the reframed Fig 2 to keep the two LLM subgroups adjacent, then ML).
    row_cls = d.drop_duplicates("_short").set_index("_short")[cls].to_dict() if cls else {}
    row_mean = mat.mean(axis=1)
    rank_list = class_rank if class_rank is not None else FIG_CLASS_ORDER
    if cls:
        def _cls_rank(c):
            return rank_list.index(c) if c in rank_list else len(rank_list)
        row_order = sorted(mat.index,
                           key=lambda r: (_cls_rank(row_cls.get(r)), -row_mean[r]))
    else:
        row_order = row_mean.sort_values(ascending=False).index
    mat = mat.reindex(row_order)

    # Single-hue sequential colormap matching the palette (white -> palette blue).
    seq = LinearSegmentedColormap.from_list(
        "auc_seq", ["#ffffff", "#dce6f5", "#9bb8e0", fs.PALETTE["blue"], "#2a4a7a"])
    sns.heatmap(mat, annot=True, fmt=".2f", cmap=seq, vmin=0.5, vmax=0.9,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "ROC AUC", "shrink": 0.6}, ax=ax)
    # highlight the best model for each outcome (per-column max)
    for cj, col in enumerate(mat.columns):
        colvals = mat[col].values.astype(float)
        if np.all(np.isnan(colvals)):
            continue
        ri = int(np.nanargmax(colvals))
        ax.add_patch(Rectangle((cj, ri), 1, 1, fill=False,
                               edgecolor="#f1c40f", lw=2.6, zorder=6))
    ax.set(xlabel="", ylabel="", title="All models \u00d7 outcomes (gold = best per outcome)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=7.5)

    # color y labels by algorithm class + draw horizontal dividers between blocks
    if cls:
        for t in ax.get_yticklabels():
            t.set_color(class_color(row_cls.get(t.get_text())))
            t.set_fontsize(8)
        # divider lines at each class boundary
        blocks = [row_cls.get(r) for r in mat.index]
        for i in range(1, len(blocks)):
            if blocks[i] != blocks[i - 1]:
                ax.axhline(i, color=fs.INK, lw=1.4, zorder=5)
        # class-color legend (what the row-label colors mean). By default drawn
        # below this panel, but can be suppressed so build_fig2 draws ONE shared
        # legend spanning panels B and C (avoids the tall-whitespace problem).
        if show_legend:
            present = class_order([c for c in blocks if c is not None])
            ax.legend(handles=[Patch(facecolor=class_color(c), label=CLASS_PRETTY.get(c, c))
                               for c in present],
                      frameon=False, fontsize=6.5, loc="upper center",
                      bbox_to_anchor=(0.5, -0.38), ncol=3,
                      title="Model class", title_fontsize=7)


# Supergroup structure for the reframed Fig 2: two families, each split into two
# subgroups. Multi-agent is intentionally excluded from the main landscape (its
# single naive implementation is discussed in the supplement).
SUPERGROUP = {
    "classical_ml":          ("ML",  "Classical"),
    "temporal_ml":           ("ML",  "Temporal"),
    "single_agent_llm":      ("LLM", "Single LLM"),
    "deliberative_ensemble": ("LLM", f"{METHOD_NAME}"),
}
# left-to-right subgroup order within the grouped panel
SUBGROUP_ORDER = ["classical_ml", "temporal_ml", "single_agent_llm",
                  "deliberative_ensemble"]
SUPER_COLOR = {"ML": fs.CLASS_COLORS["classical_ml"],
               "LLM": fs.CLASS_COLORS["single_agent_llm"]}
# Heatmap row-block order for the reframed Fig 2: LLM block on top (CEDAR then
# single LLM, the two adjacent LLM subgroups), then the ML block.
_FIG2_ROW_RANK = ["deliberative_ensemble", "single_agent_llm",
                  "classical_ml", "temporal_ml"]


def _plot_ml_vs_llm_box(ax, cost, title):
    """Grouped AUC box/strip: two supergroups (ML, LLM), each split into two
    subgroups (ML: Classical, Temporal; LLM: Single LLM, CEDAR). A bracket marks
    each supergroup and the ML-vs-LLM significance is drawn between them. Multi-
    agent is excluded. This is the primary panel for Results section 1."""
    a = pick(cost, AUC, what="auc"); cls = pick(cost, CLS, required=False, what="class")
    if cls is None:
        empty_panel(ax, "no algorithm_class column"); return
    d = cost.dropna(subset=[a]).copy()
    d = d[d[cls].isin(SUBGROUP_ORDER)]                      # drop multi-agent
    order = [c for c in SUBGROUP_ORDER if c in set(d[cls])]
    pos = {c: i for i, c in enumerate(order)}
    # box + strip per subgroup, colored by its own class color
    for c in order:
        v = d[d[cls] == c][a].values
        col = class_color(c)
        bp = ax.boxplot(v, positions=[pos[c]], widths=0.55, showfliers=False,
                        patch_artist=True, medianprops=dict(color=fs.INK, lw=1.2),
                        whiskerprops=dict(color=col, lw=1.0),
                        capprops=dict(color=col, lw=1.0),
                        boxprops=dict(facecolor=fs._lighten(col, 0.45),
                                      edgecolor=col, lw=1.2))
        jit = (np.random.RandomState(len(v)).rand(len(v)) - 0.5) * 0.28
        ax.scatter(pos[c] + jit, v, s=10, color=fs.INK, alpha=0.4, zorder=3)
    ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SUPERGROUP[c][1] for c in order], fontsize=9)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set(ylabel="ROC AUC", title=title)

    # supergroup brackets under the axis + ML vs LLM significance above
    ml_pos  = [pos[c] for c in order if SUPERGROUP[c][0] == "ML"]
    llm_pos = [pos[c] for c in order if SUPERGROUP[c][0] == "LLM"]
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    for grp, gp in [("ML", ml_pos), ("LLM", llm_pos)]:
        if not gp:
            continue
        lo, hi = min(gp), max(gp)
        yb = y0 - 0.11 * span
        ax.plot([lo - 0.32, hi + 0.32], [yb, yb], color=SUPER_COLOR[grp],
                lw=2.4, clip_on=False, solid_capstyle="round")
        ax.text((lo + hi) / 2, yb - 0.045 * span, grp, ha="center", va="top",
                fontsize=11, fontweight="bold", color=SUPER_COLOR[grp],
                clip_on=False)
    ax.set_ylim(y0 - 0.02 * span, y1)
    # ML-vs-LLM Mann-Whitney over pooled model x outcome AUCs
    if ml_pos and llm_pos:
        ml_v = d[d[cls].isin([c for c in order if SUPERGROUP[c][0] == "ML"])][a].values
        llm_v = d[d[cls].isin([c for c in order if SUPERGROUP[c][0] == "LLM"])][a].values
        star = fs.stars(fs.mwu_p(ml_v, llm_v))
        if star:
            xml = np.mean(ml_pos); xllm = np.mean(llm_pos)
            yb = y1 - 0.04 * span
            ax.plot([xml, xml, xllm, xllm],
                    [yb - 0.02 * span, yb, yb, yb - 0.02 * span],
                    color=fs.INK, lw=1.1)
            ax.text((xml + xllm) / 2, yb, star, ha="center", va="bottom",
                    fontsize=12, fontweight="bold")


def _plot_class_box(ax, cost):
    a = pick(cost, AUC, what="auc"); cls = pick(cost, CLS, required=False, what="class")
    if cls is None:
        empty_panel(ax, "no algorithm_class column"); return
    d = cost.dropna(subset=[a]); order = fig_class_order(d[cls].unique())
    pal = {c: class_color(c) for c in order}
    sns.boxplot(data=d, x=cls, y=a, order=order, hue=cls, hue_order=order, palette=pal,
                legend=False, showfliers=False, width=0.55, linewidth=0.9, ax=ax)
    sns.stripplot(data=d, x=cls, y=a, order=order, ax=ax, color=fs.INK,
                  size=2.5, alpha=0.45, jitter=0.2)
    ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CLASS_PRETTY.get(c, c).replace(" ", "\n") for c in order], fontsize=7)
    ax.set(xlabel="", ylabel="ROC AUC", title="AUC by paradigm")

    # Significance: deliberative ensemble (CEDAR) vs each other paradigm
    # (Mann-Whitney U over pooled model x outcome AUCs). Stars stack inside the
    # panel — expand the y-limit first so the brackets never overflow the axes.
    ref = "deliberative_ensemble"
    if ref in order:
        vals = {c: d[d[cls] == c][a].values for c in order}
        n_other = sum(1 for c in order if c != ref
                      and fs.stars(fs.mwu_p(vals[ref], vals[c])))
        y0, y1 = ax.get_ylim()
        data_top = max(np.nanmax(v) for v in vals.values() if len(v))
        # reserve headroom above the tallest box for the stacked brackets
        ax.set_ylim(y0, max(y1, data_top + 0.06 * (n_other + 1)))
        fs.add_sig_over_reference(ax, vals, ref, order,
                                  top=data_top, start_frac=0.02, gap_frac=0.055)


def _drop_multiagent(cost):
    """Return a copy of the cost table with the naive multi-agent class removed,
    for the reframed landscape (multi-agent lives in the supplement)."""
    if cost is None:
        return None
    cls = pick(cost, CLS, required=False, what="class")
    if cls is None:
        return cost.copy()
    return cost[cost[cls] != "multi_agent_llm"].copy()


def build_fig2(ctx_t2d, ctx_t1d):
    """Reframed performance landscape: the story is ML vs LLM. Two supergroups
    (ML = classical + temporal; LLM = single LLM + CEDAR), CEDAR shown as an LLM
    subgroup next to the single LLM. The naive multi-agent system is EXCLUDED here
    (discussed in the supplement). Layout per cohort: grouped ML-vs-LLM box (with
    supergroup brackets) + the model x outcome heatmap; T2D on top, T1D below;
    plus the per-outcome best-ML-vs-best-LLM dumbbell for T2D."""
    cost2 = _drop_multiagent(ctx_t2d.get("cost"))
    cost1 = _drop_multiagent(ctx_t1d.get("cost"))
    if cost2 is None:
        print("  [skip] fig2: nb7_cost_comparison_t2d unavailable."); return
    fig = plt.figure(figsize=(13, 14))
    # 3 rows: T2D (box | heatmap) | T1D (box | heatmap) | dumbbell (centered).
    outer = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.1, 1.0], hspace=0.36)

    r1 = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.6], wspace=0.28)
    axA = fig.add_subplot(r1[0, 0]); panel_label(axA, "A")
    _plot_ml_vs_llm_box(axA, cost2, "T2D: ML vs LLM (AUC by subgroup)")
    axB = fig.add_subplot(r1[0, 1]); panel_label(axB, "B")
    _plot_config_heatmap(axB, cost2, show_legend=False, class_rank=_FIG2_ROW_RANK)
    axB.set_title("T2D: models × outcomes (gold = best per outcome)")

    r2 = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.6], wspace=0.28)
    if cost1 is not None:
        axC = fig.add_subplot(r2[0, 0]); panel_label(axC, "C")
        _plot_ml_vs_llm_box(axC, cost1, "T1D: ML vs LLM (AUC by subgroup)")
        axD = fig.add_subplot(r2[0, 1]); panel_label(axD, "D")
        _plot_config_heatmap(axD, cost1, show_legend=False, class_rank=_FIG2_ROW_RANK)
        axD.set_title("T1D: models × outcomes (gold = best per outcome)")

    r3 = outer[2].subgridspec(1, 3, width_ratios=[0.15, 0.70, 0.15])
    axE = fig.add_subplot(r3[0, 1]); panel_label(axE, "E")
    _plot_paradigm_ladder(axE, cost2,
                          "T2D: best ML vs best LLM per outcome (LLM$-$ML gain)")
    save_fig(fig, "fig2")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Mechanism
# ══════════════════════════════════════════════════════════════════
def _auc_by_outcome(df, config_val=None):
    """Return Series outcome->max AUC (optionally for a single config)."""
    if df is None:
        return pd.Series(dtype=float)
    o = pick(df, OUT, what="outcome"); a = pick(df, AUC, what="auc")
    c = pick(df, CFG, required=False, what="config")
    d = df.copy(); d[o] = d[o].map(pretty_outcome)
    if config_val is not None and c is not None:
        d = d[d[c] == config_val]
    if d.empty:
        return pd.Series(dtype=float)
    return d.groupby(o)[a].max()


def _plot_collapse(ax, agentic, model_d):
    # keys map to technical config ids so colors come from MODEL_COLORS
    series = {
        "model_d_full": (METHOD_NAME, _auc_by_outcome(model_d, "model_d_full")
                         if model_d is not None else pd.Series(dtype=float)),
        "model_a": ("Vanilla (single-agent)", _auc_by_outcome(agentic, "model_a")),
        "model_c": ("CoT (single-agent)", _auc_by_outcome(agentic, "model_c")),
    }
    if all(s.empty for _, s in series.values()):
        empty_panel(ax, "no agentic / Model D results"); return
    outcomes = order_outcomes(sorted(set().union(
        *[set(s.index) for _, s in series.values() if not s.empty])))
    x = np.arange(len(outcomes)); width = 0.8 / len(series)
    for k, (cfg, (name, s)) in enumerate(series.items()):
        vals = s.reindex(outcomes).values
        ax.bar(x + k * width, np.nan_to_num(vals), width, label=name,
               color=model_color(cfg))
    ax.axhline(0.5, ls="--", color="black", lw=1.2, alpha=0.8)
    ax.text(len(outcomes) - 0.4, 0.505, "chance", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(outcomes, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set(ylabel="ROC AUC",
           title=f"Naive single agents drop toward/below chance; {METHOD_NAME} holds")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))


def _plot_progression(ax, agentic, model_d):
    """Architecture progression: Vanilla -> Multi -> CoT -> CEDAR per outcome.
    This subsumes the old 'collapse' panel (naive single/multi agents drop toward
    chance) AND the progression story. Soft bars for baselines (colour outline +
    light fill + hatch); CEDAR drawn SOLID red to draw the eye."""
    steps = [("model_a", "Vanilla", agentic), ("model_b", "Multi", agentic),
             ("model_c", "CoT", agentic), ("model_d_full", METHOD_NAME, model_d)]
    cols = {}
    for cfg, lab, src in steps:
        s = _auc_by_outcome(src, cfg)
        if not s.empty:
            cols[cfg] = (lab, s)
    if not cols:
        empty_panel(ax, "no progression data"); return
    keys = list(cols.keys())
    hatch_by = {"model_a": "///", "model_b": "...", "model_c": "\\\\\\", "model_d_full": ""}
    outcomes = order_outcomes(sorted(set().union(*[set(s.index) for _, s in cols.values()])))
    x = np.arange(len(outcomes)); width = 0.82 / len(cols)
    for k, cfg in enumerate(keys):
        lab, s = cols[cfg]
        vals = np.nan_to_num(s.reindex(outcomes).values)
        col = model_color(cfg)
        emph = cfg == "model_d_full"
        face = col if emph else fs._lighten(col, 0.5)
        edge = "white" if emph else col
        ax.bar(x + k * width, vals, width, label=lab, color=face,
               edgecolor=edge, linewidth=1.2, hatch=(None if emph else hatch_by.get(cfg)))
    ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.8)
    ax.text(len(outcomes) - 0.35, 0.505, "chance", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x + width * (len(cols) - 1) / 2)
    ax.set_xticklabels(outcomes, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set(ylabel="ROC AUC",
           title=f"Architecture progression: naive agents drop toward chance, {METHOD_NAME} holds")
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.14))


def _plot_feature_attention(ax, ctx):
    """Fig 3C: which features CEDAR cites/attends to, per outcome (mean cited
    importance). Reads nb6_feature_attention_t2d (see
    additional_analysis/FEATURE_IMPORTANCE_INSTRUCTIONS.md). Placeholder until
    that small export is available."""
    fa = ctx.get("feat_attn")
    if fa is None or getattr(fa, "empty", True):
        empty_panel(ax, "feature attention\n(run nb6_feature_attention_t2d export;\n"
                        "see FEATURE_IMPORTANCE_INSTRUCTIONS.md)"); return
    o = pick(fa, OUT, what="outcome"); feat = pick(fa, ["feature"], what="feature")
    imp = pick(fa, ["mean_importance", "importance"], what="importance")
    cit = pick(fa, ["n_citations", "citations"], required=False, what="citations")
    d = fa.copy(); d[o] = d[o].map(pretty_outcome)
    d[feat] = d[feat].map(pretty_feature)
    # Rank features so the matrix is dense and interpretable: features cited
    # across MANY outcomes first (breadth), then by how often they are cited.
    # This surfaces the shared clinical drivers rather than one-off citations.
    agg = d.groupby(feat).agg(n_out=(o, "nunique"),
                              tot_cit=((cit or imp), "sum"),
                              mimp=(imp, "mean"))
    top_feats = agg.sort_values(["n_out", "tot_cit", "mimp"],
                                ascending=False).head(14).index
    mat = (d[d[feat].isin(top_feats)]
           .pivot_table(index=feat, columns=o, values=imp, aggfunc="mean")
           .reindex(index=top_feats))
    mat = mat.reindex(columns=order_outcomes(mat.columns))
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("imp", ["#ffffff", "#f0c3b8", fs.CEDAR])
    # annotate the cited-importance value in each populated cell (blank for NaN);
    # white text on the dark high-importance cells, ink on the light ones.
    annot = mat.applymap(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    sns.heatmap(mat, cmap=seq, vmin=0, vmax=1, linewidths=0.6, linecolor="white",
                cbar_kws={"label": "Mean cited importance", "shrink": 0.55},
                annot=annot.values, fmt="", annot_kws={"fontsize": 6.5}, ax=ax)
    # recolor annotation text for contrast against the cell fill
    for t in ax.texts:
        try:
            val = float(t.get_text())
        except ValueError:
            continue
        t.set_color("white" if val >= 0.6 else fs.INK)
    # Mark cells the model never cited (structurally empty) with a light hatch so
    # readers don't confuse "not cited" with "cited, low importance".
    for iy in range(mat.shape[0]):
        for ix in range(mat.shape[1]):
            if pd.isna(mat.iloc[iy, ix]):
                ax.add_patch(plt.Rectangle((ix, iy), 1, 1, fill=True,
                             facecolor="#f4f4f4", edgecolor="white", lw=0.6, zorder=1))
    ax.set(xlabel="", ylabel="", title="What CEDAR cites (features shared across outcomes)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=7.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7.5)


def _plot_mechanism(ax, model_d):
    if model_d is None:
        empty_panel(ax, "no nb6_model_d_results"); return
    o = pick(model_d, OUT, what="outcome"); a = pick(model_d, AUC, what="auc")
    c = pick(model_d, CFG, what="config")
    lo = pick(model_d, CILO, required=False, what="ci_low")
    hi = pick(model_d, CIHI, required=False, what="ci_high")
    order = ["model_d_full", "model_c_plus_verify", "model_c_plus_sc"]
    present = [cc for cc in order if cc in set(model_d[c])] or list(model_d[c].unique())
    means = model_d.groupby(c)[a].mean().reindex(present)
    if lo and hi:
        hw = ((model_d[hi] - model_d[lo]) / 2)
        err = model_d.assign(_hw=hw).groupby(c)["_hw"].mean().reindex(present)
        yerr = np.nan_to_num(err.values)
    else:
        yerr = None
    x = np.arange(len(present))
    colors = [model_color(cc) for cc in present]   # CEDAR-family reds (consistent)
    # CEDAR-full solid; the two ablations soft (light fill + colour outline).
    emph = present.index("model_d_full") if "model_d_full" in present else None
    fs.soft_bars(ax, x, means.values, colors, width=fs.BAR_W, emphasis=emph, yerr=yerr)
    for xi, v in zip(x, means.values):
        if pd.notna(v):
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.8)
    ax.set_xticks(x)
    labmap = {"model_d_full": f"{METHOD_NAME}\n(full)", "model_c_plus_verify": "CoT\n+Verify",
              "model_c_plus_sc": "CoT\n+SC"}
    ax.set_xticklabels([labmap.get(cc, cc) for cc in present], fontsize=8)
    ax.set_xlim(-0.7, len(present) - 0.3)
    ax.set_ylim(0.5, 1.0)
    ax.set(ylabel="Mean ROC AUC (across outcomes)", title="Mechanism ablation")

    # Significance: full CEDAR vs each ablation (Mann-Whitney U over per-outcome
    # AUCs). Expected n.s. — that IS the result: the deliberative variants are
    # statistically indistinguishable. Start the brackets clearly ABOVE the SD
    # whiskers so the n.s. markers don't overlap the error bars.
    if "model_d_full" in present:
        vals = {cc: model_d[model_d[c] == cc][a].values for cc in present}
        whisker_top = float(np.nanmax(means.values + (yerr if yerr is not None else 0)))
        fs.add_sig_over_reference(ax, vals, "model_d_full", present,
                                  top=whisker_top + 0.035,
                                  start_frac=0.0, gap_frac=0.055)


def _plot_worked_example(ax, ctx):
    """A CEDAR output rendered as a clinician-facing 'card': the predicted
    probability plus a short table of cited (feature, visit, value) evidence and
    the verifier status. This is what a single LLM's bare probability cannot
    provide.

    SYNTHETIC EXAMPLE. Every number and reasoning phrase below is a fabricated
    composite, not any real patient's chart -- constructed to be representative
    of a high-grounding, heavily-cited CEDAR output (see Section 2 of the
    Results/faithfulness audit for the real aggregate statistics this panel
    illustrates). Do not repopulate this from a specific patient row; an earlier
    version of this panel used one real, identifiable patient's actual cited
    values and reasoning text, which is a PHI exposure risk in a public
    manuscript figure and repository -- keep this panel synthetic."""
    ax.axis("off")
    prob = 0.83
    n_cit, n_ver = 46, 46
    outcome = "Optimal Glycemic Control"

    # Fabricated (feature, visit, value, direction) rows -- illustrative only,
    # not drawn from any real patient record. See the docstring above.
    cited = [
        ("Fasting glucose", "v1 / v3", "96 / 89 mg/dL", "favorable"),
        ("Serum C-peptide", "v1", "3.6 ng/mL", "favorable"),
        ("Glycemic control (prior)", "v1–v3", "controlled x3", "favorable"),
        ("Insulin independence (prior)", "v2–v3", "achieved", "favorable"),
        ("Metformin response (prior)", "v3", "achieved", "favorable"),
    ]

    # header card
    ax.text(0.0, 1.0, f"CEDAR output — {outcome}", fontsize=12.5, fontweight="bold",
            color=fs.CEDAR, transform=ax.transAxes, va="top")
    ax.text(0.0, 0.925, f"Predicted probability: {prob:.2f}", fontsize=12,
            fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(0.0, 0.86,
            f"{n_ver} of {n_cit} cited features verified against the record",
            fontsize=10, color=fs.INK, transform=ax.transAxes, va="top")
    ax.add_patch(plt.Rectangle((-0.01, 0.83), 1.02, 0.19, transform=ax.transAxes,
                 facecolor=fs._lighten(fs.CEDAR, 0.85), edgecolor=fs.CEDAR,
                 lw=1.0, zorder=0, clip_on=False))
    # cited-evidence table header
    y = 0.75
    ax.text(0.0, y, "Cited evidence (checkable against the chart):", fontsize=10.5,
            fontweight="bold", transform=ax.transAxes, va="top")
    y -= 0.075
    ax.text(0.02, y, "Feature", fontsize=9.5, fontweight="bold", color="#555",
            transform=ax.transAxes, va="top")
    ax.text(0.52, y, "Visit", fontsize=9.5, fontweight="bold", color="#555",
            transform=ax.transAxes, va="top")
    ax.text(0.70, y, "Value", fontsize=9.5, fontweight="bold", color="#555",
            transform=ax.transAxes, va="top")
    y -= 0.018
    ax.plot([0.0, 1.0], [y, y], color="#ccc", lw=0.8, transform=ax.transAxes)
    y -= 0.062
    for feat, visit, val, direction in cited:
        ax.text(0.02, y, feat, fontsize=9.6, color=fs.INK, transform=ax.transAxes, va="top")
        ax.text(0.52, y, visit, fontsize=9.4, color="#444", transform=ax.transAxes, va="top")
        ax.text(0.70, y, val, fontsize=9.4, color="#444", transform=ax.transAxes, va="top")
        ax.scatter(0.975, y - 0.015, s=30, marker="^", color="#3a7d3a",
                   transform=ax.transAxes, clip_on=False)
        y -= 0.078
    # Fabricated reasoning excerpt in CEDAR's S1 chain-of-thought style --
    # illustrative only, not drawn from any real patient's output. Shows the
    # agent EXPLAINS its call, unlike a classical model that returns only a
    # score. See the docstring above.
    y -= 0.012
    ax.text(0.0, y, "Model's stated reasoning (excerpt):", fontsize=9.3,
            fontweight="bold", transform=ax.transAxes, va="top")
    y -= 0.05
    reasoning = [
        "“Maintained optimal control at all three visits, with",
        "near-normal fasting glucose (96→89 mg/dL) — the single",
        "strongest predictor of continued control. Robust C-peptide",
        "(3.6 ng/mL) indicates preserved beta-cell reserve; per",
        "ADA 2024 / TODAY this favors glycemic durability.",
        "→ High probability of continued optimal control.”",
    ]
    for ln in reasoning:
        ax.text(0.02, y, ln, fontsize=8.0, color="#333", style="italic",
                transform=ax.transAxes, va="top")
        y -= 0.046
    y -= 0.01
    ax.text(0.0, y,
            f"A classical model returns only the {prob:.2f} — no citations, no rationale.",
            fontsize=8.0, color="#666", transform=ax.transAxes, va="top")
    ax.set_title("What CEDAR adds beyond a risk score", fontsize=11,
                 fontweight="bold", loc="left")


def build_fig3(ctx, cohort):
    """Reframed Fig 3 — 'what CEDAR adds'. Because CEDAR ties a single LLM on
    accuracy, this figure shows its distinct value: faithful, auditable reasoning.
    (A) a synthetic worked CEDAR output (cited evidence + verification), illustrating
    the real aggregate statistics in panel B; (B) faithfulness
    metrics; (C) group-masking (evidence is collectively load-bearing); (D) the
    features CEDAR cites across outcomes. The old architecture-progression and
    mechanism-ablation panels move to the supplement."""
    faith, gm = ctx.get("faith"), ctx.get("gmask")
    if faith is None and gm is None:
        print("  [skip] fig3: no faithfulness datasets."); return
    fig = plt.figure(figsize=(14, 10.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.42)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22)
    axA = fig.add_subplot(top[0, 0]); panel_label(axA, "A")
    _plot_worked_example(axA, ctx)
    axB = fig.add_subplot(top[0, 1]); panel_label(axB, "B")
    _plot_faith_metrics(axB, faith, legend=True)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.28)
    axC = fig.add_subplot(bottom[0, 0]); panel_label(axC, "C")
    _plot_masking_curve(axC, gm, legend=True, cohort="t2d")
    axD = fig.add_subplot(bottom[0, 1]); panel_label(axD, "D")
    _plot_feature_attention(axD, ctx)
    save_fig(fig, "fig3")


def build_supp_mechanism(ctx, cohort):
    """Supplementary: the naive multi-agent collapse + architecture progression +
    CEDAR mechanism ablation (CoT+Verify / CoT+SC). Moved out of the main text
    because CEDAR ties a single LLM, so an architecture-progression framing would
    overstate the gain; retained as supporting evidence for 'a single pass is
    enough'."""
    agentic, model_d = ctx.get("agentic"), ctx.get("model_d")
    if agentic is None and model_d is None:
        print("  [skip] supp mechanism: no agentic/Model D results."); return
    fig = plt.figure(figsize=(14, 10.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.5)
    axA = fig.add_subplot(outer[0]); panel_label(axA, "A")
    _plot_progression(axA, agentic, model_d)
    axB = fig.add_subplot(outer[1]); panel_label(axB, "B")
    _plot_mechanism(axB, model_d)
    save_fig(fig, "figS_mechanism")


# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — Faithfulness & trust
# ══════════════════════════════════════════════════════════════════
def _faith_boxplot(ax, d, o, col, outcomes, color, ylabel, show_xticks):
    """Per-outcome box plot with jittered per-patient points for one metric."""
    data = [pd.to_numeric(d[d[o] == oc][col], errors="coerce").dropna().values
            for oc in outcomes]
    x = np.arange(len(outcomes))
    bp = ax.boxplot(data, positions=x, widths=0.6, patch_artist=True,
                    showfliers=False, medianprops=dict(color="#333", lw=1.4),
                    whiskerprops=dict(color=color, lw=1.0),
                    capprops=dict(color=color, lw=1.0),
                    boxprops=dict(facecolor=fs._lighten(color, 0.55),
                                  edgecolor=color, lw=1.2))
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        jx = x[i] + rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(jx, vals, s=12, color=color, edgecolor="white",
                   linewidth=0.4, alpha=0.7, zorder=3)
    ax.set_xticks(x)
    if show_xticks:
        ax.set_xticklabels(outcomes, rotation=25, ha="right", fontsize=7.5)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(ylabel, fontsize=8)


def _plot_faith_metrics(ax, faith, legend=True, title="Faithfulness: per-patient spread"):
    """Two stacked box-plot panels (sufficiency + comprehensiveness) showing the
    per-patient spread; extractive grounding is reported as text because it is a
    near-ceiling fabrication check (nearly every cited value verifies), so a bar
    at 1.0 is uninformative and looks like a glitch."""
    if faith is None or faith.empty:
        empty_panel(ax, "no nb6_faithfulness_results"); return
    o = pick(faith, OUT, what="outcome")
    suf = pick(faith, ["sufficiency_score", "sufficiency"], False, "suf")
    cmp_ = pick(faith, ["comprehensiveness_score", "comprehensiveness"], False, "cmp")
    d = faith.copy(); d[o] = d[o].map(pretty_outcome)
    outcomes = order_outcomes(d[o].unique())

    # extractive-grounding counts for the annotation
    nc = pd.to_numeric(faith.get("extractive_n_correct"), errors="coerce").sum()
    nt = pd.to_numeric(faith.get("extractive_n_total"), errors="coerce").sum()
    if nt and nt > 0:
        pct = 100 * nc / nt
        ground_txt = (f"Extractive grounding (audited subset): {int(nc):,}/{int(nt):,} "
                      f"cited values verified ({pct:.1f}%); {int(nt - nc)} mismatched")
    else:
        ground_txt = "Extractive grounding: near-ceiling (fabrication check)"

    # split the caller's cell into two stacked sub-axes
    fig = ax.figure
    ax.axis("off")
    sub = ax.get_subplotspec().subgridspec(2, 1, height_ratios=[1.6, 1.0],
                                           hspace=0.12)
    ax_top = fig.add_subplot(sub[0, 0])
    ax_bot = fig.add_subplot(sub[1, 0])

    if suf is not None:
        _faith_boxplot(ax_top, d, o, suf, outcomes, "#2980b9",
                       "Sufficiency", show_xticks=False)
    # title above, then the grounding annotation on a separate line below it
    ax_top.set_title(title, fontsize=10, fontweight="bold", pad=22)
    ax_top.text(0.5, 1.035, ground_txt, transform=ax_top.transAxes,
                ha="center", va="bottom", fontsize=7.6, color="#27ae60",
                fontweight="bold")

    if cmp_ is not None:
        _faith_boxplot(ax_bot, d, o, cmp_, outcomes, "#e67e22",
                       "Comprehensiveness", show_xticks=True)
    ax_bot.text(0.02, 0.92,
                "low = no single citation individually decisive",
                transform=ax_bot.transAxes, ha="left", va="top",
                fontsize=6.8, style="italic", color="#e67e22")


def _plot_masking_curve(ax, gm, legend=True, cohort="t2d"):
    """Per-outcome CITED masking curves plus a pooled RANDOM-masking baseline.

    Preferred source is the cited-vs-random baseline dump
    (masking_group_analysis_baseline/); if absent, falls back to the old
    per-outcome-only curve from nb6_group_masking_curve (`gm`)."""
    mb = load_masking_baseline(cohort)

    if mb is not None and mb.get("per_outcome") is not None:
        po = mb["per_outcome"].copy()
        cited = po[po["arm"].astype(str).str.lower() == "cited"].copy()
        cited["outcome"] = cited["outcome"].map(pretty_outcome)
        outcomes = order_outcomes(cited["outcome"].unique())
        palette = sns.color_palette("tab10", len(outcomes))
        for i, oc in enumerate(outcomes):
            s = (cited[cited["outcome"] == oc]
                 .groupby("k")["mean_abs_delta"].mean().sort_index())
            ax.plot(s.index, s.values, marker="o", lw=1.6, color=palette[i],
                    label=oc, zorder=3)
        # pooled RANDOM baseline (dashed control line + SD band)
        pooled = mb.get("pooled")
        if pooled is not None:
            rnd = (pooled[pooled["arm"].astype(str).str.lower() == "random"]
                   .groupby("k")[["mean_abs_delta", "sd_abs_delta"]]
                   .mean().sort_index())
            if not rnd.empty:
                x = rnd.index.values
                y = rnd["mean_abs_delta"].values
                sd = rnd["sd_abs_delta"].fillna(0).values
                ax.plot(x, y, ls="--", lw=2.0, color="#555555", marker="s",
                        markersize=4, label="Random-masking baseline (pooled)",
                        zorder=4)
                ax.fill_between(x, y - sd, y + sd, color="#555555", alpha=0.12,
                                zorder=1)
        ax.set(xlabel="Top-K features masked together", ylabel="Mean |\u0394 prob|",
               title="Cited evidence is load-bearing (group-masking)")
        ax.set_ylim(bottom=0)
        if legend:
            ax.legend(frameon=False, fontsize=6.0, ncol=2, loc="upper left",
                      handlelength=1.4, columnspacing=0.9, labelspacing=0.3)
        return

    # ---- fallback: original per-outcome-only curve ----
    if gm is None or gm.empty:
        empty_panel(ax, "no masking baseline / nb6_group_masking_curve"); return
    o = pick(gm, OUT, what="outcome")
    delta = pick(gm, ["delta", "counterfactual_delta"], what="delta")
    kcol = pick(gm, ["k", "n_masked", "top_k"], what="k")
    d = gm.copy(); d[o] = d[o].map(pretty_outcome)
    outcomes = order_outcomes(d[o].unique())
    palette = sns.color_palette("tab10", len(outcomes))
    for i, oc in enumerate(outcomes):
        s = d[d[o] == oc].groupby(kcol)[delta].mean().sort_index()
        ax.plot(s.index, s.values, marker="o", lw=1.8, color=palette[i], label=oc)
    ax.set(xlabel="Top-K cited features masked together", ylabel="Mean |\u0394 prob|",
           title="Evidence is load-bearing (group-masking)")
    if legend:
        ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="lower right",
                  handlelength=1.2, columnspacing=0.9, labelspacing=0.3)


def _plot_evidence_audit(ax, ev):
    if ev is None or ev.empty:
        empty_panel(ax, "no nb6_evidence_detail"); return
    imp = pick(ev, ["importance"], required=False, what="importance")
    cfd = pick(ev, ["counterfactual_delta", "delta"], required=False, what="cf delta")
    ver = pick(ev, ["verified", "is_verified"], required=False, what="verified")
    if imp is None or cfd is None:
        empty_panel(ax, "no importance/delta columns"); return
    d = ev.copy(); d["_abs"] = d[cfd].abs()
    if ver is not None:
        d["_v"] = d[ver].map(as_bool)
        # If this is the slim/downsampled export, the legend should report the
        # TRUE totals (carried in n_verified_total / n_hallucinated_total),
        # not the number of plotted points.
        true_tot = {}
        if "n_verified_total" in d.columns:
            true_tot[True] = int(pd.to_numeric(d["n_verified_total"], errors="coerce").dropna().max())
        if "n_hallucinated_total" in d.columns:
            true_tot[False] = int(pd.to_numeric(d["n_hallucinated_total"], errors="coerce").dropna().max())
        for flag, color, lab in [(True, "#27ae60", "verified"), (False, "#c0392b", "hallucinated")]:
            sub = d[d["_v"] == flag]
            n = true_tot.get(flag, len(sub))   # true total if available, else plotted count
            ax.scatter(sub[imp], sub["_abs"], s=16, alpha=0.5, color=color,
                       edgecolor="none", label=f"{lab} (n={n})")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    else:
        ax.scatter(d[imp], d["_abs"], s=16, alpha=0.5, color="#34495e", edgecolor="none")
    ax.axhline(0.10, ls="--", color="gray", lw=1, alpha=0.8)
    ax.text(ax.get_xlim()[1], 0.105, "0.10 move threshold", ha="right", va="bottom",
            fontsize=7.5, color="gray")
    ax.set(xlabel="Model's self-reported importance",
           ylabel="|counterfactual \u0394| when masked",
           title="Evidence audit: cited importance vs actual impact")


def build_supp_evidence_audit(ctx, cohort):
    """Supplementary: the detailed evidence audit (cited self-reported importance
    vs actual counterfactual impact when masked, colored by verification status).
    The headline faithfulness metrics + group-masking now live in main Fig 3; this
    keeps the fuller per-citation scatter available."""
    ev = ctx.get("evidence")
    if ev is None:
        print("  [skip] supp evidence audit: no nb6_evidence_detail."); return
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111); panel_label(ax, "A")
    _plot_evidence_audit(ax, ev)
    save_fig(fig, "figS_evidence_audit")


# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — Clinical economics
# ══════════════════════════════════════════════════════════════════
def _plot_cost_by_class(ax, cost):
    c = pick(cost, COST, what="cost"); cls = pick(cost, CLS, required=False, what="class")
    if cls is None:
        empty_panel(ax, "no algorithm_class column"); return
    d = cost[cost[c] > 0]
    means = d.groupby(cls)[c].mean()
    order = [x for x in class_order(means.index) if x in means.index]
    vals = means.reindex(order).values
    y = np.arange(len(order))
    base_cols = [class_color(x) for x in order]
    fs.soft_bars(ax, y, vals, base_cols, width=fs.BAR_W,
                 emphasis=[i for i, x in enumerate(order) if x == "deliberative_ensemble"],
                 horizontal=True)
    ax.set_xscale("log")
    xmin = min(v for v in vals if v > 0)
    ax.set_xlim(xmin / 3, max(vals) * 4)      # headroom so inside-labels fit
    for yi, v, cc in zip(y, vals, order):
        if pd.notna(v) and v > 0:
            # place the cost label INSIDE the bar end when there's room, else outside
            span_dec = np.log10(v) - np.log10(ax.get_xlim()[0])
            inside = span_dec > 1.0            # bar long enough (>1 decade) to hold text
            txt = f"${v:,.1e}"
            if inside:
                ax.text(v, yi, txt + "  ", va="center", ha="right", fontsize=7.5,
                        color="white", fontweight="bold")
            else:
                ax.text(v, yi, "  " + txt, va="center", ha="left", fontsize=7.5, color=fs.INK)
    ax.set_yticks(y); ax.set_yticklabels([CLASS_PRETTY.get(x, x) for x in order], fontsize=8)
    ax.set(xlabel="Cost per patient (USD, log)", ylabel="",
           title="Cost by paradigm ($\\sim$$10^6\\times$ spread)")


def _plot_pareto(ax, cost):
    a = pick(cost, AUC, what="auc"); c = pick(cost, COST, what="cost")
    mod = pick(cost, ["model_id", "model_label", "model"], what="model")
    cls = pick(cost, CLS, required=False, what="class")
    agg = cost.groupby(mod).agg(auc=(a, "mean"), cost=(c, "mean")).reset_index()
    if cls:
        agg["cls"] = cost.groupby(mod)[cls].first().reindex(agg[mod]).values
    else:
        agg["cls"] = "unknown"
    agg = agg[agg["cost"] > 0].dropna(subset=["auc", "cost"])
    if agg.empty:
        empty_panel(ax, "no cost/AUC data"); return
    srt = agg.sort_values("cost").reset_index(drop=True)
    fidx, best = [], -np.inf
    for i, r in srt.iterrows():
        if r["auc"] >= best:
            fidx.append(i); best = r["auc"]
    front = srt.loc[fidx].sort_values("cost")
    for cc in class_order(agg["cls"].unique()):
        dd = agg[agg["cls"] == cc]
        ax.scatter(dd["cost"], dd["auc"], s=65, color=class_color(cc),
                   edgecolor="#2c3e50", linewidth=0.5, zorder=3, label=CLASS_PRETTY.get(cc, cc))
    ax.plot(front["cost"], front["auc"], "-o", color="#27ae60", lw=2, zorder=2,
            markersize=5, label="Pareto frontier")
    ax.set_xscale("log")
    ax.set(xlabel="Cost per patient (USD, log)", ylabel="Mean ROC AUC",
           title="Cost–performance Pareto frontier")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3)


_ML_CLASSES = {"classical_ml", "temporal_ml"}


def _plot_cascade(ax, cost):
    """Cost-effectiveness of spending on a single LLM, as a lollipop per outcome.
    The claim the panel makes is deliberately simple: *if you pay for an LLM you buy
    accuracy over the best classical/temporal ML model, and the single LLM already
    captures that gain* --- the costlier multi-step ensemble is not needed to make
    the point. For each outcome we take the best ML model (max ROC-AUC among
    classical + temporal ML) and the single LLM (`model_a`, ~\\$0.006/patient), and
    plot the cost-effectiveness ratio = (single-LLM AUC $-$ best-ML AUC) / single-LLM
    cost, i.e. ROC-AUC points of gain per dollar. Outcomes are sorted best-to-worst;
    a stem colour indicates whether the LLM gain is statistically distinguishable
    from ML (non-overlapping 95\\% CIs, solid \\cedar-tier colour) or numerical only
    (overlapping CIs, pale). Each stem is annotated with the AUC gain that produced
    the ratio. This is honest about significance: the two big, cheap wins
    (microalbuminuria, dyslipidemia) are significant; the rest are gains the study is
    underpowered to certify. Distinct from Fig 2E (best-ML-vs-best-LLM AUC, no cost
    axis) because the x-axis here is dollars-normalized cost-effectiveness."""
    if cost is None or cost.empty:
        empty_panel(ax, "no nb7_cost_comparison"); return
    o = pick(cost, OUT, what="outcome")
    a = pick(cost, AUC, what="auc")
    c = pick(cost, COST, what="cost")
    cls = pick(cost, CLS, required=False, what="class")
    mod = pick(cost, ["model_id", "model_label", "model"], what="model")
    lo = "roc_auc_ci_low" if "roc_auc_ci_low" in cost.columns else None
    hi = "roc_auc_ci_high" if "roc_auc_ci_high" in cost.columns else None
    if cls is None:
        empty_panel(ax, "no algorithm_class column"); return

    rows = []
    for oc, grp in cost.groupby(o):
        ml = grp[grp[cls].isin(_ML_CLASSES)]
        llm = grp[grp[mod].astype(str).str.lower() == "model_a"]  # the single LLM
        if ml.empty or llm.empty:
            continue
        bml = ml.loc[ml[a].astype(float).idxmax()]
        s = llm.iloc[0]
        gain = float(s[a]) - float(bml[a])
        scost = float(s[c])
        if scost <= 0:
            continue
        sig = bool(lo and hi and float(s[lo]) > float(bml[hi]))  # LLM CI-low > ML CI-high
        rows.append((pretty_outcome(oc), gain / scost, gain, scost, sig))
    if not rows:
        empty_panel(ax, "no ML/LLM rows to compare"); return

    rows.sort(key=lambda r: r[1])  # ascending -> best on top
    labels = [r[0] for r in rows]
    ratio  = np.array([r[1] for r in rows])
    gains  = np.array([r[2] for r in rows])
    sig    = np.array([r[4] for r in rows])
    y = np.arange(len(rows))

    sig_col, ns_col = fs.PALETTE["blue"], "#c9d6e8"
    xmin = min(0.0, ratio.min() * 1.15)
    xmax = ratio.max() * 1.32
    ax.set_xlim(xmin, xmax); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.axvline(0, color=fs.PALETTE["grey"], lw=1.0, ls="--", zorder=1)

    for yi, r, s in zip(y, ratio, sig):
        col = sig_col if s else ns_col
        ax.plot([0, r], [yi, yi], color=col, lw=2.0, alpha=0.75, zorder=2)
        ax.scatter(r, yi, s=95, color=col, edgecolor="#2c3e50",
                   linewidth=0.6, zorder=4)
    off = (xmax - xmin) * 0.015
    for yi, r, g in zip(y, ratio, gains):
        # near-zero gains: always label to the right so the text clears the axis.
        right = r >= 0 or abs(r) < (xmax - xmin) * 0.04
        txt = f"$+{g:.02f}$" if g >= 0 else f"${g:.02f}$"
        ax.text(r + (off if right else -off), yi, txt,
                va="center", ha="left" if right else "right",
                fontsize=7.2, color=fs.INK)

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set(xlabel="Cost-effectiveness: ROC-AUC gain over best ML per \\$",
           ylabel="",
           title="Value of paying for a single LLM (by outcome)")
    handles = [
        plt.Line2D([], [], marker="o", ls="", markersize=9, color=sig_col,
                   label="LLM $>$ ML (95% CIs disjoint)"),
        plt.Line2D([], [], marker="o", ls="", markersize=9, color=ns_col,
                   label="Gain not significant"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.0, loc="center right",
              bbox_to_anchor=(1.0, 0.42),
              title="LLM vs. ML", title_fontsize=7.2)


def _plot_cost_scale(ax, cost):
    """Total deployment cost vs cohort size, per paradigm (log-log). The point:
    the LLM/classical cost gap is a fixed per-patient ratio, so at this study's
    scale (a few thousand patients) the LLM's absolute cost is trivial and its
    accuracy + auditability are worth it, whereas at population scale (10s-100s of
    millions) the same ratio makes classical ML the pragmatic choice. Replaces the
    arbitrary +0.03 'worth-it' threshold with the real scale-dependent argument."""
    c = pick(cost, COST, what="cost"); cls = pick(cost, CLS, required=False, what="class")
    if cls is None:
        empty_panel(ax, "no algorithm_class column"); return
    d = cost[cost[c] > 0]
    # representative per-patient cost = class mean; take the three tiers of interest
    per = d.groupby(cls)[c].mean()
    tiers = [("classical_ml", "Classical ML"), ("single_agent_llm", "Single LLM"),
             ("deliberative_ensemble", METHOD_NAME)]
    tiers = [(k, lab) for k, lab in tiers if k in per.index]
    N = np.logspace(3, 8, 100)   # 1e3 .. 1e8 patients
    # set log scales and explicit limits BEFORE plotting/annotating so autoscale
    # never runs on a linear axis over these huge ranges (which broke the bbox).
    ax.set_xscale("log"); ax.set_yscale("log")
    all_tot = np.concatenate([per[k] * N for k, _ in tiers])
    ax.set_xlim(1e3, 1e8)
    ax.set_ylim(max(all_tot.min() / 5, 1e-4), all_tot.max() * 5)
    for k, lab in tiers:
        ax.plot(N, per[k] * N, lw=2.2, color=class_color(k), label=lab)
    y_lo = ax.get_ylim()[0]
    for nx, note in [(6.5e3, "this study\n(~6.5k)"), (1e8, "population\n(100M)")]:
        ax.axvline(nx, ls=":", color=fs.PALETTE["grey"], lw=1.1)
        ax.text(nx, y_lo * 1.5, "  " + note, rotation=90, fontsize=6.8,
                color="#666", va="bottom", ha="left")
    ax.set(xlabel="Cohort size (patients, log)",
           ylabel="Total inference cost (USD, log)",
           title="Cost scales with cohort: LLMs are cheap at study scale")
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def _two_prop_p(p1, n1, p2, n2):
    """Two-proportion z-test p-value (pooled), for comparing error rates between
    two paradigms within a prevalence tertile."""
    if min(n1, n2) < 1:
        return np.nan
    x1, x2 = p1 * n1, p2 * n2
    pp = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan
    from math import erf, sqrt
    z = (p1 - p2) / se
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def _plot_prevalence_tertile(ax, strat, title):
    """Error rate (0.5 threshold) per paradigm within outcome-prevalence tertiles,
    with binomial SE error bars and ML-vs-LLM significance. Shows the crossover:
    in the LOW-prevalence tertile classical/temporal ML are safe-but-blind (low
    error by majority-class prediction) while the LLMs carry real signal; in
    mid/high prevalence the LLMs win. The clinically important, data-scarce end is
    exactly where LLMs help most."""
    if strat is None or getattr(strat, "empty", True):
        empty_panel(ax, "no nb_error_by_patient_stratum"); return
    d = strat[strat["stratum_type"] == "outcome_prevalence_tertile"].copy()
    order = ["low", "mid", "high"]
    d = d.set_index("stratum_value").reindex(order)
    cols = [("err_classical", "n_classical", fs.CLASS_COLORS["classical_ml"], "Classical ML"),
            ("err_temporal", "n_temporal", fs.CLASS_COLORS["temporal_ml"], "Temporal ML"),
            ("err_single_llm", "n_single_llm", fs.CLASS_COLORS["single_agent_llm"], "Single LLM"),
            ("err_cedar", "n_cedar", fs.CEDAR, "CEDAR")]
    x = np.arange(len(order)); w = 0.2
    for k, (ec, nc, color, lab) in enumerate(cols):
        p = d[ec].astype(float).values; n = d[nc].astype(float).values
        se = np.sqrt(np.clip(p * (1 - p), 0, None) / np.where(n > 0, n, np.nan))
        ax.bar(x + (k - 1.5) * w, p, w, yerr=se, capsize=2.5,
               color=fs._lighten(color, 0.3), edgecolor=color, linewidth=1.0,
               error_kw={"lw": 0.8, "ecolor": fs.INK}, label=lab)
    # ML-vs-LLM significance per tertile: classical (best ML at low prev) vs CEDAR
    ymax = 0
    for j, t in enumerate(order):
        if t not in d.index or pd.isna(d.loc[t, "err_classical"]):
            continue
        p_ml, n_ml = float(d.loc[t, "err_classical"]), float(d.loc[t, "n_classical"])
        p_llm, n_llm = float(d.loc[t, "err_cedar"]), float(d.loc[t, "n_cedar"])
        pval = _two_prop_p(p_ml, n_ml, p_llm, n_llm)
        star = fs.stars(pval) if pval == pval else ""
        top = max(p_ml, p_llm) + 0.06
        ymax = max(ymax, top)
        if star and star != "n.s.":
            ax.text(x[j], top, star, ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color=fs.INK)
    ax.set_xticks(x); ax.set_xticklabels([t.capitalize() for t in order], fontsize=10)
    ax.set_xlabel("Outcome-prevalence tertile", fontsize=10)
    ax.set_ylabel("Error rate (threshold 0.5)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(0.7, ymax + 0.06))
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")


def build_fig4(ctx, cohort):
    """Reframed economics (was Fig 5). (A) cost-performance Pareto frontier;
    (B) cascade analysis --- the per-outcome ROC-AUC gain from escalating from the
    cheapest traditional-ML baseline up to the best LLM/CEDAR model, which is the
    accuracy payoff that justifies moving off classical ML. Naive multi-agent
    excluded to match the reframed landscape. The cost-scale panel (formerly C) was
    removed. NOTE: the measured cost-per-paradigm bar and the prevalence-tertile
    error-rate plot both stay in the supplement --- at a 0.5 threshold classical ML
    looks better at low prevalence only by predicting the majority class, so it is
    NOT an LLM-superiority claim; the LLM advantage is on AUC (Fig 2)."""
    cost = _drop_multiagent(ctx.get("cost"))
    if cost is None:
        print("  [skip] fig4 (economics): nb7_cost_comparison unavailable."); return
    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(1, 2, wspace=0.32, width_ratios=[1.05, 1.0])
    axA = fig.add_subplot(gs[0, 0]); panel_label(axA, "A")
    _plot_pareto(axA, cost)
    axB = fig.add_subplot(gs[0, 1]); panel_label(axB, "B")
    _plot_cascade(axB, cost)
    save_fig(fig, "fig4")


# ══════════════════════════════════════════════════════════════════
# FIGURE 6 — Data-regime contrast & modality value
# ══════════════════════════════════════════════════════════════════
# Human-readable modality labels + a fixed display order (baseline first).
_MODALITY_LABELS = {"EHR_only": "EHR only", "EHR_SDOH": "EHR + SDoH",
                    "EHR_SDOH_CGM": "EHR + SDoH + CGM"}
_MODALITY_ORDER = ["EHR_only", "EHR_SDOH", "EHR_SDOH_CGM"]


def _plot_modality(ax, modality, title):
    """Grouped bars: per outcome, ROC-AUC (mean over models) for each feature set.
    The augmented sets (+SDoH, +CGM) are tested against EHR_only per outcome
    (Wilcoxon signed-rank, paired by model) and marked with significance stars."""
    if modality is None or modality.empty:
        empty_panel(ax, "no nb4_modality_ablation"); return
    a = pick(modality, AUC, what="auc")
    mod = pick(modality, ["modality_set", "modality", "feature_set", "ablation"],
               required=False, what="modality")
    mdl = pick(modality, ["model", "model_id", "model_label"], required=False, what="model")
    astd = pick(modality, ["roc_auc_std", "auc_std"], required=False, what="auc_std")
    nfold = pick(modality, ["n_folds", "n_fold"], required=False, what="n_folds")
    if mod is None or "outcome" not in modality.columns:
        empty_panel(ax, "no modality/outcome column"); return
    d = modality.copy(); d["outcome"] = d["outcome"].map(pretty_outcome)
    grp = d.groupby(["outcome", mod])[a].mean().reset_index()
    outcomes = order_outcomes(grp["outcome"].unique())
    msets = [m for m in _MODALITY_ORDER if m in set(grp[mod])] + \
            [m for m in grp[mod].unique() if m not in _MODALITY_ORDER]
    base = "EHR_only" if "EHR_only" in msets else msets[0]
    pivot = grp.pivot_table(index="outcome", columns=mod, values=a).reindex(outcomes)
    x = np.arange(len(outcomes)); width = 0.82 / max(len(msets), 1)
    # soft green→teal→blue ramp; baseline (EHR only) darkest/anchor
    ramp = {"EHR_only": fs.PALETTE["blue"], "EHR_SDOH": fs.PALETTE["green"],
            "EHR_SDOH_CGM": fs.PALETTE["purple"]}
    for k, ms in enumerate(msets):
        if ms not in pivot.columns:
            continue
        col = ramp.get(ms, sns.color_palette("crest", len(msets))[k])
        xpos = x + k * width
        ax.bar(xpos, pivot[ms].values, width, label=_MODALITY_LABELS.get(ms, str(ms)),
               color=fs._lighten(col, 0.35), edgecolor=col, linewidth=1.1)
    ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.8)

    # Per-outcome significance: each augmented modality vs EHR_only. With only 4
    # models per cell the paired rank test bottoms out at p=0.125 (never
    # significant), so we test the CLINICALLY relevant comparison instead — the
    # best model under the augmented set vs the SAME model under EHR_only, using
    # each estimate's 5-fold mean+SD via Welch's t-test. This has the power to
    # detect the one real effect (CGM for glycemic control) while correctly
    # leaving everything else n.s.
    if mdl is not None and astd is not None and nfold is not None:
        for j, oc in enumerate(outcomes):
            sub = d[d["outcome"] == oc]
            base_sub = sub[sub[mod] == base].set_index(mdl)
            for k, ms in enumerate(msets):
                if ms == base or ms not in pivot.columns:
                    continue
                aug = sub[sub[mod] == ms]
                if aug.empty:
                    continue
                brow = aug.loc[aug[a].idxmax()]          # best augmented model
                m = brow[mdl]
                if m not in base_sub.index:
                    continue
                r0 = base_sub.loc[m]
                p = fs.welch_p(brow[a], brow[astd], brow[nfold],
                               r0[a], r0[astd], r0[nfold])
                st = fs.stars(p)
                if st and st != "n.s.":
                    xk = x[j] + k * width
                    ytop = pivot[ms].iloc[j]
                    ax.text(xk, ytop + 0.012, st, ha="center", va="bottom",
                            fontsize=10, fontweight="bold", color=fs.INK)
    ax.set_xticks(x + width * (len(msets) - 1) / 2)
    ax.set_xticklabels(outcomes, rotation=25, ha="right", fontsize=7.5)
    ax.set_ylim(0.4, 1.02)
    ax.set(ylabel="ROC AUC (mean over models)", title=title)
    ax.legend(frameon=False, fontsize=7.5, title="Feature set", title_fontsize=8,
              loc="upper right")


def _plot_modality_by_class(ax, abl, cohort, title):
    """Grouped bars: per model class, the change in mean ROC-AUC from adding each
    modality on top of EHR-only (Delta from EHR_only), averaged over models and
    outcomes within the class. Shows whether SDoH / CGM contribute, across ALL
    model families (classical ML, temporal ML, single LLM, CEDAR). Uses the
    standardized all-models ablation (nb_modality_ablation_all)."""
    if abl is None or getattr(abl, "empty", True):
        empty_panel(ax, "no nb_modality_ablation_all"); return
    a = "roc_auc_mean"; cls = "algorithm_class"; fscol = "feature_set"
    # mean AUC per (class, feature_set) over models x outcomes
    piv = abl.groupby([cls, fscol])[a].mean().unstack()
    order = [c for c in FIG_CLASS_ORDER if c in piv.index]
    piv = piv.reindex(order)
    added = [m for m in ["EHR_SDOH", "EHR_SDOH_CGM"] if m in piv.columns]
    lab = {"EHR_SDOH": "+ SDoH", "EHR_SDOH_CGM": "+ SDoH + CGM"}
    col = {"EHR_SDOH": fs.PALETTE["green"], "EHR_SDOH_CGM": fs.PALETTE["purple"]}

    # Per (class, feature_set) SPREAD of the modality effect: for each individual
    # model x outcome, delta = AUC(feature_set) - AUC(EHR_only); the SD of those
    # per-unit deltas within a class shows whether the (near-zero) mean effect is
    # consistent across models/outcomes or just an average of large swings.
    key = ["model", "outcome"] if "model" in abl.columns else ["outcome"]
    wide = abl.pivot_table(index=[cls] + key, columns=fscol, values=a)

    def _deltas(c, m):
        """Per-(model x outcome) deltas AUC(m) - AUC(EHR_only) within a class."""
        if c not in wide.index.get_level_values(cls) or m not in wide.columns:
            return np.array([])
        sub = wide.xs(c, level=cls)
        return (sub[m] - sub["EHR_only"]).dropna().values

    def _spread(c, m):
        d = _deltas(c, m)
        return float(d.std()) if len(d) > 1 else 0.0

    # Point + error-bar (dot) style, matching Fig 6C: one dot per (class, modality)
    # at the mean Delta, with +/-SD whiskers across models x outcomes, PLUS the
    # individual per-(model x outcome) deltas as jittered points behind the dot so
    # the spread is visible. A dot whose whiskers straddle 0 => no consistent effect.
    x = np.arange(len(order)); off = 0.16
    rng = np.random.default_rng(0)
    for k, m in enumerate(added):
        delta = (piv[m] - piv["EHR_only"]).values
        sd = np.array([_spread(c, m) for c in order])
        c = col[m]
        xpos = x + (k - (len(added) - 1) / 2) * off
        # jittered individual points
        for xi, cl in zip(xpos, order):
            pts = _deltas(cl, m)
            if len(pts):
                jx = xi + rng.uniform(-0.05, 0.05, size=len(pts))
                ax.scatter(jx, pts, s=11, color=c, alpha=0.28,
                           edgecolor="none", zorder=2)
        ax.errorbar(xpos, delta, yerr=sd, fmt="o", color=c, markersize=8,
                    markeredgecolor="white", markeredgewidth=0.8,
                    elinewidth=1.2, capsize=4, capthick=1.0, label=lab[m], zorder=3)
    ax.axhline(0, ls="--", color=fs.INK, lw=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_PRETTY.get(c, c).replace(" (CEDAR)", "\n(CEDAR)").replace(" ", "\n", 1)
                        for c in order], fontsize=8)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylabel(r"$\Delta$ ROC-AUC vs. EHR-only", fontsize=10)
    ax.set_ylim(-0.12, 0.12)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, title="Added modality (mean $\\pm$ SD across models/outcomes)",
              title_fontsize=8, loc="upper right")


def build_fig5(ctx_t2d, ctx_t1d):
    """(Was Fig 6.) Standardized modality ablation across ALL model families
    (classical ML, temporal ML, single LLM, CEDAR): how much does adding SDoH (and
    CGM for T1D) change discrimination, per paradigm? Delta from EHR-only, averaged
    within class. Fixed test set (N=100). Reads nb_modality_ablation_all_{cohort}."""
    abl2 = ctx_t2d.get("modality_all"); abl1 = ctx_t1d.get("modality_all")
    fig = plt.figure(figsize=(14, 5.5))
    gs = fig.add_gridspec(1, 2, wspace=0.26)
    axA = fig.add_subplot(gs[0, 0]); panel_label(axA, "A")
    _plot_modality_by_class(axA, abl2, "t2d", "T2D: modality contribution by model family")
    axB = fig.add_subplot(gs[0, 1]); panel_label(axB, "B")
    _plot_modality_by_class(axB, abl1, "t1d", "T1D: modality contribution by model family")
    save_fig(fig, "fig5")


# ══════════════════════════════════════════════════════════════════
# FIGURE 6 — Error analysis: why a single pass is enough
# ══════════════════════════════════════════════════════════════════
_ERR_SYS_F6 = ["classical", "temporal", "single_llm", "cedar"]
_ERR_LAB_F6 = {"classical": "Classical\nML", "temporal": "Temporal\nML",
               "single_llm": "Single\nLLM", "cedar": "CEDAR"}


def _plot_err_heatmap_f6(ax, ov, title):
    """Per-patient error-correlation (phi) matrix across the four paradigms."""
    if ov is None or getattr(ov, "empty", True):
        empty_panel(ax, "no nb_error_overlap"); return
    n = len(_ERR_SYS_F6); M = np.full((n, n), np.nan)
    for i in range(n):
        M[i, i] = 1.0
    for _, r in ov.iterrows():
        a, b = r["system_a"], r["system_b"]
        if a in _ERR_SYS_F6 and b in _ERR_SYS_F6:
            i, j = _ERR_SYS_F6.index(a), _ERR_SYS_F6.index(b)
            M[i, j] = M[j, i] = r["phi_correlation"]
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([_ERR_LAB_F6[s] for s in _ERR_SYS_F6], fontsize=8.5)
    ax.set_yticklabels([_ERR_LAB_F6[s] for s in _ERR_SYS_F6], fontsize=8.5)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(M[i, j]) > 0.55 else fs.INK)
    ax.set_title(title, fontsize=11, fontweight="bold")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Error correlation ($\\phi$)", fontsize=9); cb.ax.tick_params(labelsize=8)


def _mech_ablation_diffs_vs_vanilla(ctx_t2d):
    """Paired mean Delta ROC-AUC (config - vanilla) with SD-based CI for the CEDAR
    mechanism ablations (CoT+Verify, CoT+SC) and full CEDAR, over the 7 T2D
    outcomes. Computed on the fly because the ablations are stored as absolute AUCs
    (nb6_model_d_results) rather than pre-differenced."""
    ag = try_ds("nb5_agentic_results_t2d"); md = ctx_t2d.get("model_d")
    if ag is None or md is None:
        return {}
    a = "roc_auc_mean"; o = "outcome"
    van = ag[ag["config_id"] == "model_a"].groupby(o)[a].max()
    out = {}
    for cfg in ["model_c_plus_verify", "model_c_plus_sc", "model_d_full"]:
        s = md[md["config_id"] == cfg].groupby(o)[a].max()
        common = van.index.intersection(s.index)
        d = (s.reindex(common) - van.reindex(common)).dropna()
        if len(d):
            m, sd, n = float(d.mean()), float(d.std(ddof=1)), len(d)
            hw = 1.96 * sd / np.sqrt(n) if n > 1 else 0.0
            out[cfg] = (m, m - hw, m + hw)
    return out


def _plot_enhancement_nulls(ax, ctx_t2d, ctx_t1d):
    """Summary forest: every CEDAR configuration we evaluated, as a paired mean
    Delta ROC-AUC vs a single LLM pass. All cluster on zero --- the visual thesis
    that no configuration beats a single competent pass. Colors are consistent with
    Figure 2: the CEDAR family (full pipeline + its ablations) uses the CEDAR red;
    the enhancement strategies shown for the first time here get distinct colors.
    (The targeted tool-use result --- the one thing that DID help, on a specific
    outcome --- is a separate story shown in Supplementary Figure~ref{sfig:hybrid_b},
    not mixed into this 'everything converges' panel.)"""
    C_CEDAR = fs.CEDAR                     # matches Fig 2 CEDAR family
    C_REASON = fs.PALETTE["purple"]        # v2 re-reasoning strategies (new here)
    C_ENCODE = fs.PALETTE["orange"]        # feature engineering (new here)
    C_FUSE = fs.PALETTE["green"]           # ML fusion (new here)
    rows = []   # (label, mean, lo, hi, color)

    # 1) CEDAR family: full pipeline + mechanism ablations (vs vanilla) — CEDAR red
    mech = _mech_ablation_diffs_vs_vanilla(ctx_t2d)
    for cfg, lab in [("model_d_full", "CEDAR (full pipeline)"),
                     ("model_c_plus_verify", "CEDAR: CoT + verify"),
                     ("model_c_plus_sc", "CEDAR: CoT + self-consistency")]:
        if cfg in mech:
            m, lo, hi = mech[cfg]; rows.append((lab, m, lo, hi, C_CEDAR))

    # 2) more reasoning (CEDAR v2 screen, vs vanilla)
    v2 = try_ds("nb_cedar_v2_paired_diffs_t2d")
    if v2 is not None and not v2.empty:
        for sysid, lab in [("cedar_sr", "CEDAR + self-refinement"),
                           ("cedar_re", "CEDAR + reasoner ensemble"),
                           ("cedar_sr_re", "CEDAR + refine + ensemble")]:
            s = v2[(v2["system"] == sysid) & (v2["comparator"] == "vanilla")]
            if not s.empty:
                rows.append((lab, s["auc_diff_mean"].mean(),
                             s["auc_diff_ci_low"].mean(), s["auc_diff_ci_high"].mean(),
                             C_REASON))

    # 3) re-encoding (feature engineering, vs vanilla)
    fe = try_ds("nb_cedar_fe_diffs_t2d")
    if fe is not None and not fe.empty:
        rows.append(("CEDAR + feature engineering", fe["auc_diff_mean"].mean(),
                     fe["auc_diff_ci_low"].mean(), fe["auc_diff_ci_high"].mean(),
                     C_ENCODE))

    # 4) paradigm fusion --- Hybrid A (late fusion) vs CEDAR, pooled over T2D
    hy = try_ds("nb_hybrid_diffs_t2d")
    if hy is not None and not hy.empty:
        for var, lab in [("hybrid_mean", "CEDAR + ML fusion: mean"),
                         ("hybrid_stack", "CEDAR + ML fusion: stack")]:
            s = hy[(hy["hybrid_variant"] == var) & (hy["comparator"] == "cedar")]
            if not s.empty:
                rows.append((lab, s["auc_diff_mean"].mean(),
                             s["auc_diff_ci_low"].mean(), s["auc_diff_ci_high"].mean(),
                             C_FUSE))

    if not rows:
        empty_panel(ax, "no enhancement datasets"); return
    rows = rows[::-1]                                   # top-to-bottom as listed
    y = np.arange(len(rows))
    for yi, (lab, m, lo, hi, col) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.9, zorder=2)
        ax.scatter(m, yi, s=46, color=col, zorder=3, edgecolor="white", linewidth=0.6)
    ax.axvline(0, ls="--", color=fs.INK, lw=1.1, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("$\\Delta$ ROC-AUC vs. a single LLM pass", fontsize=10)
    ax.set_title("Every CEDAR configuration converges to a single pass",
                 fontsize=10.2, fontweight="bold")
    ax.set_xlim(-0.1, 0.1)
    ax.tick_params(axis="x", labelsize=8)


# ML-score-as-tool inside CEDAR (Architecture B). Consistent naming with the
# fusion combiners: comparators use "stack (late fusion)" etc.
_TOOL_SYS_COL = {"cedar_B1": fs.PALETTE["green"], "cedar_B2": fs.PALETTE["purple"]}
_TOOL_CMP_LAB = {"vanilla": "vs single LLM (no tool)",
                 "cedar_v1": "vs CEDAR (no tool)",
                 "hybrid_stack": "vs stack (late fusion)"}


def _plot_tooluse_forest(ax, diffs, outcome, title, show_ylabels=True):
    """Paired-bootstrap ROC-AUC differences for tool-equipped CEDAR (score /
    score+reason) vs single LLM, CEDAR-no-tool, and the stack (late-fusion)
    combiner, on one outcome. Bold marker = 95% CI excludes 0 (significant).
    show_ylabels=False hides the row labels (shared with an adjacent panel)."""
    if diffs is None or getattr(diffs, "empty", True):
        empty_panel(ax, "no nb_hybrid_b_diffs"); return
    d = diffs[diffs["outcome"] == outcome]
    rows = []
    for sysid, slab in [("cedar_B1", "score"), ("cedar_B2", "score+reason")]:
        for comp in ["vanilla", "cedar_v1", "hybrid_stack"]:
            r = d[(d["system"] == sysid) & (d["comparator"] == comp)]
            if not r.empty:
                rr = r.iloc[0]
                rows.append((f"{slab} · {_TOOL_CMP_LAB[comp]}",
                             float(rr["auc_diff_mean"]), float(rr["auc_diff_ci_low"]),
                             float(rr["auc_diff_ci_high"]), sysid))
    if not rows:
        empty_panel(ax, f"no tool diffs for {outcome}"); return
    rows = rows[::-1]
    y = np.arange(len(rows))
    for yi, (lab, m, lo, hi, sysid) in zip(y, rows):
        sig = lo > 0 or hi < 0
        col = _TOOL_SYS_COL[sysid]
        ax.plot([lo, hi], [yi, yi], color=col, lw=2.0 if sig else 1.4,
                alpha=1.0 if sig else 0.45, zorder=2)
        ax.scatter(m, yi, s=46 if sig else 32, color=col, zorder=3,
                   edgecolor=fs.INK if sig else "white",
                   linewidth=0.8 if sig else 0.5, alpha=1.0 if sig else 0.55)
    ax.axvline(0, ls="--", color=fs.INK, lw=1.0, zorder=1)
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels([r[0] for r in rows], fontsize=7.6)
    else:
        ax.set_yticklabels([])
    ax.set_ylim(-0.6, len(rows) - 0.4)   # identical data-range -> identical aspect
    ax.set_xlabel("$\\Delta$ ROC-AUC, paired (95% CI)", fontsize=10)
    # shared symmetric x-range (all CIs fall within +/-0.085) so D and E match in
    # scale and aspect; tightened to +/-0.1 so the intervals fill the panel instead
    # of clustering near the zero line.
    ax.set_xlim(-0.1, 0.1)
    ax.set_xticks([-0.1, -0.05, 0.0, 0.05, 0.1])
    ax.set_title(title, fontsize=10.2, fontweight="bold")


def build_fig6(ctx_t2d, ctx_t1d):
    """Error analysis explains why a single pass is enough. (A) per-patient error
    correlation: single LLM and CEDAR err on the same patients (phi=0.87), so
    deliberation cannot reshape the error set; (B) T1D counterpart; (C) summary
    forest of every CEDAR-enhancement attempt (mechanism ablation, v2 self-
    refinement/ensemble, feature engineering, ML fusion) --- all tie a single pass,
    because the base model already saturates the available signal."""
    ov2 = try_ds("nb_error_overlap_t2d"); ov1 = try_ds("nb_error_overlap_t1d")
    tb2 = try_ds("nb_hybrid_b_diffs_t2d"); tb1 = try_ds("nb_hybrid_b_diffs_t1d")
    # Rows 1-2 (left): A (T2D) over B (T1D) error-correlation heatmaps; C
    # (enhancement forest) spans both rows in the right column. Row 3: the two
    # tool-use forests side by side (D = ML-competitive outcome, tool helps;
    # E = ML-weak outcome, no gain) — the targeted tool-use payoff of the
    # decorrelated-errors result in A/B.
    fig = plt.figure(figsize=(13, 11.0))
    # Outer 2-block grid: top block (A/B heatmaps + C forest), bottom block (D/E).
    outer = fig.add_gridspec(2, 1, hspace=0.30, height_ratios=[2.0, 1.05])
    # Top: A over B on the left; C on the right. A trailing spacer column narrows
    # C and leaves breathing room for its long left-side y-labels.
    top = outer[0].subgridspec(2, 3, wspace=1.15, hspace=0.42,
                               width_ratios=[0.85, 0.82, 0.02])
    axA = fig.add_subplot(top[0, 0]); panel_label(axA, "A", dx=-0.24, dy=1.10)
    _plot_err_heatmap_f6(axA, ov2, "T2D: per-patient error correlation")
    axB = fig.add_subplot(top[1, 0]); panel_label(axB, "B", dx=-0.24, dy=1.10)
    _plot_err_heatmap_f6(axB, ov1, "T1D: per-patient error correlation")
    axC = fig.add_subplot(top[0:2, 1]); panel_label(axC, "C")
    _plot_enhancement_nulls(axC, ctx_t2d, ctx_t1d)
    # Bottom: D and E, equal width and identical scale/aspect. D carries the shared
    # row labels once (drawn in the leading spacer column's gap); E hides them. We do
    # NOT use sharey here: a shared y-axis lets E's set_yticklabels([]) blank the
    # labels on D too. A leading spacer column reserves room for D's long left-side
    # labels, and a trailing spacer keeps D and E the same (narrower) width, centred.
    bot = outer[1].subgridspec(1, 4, wspace=0.10,
                               width_ratios=[0.22, 1.0, 1.0, 0.12])
    axD = fig.add_subplot(bot[0, 1]); panel_label(axD, "D")
    _plot_tooluse_forest(axD, tb1, "OUTCOME_Optimal_Glycemic_Control",
                         "ML-competitive (T1D glycemic control):\n"
                         "ML-as-tool gains are significant")
    axE = fig.add_subplot(bot[0, 2]); panel_label(axE, "E")
    _plot_tooluse_forest(axE, tb2, "OUTCOME_Dyslipidemia",
                         "ML-weak (T2D dyslipidemia):\nML-as-tool adds nothing",
                         show_ylabels=False)
    save_fig(fig, "fig6")


# ══════════════════════════════════════════════════════════════════
# MANUSCRIPT NUMBERS REPORT  (writes a plain-text file with every value
# the manuscript checklist asks for — Sections A..H)
# ══════════════════════════════════════════════════════════════════
REPORT_PATH = os.path.join(REPORT_DIR, "manuscript_numbers.txt")

# report-specific column candidates
PREV_C   = ["prevalence", "test_prevalence"]
NPOS_C   = ["n_positive"]
NSAMP_C  = ["n_samples", "n_test"]
EXTRACT_C = ["extractive_grounding_accuracy", "extractive_accuracy"]
SUFF_C   = ["sufficiency_score", "sufficiency"]
COMPR_C  = ["comprehensiveness_score", "comprehensiveness"]
KCOL_C   = ["k", "n_masked", "top_k"]
GDELTA_C = ["delta", "counterfactual_delta", "masked_delta"]
IMP_C    = ["importance"]
CFDELTA_C = ["counterfactual_delta", "delta"]
VERIF_C  = ["verified", "is_verified"]
MODSET_C = ["modality_set", "feature_set", "modality", "ablation"]
MARG_C   = ["marginal_auc_per_dollar"]
MODEL_C  = ["model_id", "model_label", "model"]

OUTCOME_ABBR = {
    "Optimal Glycemic Control": "OGC", "Insulin Independence": "InsInd",
    "Metformin Response": "Metf", "GLP1RA Response": "GLP1",
    "Dyslipidemia": "Dys", "Hypertension": "HTN", "Microalbuminuria": "Micro",
}


def _hdr(t):
    return ["", "=" * 78, t, "=" * 78]


def _f(x, nd=3):
    try:
        xf = float(x)
        if np.isnan(xf):
            return "n/a"
        return f"{xf:.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def _abbr(o):
    return OUTCOME_ABBR.get(o, str(o)[:8])


def _table(headers, rows):
    allrows = [list(map(str, headers))] + [[str(c) for c in r] for r in rows]
    nc = len(headers)
    w = [max(len(a[i]) for a in allrows) for i in range(nc)]
    def line(r): return "  ".join(str(r[i]).ljust(w[i]) for i in range(nc))
    out = [line(headers), "  ".join("-" * wi for wi in w)]
    out += [line(r) for r in rows]
    return out


def _auc_map(df, cfg):
    """outcome -> max ROC-AUC for a given config_id (pretty outcome keys)."""
    if df is None or df.empty:
        return {}
    o = pick(df, OUT); a = pick(df, AUC); c = pick(df, CFG, required=False)
    d = df.copy(); d[o] = d[o].map(pretty_outcome)
    if c is not None:
        d = d[d[c] == cfg]
    if d.empty:
        return {}
    return d.groupby(o)[a].max().to_dict()


def section_A(t2d):
    L = _hdr("SECTION A - Cohort / prevalence (Fig 1C)  [nb6_model_d_results_t2d]")
    md = t2d.get("model_d")
    if md is None or md.empty:
        return L + ["  [missing] nb6_model_d_results_t2d"]
    o = pick(md, OUT); prev = pick(md, PREV_C, required=False)
    npos = pick(md, NPOS_C, required=False); nsamp = pick(md, NSAMP_C, required=False)
    d = md.copy(); d[o] = d[o].map(pretty_outcome)
    rows = []
    for oc in order_outcomes(d[o].unique()):
        s = d[d[o] == oc]
        p = _f(s[prev].mean() * 100, 1) if prev else "n/a"
        rows.append([oc, p,
                     int(s[npos].max()) if npos else "n/a",
                     int(s[nsamp].max()) if nsamp else "n/a"])
    return L + ["A1/A2. Target-visit prevalence and test sizes (year_2):"] + \
        _table(["Outcome", "Prevalence %", "n_positive", "n_samples"], rows)


def section_B(t2d):
    L = _hdr("SECTION B - Performance landscape (Fig 2)  [nb7_cost_comparison_t2d]")
    cost = t2d.get("cost")
    if cost is None or cost.empty:
        return L + ["  [missing] nb7_cost_comparison_t2d"]
    o = pick(cost, OUT); a = pick(cost, AUC)
    lo = pick(cost, CILO, required=False); hi = pick(cost, CIHI, required=False)
    cls = pick(cost, CLS, required=False); mid = pick(cost, MODEL_C)
    d = cost.copy(); d[o] = d[o].map(pretty_outcome)

    L += ["", "B1. Best model per outcome:"]
    rows = []
    for oc in order_outcomes(d[o].unique()):
        s = d[d[o] == oc]; r = s.loc[s[a].idxmax()]
        ci = f"[{_f(r[lo])}, {_f(r[hi])}]" if (lo and hi) else "n/a"
        rows.append([oc, short_model(r[mid]), r[cls] if cls else "n/a", _f(r[a]), ci])
    L += _table(["Outcome", "Best model", "Class", "ROC-AUC", "95% CI"], rows)

    L += ["", "B2. Full model x outcome ROC-AUC matrix:"]
    d["_s"] = d[mid].map(short_model)
    cols = [c for c in order_outcomes(d[o].unique())]
    mat = d.pivot_table(index="_s", columns=o, values=a, aggfunc="max").reindex(columns=cols)
    mat = mat.reindex(mat.mean(axis=1).sort_values(ascending=False).index)
    headers = ["Model"] + [_abbr(c) for c in cols]
    rows = [[idx] + [_f(mat.loc[idx, c]) for c in cols] for idx in mat.index]
    L += _table(headers, rows)
    L += ["  codes: " + "; ".join(f"{_abbr(c)}={c}" for c in cols)]

    if cls:
        L += ["", "B3. ROC-AUC distribution by algorithm class (pooled over model x outcome):"]
        rows = []
        for c in class_order(d[cls].unique()):
            v = d[d[cls] == c][a].dropna()
            rows.append([c, len(v), _f(v.mean()), _f(v.std()),
                         _f(v.median()), _f(v.quantile(.25)), _f(v.quantile(.75))])
        L += _table(["Class", "n", "mean", "sd", "median", "q25", "q75"], rows)
        de = d[d[cls] == "deliberative_ensemble"][a].dropna()
        ma = d[d[cls] == "multi_agent_llm"][a].dropna()
        L += ["", "  KEY CLAIM:",
              f"    deliberative_ensemble mean ROC-AUC = {_f(de.mean())} "
              f"(sd {_f(de.std())})  [expect ~0.80, highest+tightest]",
              f"    multi_agent_llm median ROC-AUC     = {_f(ma.median())}  [expect near 0.5]"]
    return L


def section_C(t2d):
    L = _hdr("SECTION C - Mechanism: collapse & ablation (Fig 3)  "
             "[nb5_agentic_results_t2d + nb6_model_d_results_t2d]")
    ag, md = t2d.get("agentic"), t2d.get("model_d")
    van, cot, multi = _auc_map(ag, "model_a"), _auc_map(ag, "model_c"), _auc_map(ag, "model_b")
    cedar = _auc_map(md, "model_d_full")

    L += ["", "C1. Single-agent collapse on reasoning-hard outcomes:"]
    hard = ["Optimal Glycemic Control", "Insulin Independence", "Metformin Response"]
    L += _table(["Outcome", "Vanilla", "CoT", "CEDAR"],
                [[oc, _f(van.get(oc)), _f(cot.get(oc)), _f(cedar.get(oc))] for oc in hard])
    lows = [v for v in list(van.values()) + list(cot.values()) if pd.notna(v)]
    if lows:
        L += [f"  Min single-agent ROC-AUC across all outcomes = {_f(min(lows))} "
              f"(KEY CLAIM: toward/below chance)"]

    L += ["", "C2. Architecture progression (all outcomes):"]
    outs = order_outcomes(set(list(van) + list(cot) + list(multi) + list(cedar)))
    L += _table(["Outcome", "Vanilla", "Multi", "CoT", "CEDAR"],
                [[oc, _f(van.get(oc)), _f(multi.get(oc)), _f(cot.get(oc)), _f(cedar.get(oc))]
                 for oc in outs])

    L += ["", "C3. Mechanism ablation (mean +/- SD across outcomes):"]
    if md is not None and not md.empty:
        a = pick(md, AUC); c = pick(md, CFG)
        rows = []
        for cfg, lab in [("model_d_full", "CEDAR (full)"),
                         ("model_c_plus_verify", "CoT+Verify"),
                         ("model_c_plus_sc", "CoT+SC")]:
            v = md[md[c] == cfg][a].dropna()
            rows.append([lab, _f(v.mean()), _f(v.std()), len(v)])
        L += _table(["Config", "mean ROC-AUC", "sd", "n_outcomes"], rows)
        L += ["  KEY CLAIM: the three means differ by < 1 SD -> statistically indistinguishable."]
    else:
        L += ["  [missing] nb6_model_d_results_t2d"]
    return L


def section_D(t2d):
    L = _hdr("SECTION D - Faithfulness (Fig 4)  [HIGH PRIORITY]  "
             "[nb6_faithfulness / group_masking / evidence_detail _t2d]")
    faith, gm, ev = t2d.get("faith"), t2d.get("gmask"), t2d.get("evidence")

    if faith is not None and not faith.empty:
        o = pick(faith, OUT)
        ext = pick(faith, EXTRACT_C, required=False)
        suf = pick(faith, SUFF_C, required=False)
        cmp_ = pick(faith, COMPR_C, required=False)
        d = faith.copy(); d[o] = d[o].map(pretty_outcome)
        rows = [[oc,
                 _f(d[d[o] == oc][ext].mean()) if ext else "n/a",
                 _f(d[d[o] == oc][suf].mean()) if suf else "n/a",
                 _f(d[d[o] == oc][cmp_].mean()) if cmp_ else "n/a"]
                for oc in order_outcomes(d[o].unique())]
        L += ["D1. Per-outcome faithfulness means:"]
        L += _table(["Outcome", "Extractive", "Sufficiency", "Comprehensiveness"], rows)
        if ext:
            L += [f"D2. Pooled extractive grounding accuracy (all audited patients) = {_f(d[ext].mean())}"]
        nper = d.groupby(o).size()
        L += [f"D6. Faithfulness subsample: {int(nper.min())}-{int(nper.max())} "
              f"patients/outcome (total audited rows = {len(d)})"]
    else:
        L += ["  [missing] nb6_faithfulness_results_t2d"]

    if ev is not None and not ev.empty:
        ver = pick(ev, VERIF_C, required=False)
        cf = pick(ev, CFDELTA_C, required=False)
        # If a slim/downsampled export, report TRUE totals for the counts.
        slim = "n_verified_total" in ev.columns or "n_hallucinated_total" in ev.columns
        L += [""]
        if ver:
            vv = ev[ver].map(as_bool)
            if slim:
                nver = int(pd.to_numeric(ev.get("n_verified_total"), errors="coerce").dropna().max()) \
                    if "n_verified_total" in ev.columns else int(vv.sum())
                nhal = int(pd.to_numeric(ev.get("n_hallucinated_total"), errors="coerce").dropna().max()) \
                    if "n_hallucinated_total" in ev.columns else int((~vv).sum())
                L += [f"D3. Evidence audit counts (TRUE totals): verified n={nver}, "
                      f"hallucinated n={nhal} (total cited features = {nver + nhal}); "
                      f"panel C plots a downsampled subset of {len(vv)} points."]
            else:
                L += [f"D3. Evidence audit counts: verified n={int(vv.sum())}, "
                      f"hallucinated n={int((~vv).sum())} (total cited features = {len(vv)})"]
        if cf:
            ab = ev[cf].abs()
            note = " (over the plotted subsample)" if slim else ""
            L += [f"D5. Median |counterfactual delta| = {_f(ab.median())}; "
                  f"fraction |delta| >= 0.10 = {_f((ab >= 0.10).mean())}{note}"]
    else:
        L += ["", "  [missing] nb6_evidence_detail_t2d"]

    if gm is not None and not gm.empty:
        o = pick(gm, OUT); k = pick(gm, KCOL_C); delta = pick(gm, GDELTA_C)
        d = gm.copy(); d[o] = d[o].map(pretty_outcome)
        pooled = d.groupby(k)[delta].mean()
        L += ["", "D4. Group-masking mean |delta prob| by K (pooled across outcomes):"]
        L += _table(["K", "mean|delta|"], [[int(kk), _f(pooled.loc[kk])] for kk in sorted(pooled.index)])
        piv = d.pivot_table(index=o, columns=k, values=delta, aggfunc="mean").reindex(order_outcomes(d[o].unique()))
        kcols = sorted(piv.columns)
        L += ["   per-outcome:"]
        L += _table(["Outcome"] + [f"K={int(c)}" for c in kcols],
                    [[oc] + [_f(piv.loc[oc, c]) for c in kcols] for oc in piv.index])
        mono = all(pooled.loc[kcols[i]] <= pooled.loc[kcols[i + 1]] + 1e-9 for i in range(len(kcols) - 1))
        L += [f"   KEY CLAIM: pooled curve monotonic increasing = {mono}"]
    else:
        L += ["", "  [missing] nb6_group_masking_curve_t2d"]
    return L


def section_E(t2d):
    L = _hdr("SECTION E - Clinical economics (Fig 5)  "
             "[nb7_cost_comparison_t2d + nb7_cascade_analysis_t2d]")
    cost, casc = t2d.get("cost"), t2d.get("cascade")
    if cost is not None and not cost.empty:
        a = pick(cost, AUC); c = pick(cost, COST)
        cls = pick(cost, CLS, required=False); mid = pick(cost, MODEL_C)
        if cls:
            g = cost[cost[c] > 0].groupby(cls)[c].mean()
            L += ["E1. Mean cost per patient (USD) by paradigm:"]
            L += _table(["Class", "cost/patient (USD)"],
                        [[cc, f"{g.loc[cc]:.3e}"] for cc in class_order(g.index) if cc in g.index])
        agg = cost.groupby(mid).agg(auc=(a, "mean"), cost=(c, "mean")).reset_index()
        if cls:
            agg["cls"] = cost.groupby(mid)[cls].first().reindex(agg[mid]).values
        agg = agg[agg["cost"] > 0].dropna(subset=["auc", "cost"])
        if not agg.empty:
            cheap = agg.loc[agg["cost"].idxmin()]
            L += ["", "E2. Pareto endpoints (per-model mean over outcomes):",
                  f"    cheapest: {short_model(cheap[mid])}  cost={cheap['cost']:.3e}  AUC={_f(cheap['auc'])}"]
            ced = agg[agg[mid] == "model_d_full"]
            if len(ced):
                cr = ced.iloc[0]
                L += [f"    CEDAR:    cost={cr['cost']:.3e}  AUC={_f(cr['auc'])}",
                      f"    cost ratio CEDAR/cheapest = {cr['cost'] / cheap['cost']:.3e}   "
                      f"AUC gain = {_f(cr['auc'] - cheap['auc'])}"]
            srt = agg.sort_values("cost").reset_index(drop=True)
            opt, best = [], -np.inf
            for _, r in srt.iterrows():
                if r["auc"] >= best:
                    opt.append(r[mid]); best = r["auc"]
            dominated = [m for m in agg[mid] if m not in opt]
            L += ["", "E3. Pareto-optimal: " + ", ".join(short_model(m) for m in opt),
                  "    Dominated:     " + (", ".join(short_model(m) for m in dominated) or "none")]
    else:
        L += ["  [missing] nb7_cost_comparison_t2d"]

    if casc is not None and not casc.empty:
        o = pick(casc, OUT); marg = pick(casc, MARG_C, required=False)
        rec = "cascade_recommended" if "cascade_recommended" in casc.columns else None
        d = casc.copy(); d[o] = d[o].map(pretty_outcome)
        L += ["", "E4. Cascade table:"]
        rows = []
        for oc in order_outcomes(d[o].unique()):
            s = d[d[o] == oc].iloc[0]
            rows.append([oc, _f(s[marg], 6) if marg else "n/a",
                         str(as_bool(s[rec])) if rec else "n/a"])
        L += _table(["Outcome", "marginal_AUC_per_$", "cascade_recommended"], rows)
    else:
        L += ["", "  [missing] nb7_cascade_analysis_t2d"]
    return L


def _modality_lines(df, label):
    if df is None or df.empty:
        return [f"  [missing] modality ablation ({label})"]
    a = pick(df, AUC); ms = pick(df, MODSET_C, required=False)
    if ms is None or "outcome" not in df.columns:
        return [f"  [no modality/outcome column] ({label})"]
    d = df.copy(); d["outcome"] = d["outcome"].map(pretty_outcome)
    piv = d.groupby(["outcome", ms])[a].mean().reset_index() \
           .pivot_table(index="outcome", columns=ms, values=a) \
           .reindex(order_outcomes(d["outcome"].unique()))
    out = [f"  {label} — mean ROC-AUC by feature set:"]
    out += ["    " + ln for ln in _table(["Outcome"] + [str(c) for c in piv.columns],
            [[oc] + [_f(piv.loc[oc, c]) for c in piv.columns] for oc in piv.index])]
    # identify base / +SDoH / +CGM columns by substring
    def find(subs, excl=()):
        for c in piv.columns:
            cl = str(c).lower()
            if all(s in cl for s in subs) and not any(e in cl for e in excl):
                return c
        return None
    base = find(["ehr"], excl=["sdoh", "cgm"]) or (piv.columns[0] if len(piv.columns) else None)
    sdoh = find(["sdoh"], excl=["cgm"])
    cgm = find(["cgm"])
    if base is not None and sdoh is not None:
        out += [f"    mean delta (+SDoH vs EHR-only) = {_f((piv[sdoh] - piv[base]).mean())}"]
    if sdoh is not None and cgm is not None:
        out += [f"    mean delta (+CGM vs EHR+SDoH)  = {_f((piv[cgm] - piv[sdoh]).mean())}"]
    return out


def section_F(t2d, t1d):
    L = _hdr("SECTION F - Data-regime contrast & modality (Fig 6)  "
             "[nb7_cost_comparison_t1d + nb4_modality_ablation_t2d/_t1d]")
    cost = t1d.get("cost")
    if cost is not None and not cost.empty:
        o = pick(cost, OUT); a = pick(cost, AUC)
        lo = pick(cost, CILO, required=False); hi = pick(cost, CIHI, required=False)
        cls = pick(cost, CLS, required=False); mid = pick(cost, MODEL_C)
        d = cost.copy(); d[o] = d[o].map(pretty_outcome)
        L += ["F1. T1D best model per outcome (KEY CLAIM: classical ML wins well-powered):"]
        rows = []
        for oc in order_outcomes(d[o].unique()):
            s = d[d[o] == oc]; r = s.loc[s[a].idxmax()]
            ci = f"[{_f(r[lo])}, {_f(r[hi])}]" if (lo and hi) else "n/a"
            rows.append([oc, short_model(r[mid]), r[cls] if cls else "n/a", _f(r[a]), ci])
        L += _table(["Outcome", "Best model", "Class", "ROC-AUC", "95% CI"], rows)
    else:
        L += ["  [missing] nb7_cost_comparison_t1d"]
    L += ["", "F2. T2D modality ablation:"]
    L += _modality_lines(t2d.get("modality"), "T2D")
    L += ["", "F3. T1D modality ablation:"]
    L += _modality_lines(t1d.get("modality"), "T1D")
    return L


def section_G():
    L = _hdr("SECTION G - Methods constants to CONFIRM against the code you run")
    L += [
        "  G1. LLM: Claude Sonnet 4.6 (Anthropic)                 [confirm exact id/version]",
        "  G2. K_SELF_CONSISTENCY=3, SC_TEMPERATURE=0.7, N_BOOTSTRAP=200,",
        "      N_FOLDS=5, RANDOM_STATE=42                          [confirm]",
        "  G3. Adaptive test size N_TEST_MIN/BASE/MAX = 50/150/250 [confirm; n_samples above]",
        "  G4. Token pricing $3 / $15 per 1M in/out; ML_COST_PER_SEC_USD=0.00005 [confirm]",
        "  G5. Cascade threshold auc_gain > 0.03                   [confirm]",
        "  G6. Total LLM calls / total spend                       [nice-to-have; sum n_llm_calls x n_test]",
    ]
    return L


def section_H(cohort_suffix="_t2d"):
    L = _hdr("SECTION H - Interpretability study (clinician concordance)")
    isc = try_ds(f"nb5_interpretability_scores{cohort_suffix}")
    iag = try_ds(f"nb5_interpretability_agreement{cohort_suffix}")
    got = False
    for name, df in [("scores", isc), ("agreement", iag)]:
        if df is not None and not df.empty and df.select_dtypes("number").abs().to_numpy().sum() > 0:
            got = True
            L += [f"  {name}: shape {df.shape}, columns {list(df.columns)}"]
    if not got:
        L += ["  PENDING — interpretability datasets are empty/placeholder "
              "(n_items=0, kappa=NaN). Report n raters, n chains, mean concordance,",
              "  Cohen's/Fleiss kappa once the clinician-rating study is run."]
    return L


def build_number_report(t2d, t1d, path=REPORT_PATH):
    import datetime
    lines = [
        "MANUSCRIPT NUMBERS — Pediatric Diabetes / CEDAR",
        f"generated: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        "cohorts: T2D primary (_t2d), T1D contrast (_t1d) | horizon: year_2",
        "values are exact from the datasets; replace the manuscript's approximate",
        "figures/\\todo{} markers with these.",
    ]
    sections = [
        lambda: section_A(t2d), lambda: section_B(t2d), lambda: section_C(t2d),
        lambda: section_D(t2d), lambda: section_E(t2d), lambda: section_F(t2d, t1d),
        section_G, lambda: section_H("_t2d"),
    ]
    for fn in sections:
        try:
            lines += fn()
        except Exception as e:
            lines += ["", f"  [ERROR generating section: {e}]"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"    report -> {path}")


# ══════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════
def main():
    print("Loading T2D (primary cohort) ...")
    t2d = load_all("t2d")
    print("\nLoading T1D (regime contrast) ...")
    t1d = load_all("t1d")

    print("\nBuilding manuscript figures ...")
    for name, fn in [("fig1", lambda: build_fig1(t2d, "t2d")),
                     ("fig2", lambda: build_fig2(t2d, t1d)),
                     ("fig3", lambda: build_fig3(t2d, "t2d")),
                     ("fig4", lambda: build_fig4(t2d, "t2d")),
                     ("fig5", lambda: build_fig5(t2d, t1d)),
                     ("fig6", lambda: build_fig6(t2d, t1d))]:
                     # NOTE: figS_mechanism (multi-agent + CEDAR ablation) and
                     # figS_evidence_audit are retired --- the multi-agent baseline
                     # now lives in figS_omitted_configs, the CEDAR ablation in
                     # Fig 6C, and the evidence audit in figS_faithfulness (all
                     # built by generate_supp_figures.py).
        try:
            print(f"\n  [{name}]")
            fn()
        except Exception as e:
            print(f"  [error] {name} failed: {e}")

    print("\nWriting manuscript numbers report ...")
    build_number_report(t2d, t1d)

    print(f"\nDone. Panels -> ./{OUTDIR}/ ; numbers -> ./{REPORT_PATH}")


if __name__ == "__main__":
    main()