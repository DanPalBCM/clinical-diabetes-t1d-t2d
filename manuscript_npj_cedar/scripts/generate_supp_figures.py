"""
Generate the SUPPLEMENTARY figure panels for the Pediatric Diabetes AI manuscript,
LOCALLY, from the same raw dumps as the main figures.

The AI supplementary is deliberately AGENTIC / MODEL-focused (the clinical /
descriptive EDA belongs to the companion biostatistics paper and is NOT
reproduced here). It reuses the panel builders in generate_main_figures.py and
adds the nb8 temporal-validation figure.

Supplementary figures produced (-> figures/supplementary_figures/):
  figS_agentic_t1d      T1D performance landscape (best-per-outcome / heatmap / box)
  figS_faithfulness_t1d T1D faithfulness (metrics / group-masking / evidence audit)
  figS_modality         modality ablation, T2D and T1D side by side
  figS_temporal_val     nb8 temporal validation (ML vs single-agent vs CEDAR),
                        with the train/validation split summary

Usage:
    python scripts/generate_supp_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Reuse everything from the main generator (loaders, palette, panel builders).
# Importing G also applies the shared publication style (figstyle.apply_style).
import generate_main_figures as G
import figstyle as fs

# Supplementary figures live in figures/supplementary_figures/ (pdf + svg).
SUPP_OUTDIR = os.path.join(G._PROJECT, "figures", "supplementary_figures")
os.makedirs(SUPP_OUTDIR, exist_ok=True)


def _save_supp(fig, name):
    for fmt in ("pdf", "svg"):
        p = os.path.join(SUPP_OUTDIR, f"{name}.{fmt}")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"    saved -> {p}")
    plt.close(fig)


# NOTE: the former S-Agentic (T1D) performance-landscape supplementary figure was
# removed — it is now the main-text Figure 6 (T1D landscape, panels A/B).


# ══════════════════════════════════════════════════════════════════
# S-Prevalence: outcome prevalence + assessable N, T2D vs T1D side by side.
# Documents that the two cohorts differ in prevalence and power (reviewer point
# on why T1D numbers are not in the same range as T2D).
# ══════════════════════════════════════════════════════════════════
def _prev_panel(ax, cohort, title):
    m = G.try_ds(f"nb4_matched_comparison_{cohort}")
    if m is None or getattr(m, "empty", True):
        G.empty_panel(ax, f"no nb4_matched_comparison_{cohort}"); return
    m = G.filter_h(m)
    d = m.drop_duplicates("outcome").copy()
    d["outcome"] = d["outcome"].map(G.pretty_outcome)
    d["_N"] = d["n_train"] + d["n_test"]
    d["_prev"] = d["test_prevalence"]
    d["_pos"] = (d["_N"] * d["_prev"]).round().astype(int)
    d = d.set_index("outcome")
    order = G.order_outcomes(d.index)[::-1]
    d = d.reindex(order)
    vals = (d["_prev"] * 100).values
    y = np.arange(len(order))
    fs.soft_bars(ax, y, vals, [fs.PALETTE["blue"]] * len(order),
                 width=fs.BAR_W, horizontal=True)
    for yi, oc in zip(y, order):
        v = d.loc[oc, "_prev"] * 100
        if pd.notna(v):
            ax.text(v + 1.5, yi, f"{v:.0f}%  ({int(d.loc[oc, '_pos'])}/{int(d.loc[oc, '_N'])})",
                    va="center", fontsize=9.5, color=fs.INK)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(0, max(np.nanmax(vals) * 1.65, 10))
    ax.set_xlabel("Prevalence at 2-year target visit (%)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")


def build_supp_prevalence(t2d, t1d):
    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(1, 2, wspace=0.42)
    axA = fig.add_subplot(gs[0, 0]); G.panel_label(axA, "A")
    _prev_panel(axA, "t2d", "Outcome prevalence (T2D, full cohort)")
    axB = fig.add_subplot(gs[0, 1]); G.panel_label(axB, "B")
    _prev_panel(axB, "t1d", "Outcome prevalence (T1D, full cohort)")
    _save_supp(fig, "figS_prevalence")


# ══════════════════════════════════════════════════════════════════
# S-Calibration: reliability curves + Brier/ECE for CEDAR vs naive single LLM
# vs best classical (reviewer point on the "Calibrated" claim).
# Reads nb9_calibration_{cohort} (model_a=naive, model_d_full=CEDAR) and takes
# classical Brier from nb4_matched_comparison.
# ══════════════════════════════════════════════════════════════════
_SYS_LABEL = {"model_a": "Single LLM (naive)", "model_d_full": "Multi-step LLM (CEDAR)"}
_SYS_COLOR = {"model_a": "#9bb8e0", "model_d_full": fs.CEDAR}


def _reliability_panel(ax, cal, cohort):
    """Reliability curves (pooled over outcomes, count-weighted) for naive vs CEDAR."""
    bins = cal[cal["reliability_bin"] >= 0].copy()
    if bins.empty:
        G.empty_panel(ax, "no reliability bins"); return
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, alpha=0.8, zorder=1)
    for sysid in ("model_a", "model_d_full"):
        s = bins[bins["system"] == sysid].dropna(subset=["bin_obs_freq", "bin_pred_mean"])
        if s.empty:
            continue
        # count-weighted aggregate per bin index across outcomes
        g = s.groupby("reliability_bin").apply(
            lambda d: pd.Series({
                "pred": np.average(d["bin_pred_mean"], weights=d["bin_count"]),
                "obs": np.average(d["bin_obs_freq"], weights=d["bin_count"]),
            })).reset_index()
        ax.plot(g["pred"], g["obs"], "-o", color=_SYS_COLOR[sysid], lw=2,
                markersize=4, label=_SYS_LABEL[sysid], zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability", fontsize=10)
    ax.set_ylabel("Observed frequency", fontsize=10)
    ax.set_title(f"Reliability ({cohort.upper()}, pooled)", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")


def _ece_brier_panel(ax, cal, cost, cohort, metric):
    """Grouped bars of ECE or Brier per outcome: naive vs CEDAR (+ best classical
    Brier point where available)."""
    summ = cal[cal["reliability_bin"] == -1]
    outcomes = G.order_outcomes([G.pretty_outcome(o) for o in summ["outcome"].unique()])
    piv = summ.assign(outcome=summ["outcome"].map(G.pretty_outcome)) \
              .pivot_table(index="outcome", columns="system", values=metric).reindex(outcomes)
    x = np.arange(len(outcomes)); w = 0.38
    for k, sysid in enumerate(("model_a", "model_d_full")):
        if sysid in piv.columns:
            col = _SYS_COLOR[sysid]
            ax.bar(x + (k - 0.5) * w, piv[sysid].values, w,
                   color=fs._lighten(col, 0.35), edgecolor=col, linewidth=1.1,
                   label=_SYS_LABEL[sysid])
    # best-classical Brier point (only for the Brier panel)
    if metric == "brier" and cost is not None:
        a = G.pick(cost, G.AUC, what="auc"); cls = G.pick(cost, G.CLS, required=False, what="class")
        br = G.pick(cost, ["brier_mean", "brier"], required=False, what="brier")
        o = G.pick(cost, G.OUT, what="outcome")
        if br is not None and cls is not None:
            d = cost[cost[cls].isin(["classical_ml", "temporal_ml", "deep_learning"])].copy()
            d[o] = d[o].map(G.pretty_outcome)
            best = d.loc[d.groupby(o)[a].idxmax()].set_index(o)[br].reindex(outcomes)
            ax.scatter(x, best.values, marker="D", s=34, color="#2c3e50", zorder=5,
                       label="Best classical (Brier)")
    ax.set_xticks(x); ax.set_xticklabels(outcomes, rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel("ECE" if metric == "ece" else "Brier score", fontsize=10)
    # Paired test: CEDAR (model_d_full) vs naive LLM (model_a) across outcomes
    # (Wilcoxon signed-rank on the per-outcome metric). Reported so the reader is
    # not left to eyeball whether the CEDAR-vs-LLM difference is real.
    sig_txt = ""
    if "model_a" in piv.columns and "model_d_full" in piv.columns:
        pair = piv[["model_a", "model_d_full"]].dropna()
        if len(pair) >= 3:
            p = fs.wilcoxon_p(pair["model_a"].values, pair["model_d_full"].values)
            md = float((pair["model_a"] - pair["model_d_full"]).mean())
            if p == p:
                verdict = ("CEDAR lower" if md > 0 else "naive lower") if p < 0.05 else "n.s."
                sig_txt = f"  [CEDAR vs naive: {verdict}, p={p:.2f}]"
    ax.set_title(f"{'Expected calibration error' if metric=='ece' else 'Brier score'} "
                 f"({cohort.upper()}; lower = better){sig_txt}", fontsize=9.5,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")


def build_supp_calibration(t2d, t1d):
    cal2 = G.try_ds("nb9_calibration_t2d"); cal1 = G.try_ds("nb9_calibration_t1d")
    if cal2 is None and cal1 is None:
        print("  [skip] figS_calibration: no nb9_calibration datasets"); return
    cost2 = t2d.get("cost"); cost1 = t1d.get("cost")
    fig = plt.figure(figsize=(14, 11))
    outer = fig.add_gridspec(2, 1, hspace=0.42)
    # Row 1: T2D reliability + ECE + Brier
    r1 = outer[0].subgridspec(1, 3, width_ratios=[1.0, 1.2, 1.2], wspace=0.34)
    axA = fig.add_subplot(r1[0, 0]); G.panel_label(axA, "A"); _reliability_panel(axA, cal2, "t2d")
    axB = fig.add_subplot(r1[0, 1]); G.panel_label(axB, "B"); _ece_brier_panel(axB, cal2, cost2, "t2d", "ece")
    axC = fig.add_subplot(r1[0, 2]); G.panel_label(axC, "C"); _ece_brier_panel(axC, cal2, cost2, "t2d", "brier")
    # Row 2: T1D
    r2 = outer[1].subgridspec(1, 3, width_ratios=[1.0, 1.2, 1.2], wspace=0.34)
    axD = fig.add_subplot(r2[0, 0]); G.panel_label(axD, "D"); _reliability_panel(axD, cal1, "t1d")
    axE = fig.add_subplot(r2[0, 1]); G.panel_label(axE, "E"); _ece_brier_panel(axE, cal1, cost1, "t1d", "ece")
    axF = fig.add_subplot(r2[0, 2]); G.panel_label(axF, "F"); _ece_brier_panel(axF, cal1, cost1, "t1d", "brier")
    _save_supp(fig, "figS_calibration")


# ══════════════════════════════════════════════════════════════════
# S-Faithfulness (T1D): Fig 4 counterpart on the T1D cohort
# ══════════════════════════════════════════════════════════════════
def build_supp_faithfulness(t2d, t1d):
    """Merged faithfulness supplement (former figS_faithfulness_t1d + the standalone
    T2D evidence-audit figure): the T1D counterpart of the main faithfulness figure
    (metrics + group-masking) plus the per-citation evidence audit for BOTH cohorts,
    so all the detailed faithfulness plots sit together."""
    faith, gm = t1d.get("faith"), t1d.get("gmask")
    ev1 = t1d.get("evidence"); ev2 = t2d.get("evidence")
    if faith is None and gm is None and ev1 is None and ev2 is None:
        print("  [skip] figS_faithfulness: no faithfulness datasets"); return
    fig = plt.figure(figsize=(14, 10.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.42)
    # Row 1: T1D faithfulness metrics + group-masking (the T1D counterpart of Fig 3)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.28)
    axA = fig.add_subplot(top[0, 0]); G.panel_label(axA, "A")
    G._plot_faith_metrics(axA, faith, legend=True,
                          title="T1D: faithfulness (per-patient spread)")
    axB = fig.add_subplot(top[0, 1]); G.panel_label(axB, "B")
    G._plot_masking_curve(axB, gm, legend=True, cohort="t1d")
    axB.set_title("T1D: group-masking", fontsize=11, fontweight="bold")
    # Row 2: per-citation evidence audit, T2D and T1D side by side
    bot = outer[1].subgridspec(1, 2, wspace=0.24)
    axC = fig.add_subplot(bot[0, 0]); G.panel_label(axC, "C")
    G._plot_evidence_audit(axC, ev2); axC.set_title("T2D: evidence audit", fontsize=11, fontweight="bold")
    axD = fig.add_subplot(bot[0, 1]); G.panel_label(axD, "D")
    G._plot_evidence_audit(axD, ev1); axD.set_title("T1D: evidence audit", fontsize=11, fontweight="bold")
    _save_supp(fig, "figS_faithfulness")


# ══════════════════════════════════════════════════════════════════
# S-Modality: PER-MODEL modality ablation detail. Fig 6C/D collapse the
# ablation to a mean over models; here we break it out per individual model
# (LogReg / RF / XGBoost / XGBoost-Temporal) so the reader can see the effect
# is consistent across estimators and not an averaging artefact.
# ══════════════════════════════════════════════════════════════════
def _plot_modality_per_model(ax, modality, title):
    if modality is None or getattr(modality, "empty", True):
        G.empty_panel(ax, "no nb4_modality_ablation"); return
    a = G.pick(modality, G.AUC, what="auc")
    mod = G.pick(modality, ["modality_set", "modality", "feature_set", "ablation"],
                 required=False, what="modality")
    mdl = G.pick(modality, ["model", "model_id", "model_label"], required=False, what="model")
    if mod is None or mdl is None:
        G.empty_panel(ax, "no modality/model column"); return
    d = modality.copy()
    # average across outcomes -> one AUC per (model, feature set)
    grp = d.groupby([mdl, mod])[a].mean().reset_index()
    models = sorted(grp[mdl].unique(), key=lambda m: G.short_model(m))
    msets = [m for m in G._MODALITY_ORDER if m in set(grp[mod])] + \
            [m for m in grp[mod].unique() if m not in G._MODALITY_ORDER]
    pivot = grp.pivot_table(index=mdl, columns=mod, values=a).reindex(models)
    x = np.arange(len(models)); width = 0.82 / max(len(msets), 1)
    ramp = {"EHR_only": fs.PALETTE["blue"], "EHR_SDOH": fs.PALETTE["green"],
            "EHR_SDOH_CGM": fs.PALETTE["purple"]}
    for k, ms in enumerate(msets):
        if ms not in pivot.columns:
            continue
        col = ramp.get(ms, "#888888")
        ax.bar(x + k * width, pivot[ms].values, width,
               label=G._MODALITY_LABELS.get(ms, str(ms)),
               color=fs._lighten(col, 0.35), edgecolor=col, linewidth=1.1)
    ax.set_xticks(x + width * (len(msets) - 1) / 2)
    ax.set_xticklabels([G.short_model(m) for m in models], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0.5, 0.9)
    ax.set(ylabel="ROC AUC (mean over outcomes)", title=title)
    ax.legend(frameon=False, fontsize=7.5, title="Feature set", title_fontsize=8,
              loc="upper right")


def build_supp_modality(t2d, t1d):
    # Use the standardized all-models ablation (includes single LLM + CEDAR),
    # not the old classical-only nb4_modality_ablation.
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(1, 2, wspace=0.24)
    axA = fig.add_subplot(gs[0, 0]); G.panel_label(axA, "A")
    _plot_modality_per_model(axA, t2d.get("modality_all"),
                             "Per-model modality ablation (T2D): EHR vs +SDoH")
    axB = fig.add_subplot(gs[0, 1]); G.panel_label(axB, "B")
    _plot_modality_per_model(axB, t1d.get("modality_all"),
                             "Per-model modality ablation (T1D): EHR / +SDoH / +CGM")
    _save_supp(fig, "figS_modality")


# ══════════════════════════════════════════════════════════════════
# S-TemporalValidation: nb8 — train on earlier-diagnosed, validate on recent.
# Reframed (per reviewer request) as ORIGINAL analysis vs TEMPORAL validation,
# so the reader sees directly whether performance HOLDS under a prospective-style
# temporal split rather than just the validation number in isolation.
#   (A) T2D: CEDAR original test AUC vs temporal-validation AUC, per outcome
#   (B) T1D: same
#   (C) split-summary provenance (train vs validation diagnosis-duration)
# ══════════════════════════════════════════════════════════════════
VAL_AUC   = ["val_roc_auc", "roc_auc", "roc_auc_mean"]
VAL_LO    = ["val_roc_auc_ci_low", "roc_auc_ci_low"]
VAL_HI    = ["val_roc_auc_ci_high", "roc_auc_ci_high"]

# model-id aliases: nb7 (original) uses lowercase config ids, nb8 (temporal) uses
# canonical labels. Compare the SAME model across the two analyses.
_ORIG_VS_TEMP = [
    ("model_d_full",        "Model_D_full",        "deliberative_ensemble", "CEDAR"),
    ("Random_Forest",       "Random_Forest",       "classical_ml",          "Random Forest"),
    ("XGBoost",             "XGBoost",             "classical_ml",          "XGBoost"),
    ("Logistic_Regression", "Logistic_Regression", "classical_ml",          "LogReg"),
    ("model_c",             "LLM_single_agent",    "single_agent_llm",      "Single-agent LLM"),
]


def _plot_orig_vs_temporal_cedar(ax, cost, tv, title):
    """CEDAR-only dumbbell per outcome: original CV-test AUC (open blue) vs
    temporal-validation AUC (filled red), connected by a line, delta annotated.
    Kept deliberately simple/legible (the multi-method view proved too dense)."""
    if cost is None or tv is None or getattr(tv, "empty", True):
        G.empty_panel(ax, "no original / temporal-validation data"); return
    tv = G.filter_h(tv)
    o_c = G.pick(cost, G.OUT, what="outcome"); a_c = G.pick(cost, G.AUC, what="orig auc")
    m_c = G.pick(cost, ["model_id", "model", "model_label"], what="model")
    o_t = G.pick(tv, G.OUT, what="outcome"); a_t = G.pick(tv, VAL_AUC, what="val auc")
    m_t = G.pick(tv, ["model_id", "model", "model_label"], what="model")
    oc = cost[cost[m_c] == "model_d_full"][[o_c, a_c]].copy()
    oc[o_c] = oc[o_c].map(G.pretty_outcome)
    tvv = tv[tv[m_t] == "Model_D_full"][[o_t, a_t]].copy()
    tvv[o_t] = tvv[o_t].map(G.pretty_outcome)
    d = (oc.set_index(o_c)[a_c].rename("orig")
         .to_frame().join(tvv.set_index(o_t)[a_t].rename("temp"), how="inner"))
    if d.empty:
        G.empty_panel(ax, "no matched outcomes"); return
    d = d.reindex([o for o in G.order_outcomes(d.index)][::-1])
    y = np.arange(len(d))
    c_orig = "#9bb8e0"; c_temp = fs.CEDAR
    for i, row in enumerate(d.itertuples()):
        o0, t0 = row.orig, row.temp
        ax.plot([o0, t0], [i, i], color="#c4c4c4", lw=2.6, zorder=1)
        ax.scatter(o0, i, s=95, facecolor="white", edgecolor=c_orig, linewidth=1.6, zorder=3)
        ax.scatter(t0, i, s=95, color=c_temp, edgecolor="white", linewidth=0.8, zorder=3)
        dlt = t0 - o0
        ax.text(max(o0, t0) + 0.012, i, f"{dlt:+.02f}", va="center", ha="left",
                fontsize=10, fontweight="bold",
                color=(fs.PALETTE["green"] if dlt >= -0.02 else "#b06a6a"))
    ax.axvline(0.5, ls="--", color="gray", lw=1, alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels(d.index, fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("ROC AUC", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", markersize=9, markerfacecolor="white",
                   markeredgecolor=c_orig, markeredgewidth=1.6, label="Original (CV test)"),
        plt.Line2D([], [], marker="o", ls="", markersize=9, color=c_temp,
                   label="Temporal validation"),
    ], frameon=False, fontsize=9, loc="upper left")


def _plot_class_box_orig_vs_temporal(ax, cost, tv, title):
    """Box plot of ROC-AUC by MODEL CLASS, original vs temporal, so the reader can
    see each paradigm's distribution reproduce across the two datasets. Each box
    pools the (model x outcome) AUCs of one class in one dataset. The original
    analysis is restricted to the SAME models present in the temporal run so the
    comparison is apples-to-apples."""
    if cost is None or tv is None or getattr(tv, "empty", True):
        G.empty_panel(ax, "no original / temporal-validation data"); return
    tv = G.filter_h(tv)
    a_c = G.pick(cost, G.AUC, what="orig auc"); m_c = G.pick(cost, ["model_id", "model", "model_label"], what="model")
    a_t = G.pick(tv, VAL_AUC, what="val auc");  m_t = G.pick(tv, ["model_id", "model", "model_label"], what="model")
    # model-id -> (class) using the shared alias table
    orig_ids = {oid: cls for oid, _tid, cls, _lbl in _ORIG_VS_TEMP}
    temp_ids = {tid: cls for _oid, tid, cls, _lbl in _ORIG_VS_TEMP}
    rows = []
    for _, r in cost.iterrows():
        if r[m_c] in orig_ids and pd.notna(r[a_c]):
            rows.append({"class": orig_ids[r[m_c]], "dataset": "Original", "auc": float(r[a_c])})
    for _, r in tv.iterrows():
        if r[m_t] in temp_ids and pd.notna(r[a_t]):
            rows.append({"class": temp_ids[r[m_t]], "dataset": "Temporal", "auc": float(r[a_t])})
    if not rows:
        G.empty_panel(ax, "no shared models between analyses"); return
    df = pd.DataFrame(rows)
    class_order = [c for c in G.CLASS_ORDER if c in set(df["class"])]
    pretty = [G.CLASS_PRETTY.get(c, c) for c in class_order]
    df["class"] = pd.Categorical(df["class"], categories=class_order, ordered=True)
    # colour by class (soft); split hue by dataset via box facecolor lightness
    palette = {"Original": "#c9c9c9", "Temporal": fs.CEDAR}
    G.sns.boxplot(data=df, x="class", y="auc", hue="dataset", order=class_order,
                  hue_order=["Original", "Temporal"], palette=palette,
                  showfliers=False, width=0.6, linewidth=0.9, ax=ax)
    G.sns.stripplot(data=df, x="class", y="auc", hue="dataset", order=class_order,
                    hue_order=["Original", "Temporal"], dodge=True, ax=ax,
                    color=fs.INK, size=2.6, alpha=0.5, legend=False)
    ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.8)
    ax.set_xticks(range(len(class_order)))
    ax.set_xticklabels([p.replace(" (CEDAR)", "\n(CEDAR)").replace(" ", "\n", 1) for p in pretty],
                       fontsize=8)
    ax.set(xlabel="", ylabel="ROC AUC", title=title)
    ax.set_ylim(0.3, 1.0)
    # keep a single dataset legend
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:2], l[:2], frameon=False, fontsize=8, title="Dataset", title_fontsize=8.5,
              loc="lower left")


def _plot_split_summary(ax, sp2, sp1):
    frames = []
    for tag, sp in [("T2D", sp2), ("T1D", sp1)]:
        if sp is not None and not getattr(sp, "empty", True):
            s = G.filter_h(sp).copy(); s["_cohort"] = tag
            frames.append(s)
    if not frames:
        G.empty_panel(ax, "no nb8_temporal_split_summary"); return
    d = pd.concat(frames, ignore_index=True)
    tr = G.pick(d, ["train_median_duration_yrs"], required=False, what="train dur")
    vl = G.pick(d, ["val_median_duration_yrs"], required=False, what="val dur")
    if tr is None or vl is None:
        G.empty_panel(ax, "no duration columns in split summary"); return
    # one row per cohort: median diagnosis-duration of train vs validation split
    g = d.groupby("_cohort").agg(train=(tr, "median"), val=(vl, "median")).reset_index()
    g["_ord"] = g["_cohort"].map({"T2D": 0, "T1D": 1}).fillna(9)
    g = g.sort_values("_ord").reset_index(drop=True)
    x = np.arange(len(g)); w = 0.34
    # soft bars matching the manuscript style (lightened fill + colour outline)
    c_tr, c_vl = fs.PALETTE["blue"], fs.CEDAR
    ax.bar(x - w/2, g["train"], w, color=fs._lighten(c_tr, 0.4), edgecolor=c_tr,
           linewidth=1.2, label="Train (earlier dx)")
    ax.bar(x + w/2, g["val"], w, color=fs._lighten(c_vl, 0.4), edgecolor=c_vl,
           linewidth=1.2, label="Validation (recent dx)")
    for xi, (tv_, vv_) in enumerate(zip(g["train"], g["val"])):
        ax.text(xi - w/2, tv_ + 0.05, f"{tv_:.1f}", ha="center", va="bottom", fontsize=8.5)
        ax.text(xi + w/2, vv_ + 0.05, f"{vv_:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(g["_cohort"], fontsize=10)
    ax.set_xlim(-0.6, len(g) - 0.4)
    ax.set(ylabel="Median diabetes duration (yrs)",
           title="Temporal split provenance")
    # place the legend fully outside the axes on the right so it never overlaps
    # the bars (there is ample whitespace beside this narrow panel)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)


def build_supp_temporal_val(t2d_extra, t1d_extra, t2d, t1d):
    tv2 = t2d_extra.get("temporal"); tv1 = t1d_extra.get("temporal")
    sp2 = t2d_extra.get("split");    sp1 = t1d_extra.get("split")
    cost2 = t2d.get("cost");         cost1 = t1d.get("cost")
    if tv2 is None and tv1 is None:
        print("  [skip] figS_temporal_val: no nb8_temporal_validation datasets "
              "(bring nb8_temporal_validation_{t2d,t1d} + split summaries)")
        return
    fig = plt.figure(figsize=(14, 15))
    # Row 1: CEDAR original-vs-temporal dumbbell, T2D | T1D.
    # Row 2: ROC-AUC by model class, original vs temporal, T2D | T1D.
    # Row 3: split provenance (centered, narrower).
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.7], hspace=0.32)
    r1 = outer[0].subgridspec(1, 2, wspace=0.30)
    axA = fig.add_subplot(r1[0, 0]); G.panel_label(axA, "A")
    _plot_orig_vs_temporal_cedar(axA, cost2, tv2,
                                 "T2D — CEDAR: original test vs temporal validation")
    axB = fig.add_subplot(r1[0, 1]); G.panel_label(axB, "B")
    _plot_orig_vs_temporal_cedar(axB, cost1, tv1,
                                 "T1D — CEDAR: original test vs temporal validation")
    r2 = outer[1].subgridspec(1, 2, wspace=0.22)
    axC = fig.add_subplot(r2[0, 0]); G.panel_label(axC, "C")
    _plot_class_box_orig_vs_temporal(axC, cost2, tv2,
                                     "T2D: ROC-AUC by model class (original vs temporal)")
    axD = fig.add_subplot(r2[0, 1]); G.panel_label(axD, "D")
    _plot_class_box_orig_vs_temporal(axD, cost1, tv1,
                                     "T1D: ROC-AUC by model class (original vs temporal)")
    r3 = outer[2].subgridspec(1, 3, width_ratios=[0.34, 0.32, 0.34])
    axE = fig.add_subplot(r3[0, 1]); G.panel_label(axE, "E")
    _plot_split_summary(axE, sp2, sp1)
    _save_supp(fig, "figS_temporal_val")


# ══════════════════════════════════════════════════════════════════
# S-Selective: risk-coverage curves for selective prediction (abstention).
# Honest result: abstention helps LLM prediction generally (covered-set AUC rises
# as low-confidence patients are deferred), but CEDAR's confidence signals give NO
# advantage over a single LLM's output probability -- the verifier flag almost
# never fires, so CEDAR behaves like the single LLM.
# ══════════════════════════════════════════════════════════════════
_SEL_STYLE = {  # (label, color, linestyle)
    ("vanilla", "boundary"):    ("Single LLM (probability)", "#9bb8e0", "-"),
    ("cedar", "composite"):     ("CEDAR (composite signal)", fs.CEDAR, "-"),
    ("cedar", "consistency"):   ("CEDAR (self-consistency spread)", fs.PALETTE["purple"], "--"),
    ("cedar", "verifier"):      ("CEDAR (verifier flag)", fs.PALETTE["orange"], ":"),
}


def _plot_selective_rc(ax, rc, title):
    for (sysid, sig), (lab, col, ls) in _SEL_STYLE.items():
        s = rc[(rc["system"] == sysid) & (rc["signal"] == sig)].sort_values("coverage")
        if s.empty:
            continue
        ax.plot(s["coverage"], s["roc_auc_covered"], ls, color=col, lw=2,
                marker="o", markersize=3, label=lab)
    ax.set_xlabel("Coverage (fraction of patients predicted)", fontsize=10)
    ax.set_ylabel("ROC-AUC on covered set", fontsize=10)
    ax.set_xlim(1.02, 0.08)  # reversed: full coverage on left
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")


# build_supp_selective removed: its risk-coverage panels are now merged into
# build_supp_error_analysis (figS_error_analysis) to keep the supplement compact.


# ══════════════════════════════════════════════════════════════════
# S-CEDAR-v2: pre-registered screen of two architectural additions to CEDAR
# (verifier-gated self-refinement, diverse-reasoner ensemble, and both). Clean
# null: no config beats the single LLM. Reinforces "complexity does not help".
# ══════════════════════════════════════════════════════════════════
_V2_LABEL = {"vanilla": "Single LLM", "cedar_v1": "CEDAR v1",
             "cedar_sr": "CEDAR + refine", "cedar_re": "CEDAR + ensemble",
             "cedar_sr_re": "CEDAR + refine + ensemble"}
_V2_ORDER = ["cedar_sr", "cedar_re", "cedar_sr_re"]


def _plot_v2_forest(ax, diffs, comparator, title):
    """Forest plot of AUC-difference (v2 config minus comparator) per
    system x outcome, with 95% CIs; a vertical line at 0 = no difference."""
    d = diffs[diffs["comparator"] == comparator].copy()
    if d.empty:
        G.empty_panel(ax, f"no diffs vs {comparator}"); return
    d["outcome"] = d["outcome"].map(G.pretty_outcome)
    systems = [s for s in _V2_ORDER if s in set(d["system"])]
    outcomes = sorted(d["outcome"].unique())
    colors = G.sns.color_palette("colorblind", len(systems))
    yticks, ylabels = [], []
    row = 0
    for oc in outcomes:
        for si, s in enumerate(systems):
            r = d[(d["outcome"] == oc) & (d["system"] == s)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = row
            ax.plot([r["auc_diff_ci_low"], r["auc_diff_ci_high"]], [y, y],
                    color=colors[si], lw=1.6, zorder=2)
            ax.scatter(r["auc_diff_mean"], y, s=34, color=colors[si], zorder=3,
                       edgecolor="white", linewidth=0.6)
            yticks.append(y); ylabels.append(f"{oc[:16]} · {_V2_LABEL[s]}")
            row += 1
        row += 0.6  # gap between outcome groups
    ax.axvline(0, ls="--", color=fs.INK, lw=1.0, zorder=1)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.set_xlabel(f"$\\Delta$ ROC-AUC vs. {_V2_LABEL[comparator]}", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.invert_yaxis()


def _plot_v2_means(ax, screen, title):
    """Mean ROC-AUC (over the 3 screened outcomes) per system, anchors + v2."""
    s = screen[~screen["system"].str.contains("meanpool")].copy()
    order = ["vanilla", "cedar_v1", "cedar_sr", "cedar_re", "cedar_sr_re"]
    order = [o for o in order if o in set(s["system"])]
    m = s.groupby("system")["roc_auc_mean"].mean().reindex(order)
    x = np.arange(len(order))
    cols = [fs.PALETTE["blue"] if o == "vanilla" else
            (fs.CEDAR if o == "cedar_v1" else fs.PALETTE["purple"]) for o in order]
    ax.bar(x, m.values, 0.6, color=[fs._lighten(c, 0.35) for c in cols],
           edgecolor=cols, linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([_V2_LABEL[o].replace(" + ", "\n+") for o in order],
                       fontsize=7.5)
    ax.set_ylim(0.7, 0.9)
    ax.set_ylabel("Mean ROC-AUC (3 outcomes)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")


def _plot_omitted_pooled(ax, t2d):
    """Pooled mean ROC-AUC by system on the full T2D held-out set: the single LLM
    and every CEDAR configuration cluster together, the naive multi-agent baseline
    sits far below, and late ML-fusion matches CEDAR. This is the compact
    everything-converges-except-multi-agent bar (also preserves the multi-agent
    evidence that used to be its own panel)."""
    ag = G.try_ds("nb5_agentic_results_t2d"); md = t2d.get("model_d")
    pooled = G.try_ds("nb_hybrid_pooled_t2d")
    a = "roc_auc_mean"
    vals, labs, cols = [], [], []
    def _add(v, lab, c):
        if v == v: vals.append(v); labs.append(lab); cols.append(c)
    if ag is not None:
        _add(ag[ag["config_id"] == "model_a"][a].mean(), "Single LLM", fs.PALETTE["blue"])
        _add(ag[ag["config_id"] == "model_b"][a].mean(), "Naive\nmulti-agent", fs.PALETTE["orange"])
    if md is not None:
        _add(md[md["config_id"] == "model_d_full"][a].mean(), "CEDAR", fs.CEDAR)
    if pooled is not None:
        hm = pooled[pooled["system"] == "hybrid_mean"]["mean_roc_auc"]
        if len(hm): _add(float(hm.iloc[0]), "CEDAR +\nML fusion", fs.PALETTE["green"])
    x = np.arange(len(vals))
    ax.bar(x, vals, 0.62, color=[fs._lighten(c, 0.3) for c in cols],
           edgecolor=cols, linewidth=1.2)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylim(0.5, 0.9); ax.set_ylabel("Mean ROC-AUC (T2D, 7 outcomes)", fontsize=10)
    ax.set_title("Only the naive multi-agent design underperforms",
                 fontsize=10.2, fontweight="bold")


def build_supp_omitted_configs(t2d):
    """Additional CEDAR-enhancement configurations tested, none of which beats a
    single CEDAR pass (companion to the main-text convergence result; the former
    pooled-bars panel is dropped because the main text already shows all model
    families). (A) CEDAR v2 self-refinement / ensemble variants vs single LLM,
    paired dAUC per outcome (the three screened high-headroom outcomes).
    (B) every late-fusion variant (mean / route / stack) vs CEDAR, per outcome,
    dAUC with 95% CI. (C) per-outcome Brier: CEDAR vs each fusion variant ---
    fusion consistently improves calibration across all outcomes. All T2D."""
    v2diffs = G.try_ds("nb_cedar_v2_paired_diffs_t2d")
    hydiffs2 = G.try_ds("nb_hybrid_diffs_t2d")
    perf2 = G.try_ds("nb_hybrid_perf_t2d")
    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(1, 3, wspace=0.5, width_ratios=[1.0, 1.15, 1.0])
    axA = fig.add_subplot(gs[0, 0]); G.panel_label(axA, "A")
    if v2diffs is not None:
        _plot_v2_forest(axA, v2diffs, "vanilla",
                        "CEDAR self-refine/ensemble vs single LLM\n"
                        "($\\Delta$AUC, 95% CI; 3 highest-headroom outcomes)")
    else:
        G.empty_panel(axA, "no nb_cedar_v2 diffs")
    axB = fig.add_subplot(gs[0, 1]); G.panel_label(axB, "B")
    _plot_hybrid_diffs(axB, hydiffs2, "Late ML-fusion vs CEDAR by outcome (T2D)")
    axC = fig.add_subplot(gs[0, 2]); G.panel_label(axC, "C")
    _plot_fusion_brier(axC, perf2, "Fusion improves calibration (T2D Brier)")
    _save_supp(fig, "figS_omitted_configs")


# ══════════════════════════════════════════════════════════════════
# S-Error-analysis: do the four paradigms make the SAME mistakes?
#   A) error-agreement heatmap (phi correlation of per-patient error indicators),
#      T2D + T1D — LLM systems err together; LLM vs ML errors are decorrelated.
#   B) error rate by outcome-prevalence tertile per paradigm — the prevalence
#      crossover (classical ML wins rare outcomes; LLMs win common ones).
#   C) consensus: distribution of how many paradigms miss each patient.
# ══════════════════════════════════════════════════════════════════
_ERR_SYS = ["classical", "temporal", "single_llm", "cedar"]
_ERR_SYS_LAB = {"classical": "Classical\nML", "temporal": "Temporal\nML",
                "single_llm": "Single\nLLM", "cedar": "CEDAR"}


def _plot_error_heatmap(ax, ov, title):
    """Symmetric phi-correlation matrix of per-patient error indicators."""
    n = len(_ERR_SYS)
    M = np.full((n, n), np.nan)
    for i in range(n):
        M[i, i] = 1.0
    for _, r in ov.iterrows():
        a, b = r["system_a"], r["system_b"]
        if a in _ERR_SYS and b in _ERR_SYS:
            i, j = _ERR_SYS.index(a), _ERR_SYS.index(b)
            M[i, j] = M[j, i] = r["phi_correlation"]
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    labs = [_ERR_SYS_LAB[s] for s in _ERR_SYS]
    ax.set_xticklabels(labs, fontsize=8.5); ax.set_yticklabels(labs, fontsize=8.5)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=8.5,
                        color="white" if abs(M[i, j]) > 0.55 else fs.INK)
    ax.set_title(title, fontsize=11, fontweight="bold")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Error correlation ($\\phi$)", fontsize=9)
    cb.ax.tick_params(labelsize=8)


def _plot_error_by_prevalence(ax, strat, title):
    """Grouped bars: per-paradigm error rate within each prevalence tertile."""
    d = strat[strat["stratum_type"] == "outcome_prevalence_tertile"].copy()
    order = ["low", "mid", "high"]
    d = d.set_index("stratum_value").reindex(order)
    cols = {"err_classical": fs.CLASS_COLORS["classical_ml"],
            "err_temporal": fs.CLASS_COLORS["temporal_ml"],
            "err_single_llm": fs.CLASS_COLORS["single_agent_llm"],
            "err_cedar": fs.CEDAR}
    labs = {"err_classical": "Classical ML", "err_temporal": "Temporal ML",
            "err_single_llm": "Single LLM", "err_cedar": "CEDAR"}
    x = np.arange(len(order)); w = 0.2
    for k, (col, c) in enumerate(cols.items()):
        ax.bar(x + (k - 1.5) * w, d[col].values, w, color=fs._lighten(c, 0.3),
               edgecolor=c, linewidth=1.0, label=labs[col])
    ax.set_xticks(x); ax.set_xticklabels([o.capitalize() for o in order], fontsize=9.5)
    ax.set_xlabel("Outcome prevalence tertile", fontsize=10)
    ax.set_ylabel("Error rate (threshold 0.5)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    ax.set_ylim(0, max(0.65, np.nanmax(d[list(cols)].values) * 1.25))


def _plot_error_consensus(ax, cons2, cons1):
    """How many of the 4 paradigms miss each patient (T2D vs T1D)."""
    x = np.arange(5); w = 0.4
    for k, (cons, coh, c) in enumerate(
            [(cons2, "T2D", fs.CEDAR), (cons1, "T1D", fs.PALETTE["blue"])]):
        if cons is None:
            continue
        d = cons.set_index("n_models_wrong").reindex(range(5))
        frac = d["n_patients"] / d["n_patients"].sum()
        ax.bar(x + (k - 0.5) * w, frac.values, w, color=fs._lighten(c, 0.3),
               edgecolor=c, linewidth=1.0, label=coh)
    ax.set_xticks(x); ax.set_xticklabels(range(5), fontsize=9.5)
    ax.set_xlabel("Number of paradigms wrong (of 4)", fontsize=10)
    ax.set_ylabel("Fraction of patients", fontsize=10)
    ax.set_title("Error consensus across paradigms", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)


def build_supp_error_analysis():
    """Merged error-analysis + selective-prediction supplement (former
    figS_error_analysis + figS_selective). The per-patient error-correlation
    heatmaps are now in main Figure 6A/B and are NOT duplicated here; this figure
    keeps the supporting detail: (A) error rate by prevalence tertile --- the
    prevalence crossover (mechanism behind decorrelated errors; classical ML is
    safe-but-blind at low prevalence); (B) error consensus (how many paradigms miss
    each patient); (C,~D) risk--coverage curves for selective prediction, T2D/T1D
    --- abstention helps both a single LLM and CEDAR, but CEDAR's confidence signals
    give no advantage over a single LLM's probability."""
    st2 = G.try_ds("nb_error_by_patient_stratum_t2d")
    cons2 = G.try_ds("nb_error_consensus_t2d"); cons1 = G.try_ds("nb_error_consensus_t1d")
    rc2 = G.try_ds("nb_selective_rc_curve_t2d"); rc1 = G.try_ds("nb_selective_rc_curve_t1d")
    if st2 is None and rc2 is None:
        print("  [skip] figS_error_analysis: no stratum/selective datasets"); return
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30)
    axA = fig.add_subplot(gs[0, 0]); G.panel_label(axA, "A")
    _plot_error_by_prevalence(axA, st2, "T2D: error rate by prevalence tertile")
    axB = fig.add_subplot(gs[0, 1]); G.panel_label(axB, "B")
    _plot_error_consensus(axB, cons2, cons1)
    axC = fig.add_subplot(gs[1, 0]); G.panel_label(axC, "C")
    _plot_selective_rc(axC, rc2, "T2D: selective prediction (risk--coverage)")
    axD = fig.add_subplot(gs[1, 1]); G.panel_label(axD, "D")
    _plot_selective_rc(axD, rc1, "T1D: selective prediction (risk--coverage)")
    _save_supp(fig, "figS_error_analysis")


# ══════════════════════════════════════════════════════════════════
# S-Hybrid: late-fusion of CEDAR + classical/temporal ML (Architecture A).
#   A) pooled ROC-AUC per system (fusion does NOT raise discrimination).
#   B) pooled Brier per system (fusion sharply improves calibration).
#   C) per-outcome paired dAUC of hybrid_mean vs CEDAR — wins/losses cancel,
#      tracking whether standalone ML was competitive on that outcome.
# ══════════════════════════════════════════════════════════════════
_HYB_ORDER = ["cedar", "single_llm", "best_classical", "best_temporal",
              "hybrid_mean", "hybrid_stack", "hybrid_route"]
_HYB_LAB = {"cedar": "CEDAR", "single_llm": "Single LLM",
            "best_classical": "Classical ML", "best_temporal": "Temporal ML",
            "hybrid_mean": "Hybrid\n(mean)", "hybrid_stack": "Hybrid\n(stack)",
            "hybrid_route": "Hybrid\n(route)"}


def _hyb_color(sysid):
    if sysid == "cedar":
        return fs.CEDAR
    if sysid == "single_llm":
        return fs.CLASS_COLORS["single_agent_llm"]
    if sysid == "best_classical":
        return fs.CLASS_COLORS["classical_ml"]
    if sysid == "best_temporal":
        return fs.CLASS_COLORS["temporal_ml"]
    return fs.PALETTE["green"]   # hybrids


def _plot_hybrid_pooled(ax, pooled, metric, title, ylab, ylim=None):
    order = [s for s in _HYB_ORDER if s in set(pooled["system"])]
    d = pooled.set_index("system").reindex(order)
    x = np.arange(len(order))
    cols = [_hyb_color(s) for s in order]
    ax.bar(x, d[metric].values, 0.62, color=[fs._lighten(c, 0.32) for c in cols],
           edgecolor=cols, linewidth=1.1)
    for xi, v in zip(x, d[metric].values):
        if pd.notna(v):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.8,
                    color=fs.INK)
    ax.set_xticks(x); ax.set_xticklabels([_HYB_LAB[s] for s in order], fontsize=7.8)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)


_FUSION_LAB = {"hybrid_mean": "mean", "hybrid_route": "route",
               "hybrid_stack": "stack"}
_FUSION_ORDER = ["hybrid_mean", "hybrid_route", "hybrid_stack"]


def _plot_hybrid_diffs(ax, diffs, title):
    """Per-outcome paired dAUC vs CEDAR for ALL late-fusion variants
    (mean / route / stack), each with 95% CI; vertical line at 0. Points whose
    CI excludes 0 are drawn saturated (significant), others faded."""
    d = diffs[diffs["comparator"] == "cedar"].copy()
    if d.empty:
        G.empty_panel(ax, "no fusion vs cedar diffs"); return
    d["outcome"] = d["outcome"].map(G.pretty_outcome)
    variants = [v for v in _FUSION_ORDER if v in set(d["hybrid_variant"])]
    vcol = {"hybrid_mean": fs.PALETTE["green"],
            "hybrid_route": fs.PALETTE["blue"],
            "hybrid_stack": fs.PALETTE["purple"]}
    outcomes = sorted(d["outcome"].unique())
    yticks, ylabels, row = [], [], 0
    for oc in outcomes:
        for v in variants:
            r = d[(d["outcome"] == oc) & (d["hybrid_variant"] == v)]
            if r.empty:
                continue
            r = r.iloc[0]
            sig = r["auc_diff_ci_low"] > 0 or r["auc_diff_ci_high"] < 0
            col = vcol[v]; alpha = 1.0 if sig else 0.4
            ax.plot([r["auc_diff_ci_low"], r["auc_diff_ci_high"]], [row, row],
                    color=col, lw=1.6, alpha=alpha, zorder=2)
            ax.scatter(r["auc_diff_mean"], row, s=30, color=col, alpha=alpha,
                       zorder=3, edgecolor="white", linewidth=0.6)
            yticks.append(row); ylabels.append(f"{oc[:16]} · {_FUSION_LAB[v]}")
            row += 1
        row += 0.6
    ax.axvline(0, ls="--", color=fs.INK, lw=1.0, zorder=1)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=6.8)
    ax.set_xlabel("$\\Delta$ ROC-AUC (fusion $-$ CEDAR)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    # variant legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=vcol[v], marker="o", lw=1.6,
                      label=_FUSION_LAB[v]) for v in variants]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, title="Fusion",
              title_fontsize=8, loc="lower right")


def _plot_fusion_brier(ax, perf, title):
    """Per-outcome Brier: CEDAR vs each late-fusion variant. Paired points joined
    per outcome show fusion consistently lowers Brier (better calibration) across
    all outcomes. (No paired-bootstrap CI is available for the fusion systems, so
    this shows the per-outcome replication rather than a single pooled bar.)"""
    if perf is None or getattr(perf, "empty", True):
        G.empty_panel(ax, "no nb_hybrid_perf"); return
    d = perf.copy(); d["outcome"] = d["outcome"].map(G.pretty_outcome)
    systems = ["cedar"] + [v for v in _FUSION_ORDER if v in set(d["system"])]
    lab = {"cedar": "CEDAR", **_FUSION_LAB}
    scol = {"cedar": fs.CEDAR, "hybrid_mean": fs.PALETTE["green"],
            "hybrid_route": fs.PALETTE["blue"], "hybrid_stack": fs.PALETTE["purple"]}
    outcomes = sorted(d["outcome"].unique())
    x = np.arange(len(systems))
    # per-outcome Brier, wide (outcome x system) for paired connectors + test
    wide = d.pivot_table(index="outcome", columns="system", values="brier_mean")
    for oc in outcomes:  # faint connector per outcome
        vals = [d[(d["outcome"] == oc) & (d["system"] == s)]["brier_mean"]
                for s in systems]
        vals = [float(v.iloc[0]) if len(v) else np.nan for v in vals]
        ax.plot(x, vals, color=fs.PALETTE["grey"], lw=0.8, alpha=0.5, zorder=1)
    ymax = pd.to_numeric(d["brier_mean"], errors="coerce").max()
    from scipy import stats as _st
    for xi, s in zip(x, systems):
        yv = pd.to_numeric(d[d["system"] == s]["brier_mean"], errors="coerce").dropna()
        ax.scatter([xi] * len(yv), yv, s=34, color=scol[s], alpha=0.85,
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter([xi], [yv.mean()], marker="_", s=420, color=scol[s],
                   linewidth=2.2, zorder=4)   # mean tick
        # significance star vs CEDAR (paired Wilcoxon signed-rank across outcomes)
        if s != "cedar" and "cedar" in wide.columns and s in wide.columns:
            pair = wide[[s, "cedar"]].dropna()
            if len(pair) >= 5 and (pair[s] != pair["cedar"]).any():
                try:
                    p = _st.wilcoxon(pair[s], pair["cedar"]).pvalue
                except ValueError:
                    p = np.nan
                star = ("***" if p < 0.001 else "**" if p < 0.01
                        else "*" if p < 0.05 else "")
                if star:
                    ax.text(xi, ymax * 1.04, star, ha="center", va="bottom",
                            fontsize=13, color=scol[s], fontweight="bold")
    ax.set_ylim(top=ymax * 1.12)
    ax.set_xticks(x); ax.set_xticklabels([lab[s] for s in systems], fontsize=8.5)
    ax.set_ylabel("Brier (per outcome; lower = better)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")


def build_supp_hybrid():
    """Late ML-fusion (Hybrid A): the discrimination-vs-calibration result. Pooled
    ROC-AUC is unchanged by fusion while pooled Brier improves (recalibration). The
    per-outcome fusion-vs-CEDAR diffs now live in figS_omitted_configs."""
    p2 = G.try_ds("nb_hybrid_pooled_t2d"); p1 = G.try_ds("nb_hybrid_pooled_t1d")
    if p2 is None and p1 is None:
        print("  [skip] figS_hybrid: no nb_hybrid_pooled datasets"); return
    fig = plt.figure(figsize=(14, 5.5))
    gs = fig.add_gridspec(1, 2, wspace=0.28)
    axA = fig.add_subplot(gs[0, 0]); G.panel_label(axA, "A")
    _plot_hybrid_pooled(axA, p2, "mean_roc_auc",
                        "T2D: pooled ROC-AUC (discrimination unchanged by fusion)",
                        "Mean ROC-AUC (7 outcomes)", ylim=(0.6, 0.9))
    axB = fig.add_subplot(gs[0, 1]); G.panel_label(axB, "B")
    _plot_hybrid_pooled(axB, p2, "mean_brier",
                        "T2D: pooled Brier (fusion improves calibration)",
                        "Mean Brier (lower = better)", ylim=(0, 0.28))
    _save_supp(fig, "figS_hybrid")


# ══════════════════════════════════════════════════════════════════
# S-Hybrid-B: ML-score-as-tool inside CEDAR (Architecture B).
#   A) Arm 2 (T1D optimal glycemic control, the one ML-competitive outcome):
#      paired dAUC of cedar_B1/B2 vs vanilla / cedar_v1 / stack (late fusion).
#   B) Behavioral signal: how much the injected score moved CEDAR's output
#      (mean |shift|) and whether overrides were correct, per outcome.
#   C) Arm 1 (T2D-3, weak-ML outcomes): paired dAUC vs vanilla -- null/negative.
# ══════════════════════════════════════════════════════════════════
# Fusion combiners reuse the Fig S6 naming ("mean" / "stack") for consistency;
# in Fig S6 these are the late-fusion combiners (output-level, no LLM reasoning).
_HB_LAB = {"vanilla": "vs Single LLM", "cedar_v1": "vs CEDAR (no tool)",
           "hybrid_stack": "vs stack (late fusion)", "hybrid_mean": "vs mean (late fusion)"}
_HB_SYS_COL = {"cedar_B1": fs.PALETTE["green"], "cedar_B2": fs.PALETTE["purple"]}
_HB_SYS_LAB = {"cedar_B1": "CEDAR+tool (score)", "cedar_B2": "CEDAR+tool (score+reason)"}


def _plot_hb_absolute(ax, perf, outcome, title):
    """Absolute ROC-AUC bars for the tool-use comparison on one outcome, ordered to
    tell the story directly: single LLM -> CEDAR (no tool) -> CEDAR + tool, with the
    stack (late-fusion) combiner as a reference. Tool-equipped CEDAR bars are taller
    than the no-tool bars -> equipping the agent with a competitive tool improves
    performance on this outcome."""
    if perf is None or getattr(perf, "empty", True):
        G.empty_panel(ax, "no nb_hybrid_b_perf"); return
    d = perf[perf["outcome"] == outcome].set_index("system")
    order = [("vanilla", "Single LLM\n(no tool)", fs.CLASS_COLORS["single_agent_llm"]),
             ("cedar_v1", "CEDAR\n(no tool)", fs._lighten(fs.CEDAR, 0.35)),
             ("cedar_B1", "CEDAR + tool\n(score)", fs.PALETTE["green"]),
             ("cedar_B2", "CEDAR + tool\n(score+reason)", fs._lighten(fs.PALETTE["green"], 0.3)),
             ("hybrid_stack", "stack\n(late fusion)", fs.PALETTE["grey"])]
    order = [(s, lab, c) for s, lab, c in order if s in d.index]
    x = np.arange(len(order))
    for xi, (s, lab, c) in zip(x, order):
        v = float(d.loc[s, "roc_auc_mean"])
        lo = float(d.loc[s, "roc_auc_ci_low"]); hi = float(d.loc[s, "roc_auc_ci_high"])
        edge = c if s.startswith("cedar_B") else fs.INK
        ax.bar(xi, v, 0.66, color=fs._lighten(c, 0.2) if not s.startswith("cedar_B") else c,
               edgecolor=edge, linewidth=1.2 if s.startswith("cedar_B") else 0.8,
               yerr=[[v - lo], [hi - v]], capsize=3, error_kw={"lw": 0.9, "ecolor": fs.INK})
        ax.text(xi, v + (hi - v) + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([lab.replace("\n", " ") for _, lab, _ in order],
                       fontsize=7.0, rotation=20, ha="right")
    ax.set_ylabel("ROC-AUC (95% CI)", fontsize=10)
    ax.set_ylim(0.65, 0.95)
    ax.set_title(title, fontsize=10.5, fontweight="bold")


def _plot_hb_forest(ax, diffs, outcome, comparators, title):
    d = diffs[diffs["outcome"] == outcome].copy()
    if d.empty:
        G.empty_panel(ax, f"no B diffs for {outcome}"); return
    rows, ylab = [], []
    for comp in comparators:
        for sysid in ["cedar_B1", "cedar_B2"]:
            r = d[(d["system"] == sysid) & (d["comparator"] == comp)]
            if not r.empty:
                rows.append((sysid, comp, r.iloc[0]));
    y = np.arange(len(rows))
    for yi, (sysid, comp, r) in zip(y, rows):
        sig = r["auc_diff_ci_low"] > 0 or r["auc_diff_ci_high"] < 0
        col = _HB_SYS_COL[sysid]
        ax.plot([r["auc_diff_ci_low"], r["auc_diff_ci_high"]], [yi, yi],
                color=col, lw=1.8, alpha=1.0 if sig else 0.4, zorder=2)
        ax.scatter(r["auc_diff_mean"], yi, s=40, color=col, zorder=3,
                   edgecolor="white", linewidth=0.6, alpha=1.0 if sig else 0.5)
        ylab.append(f"{_HB_SYS_LAB[sysid].split('(')[1].rstrip(')')} · {_HB_LAB[comp]}")
    ax.axvline(0, ls="--", color=fs.INK, lw=1.0, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(ylab, fontsize=8)
    ax.set_xlabel("$\\Delta$ ROC-AUC (95% CI)", fontsize=10)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.invert_yaxis()


def _plot_hb_behavior(ax, beh2, beh1, title):
    """Mean |shift vs no-tool| (x) against override-correctness (y), per outcome;
    marker = cohort. Shows the LLM moves toward good scores, ignores/mis-trusts bad."""
    # manual label offsets (pts) to avoid collisions in the clustered top-left
    _OFF = {"OUTCOME_Dyslipidemia": (6, 8), "OUTCOME_GLP1RA_Response": (6, -12),
            "OUTCOME_Optimal_Glycemic_Control": (-4, 10),
            "OUTCOME_Hypertension": (6, 4)}
    for beh, coh, mk in [(beh2, "T2D", "o"), (beh1, "T1D", "s")]:
        if beh is None:
            continue
        b = beh[beh["system"] == "cedar_B1"]
        for _, r in b.iterrows():
            x, y = r["mean_abs_shift_vs_cedar_v1"], r["frac_ml_overridden_correctly"]
            ax.scatter(x, y, s=70, marker=mk, color=fs.CEDAR, edgecolor="white",
                       linewidth=0.7, zorder=3)
            ax.annotate(f"{coh} {G.pretty_outcome(r['outcome'])[:12]}", (x, y),
                        fontsize=6.6, textcoords="offset points",
                        xytext=_OFF.get(r["outcome"], (6, 3)), color=fs.INK)
    ax.axhline(0.5, ls=":", color=fs.PALETTE["grey"], lw=1.0)
    ax.text(0.97, 0.52, "overrides help", transform=ax.get_yaxis_transform()
            if False else ax.transAxes, fontsize=6.5, color=fs.PALETTE["grey"],
            ha="right", va="bottom")
    ax.set_xlabel("Mean |shift| vs. no-tool CEDAR", fontsize=10)
    ax.set_ylabel("Fraction of overrides correct", fontsize=10)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 1.0); ax.set_xlim(0, 0.37)


def _plot_hb_paired_forest(ax, diffs, outcome, title):
    """Paired ROC-AUC differences (tool-equipped agent minus each comparator) on one
    outcome, with 95% CIs from the PAIRED bootstrap. Unlike the marginal CIs of the
    absolute-AUC bars, the paired test is powered to show significance: a bar whose
    CI excludes 0 (bold) is a significant gain."""
    if diffs is None or getattr(diffs, "empty", True):
        G.empty_panel(ax, "no nb_hybrid_b_diffs"); return
    d = diffs[diffs["outcome"] == outcome]
    rows = []
    for sysid, slab in [("cedar_B1", "CEDAR+tool (score)"),
                        ("cedar_B2", "CEDAR+tool (score+reason)")]:
        for comp, clab in [("vanilla", "vs single LLM (no tool)"),
                           ("cedar_v1", "vs CEDAR (no tool)"),
                           ("hybrid_stack", "vs stack (late fusion)")]:
            r = d[(d["system"] == sysid) & (d["comparator"] == comp)]
            if not r.empty:
                rr = r.iloc[0]
                rows.append((f"{slab.split('(')[1].rstrip(')')} · {clab}",
                             float(rr["auc_diff_mean"]), float(rr["auc_diff_ci_low"]),
                             float(rr["auc_diff_ci_high"]), sysid))
    rows = rows[::-1]
    y = np.arange(len(rows))
    for yi, (lab, m, lo, hi, sysid) in zip(y, rows):
        sig = lo > 0 or hi < 0
        col = _HB_SYS_COL[sysid]
        ax.plot([lo, hi], [yi, yi], color=col, lw=2.0 if sig else 1.4,
                alpha=1.0 if sig else 0.45, zorder=2)
        ax.scatter(m, yi, s=46 if sig else 32, color=col, zorder=3,
                   edgecolor=fs.INK if sig else "white",
                   linewidth=0.8 if sig else 0.5, alpha=1.0 if sig else 0.55)
    ax.axvline(0, ls="--", color=fs.INK, lw=1.0, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlabel("$\\Delta$ ROC-AUC, paired (95% CI)", fontsize=10)
    ax.set_title(title, fontsize=10.2, fontweight="bold")


def build_supp_hybrid_b():
    """Mechanism for the targeted tool-use result in main Fig 6D--E: how the agent
    used the injected ML score. The paired-significance forests now live in main
    Fig 6 (D = ML-competitive T1D glycemic control; E = ML-weak T2D dyslipidemia).
    Here we show the behavioral signal --- the mean shift the score induced in
    CEDAR's output vs.\ the fraction of resulting overrides that were correct."""
    beh1 = G.try_ds("nb_hybrid_b_behavior_t1d"); beh2 = G.try_ds("nb_hybrid_b_behavior_t2d")
    if beh1 is None and beh2 is None:
        print("  [skip] figS_hybrid_b: no nb_hybrid_b_behavior datasets"); return
    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_subplot(111)
    _plot_hb_behavior(ax, beh2, beh1, "How the agent used the tool's score")
    _save_supp(fig, "figS_hybrid_b")


# ══════════════════════════════════════════════════════════════════
def main():
    print("Loading cohorts (reusing main-figure loader) ...")
    t2d = G.load_all("t2d")
    t1d = G.load_all("t1d")

    # nb8 datasets are not in load_all(); pull them directly if present.
    t2d_extra = {"temporal": G.try_ds("nb8_temporal_validation_t2d"),
                 "split":    G.try_ds("nb8_temporal_split_summary_t2d")}
    t1d_extra = {"temporal": G.try_ds("nb8_temporal_validation_t1d"),
                 "split":    G.try_ds("nb8_temporal_split_summary_t1d")}

    print("\nBuilding supplementary figures ...")
    for name, fn in [
        ("figS_prevalence",        lambda: build_supp_prevalence(t2d, t1d)),
        ("figS_calibration",       lambda: build_supp_calibration(t2d, t1d)),
        ("figS_faithfulness",      lambda: build_supp_faithfulness(t2d, t1d)),
        ("figS_modality",          lambda: build_supp_modality(t2d, t1d)),
        ("figS_temporal_val",      lambda: build_supp_temporal_val(t2d_extra, t1d_extra, t2d, t1d)),
        ("figS_omitted_configs",   lambda: build_supp_omitted_configs(t2d)),
        ("figS_error_analysis",    lambda: build_supp_error_analysis()),
        ("figS_hybrid_b",          lambda: build_supp_hybrid_b()),
    ]:
        print(f"\n  [{name}]")
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  [error] {name}: {e}")
            traceback.print_exc()
    print(f"\nDone. Supplementary panels -> {SUPP_OUTDIR}/")


if __name__ == "__main__":
    main()
