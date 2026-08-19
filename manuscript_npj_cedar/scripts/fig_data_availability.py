"""
Standalone figure: data availability over follow-up (T2D vs T1D).

Shows, for each of the 10 visit windows (6-month intervals), how many patients
have observed data (and the % of the cohort). Two cohorts overlaid so the reader
sees the follow-up attrition and why the 2-year (v4) horizon is the primary one.

This is a STANDALONE figure (not wired into any composite panel yet).
Run:  python scripts/fig_data_availability.py
Out:  additional_analysis/legacy_figures/intermediates/fig_data_availability.{png,pdf}
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle as fs

fs.apply_style()
# Standalone exploratory panel, not referenced by the manuscript -> keep in legacy.
OUT = os.path.join(os.path.dirname(__file__), "..", "additional_analysis", "legacy_figures", "intermediates")
os.makedirs(OUT, exist_ok=True)

# ── Data (from the Foundry export) ────────────────────────────────────────────
T2D = pd.DataFrame({
    "visit_month":  [6, 12, 18, 24, 30, 36, 42, 48, 54, 60],
    "n_with_data":  [2223, 1630, 1521, 1257, 1108, 973, 856, 708, 592, 473],
    "pct_with_data":[0.8931, 0.6549, 0.6111, 0.5050, 0.4452,
                     0.3909, 0.3439, 0.2845, 0.2378, 0.1900],
    "n_total": 2489,
})
T1D = pd.DataFrame({
    "visit_month":  [6, 12, 18, 24, 30, 36, 42, 48, 54, 60],
    "n_with_data":  [3370, 3158, 3025, 2837, 2706, 2556, 2383, 2225, 2111, 1954],
    "pct_with_data":[0.8387, 0.7860, 0.7529, 0.7061, 0.6735,
                     0.6361, 0.5931, 0.5538, 0.5254, 0.4863],
    "n_total": 4018,
})

C_T2D = fs.PALETTE["blue"]     # primary cohort
C_T1D = fs.PALETTE["orange"]   # contrast cohort
VISITS = [f"v{i}" for i in range(1, 11)]


def _mark_task(ax):
    """Shade the input window (v1-v3) and mark the v4 target."""
    ax.axvspan(0.5, 3.5, color=fs.PALETTE["blue"], alpha=0.07, lw=0)
    ax.axvline(4, ls="--", color=fs.CEDAR, lw=1.2, alpha=0.85)


fig, axP = plt.subplots(figsize=(6.2, 4.0))
x = np.arange(1, 11)

# ── Percentage of cohort with observed data ──────────────────────────────────
_mark_task(axP)                       # shaded input window + red dashed v4 line
LABEL_VISITS = {1, 4, 10}             # annotate only start / target / final
for df, c, lab in [(T2D, C_T2D, "T2D"), (T1D, C_T1D, "T1D")]:
    pct = df["pct_with_data"].values * 100
    axP.plot(x, pct, "-o", color=c, lw=2, markersize=5,
             markeredgecolor="white", markeredgewidth=0.6, label=lab)
    for xi, p in zip(x, pct):
        if xi in LABEL_VISITS:
            axP.text(xi, p + 2.0, f"{p:.0f}%", fontsize=8, ha="center",
                     va="bottom", color=c, fontweight="bold")
axP.set_xticks(x); axP.set_xticklabels(VISITS, fontsize=8)
axP.set_ylim(0, 100)
axP.set(xlabel="Visit window (6-month intervals)",
        ylabel="Cohort with observed data (%)")
axP.legend(frameon=False, fontsize=8, loc="upper right")

fig.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(OUT, f"fig_data_availability.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    print("saved ->", p)
plt.close(fig)
