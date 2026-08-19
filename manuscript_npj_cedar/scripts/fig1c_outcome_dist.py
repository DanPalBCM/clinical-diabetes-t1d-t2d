"""
Figure 1C — T2D outcome prevalence over the FULL assessable population, sized for
the middle-right quadrant of the combined Figure 1 (portrait, to sit beside the
tall Fig1B).

Uses nb4_matched_comparison_t2d (which carries the full train+test split counts),
NOT the 250-patient agentic evaluation sample, so the distribution is consistent
with the full-cohort framing of panel B. Because the split is stratified, the
prevalence equals the sample prevalence; only the denominators grow to the full
assessable population (n_train + n_test).

Standalone: writes additional_analysis/figure_sources/fig1_panels/Fig1C.* (a Fig 1 build input).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_main_figures as G
import figstyle as fs

fs.apply_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "additional_analysis", "figure_sources", "fig1_panels")
os.makedirs(OUT, exist_ok=True)

# Portrait aspect to match Fig1B's footprint (Fig1B ~ 0.86 w/h). Slightly wider
# so the outcome labels + count annotations fit.
fig, ax = plt.subplots(figsize=(5.4, 5.6))

# Full assessable population per outcome = n_train + n_test (stratified split, so
# prevalence == test_prevalence). One row per outcome (all models share counts).
src = G.filter_h(G.try_ds("nb4_matched_comparison_t2d"))
o = "outcome"
d = src.drop_duplicates(o).copy()
d[o] = d[o].map(G.pretty_outcome)
d["_N"] = d["n_train"] + d["n_test"]
d["_prev"] = d["test_prevalence"]
d["_pos"] = (d["_N"] * d["_prev"]).round().astype(int)
d = d.set_index(o)

order = G.order_outcomes(d.index)[::-1]
d = d.reindex(order)
vals = (d["_prev"] * 100).values
y = np.arange(len(order))
# soft/pretty bars matching the rest of the manuscript (lightened fill + colour outline)
fs.soft_bars(ax, y, vals, [fs.PALETTE["blue"]] * len(order),
             width=fs.BAR_W, horizontal=True)
for yi, oc in zip(y, order):
    v = d.loc[oc, "_prev"] * 100
    if pd.notna(v):
        ax.text(v + 1.5, yi, f"{v:.0f}%  ({int(d.loc[oc, '_pos'])}/{int(d.loc[oc, '_N'])})",
                va="center", fontsize=11.5, color=fs.INK, fontweight="medium")
ax.set_yticks(y); ax.set_yticklabels(order, fontsize=11)
ax.tick_params(axis="x", labelsize=10)
ax.set_xlim(0, max(np.nanmax(vals) * 1.6, 10))   # room for the (pos/N) labels
ax.set_xlabel("Prevalence at 2-year target visit (%)", fontsize=11)
ax.set_title("Outcome prevalence (T2D, full cohort)", fontsize=12.5, fontweight="bold")

fig.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(OUT, f"Fig1C.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    print("saved ->", p)
plt.close(fig)
