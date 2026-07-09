"""
T2D OUTCOMES HEATMAP — VERSION 3 (Aligned with corrected 1/0/null encoding)
============================================================================
Now that Transform 11 properly encodes:
    1    = patient HAS the outcome
    0    = patient was ASSESSED and does NOT have the outcome
    null = patient was NOT ASSESSED (not in denominator)

Denominators are simply: count of non-null values per outcome column.
No more manual BP-flag filters or A1C-based denominator overrides —
the transform already handles all of that in the outcome encoding.

Two sets are still shown for HTN & Dyslipidemia:
  Set 1 — All assessed patients (non-null outcome), full cohort
  Set 2 — Set 1 + valid A1C at the same timepoint

Standard outcomes (Micro, Glycemic, Insulin Indep, Met Response, GLP-1):
  Denominator = non-null outcome values (already reflects who was assessed)
  Shown within A1C-at-diagnosis filtered cohort for clinical relevance.

Output: analysis/outcomes_heatmap/
"""

from foundry.transforms import Dataset
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================
OUTPUT_DIR = "analysis/outcomes_heatmap"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory ready: {OUTPUT_DIR}/")

def out(filename):
    return os.path.join(OUTPUT_DIR, filename)

# ============================================================================
# LOAD DATA
# ============================================================================
df_full = Dataset.get("t2d_outcomes").read_table(format="pandas")
print(f"\nFull dataset loaded: {len(df_full):,} rows, {len(df_full.columns)} columns")

# ============================================================================
# COLUMN MAPS
# ============================================================================
A1C_COLUMNS = {
    "At Diagnosis": "a1c_diagnosis",
    "At 2 Years":   "a1c_2yr",
    "At 5 Years":   "a1c_5yr",
}

HTN_OUTCOME_COLS = {
    "At Diagnosis": "OUTCOME_Hypertension_at_diagnosis",
    "At 2 Years":   "OUTCOME_Hypertension_at_2_years",
    "At 5 Years":   "OUTCOME_Hypertension_at_5_years",
}

DYSLIP_OUTCOME_COLS = {
    "At Diagnosis": "OUTCOME_Dyslipidemia_at_diagnosis",
    "At 2 Years":   "OUTCOME_Dyslipidemia_at_2_years",
    "At 5 Years":   "OUTCOME_Dyslipidemia_at_5_years",
}

MICRO_OUTCOME_COLS = {
    "At Diagnosis": "OUTCOME_Microalbuminuria_at_diagnosis",
    "At 2 Years":   "OUTCOME_Microalbuminuria_at_2_years",
    "At 5 Years":   "OUTCOME_Microalbuminuria_at_5_years",
}

GLYCEMIC_OUTCOME_COLS = {
    "At Diagnosis": "OUTCOME_Optimal_Glycemic_Control_at_diagnosis",
    "At 2 Years":   "OUTCOME_Optimal_Glycemic_Control_at_2_years",
    "At 5 Years":   "OUTCOME_Optimal_Glycemic_Control_at_5_years",
}

INSULIN_INDEP_COLS = {
    "At Diagnosis": "OUTCOME_Insulin_Independence_at_diagnosis",
    "At 2 Years":   "OUTCOME_Insulin_Independence_at_2_years",
    "At 5 Years":   "OUTCOME_Insulin_Independence_at_5_years",
}

MET_RESPONSE_COLS = {
    "At Diagnosis": None,
    "At 2 Years":   "OUTCOME_Metformin_Response_at_2_years",
    "At 5 Years":   "OUTCOME_Metformin_Response_at_5_years",
}

GLP1_RESPONSE_COLS = {
    "At Diagnosis": None,
    "At 2 Years":   "OUTCOME_GLP1RA_Response_at_2_years",
    "At 5 Years":   "OUTCOME_GLP1RA_Response_at_5_years",
}

TIMEPOINT_LABELS = ["At Diagnosis", "At 2 Years", "At 5 Years"]

# ============================================================================
# HELPER: compute outcome stats using 1/0/null encoding
# ============================================================================
def outcome_stats(df, col):
    """
    Given a properly encoded outcome column (1/0/null):
      - denominator = count of non-null (assessed patients)
      - numerator   = count of 1s (patients with outcome)
      - prevalence  = numerator / denominator * 100
    Returns NaN for all if col is None or not in dataframe.
    """
    if col is None or col not in df.columns:
        return np.nan, np.nan, np.nan
    series = df[col]
    assessed = series.notna()
    denom = assessed.sum()
    if denom == 0:
        return 0, 0, np.nan
    numer = (series == 1).sum()
    pct = numer / denom * 100
    return int(numer), int(denom), pct

# ============================================================================
# GLOBAL FILTER — A1C at diagnosis cohort (for standard outcomes)
# ============================================================================
a1c_dx_col = A1C_COLUMNS["At Diagnosis"]
if a1c_dx_col not in df_full.columns:
    raise ValueError(f"A1C diagnosis column '{a1c_dx_col}' not found.")

n_full = len(df_full)
t2d = df_full[df_full[a1c_dx_col].notna()].copy()
n_a1c = len(t2d)

print(f"\nGlobal filter — A1C at diagnosis not null:")
print(f"  Full cohort : {n_full:,}")
print(f"  Dropped     : {n_full - n_a1c:,}  ({(n_full - n_a1c) / n_full * 100:.1f}%)")
print(f"  Retained    : {n_a1c:,}")

# ============================================================================
# HTN & DYSLIPIDEMIA — Two sets (full cohort base)
#   Set 1: all assessed patients (outcome non-null) in full cohort
#   Set 2: Set 1 + valid A1C at same timepoint
# ============================================================================
print("\n" + "=" * 60)
print("HYPERTENSION (using non-null outcome as denominator)")
print("=" * 60)

htn_s1 = {"count": {}, "denom": {}, "pct": {}}
htn_s2 = {"count": {}, "denom": {}, "pct": {}}

for tp in TIMEPOINT_LABELS:
    oc = HTN_OUTCOME_COLS[tp]
    ac = A1C_COLUMNS[tp]

    # Set 1: full cohort, outcome non-null
    cnt, den, pct = outcome_stats(df_full, oc)
    htn_s1["count"][tp] = cnt
    htn_s1["denom"][tp] = den
    htn_s1["pct"][tp] = pct

    # Set 2: Set 1 + A1C valid
    if oc in df_full.columns and ac in df_full.columns:
        mask = df_full[oc].notna() & df_full[ac].notna()
        df_s2 = df_full[mask]
        cnt2, den2, pct2 = outcome_stats(df_s2, oc)
    else:
        cnt2, den2, pct2 = np.nan, np.nan, np.nan
    htn_s2["count"][tp] = cnt2
    htn_s2["denom"][tp] = den2
    htn_s2["pct"][tp] = pct2

    print(f"  [{tp}]")
    print(f"    Set 1 (assessed, full cohort) : {cnt:,} / {den:,}  ({pct:.1f}%)" if not np.isnan(pct) else f"    Set 1: N/A")
    print(f"    Set 2 (assessed + A1C)        : {cnt2:,} / {den2:,}  ({pct2:.1f}%)" if not np.isnan(pct2) else f"    Set 2: N/A")

print("\n" + "=" * 60)
print("DYSLIPIDEMIA (using non-null outcome as denominator)")
print("=" * 60)

dys_s1 = {"count": {}, "denom": {}, "pct": {}}
dys_s2 = {"count": {}, "denom": {}, "pct": {}}

for tp in TIMEPOINT_LABELS:
    oc = DYSLIP_OUTCOME_COLS[tp]
    ac = A1C_COLUMNS[tp]

    cnt, den, pct = outcome_stats(df_full, oc)
    dys_s1["count"][tp] = cnt
    dys_s1["denom"][tp] = den
    dys_s1["pct"][tp] = pct

    if oc in df_full.columns and ac in df_full.columns:
        mask = df_full[oc].notna() & df_full[ac].notna()
        df_s2 = df_full[mask]
        cnt2, den2, pct2 = outcome_stats(df_s2, oc)
    else:
        cnt2, den2, pct2 = np.nan, np.nan, np.nan
    dys_s2["count"][tp] = cnt2
    dys_s2["denom"][tp] = den2
    dys_s2["pct"][tp] = pct2

    print(f"  [{tp}]")
    print(f"    Set 1 (assessed, full cohort) : {cnt:,} / {den:,}  ({pct:.1f}%)" if not np.isnan(pct) else f"    Set 1: N/A")
    print(f"    Set 2 (assessed + A1C)        : {cnt2:,} / {den2:,}  ({pct2:.1f}%)" if not np.isnan(pct2) else f"    Set 2: N/A")

# ============================================================================
# STANDARD OUTCOMES — within A1C-filtered cohort (t2d)
# Denominator = non-null outcome values (proper 1/0/null encoding)
# ============================================================================
STANDARD_OUTCOMES = {
    "Microalbuminuria":          MICRO_OUTCOME_COLS,
    "Optimal Glycemic\nControl": GLYCEMIC_OUTCOME_COLS,
    "Insulin\nIndependence":     INSULIN_INDEP_COLS,
    "Metformin\nResponse":       MET_RESPONSE_COLS,
    "GLP-1RA\nResponse":         GLP1_RESPONSE_COLS,
}

STD_LABELS = list(STANDARD_OUTCOMES.keys())
n_time = len(TIMEPOINT_LABELS)
n_std = len(STD_LABELS)

std_count = np.full((n_time, n_std), np.nan)
std_denom = np.full((n_time, n_std), np.nan)
std_pct = np.full((n_time, n_std), np.nan)

print("\n" + "=" * 60)
print("STANDARD OUTCOMES (A1C-filtered cohort, non-null denominator)")
print("=" * 60)

for j, (label, col_map) in enumerate(STANDARD_OUTCOMES.items()):
    print(f"\n  {label.replace(chr(10), ' ')}:")
    for i, tp in enumerate(TIMEPOINT_LABELS):
        col = col_map[tp]
        cnt, den, pct = outcome_stats(t2d, col)
        std_count[i, j] = cnt
        std_denom[i, j] = den
        std_pct[i, j] = pct

        if not np.isnan(pct):
            print(f"    [{tp}] : {cnt:,} / {den:,}  ({pct:.1f}%)")
        else:
            print(f"    [{tp}] : N/A")

# ============================================================================
# ASSEMBLE UNIFIED DISPLAY MATRICES
# ============================================================================
ALL_LABELS = ["Hypertension", "Dyslipidemia"] + STD_LABELS
n_all = len(ALL_LABELS)

mat_pct_s1 = np.full((n_time, n_all), np.nan)
mat_cnt_s1 = np.full((n_time, n_all), np.nan)
mat_den_s1 = np.full((n_time, n_all), np.nan)
mat_pct_s2 = np.full((n_time, n_all), np.nan)
mat_cnt_s2 = np.full((n_time, n_all), np.nan)
mat_den_s2 = np.full((n_time, n_all), np.nan)

for i, tp in enumerate(TIMEPOINT_LABELS):
    # HTN
    mat_pct_s1[i, 0] = htn_s1["pct"][tp]
    mat_cnt_s1[i, 0] = htn_s1["count"][tp]
    mat_den_s1[i, 0] = htn_s1["denom"][tp]
    mat_pct_s2[i, 0] = htn_s2["pct"][tp]
    mat_cnt_s2[i, 0] = htn_s2["count"][tp]
    mat_den_s2[i, 0] = htn_s2["denom"][tp]
    # Dyslipidemia
    mat_pct_s1[i, 1] = dys_s1["pct"][tp]
    mat_cnt_s1[i, 1] = dys_s1["count"][tp]
    mat_den_s1[i, 1] = dys_s1["denom"][tp]
    mat_pct_s2[i, 1] = dys_s2["pct"][tp]
    mat_cnt_s2[i, 1] = dys_s2["count"][tp]
    mat_den_s2[i, 1] = dys_s2["denom"][tp]
    # Standard outcomes
    for j in range(n_std):
        mat_pct_s1[i, j + 2] = std_pct[i, j]
        mat_cnt_s1[i, j + 2] = std_count[i, j]
        mat_den_s1[i, j + 2] = std_denom[i, j]

# ============================================================================
# SAVE RAW NUMBERS TO CSV
# ============================================================================
rows = []
for i, tp in enumerate(TIMEPOINT_LABELS):
    for j, label in enumerate(ALL_LABELS):
        rows.append({
            "timepoint": tp,
            "outcome": label.replace("\n", " "),
            "numerator_s1": mat_cnt_s1[i, j],
            "denominator_s1": mat_den_s1[i, j],
            "prevalence_s1_pct": mat_pct_s1[i, j],
            "numerator_s2": mat_cnt_s2[i, j] if j < 2 else np.nan,
            "denominator_s2": mat_den_s2[i, j] if j < 2 else np.nan,
            "prevalence_s2_pct": mat_pct_s2[i, j] if j < 2 else np.nan,
        })

summary_df = pd.DataFrame(rows)
csv_path = out("outcome_prevalence_summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\n  ✓ Summary CSV saved → {csv_path}")

# ============================================================================
# NULL AUDIT — verify outcome encoding is working correctly
# ============================================================================
print("\n" + "=" * 60)
print("NULL AUDIT — Outcome column value distributions")
print("=" * 60)

audit_cols = {}
audit_cols.update(HTN_OUTCOME_COLS)
audit_cols.update(DYSLIP_OUTCOME_COLS)
for col_map in STANDARD_OUTCOMES.values():
    audit_cols.update(col_map)

audit_rows = []
for label, col in audit_cols.items():
    if col is not None and col in df_full.columns:
        s = df_full[col]
        n1 = (s == 1).sum()
        n0 = (s == 0).sum()
        nn = s.isna().sum()
        total = len(s)
        audit_rows.append({
            "column": col,
            "n=1 (has outcome)": n1,
            "n=0 (assessed, no outcome)": n0,
            "n=null (not assessed)": nn,
            "total": total,
            "assessed (denom)": n1 + n0,
        })

audit_df = pd.DataFrame(audit_rows)
audit_path = out("outcome_null_audit.csv")
audit_df.to_csv(audit_path, index=False)
print(audit_df.to_string(index=False))
print(f"\n  ✓ Null audit CSV saved → {audit_path}")

# ============================================================================
# CONSOLE LEGEND
# ============================================================================
print("\n" + "=" * 60)
print("PLOT ANNOTATION KEY")
print("=" * 60)
print("  Hypertension & Dyslipidemia cells:")
print("    Large bold %    = Set 1 prevalence (full cohort, assessed only)")
print("    count / n       = Set 1 numerator / denominator (non-null)")
print("    Italic line     = Set 2: + A1C-valid at same timepoint")
print()
print("  All other outcome cells:")
print("    Large bold %    = prevalence (positive / assessed)")
print("    count / n       = positive / non-null at that timepoint")
print("    Within A1C-at-diagnosis filtered cohort")
print()
print("  DENOMINATOR PRINCIPLE:")
print("    null = not assessed → excluded from denominator")
print("    0    = assessed, negative → in denominator")
print("    1    = assessed, positive → in denominator")
print("=" * 60)

# ============================================================================
# PLOT — Warm editorial white theme
# ============================================================================
def plot_heatmap(mat_pct_s1, mat_cnt_s1, mat_den_s1,
                 mat_pct_s2, mat_cnt_s2, mat_den_s2,
                 outcome_labels, timepoint_labels,
                 n_full, n_a1c, out_path):

    fig = plt.figure(figsize=(22, 7.5), facecolor="white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[17, 0.4], wspace=0.03)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("white")

    cmap = LinearSegmentedColormap.from_list(
        "amber_red", ["#FFF8F0", "#FDDCB5", "#F7A86E", "#E05A2B", "#8B1A00"], N=256
    )

    masked = np.ma.masked_invalid(mat_pct_s1)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=100)

    n_r, n_c = mat_pct_s1.shape

    # Grid lines
    for x in np.arange(-0.5, n_c, 1):
        ax.axvline(x, color="white", linewidth=3)
    for y in np.arange(-0.5, n_r, 1):
        ax.axhline(y, color="white", linewidth=3)

    # Separator between CV risk factors and glycemic outcomes
    ax.axvline(1.5, color="#333333", linewidth=2.5, alpha=0.4)

    HTN_DYSLIP = {0, 1}

    for i in range(n_r):
        for j in range(n_c):
            pct1 = mat_pct_s1[i, j]
            if np.isnan(pct1):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=12, color="#BBBBBB")
                continue

            cnt1 = int(mat_cnt_s1[i, j]) if not np.isnan(mat_cnt_s1[i, j]) else "?"
            den1 = int(mat_den_s1[i, j]) if not np.isnan(mat_den_s1[i, j]) else "?"
            tc = "white" if pct1 > 55 else "#1A1A1A"

            if j in HTN_DYSLIP:
                pct2 = mat_pct_s2[i, j]
                cnt2 = int(mat_cnt_s2[i, j]) if not np.isnan(mat_cnt_s2[i, j]) else "?"
                den2 = int(mat_den_s2[i, j]) if not np.isnan(mat_den_s2[i, j]) else "?"

                ax.text(j, i - 0.24, f"{pct1:.1f}%",
                        ha="center", va="center", fontsize=15,
                        fontweight="bold", color=tc)
                ax.text(j, i + 0.07, f"{cnt1:,} / {den1:,}",
                        ha="center", va="center", fontsize=9,
                        color=tc, alpha=0.88)
                if not np.isnan(pct2):
                    ax.text(j, i + 0.38,
                            f"A1C-filt: {pct2:.1f}%  ({cnt2:,}/{den2:,})",
                            ha="center", va="center", fontsize=8,
                            color=tc, alpha=0.78, style="italic")
            else:
                ax.text(j, i - 0.12, f"{pct1:.1f}%",
                        ha="center", va="center", fontsize=16,
                        fontweight="bold", color=tc)
                ax.text(j, i + 0.30, f"{cnt1:,} / {den1:,}",
                        ha="center", va="center", fontsize=10,
                        color=tc, alpha=0.85)

    # Category brackets
    spans = [
        ("Cardiovascular Risk Factors", 0, 1),
        ("Glycemic & Treatment Outcomes", 2, n_c - 1),
    ]
    for cat_label, x0, x1 in spans:
        mid = (x0 + x1) / 2
        ax.annotate("", xy=(x1 + 0.45, -0.62), xytext=(x0 - 0.45, -0.62),
                     xycoords=("data", "axes fraction"),
                     arrowprops=dict(arrowstyle="-", color="#888888", lw=1.5))
        ax.text(mid, -0.685, cat_label, ha="center", va="top",
                fontsize=10, color="#555555", style="italic",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(range(n_c))
    ax.set_xticklabels(outcome_labels, fontsize=12, color="#222222",
                       fontweight="600")
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(timepoint_labels, fontsize=13, color="#222222",
                       fontweight="bold")
    ax.tick_params(length=0, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar
    ax_cb = fig.add_subplot(gs[1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label("Prevalence (%)", fontsize=11, color="#444444", labelpad=10)
    cbar.ax.tick_params(labelsize=10, color="#888888")
    cbar.outline.set_edgecolor("#DDDDDD")

    # Title and subtitle
    fig.text(0.02, 0.97,
             "Clinical Outcome Prevalence Across Follow-up Timepoints",
             fontsize=17, fontweight="bold", color="#111111", va="top")
    fig.text(0.02, 0.92,
             f"Full cohort: {n_full:,}  |  A1C-filtered cohort: {n_a1c:,}"
             f"  |  Denominator = assessed patients (non-null outcome)",
             fontsize=10, color="#666666", va="top")

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  ✓ Heatmap saved → {out_path}")

# ============================================================================
# RUN
# ============================================================================
print("\n" + "=" * 60)
print("GENERATING V3 HEATMAP")
print("=" * 60)

plot_heatmap(
    mat_pct_s1, mat_cnt_s1, mat_den_s1,
    mat_pct_s2, mat_cnt_s2, mat_den_s2,
    outcome_labels=ALL_LABELS,
    timepoint_labels=TIMEPOINT_LABELS,
    n_full=n_full,
    n_a1c=n_a1c,
    out_path=out("heatmap_v3_prevalence.png"),
)

print("\n" + "=" * 60)
print("✓ Done. Outputs in analysis/outcomes_heatmap/")
print("  - heatmap_v3_prevalence.png")
print("  - outcome_prevalence_summary.csv")
print("  - outcome_null_audit.csv")
print("=" * 60)