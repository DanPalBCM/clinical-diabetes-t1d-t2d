"""
Medication Distribution Inspector
===================================
1. Grouped bar chart: positive counts per medication × timepoint
2. UpSet plots: medication combination patterns (one per timepoint)

Run AFTER preprocessing Cell 1. Self-contained.

Output: analysis/medication_distributions/
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations

OUTPUT_DIR = "analysis/medication_distributions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# COLUMN NAME HELPERS
# ============================================================================
_TP_MED = {"diagnosis": "_diagnosis", "2yr": "_2yr", "5yr": "_5yr"}

def med_col(base, tp):
    return f"{base}{_TP_MED[tp]}"

# ============================================================================
# MEDICATION LIST
# ============================================================================
MEDICATIONS = [
    ("Insulin",                "Insulins"),
    ("Metformin (Biguanide)",  "Biguanide"),
    ("GLP-1 Agonists",        "GLP1_agonists"),
]

TP_MAP = {"At Diagnosis": "diagnosis", "At 2 Years": "2yr", "At 5 Years": "5yr"}
TP_LABELS = list(TP_MAP.keys())

# ============================================================================
# COLLECT COUNTS
# ============================================================================
total_n = len(df_full)
rows = []

for display_name, base_name in MEDICATIONS:
    for tp_label, tp_key in TP_MAP.items():
        col_name = med_col(base_name, tp_key)
        if col_name in df_full.columns:
            s = pd.to_numeric(df_full[col_name], errors="coerce")
            n_pos = int((s == 1).sum())
            n_neg = int((s == 0).sum())
            n_missing = int(s.isna().sum())
        else:
            n_pos, n_neg, n_missing = 0, 0, total_n
            col_name = f"{col_name} (NOT FOUND)"

        rows.append({
            "Medication": display_name,
            "Timepoint": tp_label,
            "Column": col_name,
            "Positive (n=1)": n_pos,
            "Negative (n=0)": n_neg,
            "Missing (null)": n_missing,
            "Total": total_n,
            "Prevalence (%)": n_pos / total_n * 100 if total_n > 0 else 0,
        })

summary = pd.DataFrame(rows)

# ============================================================================
# PRINT SUMMARY
# ============================================================================
print("=" * 90)
print("MEDICATION DISTRIBUTION SUMMARY")
print("=" * 90)

for med_name in summary["Medication"].unique():
    sub = summary[summary["Medication"] == med_name]
    total_pos = sub["Positive (n=1)"].sum()
    print(f"\n  {med_name}  (total positive across timepoints: {total_pos})")
    for _, r in sub.iterrows():
        print(f"    {r['Timepoint']:15s}: "
              f"pos={r['Positive (n=1)']:>5,}  "
              f"neg={r['Negative (n=0)']:>5,}  "
              f"null={r['Missing (null)']:>5,}  "
              f"prev={r['Prevalence (%)']:.1f}%")

csv_path = os.path.join(OUTPUT_DIR, "medication_distribution_summary.csv")
summary.to_csv(csv_path, index=False)
print(f"\n  ✓ Summary CSV → {csv_path}")

# ============================================================================
# PLOT 1: Grouped bar chart — positive counts per medication × timepoint
# ============================================================================
med_names = [m[0] for m in MEDICATIONS]
n_meds = len(med_names)
n_tp = len(TP_LABELS)

fig, ax = plt.subplots(figsize=(max(14, n_meds * 1.5), 7), facecolor="white")
ax.set_facecolor("white")

x = np.arange(n_meds)
width = 0.25
colors = ["#2D6A4F", "#52B788", "#B7E4C7"]

for k, tp in enumerate(TP_LABELS):
    vals = [
        summary[(summary["Medication"] == m) & (summary["Timepoint"] == tp)]["Positive (n=1)"].values[0]
        for m in med_names
    ]
    bars = ax.bar(x + k * width, vals, width, label=tp, color=colors[k],
                  edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total_n * 0.005,
                    str(v), ha="center", va="bottom", fontsize=8, color="#333333")

ax.set_xticks(x + width)
ax.set_xticklabels(med_names, rotation=35, ha="right", fontsize=10)
ax.set_ylabel("Positive Patients (n=1)", fontsize=12)
ax.set_title("Medication Use by Timepoint — Positive Counts",
             fontsize=14, fontweight="bold", pad=15)
ax.legend(fontsize=10, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "medication_positive_counts.png")
plt.savefig(path1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  ✓ Positive counts plot → {path1}")

# ============================================================================
# UPSET PLOTS — one per timepoint
# ============================================================================

def build_binary_matrix(df, medications, tp_key):
    """Build a binary matrix of medication use for a given timepoint.
    Only includes patients on at least one medication."""
    mat = pd.DataFrame(index=df.index)
    for display_name, base_name in medications:
        col = med_col(base_name, tp_key)
        if col in df.columns:
            mat[display_name] = (pd.to_numeric(df[col], errors="coerce") == 1).astype(int)
        else:
            mat[display_name] = 0
    # Keep only patients on at least one medication
    mat = mat[mat.sum(axis=1) > 0]
    return mat


def plot_upset(binary_matrix, tp_label, out_path, top_n=25):
    """
    Hand-rolled UpSet plot:
      - Bottom: dot matrix showing which medications are in each combination
      - Top: bar chart showing count of patients with that exact combination
      - Right: horizontal bars showing total patients per medication
    """
    if binary_matrix.empty:
        print(f"    ⚠ No patients on any medication at {tp_label} — skipping")
        return

    med_labels = binary_matrix.columns.tolist()
    n_meds_local = len(med_labels)

    # Count each unique combination
    combo_counts = (
        binary_matrix.groupby(med_labels)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    n_combos = len(combo_counts)

    if n_combos == 0:
        print(f"    ⚠ No combinations found at {tp_label} — skipping")
        return

    # Per-medication totals
    med_totals = binary_matrix.sum().sort_values(ascending=True)
    med_order = med_totals.index.tolist()

    # Reorder columns to match med_order (least → most frequent, bottom → top in dot matrix)
    combo_matrix = combo_counts[med_order].values  # shape: (n_combos, n_meds)
    combo_sizes = combo_counts["count"].values

    # ---- Layout ----
    fig = plt.figure(figsize=(max(12, n_combos * 0.55), max(7, n_meds_local * 0.4 + 4)),
                     facecolor="white")

    # GridSpec: top = bar chart, bottom = dot matrix, right = med totals
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[2.5, n_meds_local * 0.35],
        width_ratios=[n_combos, max(4, n_combos * 0.25)],
        hspace=0.05, wspace=0.15
    )

    ax_bars = fig.add_subplot(gs[0, 0])   # top-left: intersection sizes
    ax_dots = fig.add_subplot(gs[1, 0])   # bottom-left: dot matrix
    ax_totals = fig.add_subplot(gs[1, 1]) # bottom-right: per-med totals
    ax_empty = fig.add_subplot(gs[0, 1])  # top-right: empty
    ax_empty.axis("off")

    x = np.arange(n_combos)

    # ---- Top: Intersection size bars ----
    bar_color = "#E05A2B"
    ax_bars.bar(x, combo_sizes, color=bar_color, edgecolor="white", linewidth=0.5, width=0.7)
    for i, v in enumerate(combo_sizes):
        ax_bars.text(i, v + max(combo_sizes) * 0.02, str(v),
                     ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333")
    ax_bars.set_ylabel("Patients", fontsize=11)
    ax_bars.set_xlim(-0.5, n_combos - 0.5)
    ax_bars.set_xticks([])
    ax_bars.spines["top"].set_visible(False)
    ax_bars.spines["right"].set_visible(False)
    ax_bars.spines["bottom"].set_visible(False)
    ax_bars.grid(axis="y", alpha=0.2)
    ax_bars.set_title(f"Medication Combinations — {tp_label}",
                      fontsize=13, fontweight="bold", pad=10)

    # ---- Bottom-left: Dot matrix ----
    for i in range(n_combos):
        active_rows = [j for j in range(n_meds_local) if combo_matrix[i, j] == 1]
        inactive_rows = [j for j in range(n_meds_local) if combo_matrix[i, j] == 0]

        # Inactive dots (light gray)
        ax_dots.scatter([i] * len(inactive_rows), inactive_rows,
                        s=60, color="#DDDDDD", zorder=2)
        # Active dots (dark)
        ax_dots.scatter([i] * len(active_rows), active_rows,
                        s=80, color="#333333", zorder=3)
        # Connect active dots with a line
        if len(active_rows) > 1:
            ax_dots.plot([i, i], [min(active_rows), max(active_rows)],
                         color="#333333", linewidth=2, zorder=1)

    ax_dots.set_xlim(-0.5, n_combos - 0.5)
    ax_dots.set_ylim(-0.5, n_meds_local - 0.5)
    ax_dots.set_yticks(range(n_meds_local))
    ax_dots.set_yticklabels(med_order, fontsize=9)
    ax_dots.set_xticks([])
    ax_dots.spines["top"].set_visible(False)
    ax_dots.spines["right"].set_visible(False)
    ax_dots.spines["bottom"].set_visible(False)
    ax_dots.grid(axis="y", alpha=0.1)

    # Alternating row background
    for j in range(n_meds_local):
        if j % 2 == 0:
            ax_dots.axhspan(j - 0.5, j + 0.5, color="#F5F5F5", zorder=0)

    # ---- Bottom-right: Per-medication totals ----
    med_total_vals = [med_totals[m] for m in med_order]
    y = np.arange(n_meds_local)
    ax_totals.barh(y, med_total_vals, color="#52B788", edgecolor="white",
                   linewidth=0.5, height=0.7)
    for i, v in enumerate(med_total_vals):
        if v > 0:
            ax_totals.text(v + max(med_total_vals) * 0.02, i, str(v),
                           ha="left", va="center", fontsize=8, color="#333")
    ax_totals.set_ylim(-0.5, n_meds_local - 0.5)
    ax_totals.set_yticks([])
    ax_totals.set_xlabel("Total patients", fontsize=9)
    ax_totals.spines["top"].set_visible(False)
    ax_totals.spines["right"].set_visible(False)

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ UpSet plot → {out_path}")


# ---- Generate one UpSet plot per timepoint ----
print("\n" + "=" * 60)
print("GENERATING UPSET PLOTS")
print("=" * 60)

for tp_label, tp_key in TP_MAP.items():
    print(f"\n  [{tp_label}]")
    mat = build_binary_matrix(df_full, MEDICATIONS, tp_key)
    print(f"    Patients on ≥1 medication: {len(mat):,}")

    safe_tp = tp_key.replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"medication_upset_{safe_tp}.png")
    plot_upset(mat, tp_label, path, top_n=25)

# ============================================================================
# COMBINATION SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 60)
print("TOP MEDICATION COMBINATIONS PER TIMEPOINT")
print("=" * 60)

combo_rows = []
for tp_label, tp_key in TP_MAP.items():
    mat = build_binary_matrix(df_full, MEDICATIONS, tp_key)
    if mat.empty:
        continue

    med_labels = mat.columns.tolist()
    combos = (
        mat.groupby(med_labels)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print(f"\n  {tp_label} (top 10):")
    for rank, (_, row) in enumerate(combos.head(10).iterrows(), 1):
        active = [m for m in med_labels if row[m] == 1]
        combo_str = " + ".join(active)
        n = int(row["count"])
        pct = n / total_n * 100
        print(f"    {rank:2d}. {combo_str:50s}  n={n:>4,}  ({pct:.1f}%)")
        combo_rows.append({
            "Timepoint": tp_label,
            "Rank": rank,
            "Combination": combo_str,
            "N_medications": len(active),
            "Count": n,
            "Prevalence (%)": round(pct, 2),
        })

combo_df = pd.DataFrame(combo_rows)
combo_path = os.path.join(OUTPUT_DIR, "medication_combinations_top10.csv")
combo_df.to_csv(combo_path, index=False)
print(f"\n  ✓ Combinations CSV → {combo_path}")

# ============================================================================
print(f"\n{'=' * 60}")
print(f"✓ Done. Outputs in {OUTPUT_DIR}/")
print(f"  - medication_distribution_summary.csv")
print(f"  - medication_positive_counts.png")
print(f"  - medication_upset_diagnosis.png")
print(f"  - medication_upset_2yr.png")
print(f"  - medication_upset_5yr.png")
print(f"  - medication_combinations_top10.csv")
print(f"{'=' * 60}")