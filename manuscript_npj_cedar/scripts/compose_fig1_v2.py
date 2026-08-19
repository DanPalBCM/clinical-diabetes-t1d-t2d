"""
Figure 1 — LANDSCAPE alternative (v2), for comparison against the portrait v1.

Layout: 2 rows x 3 columns (6 quadrants)
    Row 1:  A  (cols 1-2, wide)   |  B  (col 3)
    Row 2:  D  (cols 1-2, wide)   |  C  (col 3)

Reads the Fig 1 sub-panels from additional_analysis/figure_sources/fig1_panels/
and writes figures/main_figures/fig1_v2.{pdf,svg}. Auto-crops panel whitespace.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(HERE, "..", "additional_analysis", "figure_sources", "fig1_panels")
OUT = os.path.join(HERE, "..", "figures", "main_figures")
os.makedirs(OUT, exist_ok=True)


def _autocrop(im, pad_frac=0.01):
    a = im
    if a.ndim == 3 and a.shape[2] == 4:
        mask = a[:, :, 3] > 0.05
    else:
        rgb = a[:, :, :3] if a.ndim == 3 else a
        mask = rgb.min(axis=2) < 0.97
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return im
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    py = int((y1 - y0) * pad_frac); px = int((x1 - x0) * pad_frac)
    return im[max(0, y0 - py):min(im.shape[0], y1 + py + 1),
              max(0, x0 - px):min(im.shape[1], x1 + px + 1)]


imgs = {k: _autocrop(imread(os.path.join(MAN, f"Fig1{k}.png"))) for k in "ABCD"}


def ar(k):
    h, w = imgs[k].shape[:2]
    return w / h


def show(ax, k):
    ax.imshow(imgs[k]); ax.axis("off")
    ax.text(-0.02, 1.03, k, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="bottom", ha="right")


# 3 equal columns; wide panels (A, D) span cols 1-2, single panels (B, C) in col 3.
FIG_W = 12.0
col = FIG_W / 3.0
w_wide = 2 * col          # A, D
w_side = col              # B, C
# Row height = the taller of its two panels at their widths.
h_row1 = max(w_wide / ar("A"), w_side / ar("B"))
h_row2 = max(w_wide / ar("D"), w_side / ar("C"))
GAP = 0.12
FIG_H = h_row1 + h_row2 + GAP

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = fig.add_gridspec(2, 3, height_ratios=[h_row1, h_row2],
                      hspace=GAP / (FIG_H / 2), wspace=0.05)

axA = fig.add_subplot(gs[0, 0:2]); show(axA, "A")
axB = fig.add_subplot(gs[0, 2]);   show(axB, "B")
axD = fig.add_subplot(gs[1, 0:2]); show(axD, "D")
axC = fig.add_subplot(gs[1, 2]);   show(axC, "C")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
for ext in ("pdf", "svg"):
    p = os.path.join(OUT, f"fig1_v2.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    print("saved ->", p)
plt.close(fig)
