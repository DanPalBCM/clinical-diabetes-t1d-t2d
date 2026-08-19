"""
Shared publication style, palette, and significance helpers for the Pediatric
Diabetes AI manuscript figures (main + supplementary).

Built on PubliPlots (Botas 2025, https://github.com/jorgebotas/publiplots) for a
consistent, Nature-style look:
  - a single curated categorical palette (publiplots 'muted') used everywhere,
  - publiplots' global rcParams (init_rcparams) for typography/spines/sizing,
  - thin, consistent bar widths,
  - a significance-bracket helper (* / ** / ***) since publiplots does not ship one.

Cite: Botas, J. (2025). PubliPlots: Publication-ready plotting for Python.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import publiplots as pp
    _HAVE_PP = True
except Exception:                     # graceful fallback if not installed
    pp = None
    _HAVE_PP = False

try:
    from scipy.stats import mannwhitneyu
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ── Bar geometry (keep bars thin & consistent across every figure) ───────────
BAR_W = 0.62          # single-series bar width fraction
GROUP_W = 0.74        # total width shared by a grouped-bar cluster


# ══════════════════════════════════════════════════════════════════
# Palette — ONE categorical palette for the whole manuscript.
# publiplots 'muted' is a Nature-style, colour-blind-friendly set; we pin
# named roles to fixed hex so a model/paradigm keeps its colour everywhere.
# ══════════════════════════════════════════════════════════════════
def _hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*(int(round(c * 255)) for c in rgb))


if _HAVE_PP:
    _MUTED = [_hex(c) for c in pp.color_palette("muted", n_colors=8)]
else:
    # publiplots 'muted' values, hard-coded fallback
    _MUTED = ["#4878d0", "#ee854a", "#6acc64", "#d65f5f",
              "#956cb4", "#8c613c", "#dc7ec0", "#797979"]

# Semantic role colours (drawn from the one palette, fixed for consistency).
PALETTE = {
    "blue":   _MUTED[0],   # classical ML / single-pass
    "orange": _MUTED[1],   # multi-agent
    "green":  _MUTED[2],   # recall / favourable
    "red":    _MUTED[3],   # CEDAR / deliberative ensemble (headline)
    "purple": _MUTED[4],   # deep learning
    "brown":  _MUTED[5],
    "pink":   _MUTED[6],
    "grey":   _MUTED[7],
}
CEDAR = PALETTE["red"]
INK = "#2b2b2b"

# Algorithm-class palette (used by class-level panels) — from the one palette.
CLASS_COLORS = {
    "classical_ml":          PALETTE["blue"],    # flat-feature models
    "temporal_ml":           PALETTE["purple"],  # sequence / temporal models
    "single_agent_llm":      "#9bb8e0",          # pale blue
    "multi_agent_llm":       PALETTE["orange"],
    "deliberative_ensemble": CEDAR,
    # 'deep_learning' is remapped to 'temporal_ml' at load time (see CLASS_REMAP);
    # kept here as an alias so any stray reference still resolves to a colour.
    "deep_learning":         PALETTE["purple"],
}
# Metric palette (recall / precision / F1 style panels)
C_RECALL, C_PRECISION, C_F1 = PALETTE["green"], PALETTE["red"], PALETTE["blue"]


def apply_style():
    """Apply the global publication style once, at import time of a generator."""
    if _HAVE_PP:
        pp.init_rcparams()
    # a few overrides we rely on regardless of publiplots availability
    plt.rcParams.update({
        "savefig.dpi": 300, "figure.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif",
    })


# ══════════════════════════════════════════════════════════════════
# Significance helpers
# ══════════════════════════════════════════════════════════════════
def stars(p):
    """p-value -> conventional star string ('' if n.s.)."""
    if p is None or np.isnan(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


def mwu_p(a, b):
    """Two-sided Mann-Whitney U p-value; NaN if not computable."""
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if not _HAVE_SCIPY or len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def wilcoxon_p(a, b):
    """Two-sided Wilcoxon signed-rank p for paired arrays a,b (same order/length,
    e.g. per-model AUCs for two modalities). NaN if not computable."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if not _HAVE_SCIPY or len(a) < 2 or np.allclose(a, b):
        return np.nan
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(a, b).pvalue)
    except (ValueError, ImportError):
        return np.nan


def welch_p(m1, s1, n1, m2, s2, n2):
    """Two-sided Welch's t-test p-value from summary stats (means, SDs, n's) —
    for comparing two cross-validated estimates whose per-fold SDs are known
    (e.g. best model's ROC-AUC under two feature sets). NaN if not computable."""
    if not _HAVE_SCIPY:
        return np.nan
    try:
        n1, n2 = float(n1), float(n2)
        se2 = s1 ** 2 / n1 + s2 ** 2 / n2
        if se2 <= 0 or n1 < 2 or n2 < 2:
            return np.nan
        from scipy.stats import t as _t
        tstat = (m1 - m2) / np.sqrt(se2)
        df = se2 ** 2 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) +
                         (s2 ** 2 / n2) ** 2 / (n2 - 1))
        return float(2 * _t.sf(abs(tstat), df))
    except (ValueError, ZeroDivisionError):
        return np.nan


def sig_bracket(ax, x1, x2, y, p, *, tick=None, lw=1.0, color=INK, fs=9):
    """Draw a significance bracket between x1 and x2 at height y, labelled by p.
    `tick` is the vertical drop of the bracket ends (data units; auto if None)."""
    label = stars(p)
    if not label:
        return
    if tick is None:
        lo, hi = ax.get_ylim()
        tick = (hi - lo) * 0.015
    ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick],
            lw=lw, color=color, clip_on=True)
    ax.text((x1 + x2) / 2, y, label, ha="center",
            va="bottom", fontsize=fs, color=color, clip_on=True)


def add_sig_over_reference(ax, values_by_group, ref_key, order, *,
                           top=None, gap_frac=0.06, start_frac=0.02):
    """Given {group: array_of_values}, test ref_key vs each other group
    (Mann-Whitney U) and stack significance brackets above the reference group's
    x position. `order` is the left-to-right x ordering of the groups.

    Returns the p-values dict {other_group: p}.
    """
    if ref_key not in values_by_group:
        return {}
    xref = order.index(ref_key)
    lo, hi = ax.get_ylim()
    span = hi - lo
    y = (top if top is not None else hi) + span * start_frac
    pvals = {}
    others = [g for g in order if g != ref_key]
    for g in others:
        p = mwu_p(values_by_group[ref_key], values_by_group[g])
        pvals[g] = p
        if stars(p):                       # only draw a bracket if significant/marked
            sig_bracket(ax, xref, order.index(g), y, p)
            y += span * gap_frac
    return pvals


# ══════════════════════════════════════════════════════════════════
# Publication bar style — soft fills + darker outlines (+ optional hatch),
# matching the PubliPlots look. One "emphasis" bar can be drawn solid (e.g.
# CEDAR red) so it draws the eye while baselines recede.
# ══════════════════════════════════════════════════════════════════
def _lighten(hexc, amt=0.55):
    """Blend a hex colour toward white by `amt` (0=orig, 1=white)."""
    hexc = hexc.lstrip("#")
    r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amt); g = int(g + (255 - g) * amt); b = int(b + (255 - b) * amt)
    return f"#{r:02x}{g:02x}{b:02x}"


# rotating hatch textures for baseline bars (subtle, publication-safe)
HATCHES = ["", "///", "...", "\\\\\\", "xxx", "---"]


def soft_bars(ax, x, heights, base_colors, *, width=None, emphasis=None,
              horizontal=False, yerr=None, hatch=None, lighten=0.55, lw=1.3):
    """Draw bars with soft fills + saturated outlines.
      x, heights : positions and values
      base_colors: per-bar saturated colour; fill is a lightened version, edge
                   is the saturated colour.
      emphasis   : index (or set of indices) drawn SOLID (full colour) to pop.
      hatch      : optional list of hatch strings per bar (None = no hatch).
    Returns the bar container."""
    width = width if width is not None else BAR_W
    emph = set() if emphasis is None else ({emphasis} if isinstance(emphasis, int) else set(emphasis))
    faces, edges = [], []
    for i, c in enumerate(base_colors):
        if i in emph:
            faces.append(c); edges.append("white")          # solid, white outline
        else:
            faces.append(_lighten(c, lighten)); edges.append(c)  # soft fill, colour outline
    bar_fn = ax.barh if horizontal else ax.bar
    kw = dict(color=faces, edgecolor=edges, linewidth=lw)
    if hatch is not None:
        # hatch colour follows edge; matplotlib uses edgecolor for hatch lines
        kw["hatch"] = hatch
    if horizontal:
        return bar_fn(x, heights, width, xerr=yerr, capsize=2.5,
                      error_kw={"lw": 0.9, "ecolor": INK}, **kw)
    return bar_fn(x, heights, width, yerr=yerr, capsize=2.5,
                  error_kw={"lw": 0.9, "ecolor": INK}, **kw)
