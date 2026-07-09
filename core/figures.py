"""Figure-level outputs that build their whole figure (dendrogram, facets).
They don't go on the single-axes canvas: they are generated and exported. Reuse the engine theme."""
from __future__ import annotations
import pandas as pd
import seaborn as sns

from .plot_engine import THEMES, DEFAULT_THEME


def _theme(style):
    t = THEMES.get(getattr(style, "theme", DEFAULT_THEME), THEMES[DEFAULT_THEME])
    sns.set_theme(style=t["base"], context=getattr(style, "context", "notebook"),
                  font_scale=getattr(style, "font_scale", 1.0))
    return t


def clustermap(df, cols, z_score=True, cmap="vlag", style=None):
    """Heatmap with hierarchical clustering of rows and columns (dendrograms)."""
    t = _theme(style)
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError("Need >=2 rows and >=2 numeric columns.")
    g = sns.clustermap(data, z_score=1 if z_score else None, cmap=cmap,
                       figsize=(min(2 + 0.5 * len(cols), 14), 8), xticklabels=True)
    g.figure.patch.set_facecolor(t["fig"])
    return g.figure


def pairplot(df, cols, hue=None, style=None):
    """Pairwise scatter matrix (with distributions on the diagonal)."""
    t = _theme(style)
    use = df[cols + ([hue] if hue else [])].copy()
    for c in cols:
        use[c] = pd.to_numeric(use[c], errors="coerce")
    use = use.dropna()
    if use.shape[0] < 2:
        raise ValueError("Too few valid rows for the pairplot.")
    g = sns.pairplot(use, vars=cols, hue=hue, corner=False, diag_kind="kde",
                     plot_kws=dict(s=18, alpha=0.6))
    g.figure.patch.set_facecolor(t["fig"])
    return g.figure
