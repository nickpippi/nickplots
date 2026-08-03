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


def overview(df, cols, group=None, style=None, ncols=4):
    """Small multiples of every selected numeric column: a histogram when there is no
    group, a box + points per group when there is. One glance at the whole table."""
    import numpy as np
    from matplotlib.figure import Figure
    t = _theme(style)
    cols = [c for c in (cols or []) if c in df.columns]
    if not cols:
        raise ValueError("Select at least one numeric column.")
    n = len(cols)
    ncols = max(1, min(int(ncols), n))
    nrows = -(-n // ncols)
    fig = Figure(figsize=(3.1 * ncols, 2.5 * nrows), dpi=110)
    for i, c in enumerate(cols):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        v = pd.to_numeric(df[c], errors="coerce")
        if group and group in df.columns:
            d = pd.DataFrame({c: v, group: df[group].astype(str)}).dropna()
            if not d.empty:
                sns.boxplot(data=d, x=group, y=c, hue=group, legend=False,
                            showfliers=False, ax=ax)
                sns.stripplot(data=d, x=group, y=c, color="#1f1f1f", size=2.5,
                              alpha=0.45, jitter=0.25, ax=ax)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
        else:
            vv = v.dropna()
            if len(vv):
                ax.hist(vv, bins=min(30, max(5, int(np.sqrt(len(vv))))),
                        color="#3a6ea5", alpha=0.85)
        miss = int(v.isna().sum())
        ax.set_title(f"{c}" + (f"  ({miss} NA)" if miss else ""), fontsize=8.5, loc="left")
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(labelsize=7)
        ax.set_facecolor(t["ax"])
    fig.patch.set_facecolor(t["fig"])
    fig.tight_layout()
    return fig
