"""Declarative plot registry.

Each plot is a PlotSpec that declares its channels (x/y/hue/size/style), the column
types each channel accepts, its parameters and whether it is 'axes' (stackable) or
'figure' (standalone). The UI reads this schema and builds its panel automatically.
Adding a plot = adding a PlotSpec; no UI or engine changes needed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd
import seaborn as sns

from .data_loader import NUMBER, CATEGORY, DATETIME

ANY = (NUMBER, CATEGORY, DATETIME)


@dataclass
class Channel:
    name: str
    required: bool = False
    accepts: tuple = ANY
    label: str = ""

    def nice(self) -> str:
        return self.label or self.name.capitalize()


@dataclass
class Param:
    name: str
    kind: str                 # "float" | "int" | "bool" | "choice" | "palette"
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple = ()
    label: str = ""

    def nice(self) -> str:
        return self.label or self.name


@dataclass
class PlotSpec:
    key: str
    label: str
    level: str                # "axes" (stackable) | "figure" (standalone)
    render: Callable
    channels: list = field(default_factory=list)
    params: list = field(default_factory=list)
    pickable: bool = False    # True -> clicking a point shows its table row

    @property
    def layerable(self) -> bool:
        return self.level == "axes"


# ---- helper: resolve hue/palette avoiding the seaborn 0.13 FutureWarning ---- #
def _resolve_palette(df, level_col, p):
    """Named palette, or a {level: color} dict when the user fixed per-group colors in
    the color picker (__colors__), matching each level by its text and falling back to
    the palette for the unchosen ones. This makes the color picker apply to ANY plot
    with hue/category."""
    named = p.get("palette")
    cm = p.get("__colors__")
    if not cm or not level_col or level_col not in df.columns:
        return named
    levels = list(pd.unique(df[level_col].dropna()))
    base = sns.color_palette(named or "viridis", max(len(levels), 1))
    return {lv: cm.get(str(lv), base[i]) for i, lv in enumerate(levels)}


def _hue_kw(df, m, p, x_for_fallback=None):
    """When there's no explicit hue but a palette exists, assign hue=x with
    legend=False (seaborn's own recommendation) instead of passing palette without hue."""
    hue = m.get("hue")
    if hue:
        return dict(hue=hue, palette=_resolve_palette(df, hue, p))
    if x_for_fallback:
        return dict(hue=x_for_fallback, palette=_resolve_palette(df, x_for_fallback, p), legend=False)
    return {}


# ------------------------------- renderers ---------------------------------- #
def _r_scatter(ax, df, m, p):
    # colour-blind safety: mirror the hue on the marker shape too, so colour is not the
    # only encoding (unless an explicit style column is set).
    style_col = m.get("style") or (m.get("hue") if p.get("shape_hue") else None)
    sns.scatterplot(data=df, x=m["x"], y=m["y"], hue=m.get("hue"), size=m.get("size"),
                    style=style_col, alpha=p["alpha"],
                    palette=_resolve_palette(df, m["hue"], p) if m.get("hue") else None, ax=ax)
    if p.get("regression") and m.get("x") and m.get("y"):
        _overlay_regression(ax, df, m["x"], m["y"])


def _r_line(ax, df, m, p):
    # seaborn's default CI band bootstraps 1000x per group -> minutes on many-obs-per-x
    # data. Default to no band (instant); the band is opt-in via the "error band" param.
    eb = {"none": None, "SD": "sd", "SEM": "se", "CI95": ("ci", 95)}[p.get("errorbar", "none")]
    sns.lineplot(data=df, x=m["x"], y=m["y"], hue=m.get("hue"),
                 linewidth=p["linewidth"], alpha=p["alpha"], errorbar=eb,
                 palette=_resolve_palette(df, m["hue"], p) if m.get("hue") else None, ax=ax)


def _r_bar(ax, df, m, p):
    sns.barplot(data=df, x=m["x"], y=m["y"], alpha=p["alpha"], ax=ax, **_hue_kw(df, m, p, m["x"]))


def _sig_brackets(ax, df, m, p):
    """Significance brackets between the x groups (only when there is no hue).
    Markers pushed from Analysis > Compare groups arrive in __sig_pairs__; without
    them the plot falls back to computing its own (Welch + Holm)."""
    if p.get("sig") and m.get("x") and m.get("y") and not m.get("hue"):
        _draw_sig(ax, df, m["x"], m["y"], list(pd.unique(df[m["x"]].dropna())),
                  pairs=p.get("__sig_pairs__"))


def _r_box(ax, df, m, p):
    sns.boxplot(data=df, x=m["x"], y=m["y"], ax=ax, **_hue_kw(df, m, p, m["x"]))
    _sig_brackets(ax, df, m, p)


def _r_violin(ax, df, m, p):
    sns.violinplot(data=df, x=m["x"], y=m["y"], ax=ax, **_hue_kw(df, m, p, m["x"]))
    _sig_brackets(ax, df, m, p)


def _group_colors(names, p):
    base = sns.color_palette(p.get("palette", "viridis"), max(len(names), 1))
    cm = p.get("__colors__") or {}
    return [cm.get(str(n), base[i]) for i, n in enumerate(names)]


def _draw_sig(ax, df, xcol, ycol, cats, pairs=None):
    """Draw significance brackets between groups.

    `pairs` = markers pushed from Analysis > Compare groups (Mann-Whitney + BH), so
    the figure and the written report come from the same test. Without them the plot
    computes its own (Welch + Holm) - convenient, but a different test.
    """
    from .stats import pairwise_sig
    pos = {str(c): i for i, c in enumerate(cats)}
    if pairs:
        sig = [(str(d["g1"]), str(d["g2"]), d.get("q", 1.0), d.get("stars", "*"))
               for d in pairs if d.get("sig")]
    else:
        sig = pairwise_sig(df, ycol, xcol)
    if not sig:
        return
    ymax = pd.to_numeric(df[ycol], errors="coerce").max()
    yr = ymax - pd.to_numeric(df[ycol], errors="coerce").min() or 1
    step = yr * 0.08
    lvl = 0
    for g1, g2, _, stars in sig:
        if g1 not in pos or g2 not in pos:
            continue
        x1, x2 = pos[g1], pos[g2]
        h = ymax + step * (lvl + 1)
        ax.plot([x1, x1, x2, x2], [h - step * 0.25, h, h, h - step * 0.25], lw=1, c="#555")
        ax.text((x1 + x2) / 2, h, stars, ha="center", va="bottom", fontsize=11)
        lvl += 1


def _r_dose4pl(ax, df, m, p):
    """4PL dose-response curve with annotated IC50 (one curve per hue group)."""
    from .stats import fit_4pl
    x, y, hue = m["x"], m["y"], m.get("hue")
    groups = list(df.groupby(hue)) if hue else [(None, df)]
    names = [g for g, _ in groups]
    colors = _group_colors(names, p) if hue else ["#3a6ea5"]
    txt = []
    for (name, sub), col in zip(groups, colors):
        xv = pd.to_numeric(sub[x], errors="coerce")
        yv = pd.to_numeric(sub[y], errors="coerce")
        ax.scatter(xv, yv, s=20, alpha=p["alpha"], color=col,
                   label=str(name) if hue else None)
        try:
            fit = fit_4pl(xv, yv)
            xp = xv[xv > 0]
            xx = np.logspace(np.log10(max(xp.min(), 1e-6)), np.log10(xv.max()), 200)
            ax.plot(xx, fit["predict"](xx), color=col, lw=1.8)
            ax.axvline(fit["ic50"], ls=":", color=col, lw=1)
            txt.append(f"{(str(name)+': ') if hue else ''}IC50={fit['ic50']:.3g}  R²={fit['r2']:.3f}")
        except Exception as e:
            txt.append(f"{(str(name)+': ') if hue else ''}no fit ({e})")
    if p.get("log_x", True):
        ax.set_xscale("log")
    ax.text(0.02, 0.02, "\n".join(txt), transform=ax.transAxes, va="bottom", fontsize=8,
            bbox=dict(fc="white", alpha=0.7, ec="none"))


def _r_roc(ax, df, m, p):
    """ROC curve + AUC (with bootstrap CI95) for a score against a binary outcome.
    Overlay a second layer with another score column to compare two markers."""
    from .stats import roc_analysis
    r = roc_analysis(df, m["score"], m["label"], p.get("positive") or None,
                     n_boot=300 if p.get("ci", True) else 0)
    col = (p.get("__colors__") or {}).get(m["score"]) or sns.color_palette(
        p.get("palette", "viridis"), 4)[0]
    lab = f"{m['score']}  AUC={r['auc']:.3f}"
    if p.get("ci", True) and r["ci"][0] == r["ci"][0]:
        lab += f" [{r['ci'][0]:.2f}-{r['ci'][1]:.2f}]"
    ax.plot(r["fpr"], r["tpr"], lw=1.8, color=col, label=lab)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#999")     # chance line
    if p.get("mark_cutoff", True):
        ax.plot(1 - r["spec"], r["sens"], "o", ms=6, mfc="none", mew=1.6, color=col)
        ax.annotate(f"{r['cutoff']:.3g}", (1 - r["spec"], r["sens"]),
                    textcoords="offset points", xytext=(7, -9), fontsize=8, color=col)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=8, frameon=False)


def _r_bland(ax, df, m, p):
    """Bland-Altman: difference vs mean of two methods, with bias and 95% limits."""
    from .stats import bland_altman
    r = bland_altman(df, m["m1"], m["m2"])
    hue = m.get("hue")
    if hue:
        d = df[[m["m1"], m["m2"], hue]].apply(
            lambda c: pd.to_numeric(c, errors="coerce") if c.name != hue else c).dropna()
        names = list(pd.unique(d[hue].astype(str)))
        for name, col in zip(names, _group_colors(names, p)):
            sub = d[d[hue].astype(str) == name]
            a, b = sub[m["m1"]].to_numpy(float), sub[m["m2"]].to_numpy(float)
            ax.scatter((a + b) / 2, a - b, s=p["psize"] * 6, alpha=p["alpha"],
                       color=col, label=str(name))
        ax.legend(title=hue, fontsize=8)
    else:
        ax.scatter(r["avg"], r["diff"], s=p["psize"] * 6, alpha=p["alpha"],
                   color="#3a6ea5")
    for y, ls, txt in ((r["bias"], "-", "bias"), (r["lo"], "--", "-1.96 SD"),
                       (r["hi"], "--", "+1.96 SD")):
        ax.axhline(y, ls=ls, lw=1.2, color="#c2410c" if ls == "-" else "#777")
        ax.annotate(f"{txt}  {y:.3g}", (1.0, y), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(-4, 3), ha="right", fontsize=7,
                    color="#c2410c" if ls == "-" else "#777")
    ax.set_xlabel(f"Mean of {m['m1']} and {m['m2']}")
    ax.set_ylabel(f"Difference ({m['m1']} - {m['m2']})")


def _r_bar_err(ax, df, m, p):
    """Bar or point with mean/median ± error + individual points (+ significance)."""
    x, y, hue = m["x"], m["y"], m.get("hue")
    est = np.mean if p["center"] == "mean" else np.median
    ebar = {"SD": "sd", "SEM": "se", "CI95": ("ci", 95)}[p["error"]]
    pal = p.get("__colors__") or p["palette"]
    common = dict(data=df, x=x, y=y, ax=ax)
    if p["kind"] == "bar":
        sns.barplot(hue=hue or x, estimator=est, errorbar=ebar, palette=pal,
                    legend=bool(hue), alpha=0.9, **common)
    else:
        sns.pointplot(hue=hue or x, estimator=est, errorbar=ebar, palette=pal,
                      linestyle="none", legend=bool(hue), **common)
    if p["points"]:
        sns.stripplot(data=df, x=x, y=y, hue=hue, dodge=bool(hue) and hue != x, jitter=0.18,
                      size=p["psize"], alpha=0.55, color="#1f1f1f",
                      edgecolor="white", linewidth=0.3, legend=False, ax=ax)
    _sig_brackets(ax, df, m, p)


def _annot_kws(p, auto=None):
    """Font size of the numbers written inside heatmap cells.
    0 (the default) keeps each heatmap's previous behaviour, so old projects and
    templates render identically."""
    sz = float(p.get("annot_size") or 0)
    if sz > 0:
        return {"fontsize": sz}
    return {"fontsize": auto} if auto else {}


def _r_heatmap_xy(ax, df, m, p):
    """Heatmap of a value across two categorical axes (model x task, gene x sample...).

    Pivots a tidy table: one row per (row, col) pair. Duplicated pairs are collapsed
    with the chosen aggregation instead of silently keeping the last one.
    """
    row, col, val = m["row"], m["col"], m["value"]
    d = df[[row, col, val]].copy()
    d[val] = pd.to_numeric(d[val], errors="coerce")
    d = d.dropna(subset=[val])
    if d.empty:
        raise ValueError(f"No numeric value left in '{val}' for these rows.")
    agg = p.get("agg", "mean")
    mat = d.pivot_table(index=d[row].astype(str), columns=d[col].astype(str),
                        values=val, aggfunc=agg)
    rl = p.get("__relabel__") or {}
    al = p.get("__aliases__") or {}
    mat.index = [rl.get(i, i) for i in mat.index]
    mat.columns = [rl.get(c, c) for c in mat.columns]
    dec = int(p.get("decimals", 1))
    sns.heatmap(mat, cmap=p.get("palette", "viridis"), annot=bool(p.get("annot", True)),
                fmt=f".{dec}f", annot_kws=_annot_kws(p, 8 if mat.shape[1] <= 12 else 6),
                cbar=True, linewidths=float(p.get("gap", 0.0)), ax=ax)
    ax.set_xlabel(al.get(col, col)); ax.set_ylabel(al.get(row, row))
    ax.tick_params(axis="y", rotation=0)


def _r_heatmap_matrix(ax, df, m, p):
    num = df.select_dtypes("number")
    data = (num - num.mean()) / num.std() if p["zscore"] else num
    lab = m.get("labels")
    if lab and lab in df.columns:          # name the rows instead of 0,1,2...
        rl = p.get("__relabel__") or {}
        data = data.set_axis([rl.get(str(v), str(v)) for v in df[lab]], axis=0)
    sns.heatmap(data, cmap=p["palette"], annot=p["annot"], annot_kws=_annot_kws(p),
                cbar=True, ax=ax)
    if lab:
        ax.tick_params(axis="y", rotation=0)


def _r_ecdf(ax, df, m, p):
    hue = m.get("hue")
    kw = dict(palette=_resolve_palette(df, hue, p)) if hue else {}
    sns.ecdfplot(data=df, x=m["x"], hue=hue, ax=ax, **kw)


def _r_regband(ax, df, m, p):
    x, y, hue = m["x"], m["y"], m.get("hue")
    if hue:
        groups = list(df.groupby(hue))
        for (name, sub), col in zip(groups, _group_colors([g for g, _ in groups], p)):
            sns.regplot(data=sub, x=x, y=y, ax=ax, ci=int(p["ci"]), color=col,
                        scatter_kws=dict(s=16, alpha=p["alpha"]), line_kws=dict(color=col))
    else:
        sns.regplot(data=df, x=x, y=y, ax=ax, ci=int(p["ci"]),
                    scatter_kws=dict(s=16, alpha=p["alpha"]))


def _r_paired(ax, df, m, p):
    """Before-after: connect the same sample (id) across the categories of x."""
    x, y, sid = m["x"], m["y"], m.get("id")
    cats = list(pd.unique(df[x].dropna()))
    pos = {c: i for i, c in enumerate(cats)}
    for _, sub in df.groupby(sid):
        sub = sub.dropna(subset=[y])
        xs = np.array([pos[c] for c in sub[x] if c in pos], float)
        ys = pd.to_numeric(sub[y], errors="coerce").values[:len(xs)]
        o = np.argsort(xs)
        ax.plot(xs[o], ys[o], color="#6a7385", alpha=p["alpha"], lw=0.8, marker="o", ms=4)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats)


def _r_scatter_density(ax, df, m, p):
    """Scatter with a density background: each region is tinted by the color of the
    group that dominates there. Requires hue."""
    from scipy.stats import gaussian_kde
    import matplotlib.colors as mcolors
    x, y, hue = m["x"], m["y"], m.get("hue")
    size_col, style_col = m.get("size"), m.get("style")

    # Unique columns to avoid duplicate errors when slicing the DataFrame
    cols = list(dict.fromkeys([c for c in [x, y, hue, size_col, style_col] if c]))
    
    sub = df[cols].copy()
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub = sub.dropna(subset=[x, y, hue])
    if sub.empty:
        return
    
    groups = list(sub.groupby(hue))
    names = [g for g, _ in groups]
    colors = _group_colors(names, p)
    palette = {n: colors[i] for i, n in enumerate(names)}

    if p.get("contour"):
        # "waves" mode: per-group KDE contours, unfilled -> group overlap stays
        # visible (imshow tints each pixel by the dominant group, hiding overlap;
        # here the curves cross each other).
        try:
            sns.kdeplot(data=sub, x=x, y=y, hue=hue, hue_order=names,
                        levels=int(p.get("levels", 10)), palette=palette,
                        linewidths=1.0, alpha=0.9, common_norm=False,
                        legend=False, ax=ax, zorder=0)
        except Exception:
            pass
    else:
        xv, yv = sub[x].to_numpy(float), sub[y].to_numpy(float)
        padx = (xv.max() - xv.min()) * 0.05 or 1.0
        pady = (yv.max() - yv.min()) * 0.05 or 1.0
        x0, x1 = xv.min() - padx, xv.max() + padx
        y0, y1 = yv.min() - pady, yv.max() + pady

        res = int(p.get("grid", 120))
        gx = np.linspace(x0, x1, res); gy = np.linspace(y0, y1, res)
        GX, GY = np.meshgrid(gx, gy)
        pos = np.vstack([GX.ravel(), GY.ravel()])
        dens = []

        for _, g in groups:
            gx2 = g[x].to_numpy(float); gy2 = g[y].to_numpy(float)
            if len(gx2) < 3 or np.ptp(gx2) == 0 or np.ptp(gy2) == 0:
                dens.append(np.zeros((res, res)))
                continue
            try:
                k = gaussian_kde(np.vstack([gx2, gy2]))
                dens.append(k(pos).reshape(res, res) * len(gx2))
            except Exception:
                dens.append(np.zeros((res, res)))

        dens = np.array(dens)
        dom = np.argmax(dens, axis=0)
        mx = dens.max(axis=0)
        norm = mx / mx.max() if mx.max() > 0 else mx
        a = p.get("alpha", 0.35)
        rgba = np.zeros((res, res, 4))

        for i, c in enumerate(colors):
            r, gg, b = mcolors.to_rgb(c)
            mask = dom == i
            rgba[mask, 0], rgba[mask, 1], rgba[mask, 2] = r, gg, b
            rgba[mask, 3] = norm[mask] * a

        ax.imshow(rgba, extent=[x0, x1, y0, y1], origin="lower", aspect="auto",
                  interpolation="bilinear", zorder=0)

    # Build the sns.scatterplot kwargs dynamically (points on top)
    scatter_kws = dict(
        data=sub, x=x, y=y, hue=hue,
        palette=palette,
        alpha=0.9, ax=ax, zorder=2, edgecolor="white", linewidth=0.3
    )
    
    if size_col:
        scatter_kws["size"] = size_col
    else:
        scatter_kws["s"] = p.get("psize", 22)
        
    if style_col:
        scatter_kws["style"] = style_col

    sns.scatterplot(**scatter_kws)


def _r_strip(ax, df, m, p):
    sns.stripplot(data=df, x=m["x"], y=m["y"], alpha=p["alpha"], ax=ax, **_hue_kw(df, m, p, m["x"]))
    _sig_brackets(ax, df, m, p)


def _r_box_points(ax, df, m, p):
    """Boxplot + individual points aligned to each group (jitter)."""
    x, y, hue = m["x"], m["y"], m.get("hue")
    if hue:
        sns.boxplot(data=df, x=x, y=y, hue=hue, palette=p["palette"], showfliers=False, ax=ax)
        # dodge only when hue subdivides x; hue==x (color-only) must stay centered on the box
        sns.stripplot(data=df, x=x, y=y, hue=hue, palette=p["palette"], dodge=hue != x,
                      jitter=p["jitter"], size=p["size"], alpha=p["alpha"],
                      edgecolor="white", linewidth=0.4, legend=False, ax=ax)
    else:
        sns.boxplot(data=df, x=x, y=y, hue=x, palette=p["palette"], legend=False,
                    showfliers=False, ax=ax)
        sns.stripplot(data=df, x=x, y=y, jitter=p["jitter"], size=p["size"],
                      alpha=p["alpha"], color="#1f1f1f", edgecolor="white", linewidth=0.4, ax=ax)
    _sig_brackets(ax, df, m, p)


def _r_violin_points(ax, df, m, p):
    """Violin + individual points per group."""
    x, y, hue = m["x"], m["y"], m.get("hue")
    if hue:
        sns.violinplot(data=df, x=x, y=y, hue=hue, palette=p["palette"], inner=None, ax=ax)
        # dodge only when hue subdivides x; hue==x (color-only) must stay centered
        sns.stripplot(data=df, x=x, y=y, hue=hue, palette=p["palette"], dodge=hue != x,
                      jitter=p["jitter"], size=p["size"], alpha=p["alpha"],
                      edgecolor="white", linewidth=0.4, legend=False, ax=ax)
    else:
        sns.violinplot(data=df, x=x, y=y, hue=x, palette=p["palette"], legend=False, inner=None, ax=ax)
        sns.stripplot(data=df, x=x, y=y, jitter=p["jitter"], size=p["size"],
                      alpha=p["alpha"], color="#1f1f1f", edgecolor="white", linewidth=0.4, ax=ax)
    _sig_brackets(ax, df, m, p)


def _r_hist(ax, df, m, p):
    hue = m.get("hue")
    kw = dict(palette=_resolve_palette(df, hue, p)) if hue else {}
    sns.histplot(data=df, x=m["x"], hue=hue, bins=p["bins"],
                 kde=p["kde"], alpha=p["alpha"], ax=ax, **kw)


def _r_kde(ax, df, m, p):
    hue = m.get("hue")
    kw = dict(palette=_resolve_palette(df, hue, p)) if hue else {}
    sns.kdeplot(data=df, x=m["x"], hue=hue, fill=p["fill"], alpha=p["alpha"], ax=ax, **kw)


def _r_ridge(ax, df, m, p):
    """Ridgeline: one KDE per group, stacked vertically (reveals bimodality and
    per-condition distribution shift better than a 2D scatter)."""
    from scipy.stats import gaussian_kde
    x, hue = m["x"], m["hue"]
    d = df[[x, hue]].copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d = d.dropna()
    cats = sorted(d[hue].astype(str).unique())
    n = len(cats)
    if n == 0:
        return
    # honour hand-picked per-group colours like every other categorical plot
    # (the UI always sends "palette", so _group_colors' own fallback never applies)
    base = _group_colors(cats, p)
    overlap = float(p.get("overlap", 1.1)); alpha = float(p.get("alpha", 0.85))
    xs = d[x].to_numpy(float)
    lo, hi = float(np.nanmin(xs)), float(np.nanmax(xs))
    pad = (hi - lo) * 0.05 or 1.0
    grid = np.linspace(lo - pad, hi + pad, 256)
    for i, c in enumerate(cats):
        vals = d.loc[d[hue].astype(str) == c, x].to_numpy(float)
        off = float(n - 1 - i)                       # cats[0] on top
        if len(vals) >= 2 and np.nanstd(vals) > 0:
            dens = gaussian_kde(vals)(grid)
            dens = dens / dens.max() * overlap
        else:
            dens = np.zeros_like(grid)
        ax.fill_between(grid, off, off + dens, color=base[i], alpha=alpha,
                        lw=0, zorder=n - i)
        ax.plot(grid, off + dens, color="white", lw=0.8, zorder=n - i)
    # this plot has no legend: its categories ARE the y ticks, so apply the renames
    # (Style > Rename items) and the column aliases (Rename columns) here.
    rl = p.get("__relabel__") or {}
    al = p.get("__aliases__") or {}
    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels([rl.get(c, al.get(c, c)) for c in cats])
    ax.set_xlabel(al.get(x, x)); ax.set_ylabel(al.get(hue, hue))
    ax.set_ylim(-0.2, n - 1 + overlap + 0.3)


def _r_heatmap(ax, df, m, p):
    num = df.select_dtypes("number")
    if not p.get("stars"):
        sns.heatmap(num.corr(numeric_only=True), annot=p["annot"], cmap=p["palette"],
                    annot_kws=_annot_kws(p), ax=ax)
        return
    from .stats import corr_with_p                 # r + BH-corrected significance stars
    r, _pv, stars = corr_with_p(df, list(num.columns), p.get("method", "pearson"))
    labels = r.round(2).astype(str) + "\n" + stars if p["annot"] else stars
    sns.heatmap(r, annot=labels.to_numpy(), fmt="", cmap=p["palette"],
                vmin=-1, vmax=1, annot_kws=_annot_kws(p, 7), ax=ax)


def _r_michaelis(ax, df, m, p):
    """Michaelis-Menten fit: v = Vmax*[S]/(Km+[S]), with Km/Vmax annotated per group."""
    from .stats import michaelis_menten
    x, y, hue = m["x"], m["y"], m.get("hue")
    groups = list(df.groupby(hue)) if hue else [(None, df)]
    colors = _group_colors([g for g, _ in groups], p) if hue else ["#3a6ea5"]
    txt = []
    for (name, sub), col in zip(groups, colors):
        s = pd.to_numeric(sub[x], errors="coerce"); v = pd.to_numeric(sub[y], errors="coerce")
        ax.scatter(s, v, s=22, alpha=p["alpha"], color=col, label=str(name) if hue else None)
        try:
            fit = michaelis_menten(s, v)
            xx = np.linspace(0, float(np.nanmax(s)) * 1.05, 200)
            ax.plot(xx, fit["predict"](xx), color=col, lw=1.6)
            txt.append(f"{(str(name)+': ') if hue else ''}Vmax={fit['Vmax']:.3g}  "
                       f"Km={fit['Km']:.3g}  R²={fit['r2']:.3f}")
            if p.get("show_km"):
                ax.axvline(fit["Km"], ls=":", color=col, lw=1)
        except Exception as e:
            txt.append(f"{(str(name)+': ') if hue else ''}no fit ({e})")
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.text(0.98, 0.03, "\n".join(txt), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(fc="white", alpha=0.75, ec="none"))
    if hue:
        ax.legend()


def _r_growth(ax, df, m, p):
    """Growth curve with an exponential fit and the doubling time per group."""
    from .stats import growth_fit
    x, y, hue = m["x"], m["y"], m.get("hue")
    groups = list(df.groupby(hue)) if hue else [(None, df)]
    colors = _group_colors([g for g, _ in groups], p) if hue else ["#3a6ea5"]
    lo, hi = p.get("t_min"), p.get("t_max")
    txt = []
    for (name, sub), col in zip(groups, colors):
        t = pd.to_numeric(sub[x], errors="coerce"); v = pd.to_numeric(sub[y], errors="coerce")
        d = pd.DataFrame({"t": t, "v": v}).dropna().groupby("t", as_index=False)["v"].mean()
        ax.plot(d["t"], d["v"], "o-", ms=4, lw=1.2, color=col,
                label=str(name) if hue else None)
        win = d
        if lo not in (None, "") or hi not in (None, ""):     # fit only the chosen window
            if lo not in (None, ""):
                win = win[win["t"] >= float(lo)]
            if hi not in (None, ""):
                win = win[win["t"] <= float(hi)]
        try:
            g = growth_fit(win["t"], win["v"])
            xx = np.linspace(win["t"].min(), win["t"].max(), 100)
            ax.plot(xx, g["y0"] * np.exp(g["mu"] * xx), "--", lw=1.4, color=col)
            txt.append(f"{(str(name)+': ') if hue else ''}td={g['doubling_time']:.3g}  "
                       f"µ={g['mu']:.3g}  R²={g['r2']:.3f}")
        except Exception as e:
            txt.append(f"{(str(name)+': ') if hue else ''}no fit ({e})")
    if p.get("log_y"):
        ax.set_yscale("log")
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.text(0.02, 0.97, "\n".join(txt), transform=ax.transAxes, ha="left", va="top",
            fontsize=8, bbox=dict(fc="white", alpha=0.75, ec="none"))
    if hue:
        ax.legend()


def _r_stdcurve(ax, df, m, p):
    """Standard curve: the standards, the fit and R². Unknowns are interpolated in the
    Bench math panel (this is the picture of that fit)."""
    from .stats import standard_curve
    x, y = m["x"], m["y"]
    d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    ax.scatter(d[x], d[y], s=30, color="#3a6ea5", zorder=3, label="standards")
    try:
        r = standard_curve(df, x, y, p.get("model", "linear"))
        xx = np.linspace(float(d[x].min()), float(d[x].max()), 200)
        if p.get("model") == "4pl":
            a, b, c, dd = (r["params"][k] for k in ("a", "b", "c", "d"))
            yy = dd + (a - dd) / (1.0 + (xx / c) ** b)
        else:
            yy = r["params"]["slope"] * xx + r["params"]["intercept"]
        ax.plot(xx, yy, color="#e2683b", lw=1.6, zorder=2)
        ax.text(0.02, 0.97, r["text"], transform=ax.transAxes, ha="left", va="top",
                fontsize=8, bbox=dict(fc="white", alpha=0.75, ec="none"))
    except Exception as e:
        ax.text(0.5, 0.5, f"no fit: {e}", transform=ax.transAxes, ha="center")
    ax.set_xlabel(x); ax.set_ylabel(y)


def _r_volcano(ax, df, m, p):
    fc = df[m["x"]].to_numpy(float)
    pv = df[m["y"]].to_numpy(float)
    nlp = -np.log10(np.clip(pv, 1e-300, None))
    up = (fc >= p["fc_thr"]) & (pv <= p["p_thr"])
    dn = (fc <= -p["fc_thr"]) & (pv <= p["p_thr"])
    ns = ~(up | dn)
    ax.scatter(fc[ns], nlp[ns], s=12, c="lightgrey", alpha=p["alpha"])
    ax.scatter(fc[up], nlp[up], s=14, c="tab:red", alpha=p["alpha"], label="up")
    ax.scatter(fc[dn], nlp[dn], s=14, c="tab:blue", alpha=p["alpha"], label="down")
    ax.axhline(-np.log10(p["p_thr"]), ls="--", c="grey", lw=0.8)
    ax.axvline(p["fc_thr"], ls="--", c="grey", lw=0.8)
    ax.axvline(-p["fc_thr"], ls="--", c="grey", lw=0.8)
    ax.set_xlabel("log2 Fold Change"); ax.set_ylabel("-log10(p)")
    if up.any() or dn.any():
        ax.legend()


def _overlay_regression(ax, df, xcol, ycol):
    sub = df[[xcol, ycol]].apply(__import__("pandas").to_numeric, errors="coerce").dropna()
    x, y = sub[xcol].to_numpy(float), sub[ycol].to_numpy(float)
    if len(x) < 2:
        return
    a, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ss_res = np.sum((y - (a * x + b)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    ax.plot(xs, a * xs + b, c="black", lw=1.2)
    ax.text(0.02, 0.97, f"y={a:.3g}x+{b:.3g}\nR²={r2:.3f}", transform=ax.transAxes,
            va="top", fontsize=8, bbox=dict(fc="white", alpha=0.7, ec="none"))


def _r_superplot(ax, df, m, p):
    """SuperPlot (Lord et al., JCB 2020): faint raw points + one big marker per
    replicate mean, colored by replicate. Test on the replicate means, not the cells —
    this is the honest way to plot data with many cells per replicate."""
    x, y, rep = m["x"], m["y"], m.get("rep")
    d = df[[c for c in [x, y, rep] if c]].copy()
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=[x, y])
    if rep:
        d[rep] = d[rep].astype(str)           # numeric replicate ids -> discrete colors
        pal = _resolve_palette(d, rep, p)
        sns.stripplot(data=d, x=x, y=y, hue=rep, palette=pal, dodge=False, jitter=0.28,
                      size=p["size"] * 0.6, alpha=0.35, legend=False, ax=ax)
        means = d.groupby([x, rep], observed=True)[y].mean().reset_index()
        # legend only helps with few replicates (the SuperPlot use case); many track ids
        # would overflow the figure, so suppress it there.
        sns.stripplot(data=means, x=x, y=y, hue=rep, palette=pal, dodge=False, jitter=0.06,
                      size=p["size"] * 2.4, alpha=1.0, edgecolor="black", linewidth=1.1,
                      legend=(d[rep].nunique() <= 12), ax=ax)
    else:
        sns.stripplot(data=d, x=x, y=y, color="#9aa0aa", jitter=0.28,
                      size=p["size"] * 0.6, alpha=0.35, ax=ax)
    # thick line at each condition's grand mean
    for i, c in enumerate(list(pd.unique(d[x]))):
        gm = d.loc[d[x] == c, y].mean()
        ax.plot([i - 0.32, i + 0.32], [gm, gm], color="#333", lw=2.2, zorder=5)


def _r_qq(ax, df, m, p):
    """Normal Q-Q plot per group: points on the diagonal => normally distributed."""
    from scipy import stats as st
    y, hue = m["val"], m.get("hue")
    groups = list(df.groupby(hue)) if hue else [(None, df)]
    colors = _group_colors([g for g, _ in groups], p) if hue else ["#3a6ea5"]
    for (name, sub), col in zip(groups, colors):
        v = pd.to_numeric(sub[y], errors="coerce").dropna().to_numpy()
        if len(v) < 3:
            continue
        (osm, osr), (slope, inter, _r) = st.probplot(v, dist="norm")
        ax.scatter(osm, osr, s=14, color=col, alpha=p["alpha"],
                   label=str(name) if hue else None)
        ax.plot(osm, slope * osm + inter, color=col, lw=1.0)
    ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
    if hue:
        ax.legend()


def _r_km(ax, df, m, p):
    """Kaplan-Meier survival curves (+ log-rank p when grouped). 'time' = duration,
    'event' = status column, 'event value(s)' = which status counts as the event
    (comma-separated; empty = every non-empty status counts)."""
    from .stats import km_curves
    time, event, hue = m["time"], m.get("event"), m.get("hue")
    ev = [s.strip() for s in str(p.get("event_value", "")).split(",") if s.strip()]
    curves, lr = km_curves(df, time, event, ev, hue)
    colors = _group_colors(list(curves), p) if hue else ["#3a6ea5"]
    for (name, (t, s, cens)), col in zip(curves.items(), colors):
        ax.step(t, s, where="post", color=col, lw=1.6, label=(str(name) if hue else None))
        if p.get("censors", True) and len(cens):
            sc = np.interp(cens, t, s)        # survival at each censoring time
            ax.plot(cens, sc, "|", color=col, ms=7, mew=1.2)
    ax.set_ylim(0, 1.03); ax.set_xlabel(time); ax.set_ylabel("Survival probability")
    if lr:
        ax.text(0.02, 0.05, f"log-rank p = {lr['p']:.3g}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9,
                bbox=dict(fc="white", alpha=0.7, ec="none"))
    if hue:
        ax.legend(title=hue)


def _r_traj(ax, df, m, p):
    """XY trajectories: one path per track (id), optionally ordered by a time column and
    colored by group (hue)."""
    x, y, tid, hue, tcol = m["x"], m["y"], m["id"], m.get("hue"), m.get("time")
    cmap, seen = None, set()
    if hue:
        groups = list(df.groupby(hue))
        cols = _group_colors([g for g, _ in groups], p)
        cmap = {g: cols[i] for i, (g, _) in enumerate(groups)}
    for _, sub in df.groupby(tid):
        if tcol:
            sub = sub.sort_values(tcol)
        c = cmap[sub[hue].iloc[0]] if hue else None
        lbl = None
        if hue and sub[hue].iloc[0] not in seen:      # one legend entry per group
            lbl = str(sub[hue].iloc[0]); seen.add(sub[hue].iloc[0])
        ax.plot(pd.to_numeric(sub[x], errors="coerce"),
                pd.to_numeric(sub[y], errors="coerce"),
                lw=p["linewidth"], alpha=p["alpha"], color=c, label=lbl)
    ax.set_xlabel(x); ax.set_ylabel(y)
    if p.get("equal", True):
        ax.set_aspect("equal", adjustable="datalim")
    if hue:
        ax.legend(title=hue)


def _r_msd(ax, df, m, p):
    """Mean squared displacement vs lag, averaged across tracks, per group."""
    from .stats import msd_by_track
    curves = msd_by_track(df, m["id"], m["px"], m["py"], m.get("time"), m.get("hue"))
    if not curves:
        return
    colors = _group_colors(list(curves), p)
    for (name, (lags, msd, ntr)), col in zip(curves.items(), colors):
        ax.plot(lags, msd, marker="o", ms=3, lw=1.4, color=col,
                label=(f"{name} (n={ntr})" if name is not None else None))
    ax.set_xlabel("lag (frames)"); ax.set_ylabel("MSD")
    if p.get("loglog"):
        ax.set_xscale("log"); ax.set_yscale("log")
    if any(n is not None for n in curves):
        ax.legend()


def _r_stacked(ax, df, m, p):
    """Composition bars: for each x category, the proportion (or count) of each hue
    category, stacked. Optional chi-square/Fisher p annotation."""
    x, hue = m["x"], m["hue"]
    ct = pd.crosstab(df[x].astype(str), df[hue].astype(str))
    if p.get("proportion", True):
        ct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)
    cats = list(ct.columns)
    colors = _group_colors(cats, p)
    bottom = np.zeros(len(ct))
    xs = np.arange(len(ct))
    for c, col in zip(cats, colors):
        vals = ct[c].to_numpy(float)
        ax.bar(xs, vals, bottom=bottom, color=col, width=0.8, label=str(c),
               edgecolor="white", linewidth=0.5)
        bottom += np.nan_to_num(vals)
    ax.set_xticks(xs); ax.set_xticklabels(ct.index)
    ax.set_ylabel("proportion" if p.get("proportion", True) else "count")
    if p.get("show_p"):
        try:
            from .stats import contingency_test
            pl = contingency_test(df, x, hue)["text"].splitlines()
            ptxt = next((ln for ln in pl if ln.startswith("Chi-square")), "")
            # inside top-left with a white box: the engine overwrites the title, and 1.0+
            # would sit under it. White bbox keeps it readable over the bars.
            ax.text(0.02, 0.98, ptxt, transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    bbox=dict(fc="white", alpha=0.75, ec="none"))
        except Exception:
            pass
    ax.legend(title=hue)


_ALPHA = Param("alpha", "float", 0.8, 0.0, 1.0)
_PAL = Param("palette", "palette", "colorblind",
             choices=("colorblind", "cividis", "viridis", "mako", "crest", "flare",
                      "Set2", "magma", "rocket", "Set1", "tab10", "coolwarm", "Spectral"))
# Heatmaps use the "palette" as a matplotlib colormap, so it must be a valid
# colormap name (a categorical palette like "colorblind" is not a colormap).
_ANNSZ = Param("annot_size", "float", 0, 0, 24, label="number size (0 = auto)")
_CMAP = Param("palette", "palette", "viridis",
              choices=("viridis", "magma", "rocket", "mako", "crest", "flare",
                       "cividis", "coolwarm", "Spectral", "vlag", "icefire"))

REGISTRY: dict[str, PlotSpec] = {s.key: s for s in [
    PlotSpec("scatter", "Scatter", "axes", _r_scatter, pickable=True,
             channels=[Channel("x", True, (NUMBER, DATETIME)), Channel("y", True, (NUMBER,)),
                       Channel("hue", accepts=(CATEGORY, NUMBER)),
                       Channel("size", accepts=(NUMBER,)), Channel("style", accepts=(CATEGORY,))],
             params=[_ALPHA, _PAL, Param("regression", "bool", False, label="Regression + R²"),
                     Param("shape_hue", "bool", False, label="shape by hue (colour-blind safe)")]),
    PlotSpec("line", "Line", "axes", _r_line,
             channels=[Channel("x", True), Channel("y", True, (NUMBER,)), Channel("hue", accepts=(CATEGORY,))],
             params=[Param("linewidth", "float", 1.5, 0.5, 6.0),
                     Param("errorbar", "choice", "none", choices=("none", "SD", "SEM", "CI95"), label="error band"),
                     _ALPHA, _PAL]),
    PlotSpec("bar", "Bar", "axes", _r_bar,
             channels=[Channel("x", True, (CATEGORY,)), Channel("y", True, (NUMBER,)), Channel("hue", accepts=(CATEGORY,))],
             params=[_ALPHA, _PAL]),
    PlotSpec("box", "Boxplot", "axes", _r_box,
             channels=[Channel("x", True, (CATEGORY,)), Channel("y", True, (NUMBER,)), Channel("hue", accepts=(CATEGORY,))],
             params=[Param("sig", "bool", False, label="significance brackets (no hue)"), _PAL]),
    PlotSpec("violin", "Violin", "axes", _r_violin,
             channels=[Channel("x", True, (CATEGORY,)), Channel("y", True, (NUMBER,)), Channel("hue", accepts=(CATEGORY,))],
             params=[Param("sig", "bool", False, label="significance brackets (no hue)"), _PAL]),
    PlotSpec("strip", "Stripplot", "axes", _r_strip,
             channels=[Channel("x", True, (CATEGORY,)), Channel("y", True, (NUMBER,)), Channel("hue", accepts=(CATEGORY,))],
             params=[Param("sig", "bool", False, label="significance brackets (no hue)"), _ALPHA, _PAL]),
    PlotSpec("box_points", "Box + points", "axes", _r_box_points,
             channels=[Channel("x", True, (CATEGORY,)), Channel("y", True, (NUMBER,)), Channel("hue", accepts=(CATEGORY,))],
             params=[Param("size", "float", 4, 1, 12, label="point size"),
                     Param("jitter", "float", 0.2, 0.0, 0.4, label="jitter"),
                     Param("sig", "bool", False, label="significance brackets (no hue)"),
                     Param("alpha", "float", 0.7, 0.0, 1.0), _PAL]),
    PlotSpec("violin_points", "Violin + points", "axes", _r_violin_points,
             channels=[Channel("x", True, (CATEGORY,)), Channel("y", True, (NUMBER,)), Channel("hue", accepts=(CATEGORY,))],
             params=[Param("size", "float", 4, 1, 12, label="point size"),
                     Param("jitter", "float", 0.2, 0.0, 0.4, label="jitter"),
                     Param("sig", "bool", False, label="significance brackets (no hue)"),
                     Param("alpha", "float", 0.7, 0.0, 1.0), _PAL]),
    PlotSpec("scatter_density", "Scatter + group background", "axes", _r_scatter_density, pickable=True,
             channels=[Channel("x", True, (NUMBER,)), Channel("y", True, (NUMBER,)),
                       Channel("hue", True, (CATEGORY,)),
                       Channel("size", accepts=(NUMBER,)), Channel("style", accepts=(CATEGORY,))],
             params=[Param("psize", "float", 22, 4, 80, label="point size"),
                     Param("alpha", "float", 0.35, 0.05, 0.8, label="background intensity"),
                     Param("grid", "int", 120, 40, 240, label="resolution (solid mode)"),
                     Param("contour", "bool", False, label="contour background (KDE waves)"),
                     Param("levels", "int", 10, 3, 24, label="number of waves (contour mode)"), _PAL]),
    PlotSpec("dose4pl", "Dose-response (4PL + IC50)", "axes", _r_dose4pl, pickable=True,
             channels=[Channel("x", True, (NUMBER,), "dose"), Channel("y", True, (NUMBER,), "response"),
                       Channel("hue", accepts=(CATEGORY,))],
             params=[Param("log_x", "bool", True, label="log X axis"),
                     Param("alpha", "float", 0.8, 0.0, 1.0), _PAL]),
    PlotSpec("bar_err", "Bar/point ± error", "axes", _r_bar_err,
             channels=[Channel("x", True, (CATEGORY,)), Channel("y", True, (NUMBER,)), Channel("hue", accepts=(CATEGORY,))],
             params=[Param("kind", "choice", "bar", choices=("bar", "point")),
                     Param("center", "choice", "mean", choices=("mean", "median")),
                     Param("error", "choice", "SEM", choices=("SD", "SEM", "CI95")),
                     Param("points", "bool", True, label="show points"),
                     Param("psize", "float", 3.5, 1, 10, label="point size"),
                     Param("sig", "bool", False, label="significance bars (no hue)"), _PAL]),
    PlotSpec("roc", "ROC curve (+ AUC)", "axes", _r_roc,
             channels=[Channel("score", True, (NUMBER,), "score / marker"),
                       Channel("label", True, (CATEGORY, NUMBER), "outcome (2 levels)")],
             params=[Param("positive", "text", "", label="positive level (empty = last)"),
                     Param("ci", "bool", True, label="bootstrap CI95 of the AUC"),
                     Param("mark_cutoff", "bool", True, label="mark the Youden cutoff"),
                     _PAL]),
    PlotSpec("bland_altman", "Bland-Altman (agreement)", "axes", _r_bland,
             channels=[Channel("m1", True, (NUMBER,), "method A"),
                       Channel("m2", True, (NUMBER,), "method B"),
                       Channel("hue", accepts=(CATEGORY,))],
             params=[Param("psize", "float", 3.5, 1, 10, label="point size"),
                     _ALPHA, _PAL]),
    PlotSpec("heatmap_xy", "Heatmap (rows x columns)", "axes", _r_heatmap_xy,
             channels=[Channel("row", True, (CATEGORY,), "rows (Y)"),
                       Channel("col", True, (CATEGORY,), "columns (X)"),
                       Channel("value", True, (NUMBER,), "value in each cell")],
             params=[Param("annot", "bool", True, label="write the value in each cell"),
                     Param("decimals", "int", 1, 0, 3, label="decimals"),
                     Param("agg", "choice", "mean",
                           choices=("mean", "median", "sum", "max", "min", "count"),
                           label="if a pair repeats"),
                     Param("gap", "float", 0.0, 0.0, 2.0, label="gap between cells"),
                     _ANNSZ, _CMAP]),
    PlotSpec("heatmap_matrix", "Matrix heatmap", "axes", _r_heatmap_matrix,
             channels=[Channel("labels", accepts=(CATEGORY,), label="row names (optional)")],
             params=[Param("zscore", "bool", True, label="z-score per column"),
                     Param("annot", "bool", False), _ANNSZ, _CMAP]),
    PlotSpec("ecdf", "ECDF (cumulative distribution)", "axes", _r_ecdf,
             channels=[Channel("x", True, (NUMBER,)), Channel("hue", accepts=(CATEGORY,))], params=[]),
    PlotSpec("regband", "Regression + CI band", "axes", _r_regband, pickable=True,
             channels=[Channel("x", True, (NUMBER,)), Channel("y", True, (NUMBER,)), Channel("hue", accepts=(CATEGORY,))],
             params=[Param("ci", "int", 95, 80, 99, label="CI (%)"),
                     Param("alpha", "float", 0.6, 0.0, 1.0), _PAL]),
    PlotSpec("paired", "Paired (before-after)", "axes", _r_paired,
             channels=[Channel("x", True, (CATEGORY,)), Channel("y", True, (NUMBER,)),
                       Channel("id", True, label="sample (id)")],
             params=[Param("alpha", "float", 0.6, 0.0, 1.0)]),
    PlotSpec("hist", "Histogram", "axes", _r_hist,
             channels=[Channel("x", True, (NUMBER,)), Channel("hue", accepts=(CATEGORY,))],
             params=[Param("bins", "int", 30, 5, 200), Param("kde", "bool", False), _ALPHA]),
    PlotSpec("kde", "KDE", "axes", _r_kde,
             channels=[Channel("x", True, (NUMBER,)), Channel("hue", accepts=(CATEGORY,))],
             params=[Param("fill", "bool", True), _ALPHA]),
    PlotSpec("ridge", "Ridgeline (per-group distribution)", "axes", _r_ridge,
             channels=[Channel("x", True, (NUMBER,)), Channel("hue", True, (CATEGORY,))],
             params=[Param("overlap", "float", 1.1, 0.3, 3.0, label="overlap"), _ALPHA, _PAL]),
    PlotSpec("heatmap", "Heatmap (correlation)", "axes", _r_heatmap,
             channels=[], params=[Param("annot", "bool", True), _ANNSZ,
                                  Param("stars", "bool", False, label="significance stars (BH)"),
                                  Param("method", "choice", "pearson", choices=("pearson", "spearman")),
                                  _CMAP]),
    PlotSpec("michaelis", "Michaelis-Menten (Km / Vmax)", "axes", _r_michaelis,
             channels=[Channel("x", True, (NUMBER,), "[S] substrate"),
                       Channel("y", True, (NUMBER,), "v (rate)"),
                       Channel("hue", accepts=(CATEGORY,), label="group")],
             params=[Param("show_km", "bool", True, label="mark Km"),
                     Param("alpha", "float", 0.85, 0.0, 1.0), _PAL]),
    PlotSpec("growth", "Growth curve (doubling time)", "axes", _r_growth,
             channels=[Channel("x", True, (NUMBER,), "time"),
                       Channel("y", True, (NUMBER,), "OD / signal"),
                       Channel("hue", accepts=(CATEGORY,), label="group")],
             params=[Param("log_y", "bool", False, label="log Y"),
                     Param("t_min", "text", "", label="fit from time"),
                     Param("t_max", "text", "", label="fit to time"), _PAL]),
    PlotSpec("stdcurve", "Standard curve (+ R²)", "axes", _r_stdcurve,
             channels=[Channel("x", True, (NUMBER,), "known concentration"),
                       Channel("y", True, (NUMBER,), "signal")],
             params=[Param("model", "choice", "linear", choices=("linear", "4pl"))]),
    PlotSpec("volcano", "Volcano Plot", "axes", _r_volcano, pickable=True,
             channels=[Channel("x", True, (NUMBER,), "log2FC"), Channel("y", True, (NUMBER,), "p-value")],
             params=[Param("fc_thr", "float", 1.0, 0.0, 10.0, label="|log2FC| threshold"),
                     Param("p_thr", "float", 0.05, 0.0, 1.0, label="p threshold"), _ALPHA]),
    PlotSpec("superplot", "SuperPlot (cells + replicate means)", "axes", _r_superplot,
             channels=[Channel("x", True, (CATEGORY,), "condition"), Channel("y", True, (NUMBER,)),
                       Channel("rep", accepts=(CATEGORY, NUMBER), label="replicate")],
             params=[Param("size", "float", 5, 1, 12, label="point size"),
                     Param("alpha", "float", 0.35, 0.0, 1.0, label="raw points alpha"), _PAL]),
    PlotSpec("qq", "Q-Q plot (normality)", "axes", _r_qq,
             channels=[Channel("val", True, (NUMBER,), "value"), Channel("hue", accepts=(CATEGORY,))],
             params=[_ALPHA, _PAL]),
    PlotSpec("km", "Kaplan-Meier (survival + log-rank)", "axes", _r_km,
             channels=[Channel("time", True, (NUMBER,), "time / duration"),
                       Channel("event", accepts=(CATEGORY, NUMBER), label="event (status)"),
                       Channel("hue", accepts=(CATEGORY,), label="group")],
             params=[Param("event_value", "text", "", label="event value(s), comma-sep"),
                     Param("censors", "bool", True, label="mark censored (ticks)"), _PAL]),
    PlotSpec("traj", "Trajectories (XY tracks)", "axes", _r_traj,
             channels=[Channel("x", True, (NUMBER,), "pos x"), Channel("y", True, (NUMBER,), "pos y"),
                       Channel("id", True, label="track id"),
                       Channel("time", accepts=(NUMBER,), label="time (order)"),
                       Channel("hue", accepts=(CATEGORY,), label="group")],
             params=[Param("linewidth", "float", 0.9, 0.2, 4.0),
                     Param("alpha", "float", 0.7, 0.05, 1.0),
                     Param("equal", "bool", True, label="equal aspect"), _PAL]),
    PlotSpec("msd", "MSD (mean squared displacement)", "axes", _r_msd,
             channels=[Channel("id", True, label="track id"),
                       Channel("px", True, (NUMBER,), "pos x"), Channel("py", True, (NUMBER,), "pos y"),
                       Channel("time", accepts=(NUMBER,), label="time (order)"),
                       Channel("hue", accepts=(CATEGORY,), label="group")],
             params=[Param("loglog", "bool", False, label="log-log axes"), _PAL]),
    PlotSpec("stacked", "Stacked composition (+ chi2/Fisher)", "axes", _r_stacked,
             channels=[Channel("x", True, (CATEGORY,), "condition"),
                       Channel("hue", True, (CATEGORY,), "category")],
             params=[Param("proportion", "bool", True, label="proportion (else count)"),
                     Param("show_p", "bool", True, label="chi-square p in title"), _PAL]),
]}
