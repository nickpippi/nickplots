"""Statistical analysis: dimensionality reduction, group tests, correlation.
No UI dependency. Heavy functions run via run_async so the window doesn't freeze.

UMAP is a lazy import: the app works without it installed; it only fails if you
choose 'UMAP' without the package (the UI shows a friendly message)."""
from __future__ import annotations
import threading
import numpy as np
import pandas as pd


def _matrix(df: pd.DataFrame, cols: list[str]):
    X = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if X.shape[0] < 3:
        raise ValueError("Too few valid rows (after dropping NaN) for the reduction.")
    if X.shape[1] < 2:
        raise ValueError("Select at least 2 numeric columns.")
    return X


def compute_embedding(df, cols, method="PCA", n_components=2, perplexity=30):
    """Apply PCA / t-SNE / UMAP to the chosen columns and return (new_df, column_names).
    The new columns (PC1/PC2, tSNE1..., UMAP1...) are appended to the dataframe and are plottable."""
    from sklearn.preprocessing import StandardScaler
    X = _matrix(df, cols)
    idx = X.index
    Xs = StandardScaler().fit_transform(X.values)
    method = method.upper().replace("-", "")

    if method == "PCA":
        from sklearn.decomposition import PCA
        emb = PCA(n_components=n_components).fit_transform(Xs)
        prefix = "PC"
    elif method == "TSNE":
        from sklearn.manifold import TSNE
        perp = min(perplexity, max(5, len(Xs) - 1))
        emb = TSNE(n_components=n_components, perplexity=perp,
                   init="pca", random_state=0).fit_transform(Xs)
        prefix = "tSNE"
    elif method == "UMAP":
        try:
            import umap
        except ImportError:
            raise RuntimeError("UMAP is not installed. Run: pip install umap-learn")
        emb = umap.UMAP(n_components=n_components, random_state=0).fit_transform(Xs)
        prefix = "UMAP"
    else:
        raise ValueError(f"Unknown method: {method}")

    names = [f"{prefix}{i + 1}" for i in range(n_components)]
    out = df.copy()
    for i, name in enumerate(names):
        out[name] = np.nan
        out.loc[idx, name] = emb[:, i]
    return out, names


def group_test(df, value_col, group_col) -> str:
    """Welch t-test (2 groups) or one-way ANOVA (3+ groups)."""
    from scipy import stats
    parts = [(str(k), g[value_col].dropna().to_numpy(float))
             for k, g in df.groupby(group_col)]
    parts = [(n, v) for n, v in parts if len(v) > 1]
    if len(parts) < 2:
        return "Need at least 2 groups with more than 1 observation each."
    if len(parts) == 2:
        (n1, a), (n2, b) = parts
        t, p = stats.ttest_ind(a, b, equal_var=False)
        return (f"Welch t-test — {n1} (n={len(a)}) vs {n2} (n={len(b)})\n"
                f"t = {t:.4f}   p = {p:.4g}")
    f, p = stats.f_oneway(*[v for _, v in parts])
    return (f"One-way ANOVA — {len(parts)} groups ({', '.join(n for n, _ in parts)})\n"
            f"F = {f:.4f}   p = {p:.4g}")


def describe(df, cols) -> str:
    d = df[cols].apply(pd.to_numeric, errors="coerce").describe().round(4)
    return d.to_string()


def run_async(fn, on_done, on_error):
    """Run fn() in a daemon thread. on_done(result) / on_error(exc) are called back.
    The UI must re-enter the main thread with self.after(0, ...) inside those callbacks."""
    def worker():
        try:
            res = fn()
        except Exception as e:
            on_error(e)
            return
        on_done(res)
    threading.Thread(target=worker, daemon=True).start()


# ============================ extensions ============================ #
def fit_4pl(x, y):
    """Four-parameter logistic dose-response fit.
    y = d + (a-d)/(1+(x/c)**b). Returns params, IC50(=c), R² and a predictor function."""
    from scipy.optimize import curve_fit
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        raise ValueError("Too few points to fit 4PL (minimum 4).")

    def f(xx, a, b, c, d):
        return d + (a - d) / (1.0 + (xx / c) ** b)

    a0, d0 = y.max(), y.min()
    c0 = np.median(x[x > 0]) if np.any(x > 0) else 1.0
    p0 = [a0, 1.0, c0, d0]
    bounds = ([-np.inf, -np.inf, 1e-9, -np.inf], [np.inf, np.inf, np.inf, np.inf])
    popt, _ = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=20000)
    yhat = f(x, *popt)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return dict(a=popt[0], b=popt[1], c=popt[2], d=popt[3], ic50=popt[2], r2=r2,
                predict=lambda xx: f(np.asarray(xx, float), *popt))


def describe_by_group(df, value_cols, group_col=None):
    """Descriptive statistics table (N, Mean, SD, SEM, Median, Min, Max).
    Per group if group_col is given, otherwise overall. Returns a display/export-ready DataFrame."""
    def stats_block(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        n = len(s)
        sd = s.std()
        return dict(N=n, Mean=round(s.mean(), 4), SD=round(sd, 4),
                    SEM=round(sd / np.sqrt(n), 4) if n else np.nan,
                    Median=round(s.median(), 4), Min=round(s.min(), 4), Max=round(s.max(), 4))
    rows = []
    if group_col:
        for g, sub in df.groupby(group_col):
            for col in value_cols:
                rows.append({"Group": str(g), "Variable": col, **stats_block(sub[col])})
    else:
        for col in value_cols:
            rows.append({"Variable": col, **stats_block(df[col])})
    return pd.DataFrame(rows)


def gating_coords(df, x_col, y_col, pos=None):
    """Numeric coordinates for gating. On a CATEGORICAL axis (strip/swarm/box... with a
    group on the axis) it uses the category's POSITION on the axis (0,1,2...), passed in
    `pos={'x':{label:position}, 'y':...}`. On a continuous axis it uses the column value.
    This makes gating work on any point plot, reproducibly (independent of the random
    jitter, which is only visual)."""
    pos = pos or {}

    def axis(col, key):
        m = pos.get(key)
        if m:
            return df[col].astype(str).map(m).to_numpy(dtype=float)
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    return axis(x_col, "x"), axis(y_col, "y")


def assign_regions(df, x_col, y_col, gates, outside="(outside)", pos=None):
    """Series aligned to df.index with the name of the FIRST region (gate) whose polygon
    contains each point (list order = priority on overlap), or `outside` if none. Points
    without a numeric X/Y become NaN. Uses matplotlib.path."""
    from matplotlib.path import Path
    x, y = gating_coords(df, x_col, y_col, pos)
    pts = np.column_stack([x, y])
    valid = np.isfinite(x) & np.isfinite(y)
    res = np.array([outside] * len(df), dtype=object)
    assigned = np.zeros(len(df), bool)
    for i, g in enumerate(gates):
        poly = g.get("points") or []
        if len(poly) < 3:
            continue
        name = g.get("name") or f"R{i + 1}"
        inside = Path(np.asarray(poly, float)).contains_points(pts) & valid & ~assigned
        res[inside] = name
        assigned |= inside
    out = pd.Series(res, index=df.index, dtype=object)
    out[~valid] = np.nan
    return out


def gate_breakdown(df, x_col, y_col, gates, hue_col=None, pos=None):
    """Count observations per region (polygon), optionally per group (hue).
    Columns: Region | <one per group> | Total | Groups (number of groups present; >=2 = overlap)."""
    if not x_col or not y_col:
        raise ValueError("Set the X and Y columns of the plot first.")
    if not gates:
        raise ValueError("Add at least one region.")
    reg = assign_regions(df, x_col, y_col, gates, pos=pos)
    tmp = pd.DataFrame({"_reg": reg})
    if hue_col:
        tmp[hue_col] = df[hue_col].astype(str).to_numpy()
    tmp = tmp.dropna(subset=["_reg"])

    named = []
    for i, g in enumerate(gates):
        nm = g.get("name") or f"R{i + 1}"
        if nm not in named:
            named.append(nm)

    if hue_col:
        groups = sorted(tmp[hue_col].unique()) if not tmp.empty else []
        ct = pd.crosstab(tmp["_reg"], tmp[hue_col]) if not tmp.empty else pd.DataFrame()
        # ensure ALL named regions (even empty) and all group columns
        idx = list(named) + (["(outside)"] if "(outside)" in ct.index else [])
        ct = ct.reindex(index=idx, columns=groups, fill_value=0).fillna(0).astype(int)
        n_groups = (ct > 0).sum(axis=1)
        ct["Total"] = ct.sum(axis=1)
        ct["Groups"] = n_groups
        ct = ct.reset_index()
        return ct.rename(columns={"_reg": "Region", "index": "Region"})

    counts = tmp.groupby("_reg").size() if not tmp.empty else pd.Series(dtype=int)
    idx = list(named) + (["(outside)"] if "(outside)" in counts.index else [])
    out = counts.reindex(idx, fill_value=0).astype(int).rename("Total").reset_index()
    return out.rename(columns={"_reg": "Region", "index": "Region"})


def region_breakdown(df, x_col, y_col, lines, hue_col=None):
    """Split the X×Y plane by one or more threshold lines and count the observations in
    each region, optionally separated by group (hue).

    Each line is {x, y, angle}: it passes through (x,y) with `angle` degrees ON THE DATA
    SCALE (slope = tan(angle); 0=horizontal, 90=vertical). A point is 'below' if
    y < y0 + tan(angle)*(x - x0) (or 'left/right' for a vertical line). A point's region
    is the combination of the sides of all lines.

    Returns DataFrame: Region | <one per group> | Total | Groups. The 'Groups' column is
    the number of groups present there (>=2 indicates overlap). No hue: Region | Total."""
    if not x_col or not y_col:
        raise ValueError("Set the X and Y columns of the plot first.")
    if not lines:
        raise ValueError("Add at least one threshold line.")
    # extract X and Y by name (works even with X == Y, which would duplicate columns in df[[...]])
    xs = pd.to_numeric(df[x_col], errors="coerce")
    ys = pd.to_numeric(df[y_col], errors="coerce")
    mask = xs.notna() & ys.notna()
    if not mask.any():
        raise ValueError("No valid numeric data in X/Y.")
    X = xs[mask].to_numpy(float); Y = ys[mask].to_numpy(float)
    hue_vals = df.loc[mask, hue_col].astype(str).to_numpy() if hue_col else None

    sides = []
    for i, ln in enumerate(lines):
        ang = float(ln.get("angle", 0.0)) % 180.0
        x0 = float(ln.get("x", 0.0)); y0 = float(ln.get("y", 0.0))
        if abs(ang - 90.0) < 1e-9:
            s = np.where(X < x0, "left", "right")
        else:
            m = np.tan(np.deg2rad(ang))
            s = np.where(Y < y0 + m * (X - x0), "below", "above")
        sides.append((f"L{i + 1}", s))

    regions = [" · ".join(f"{nm} {s[j]}" for nm, s in sides) for j in range(len(X))]

    if hue_col:
        sub = pd.DataFrame({"_reg": regions, hue_col: hue_vals})
        ct = pd.crosstab(sub["_reg"], sub[hue_col])
        n_groups = (ct > 0).sum(axis=1)
        ct["Total"] = ct.sum(axis=1)
        ct["Groups"] = n_groups
        ct = ct.sort_values("Total", ascending=False).reset_index()
        return ct.rename(columns={"_reg": "Region"})
    out = (pd.Series(regions).value_counts().rename("Total")
           .rename_axis("Region").reset_index())
    return out


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    s = 0.0
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]; x2, y2 = poly[(k + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _clip_halfplane(poly, a, b, c, neg, eps=1e-9):
    """Clip a convex polygon by the half-plane a*x+b*y+c <=0 (neg) or >=0."""
    def f(p):
        return a * p[0] + b * p[1] + c

    def _inter(p, q, fp, fq):
        d = fp - fq
        if abs(d) < 1e-15:
            return q
        t = fp / d
        return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))

    out = []
    n = len(poly)
    for k in range(n):
        cur = poly[k]; prv = poly[k - 1]
        fc = f(cur); fp = f(prv)
        ci = (fc <= eps) if neg else (fc >= -eps)
        pi = (fp <= eps) if neg else (fp >= -eps)
        if ci:
            if not pi:
                out.append(_inter(prv, cur, fp, fc))
            out.append(cur)
        elif pi:
            out.append(_inter(prv, cur, fp, fc))
    return out


def merge_gates_geometric(gates):
    """Geometrically merge regions with the SAME name into a single polygon, erasing the
    inner edges. Regions with different names stay intact. Same name in disconnected
    pieces becomes several polygons (with the same name)."""
    try:
        from shapely.geometry import Polygon as SPoly
        from shapely.ops import unary_union
    except ImportError:
        raise RuntimeError("Geometric merge needs shapely. Install: pip install shapely")
    from collections import OrderedDict
    groups = OrderedDict()
    for g in gates or []:
        groups.setdefault(g.get("name") or "R", []).append(g)

    out = []
    for name, gs in groups.items():
        if len(gs) == 1:
            out.append(gs[0])
            continue
        polys = []
        for g in gs:
            pts = g.get("points") or []
            if len(pts) >= 3:
                p = SPoly([(float(a), float(b)) for a, b in pts])
                if not p.is_valid:
                    p = p.buffer(0)
                polys.append(p)
        if not polys:
            out.extend(gs)
            continue
        u = unary_union(polys).simplify(0)          # simplify(0) removes collinear vertices
        color = gs[0].get("color") or "#e23b3b"
        geoms = list(u.geoms) if u.geom_type == "MultiPolygon" else [u]
        for geom in geoms:
            ext = list(geom.exterior.coords)[:-1]   # drop the closing (duplicate) point
            if len(ext) >= 3:
                out.append(dict(name=name, color=color,
                                points=[[round(x, 4), round(y, 4)] for x, y in ext]))
    return out


def merge_gates_geometric_available():
    try:
        import shapely  # noqa
        return True
    except ImportError:
        return False


def cells_from_lines(lines, xlim, ylim):
    """Build the polygonal cells formed by the lines inside the plot box. Each line splits
    the existing cells into two half-planes. Returns a list of {name, color, points} — the
    name follows the line-analysis convention (e.g. 'L1 above · L2 right'), ready to become
    named regions (gates)."""
    import math
    x0, x1 = float(xlim[0]), float(xlim[1]); y0, y1 = float(ylim[0]), float(ylim[1])
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    rect = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    cells = [(rect, [])]

    for i, ln in enumerate(lines):
        ang = float(ln.get("angle", 0.0)) % 180.0
        ax = float(ln.get("x", 0.0)); ay = float(ln.get("y", 0.0))
        if abs(ang - 90.0) < 1e-9:
            a, b, c = 1.0, 0.0, -ax; lo, hi = "left", "right"       # x - ax
        else:
            m = math.tan(math.radians(ang))
            a, b, c = -m, 1.0, -(ay - m * ax); lo, hi = "below", "above"  # y - m*x - (...)
        nxt = []
        for poly, labs in cells:
            neg = _clip_halfplane(poly, a, b, c, True)
            pos = _clip_halfplane(poly, a, b, c, False)
            if _poly_area(neg) > 1e-12:
                nxt.append((neg, labs + [f"L{i + 1} {lo}"]))
            if _poly_area(pos) > 1e-12:
                nxt.append((pos, labs + [f"L{i + 1} {hi}"]))
        if nxt:
            cells = nxt

    palette = ["#e23b3b", "#2a7de1", "#27b85e", "#e6a33b", "#9b51e0",
               "#1ab6c4", "#d94f9a", "#7a8a3a"]
    out = []
    for k, (poly, labs) in enumerate(cells):
        out.append(dict(name=" · ".join(labs) if labs else "(all)",
                        color=palette[k % len(palette)],
                        points=[[round(px, 4), round(py, 4)] for px, py in poly]))
    return out


def rank_separating_pairs(df, group_col, value_cols=None, sample_cap=400):
    """For EACH pair of numeric columns, measure how well that pair separates the groups
    of `group_col`. Rank from the pair that best splits the groups to the worst.

    Two readings, because 'separate' has no single definition:
      - Silhouette: geometric separation of the groups in the standardized 2D plane
        (what you SEE in a scatter; -1 to 1, higher = more separated). It is the sort key,
        matching the intent 'which pair shows the groups split apart'.
      - Bal. LDA acc.: BALANCED accuracy of a cross-validated LDA — the classification
        reading ('this pair classifies the group with X%'). Balanced so it isn't fooled
        by unbalanced groups.

    Honest limits: both assume ~linear/convex separation on the standardized axes. A pair
    with a curved boundary may separate well and still score low here."""
    from itertools import combinations
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    if not group_col or group_col not in df.columns:
        raise ValueError("Choose the categorical group column.")

    if not value_cols:
        value_cols = list(df.columns)
    num = []
    for c in value_cols:
        if c == group_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 3:
            num.append(c)
    if len(num) < 2:
        raise ValueError("Need at least 2 valid numeric columns.")

    rows = []
    for a, b in combinations(num, 2):
        sub = df[[a, b, group_col]].copy()
        sub[a] = pd.to_numeric(sub[a], errors="coerce")
        sub[b] = pd.to_numeric(sub[b], errors="coerce")
        sub = sub.dropna()
        y = sub[group_col].astype(str).to_numpy()
        labels, counts = np.unique(y, return_counts=True)
        # silhouette requires 2 <= number of labels <= n-1
        if len(labels) < 2 or sub.shape[0] < len(labels) + 2:
            continue
        X = StandardScaler().fit_transform(sub[[a, b]].to_numpy(float))

        try:
            ss = float(silhouette_score(
                X, y, sample_size=(sample_cap if len(y) > sample_cap else None),
                random_state=0))
        except Exception:
            ss = float("nan")

        acc = float("nan"); acc_lo = acc_hi = float("nan")
        if counts.min() >= 2:
            try:
                k = int(min(5, counts.min()))
                scores = []
                for rep in range(5):                       # repeated CV -> honest CI
                    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=rep)
                    scores.extend(cross_val_score(LinearDiscriminantAnalysis(), X, y,
                                                  cv=cv, scoring="balanced_accuracy"))
                sc = np.asarray(scores, float)
                acc = float(sc.mean())
                acc_lo = float(np.percentile(sc, 2.5)); acc_hi = float(np.percentile(sc, 97.5))
            except Exception:
                pass

        rows.append((a, b, ss, acc, acc_lo, acc_hi, int(sub.shape[0])))

    if not rows:
        raise ValueError("No evaluable pair (insufficient groups after dropping NaN).")

    # sort by silhouette desc; NaN goes to the end
    rows.sort(key=lambda r: (r[2] if r[2] == r[2] else -1e9), reverse=True)
    def _ci(lo, hi):
        return "—" if (lo != lo or hi != hi) else f"{lo:.2f}–{hi:.2f}"
    data = [(a, b, round(ss, 4) if ss == ss else float("nan"),
             round(ac, 4) if ac == ac else float("nan"), _ci(lo, hi), n)
            for (a, b, ss, ac, lo, hi, n) in rows]
    out = pd.DataFrame(data, columns=["Var X", "Var Y", "Silhouette",
                                      "Bal. LDA acc.", "Acc CI95", "N"])
    return out


def pairwise_sig(df, value_col, group_col, alpha=0.05):
    """Welch t-tests between all pairs of groups, corrected by Holm.
    Returns a list [(g1, g2, corrected_p, stars)] of the significant pairs only."""
    from scipy import stats
    from itertools import combinations
    groups = [(str(k), pd.to_numeric(v[value_col], errors="coerce").dropna().values)
              for k, v in df.groupby(group_col)]
    groups = [(n, v) for n, v in groups if len(v) > 1]
    pairs = list(combinations(groups, 2))
    if not pairs:
        return []
    raw = []
    for (n1, a), (n2, b) in pairs:
        _, p = stats.ttest_ind(a, b, equal_var=False)
        raw.append([n1, n2, p])
    # Holm correction
    order = sorted(range(len(raw)), key=lambda i: raw[i][2])
    m = len(raw)
    out = []
    for rank, idx in enumerate(order):
        p_adj = min(1.0, raw[idx][2] * (m - rank))
        raw[idx].append(p_adj)
    for n1, n2, p, p_adj in raw:
        if p_adj <= alpha:
            stars = "***" if p_adj < 1e-3 else "**" if p_adj < 1e-2 else "*"
            out.append((n1, n2, p_adj, stars))
    return out


# ============================ diversity (Shannon) ============================
def _shannon_H(counts):
    c = np.asarray([x for x in counts if x > 0], float)
    if c.sum() <= 0:
        return 0.0, 0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum()), int(len(c))     # H in nats, richness S


def shannon_breakdown(df, group_col, region_series=None):
    """Shannon (diversity) index of the EXPERIMENTAL GROUPS of `group_col`.
    With `region_series`: one row per region (gating), showing the count of each group
    INSIDE the region + the Shannon of that composition. Without regions: the overall
    composition. H in nats; J = Pielou (H/ln S), 0..1; Richness = number of groups present."""
    if not group_col or group_col not in df.columns:
        raise ValueError("Choose the group column (categorical).")
    g = df[group_col].astype(str)
    groups = sorted(g.unique())                       # one column per experimental group

    def row(label, gg):
        vc = gg.value_counts()
        counts = [int(vc.get(k, 0)) for k in groups]
        H, S = _shannon_H(counts)
        J = H / np.log(S) if S > 1 else 0.0
        return [label] + counts + [int(sum(counts)), int(S), round(H, 4) or 0.0, round(J, 4) or 0.0]

    cols = ["Region"] + groups + ["n", "Richness", "H (Shannon)", "J (Pielou)"]
    rows = []
    if region_series is not None:
        reg = pd.Series(np.asarray(region_series, dtype=object), index=df.index)
        for r in sorted(pd.unique(reg[reg.notna()]), key=lambda v: str(v)):
            rows.append(row(str(r), g[reg == r]))
    rows.append(row("— Overall —", g))
    return pd.DataFrame(rows, columns=cols)


def shannon_occupancy(df, group_col, region_series):
    """RELATIVE Shannon: for each GROUP/treatment, the diversity of the REGIONS it occupies.
    Occupying many regions evenly -> high H; concentrating in few -> low H. '(outside)' is
    not a region, so it is excluded. Richness = number of regions occupied by the group."""
    if region_series is None:
        raise ValueError("This view needs regions (gating) + X/Y axes.")
    if not group_col or group_col not in df.columns:
        raise ValueError("Choose the group column (categorical).")
    g = df[group_col].astype(str)
    reg = pd.Series(np.asarray(region_series, dtype=object), index=df.index).astype(str)
    keep = reg != "(outside)"
    g, reg = g[keep], reg[keep]
    regions = sorted(reg.unique())
    groups = sorted(g.unique())
    if not regions:
        raise ValueError("No observation inside regions.")
    ct = pd.crosstab(g, reg).reindex(index=groups, columns=regions, fill_value=0)

    rows = []
    for grp in groups:
        counts = [int(ct.loc[grp, r]) for r in regions]
        H, S = _shannon_H(counts)
        J = H / np.log(S) if S > 1 else 0.0
        rows.append([grp] + counts + [int(sum(counts)), int(S), round(H, 4) or 0.0, round(J, 4) or 0.0])
    cols = ["Treatment"] + regions + ["n", "Regions occ.", "H (Shannon)", "J (Pielou)"]
    return pd.DataFrame(rows, columns=cols)


# ===================== robust comparison statistics ======================
def bh_correct(pvals):
    """Benjamini-Hochberg: p-values -> q-values (FDR). Keeps the input order."""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return np.clip(q, 0.0, 1.0)


def _cap(a, n=2000, seed=0):
    a = np.asarray(a, float)
    if len(a) <= n:
        return a
    return a[np.random.default_rng(seed).choice(len(a), n, replace=False)]


def _cliffs_delta(a, b):
    from scipy.stats import mannwhitneyu
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    try:
        U = float(mannwhitneyu(a, b, alternative="two-sided").statistic)  # #(a>b)+0.5*ties
    except Exception:
        return float("nan")
    return 2.0 * U / (na * nb) - 1.0


def _hedges_g(a, b):
    na, nb = len(a), len(b)
    if na + nb <= 2:
        return float("nan")
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))
    if not np.isfinite(sp) or sp == 0:
        return float("nan")
    d = (np.mean(a) - np.mean(b)) / sp
    J = 1 - 3 / (4 * (na + nb) - 9)
    return d * J


def _boot_ci(fn, a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = _cap(a), _cap(b)
    out = np.empty(n)
    for i in range(n):
        out[i] = fn(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))])
    out = out[np.isfinite(out)]
    if len(out) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _mag(v, cuts, names):
    v = abs(v)
    for c, nm in zip(cuts, names):
        if v < c:
            return nm
    return names[-1]


def robust_compare(df, value_col, group_col):
    """Rigorous group comparison: checks normality, gives both the non-parametric AND the
    parametric test, effect size WITH CI95 (bootstrap), and — for 3+ groups — pairwise
    with Benjamini-Hochberg correction. Returns display-ready text."""
    from scipy import stats as st
    from itertools import combinations
    if not value_col or not group_col:
        raise ValueError("Choose the value column and the group column.")
    d = df[[value_col, group_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna()
    d[group_col] = d[group_col].astype(str)
    arrs = {g: sub[value_col].to_numpy(float) for g, sub in d.groupby(group_col)}
    arrs = {g: a for g, a in arrs.items() if len(a) >= 2}
    groups = list(arrs.keys())
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups with n>=2 after dropping blanks.")

    L = []
    L.append("n per group: " + ", ".join(f"{g}={len(a)}" for g, a in arrs.items()))
    norm = {}
    for g, a in arrs.items():
        norm[g] = float(st.shapiro(a).pvalue) if 3 <= len(a) <= 5000 else float("nan")
    shown = [f"{g}={norm[g]:.2g}" for g in groups if norm[g] == norm[g]]
    if shown:
        L.append("Normality (Shapiro p): " + ", ".join(shown))
    n_ok = all((p > 0.05) for p in norm.values() if p == p) and len(shown) > 0
    L.append("-> " + ("looks normal" if n_ok else "NON-normal distribution -> prefer the non-parametric"))
    L.append("")

    dmag = ([0.147, 0.33, 0.474], ["negligible", "small", "medium", "large"])
    gmag = ([0.2, 0.5, 0.8], ["negligible", "small", "medium", "large"])

    if len(groups) == 2:
        a, b = arrs[groups[0]], arrs[groups[1]]
        L.append(f"Comparing: {groups[0]}  vs  {groups[1]}")
        U, pmw = st.mannwhitneyu(a, b, alternative="two-sided")
        delta = _cliffs_delta(a, b); dlo, dhi = _boot_ci(_cliffs_delta, a, b)
        L.append(f"[Non-parametric] Mann-Whitney U={U:.0f}  p={pmw:.3g}")
        L.append(f"   Cliff's δ = {delta:+.3f}  CI95[{dlo:+.3f}, {dhi:+.3f}]  ({_mag(delta, *dmag)})")
        t, pt = st.ttest_ind(a, b, equal_var=False)
        gg = _hedges_g(a, b); glo, ghi = _boot_ci(_hedges_g, a, b)
        L.append(f"[Parametric] Welch t={t:.3g}  p={pt:.3g}")
        L.append(f"   Hedges g = {gg:+.3f}  CI95[{glo:+.3f}, {ghi:+.3f}]  ({_mag(gg, *gmag)})")
        L.append("")
        L.append("Recommended: " + ("Mann-Whitney + Cliff's δ" if not n_ok
                                     else "either — report the effect with CI, not just the p"))
    else:
        H, pk = st.kruskal(*arrs.values())
        N = sum(len(a) for a in arrs.values()); k = len(groups)
        eps2 = (H - k + 1) / (N - k) if N > k else float("nan")
        L.append(f"[Omnibus] Kruskal-Wallis H={H:.3g}  p={pk:.3g}  ε²={eps2:.3f}")
        try:
            f, pf = st.f_oneway(*arrs.values())
            L.append(f"[Parametric omnibus] ANOVA F={f:.3g}  p={pf:.3g}")
        except Exception:
            pass
        L.append("")
        L.append("Pairs — Mann-Whitney, q = corrected p (Benjamini-Hochberg):")
        pairs, praw = [], []
        for ga, gb in combinations(groups, 2):
            U, pp = st.mannwhitneyu(arrs[ga], arrs[gb], alternative="two-sided")
            pairs.append((ga, gb, pp, _cliffs_delta(arrs[ga], arrs[gb]))); praw.append(pp)
        qs = bh_correct(praw)
        for (ga, gb, pp, dl), qq in zip(pairs, qs):
            star = "*" if qq < 0.05 else " "
            L.append(f"  {ga} vs {gb}:  p={pp:.3g}  q={qq:.3g} {star}  δ={dl:+.2f} ({_mag(dl, *dmag)})")
        L.append("")
        L.append("* q<0.05 after correction. ε² and δ are effect sizes (independent of n).")

    L.append("")
    L.append("Note: each row counts as 1 independent observation. If the cells come from")
    L.append("wells/replicates, aggregate by replicate first (avoids pseudo-replication).")
    return "\n".join(L)
