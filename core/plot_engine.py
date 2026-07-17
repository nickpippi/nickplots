"""Plotting engine: validates, applies THEME (coherent background+text), legend, layers and export."""
from __future__ import annotations
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .data_loader import column_kinds, CATEGORY
from .plot_registry import REGISTRY, PlotSpec


class PlotConfigError(Exception):
    """Friendly configuration error."""


# Themes: each defines seaborn base + figure/axes background + text/axis color + grid.
THEMES = {
    "Light · grid":   dict(base="whitegrid", fig="#ffffff", ax="#ffffff", fg="#222222", grid="#d0d0d0"),
    "Light · ticks":  dict(base="ticks",     fig="#ffffff", ax="#ffffff", fg="#222222", grid=None),
    "Light · clean":  dict(base="white",     fig="#ffffff", ax="#ffffff", fg="#222222", grid=None),
    "Dark · grid":    dict(base="darkgrid",  fig="#1e1e2e", ax="#181825", fg="#e6e6e6", grid="#45475a"),
    "Dark · plain":   dict(base="dark",      fig="#15171c", ax="#1b1e26", fg="#e6e6e6", grid=None),
    "Slate":          dict(base="darkgrid",  fig="#22272e", ax="#2d333b", fg="#e8e8e8", grid="#444c56"),
}
DEFAULT_THEME = "Light · grid"


@dataclass
class Style:
    theme: str = DEFAULT_THEME
    context: str = "notebook"        # paper | notebook | talk | poster
    font_scale: float = 1.0
    fig_bg: str | None = None        # optional theme override
    ax_bg: str | None = None
    grid: bool = True
    logx: bool = False
    logy: bool = False
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None
    despine: bool = False
    tick_size: float = 0.0           # 0 = default
    fig_w_mm: float | None = None    # if both set, they override the size
    fig_h_mm: float | None = None
    colors: dict | None = None       # {category: color} hand-picked per-group colors
    show_n: bool = False             # append (n=k) to categorical x tick labels


def _apply_axes(ax, st):
    if st.logx:
        ax.set_xscale("log")
    if st.logy:
        ax.set_yscale("log")
    if st.xmin is not None or st.xmax is not None:
        ax.set_xlim(left=st.xmin, right=st.xmax)
    if st.ymin is not None or st.ymax is not None:
        ax.set_ylim(bottom=st.ymin, top=st.ymax)
    if st.tick_size and st.tick_size > 0:
        ax.tick_params(labelsize=st.tick_size)
    if st.despine:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)


@dataclass
class Legend:
    show: bool = True
    pos: str = "upper right"         # named loc | "outside" | "free"
    x: float = 0.74                  # in pos="free": FIGURE fraction (legend lower-left corner)
    y: float = 0.72
    fontsize: float = 9.0
    ncol: int = 1
    frame: bool = True
    fit: bool = True                 # free: open space (outside) vs float freely (inside)
    relabel: dict | None = None      # {original_label: new_name}
    title: str | None = None         # legend title (None = default = hue name)


def validate(spec: PlotSpec, df, mapping: dict) -> None:
    kinds = column_kinds(df)
    for ch in spec.channels:
        col = mapping.get(ch.name)
        if ch.required and not col:
            raise PlotConfigError(f"Plot '{spec.label}' requires the '{ch.nice()}' channel.")
        if col:
            if col not in df.columns:
                raise PlotConfigError(f"Column '{col}' does not exist in the data.")
            if kinds[col] not in ch.accepts:
                raise PlotConfigError(
                    f"'{ch.nice()}' of {spec.label} expects a {'/'.join(ch.accepts)} column, "
                    f"but '{col}' is {kinds[col]}.")


def _fill(spec, params):
    out = {p.name: p.default for p in spec.params}
    out.update(params or {})
    return out


class Layer:
    def __init__(self, spec_key, mapping, params=None):
        self.spec = REGISTRY[spec_key]
        self.mapping = mapping
        self.params = _fill(self.spec, params)


def _apply_theme(fig, ax, style: Style):
    t = THEMES.get(style.theme, THEMES[DEFAULT_THEME])
    fg = t["fg"]
    fig.patch.set_facecolor(style.fig_bg or t["fig"])
    ax.set_facecolor(style.ax_bg or t["ax"])
    ax.title.set_color(fg)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.tick_params(colors=fg)
    for sp in ax.spines.values():
        sp.set_edgecolor(fg)
    if style.grid and t["grid"]:
        ax.grid(True, color=t["grid"], linewidth=0.6)
    else:
        ax.grid(style.grid)
    return t, fg


def _is_legend_header(h):
    """Seaborn sub-header (column name): a handle with no drawn marker."""
    try:
        mk = h.get_marker()
        ms = h.get_markersize() or 0
        return (mk in ("", "None", None)) and ms == 0
    except Exception:
        return False


def _section_columns(handles, labels):
    """Reorganize a combined legend (several sub-headered sections) so each section
    fills a whole column, with its title on top. Returns (handles, labels, ncol).
    With a single section, returns (None ncol) and leaves it untouched."""
    from matplotlib.lines import Line2D
    heads = [i for i, h in enumerate(handles) if _is_legend_header(h)]
    if len(heads) < 2:
        return handles, labels, None                      # 1 section: default reflow is fine
    bounds = heads + [len(handles)]
    sections = [(bounds[k], bounds[k + 1]) for k in range(len(heads))]
    rows = max(e - s for s, e in sections)                # height = largest section
    out_h, out_l = [], []
    for s, e in sections:
        seg_h, seg_l = list(handles[s:e]), list(labels[s:e])
        for _ in range(rows - (e - s)):                   # pad with empty entries
            seg_h.append(Line2D([], [], linestyle="none", marker=""))
            seg_l.append("")
        out_h += seg_h
        out_l += seg_l
    return out_h, out_l, len(sections)                    # column-major: 1 column/section


def _strip_auto_legend(ax):
    """Remove the legend seaborn draws itself when plotting with hue (scatter/line/
    bar/box/violin/hist/kde...) BEFORE tight_layout/subplots_adjust: otherwise the layout
    reserves fixed space for it and, since it's always replaced by _apply_legend (which
    actually decides show/position), that space never returns to the axes -- even with the
    legend hidden. Returns (title, handles, labels) captured from it: distribution plots
    (histplot/kdeplot/ecdfplot) build the legend 'by hand' and don't expose the handles
    afterwards via ax.get_legend_handles_labels() -- without this the legend never
    reappears on those plots."""
    old = ax.get_legend()
    if old is None:
        return "", [], []
    title = old.get_title().get_text()
    handles = list(old.legend_handles)
    labels = [t.get_text() for t in old.get_texts()]
    old.remove()
    return title, handles, labels


def _apply_legend(fig, ax, lg: Legend, fg, theme, aliases=None, auto_title="",
                   auto_handles=None, auto_labels=None):
    aliases = aliases or {}
    handles, labels = ax.get_legend_handles_labels()
    if not handles and auto_handles:      # fallback: distribution plots (see _strip_auto_legend)
        handles, labels = list(auto_handles), list(auto_labels or [])
    if not lg.show or not handles:
        return None
    # column alias (hue/size sub-headers) + explicit item relabel; relabel wins.
    relabel = {**aliases, **(lg.relabel or {})}
    if relabel:
        labels = [relabel.get(str(l), l) for l in labels]
    kw = dict(fontsize=lg.fontsize, ncol=max(1, int(lg.ncol)), frameon=lg.frame)
    title = lg.title or aliases.get(auto_title)     # alias the hue title when present
    if title:
        kw["title"] = title

    # Combined legend (hue + size/style): seaborn emits sub-headers (column name, no
    # marker). At ncol>1 matplotlib reflows the flattened list column-major and SPLITS
    # the sections -- the size title in one column, its items in another. Fix: detect the
    # sections and give each a whole column (nrows = largest section, padding the rest
    # with blank entries) so each title sits at the top of its own column.
    if kw["ncol"] > 1:
        handles, labels, ncol_fix = _section_columns(handles, labels)
        if ncol_fix:
            kw["ncol"] = ncol_fix

    if lg.pos == "free":
        leg = ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(lg.x, lg.y),
                        bbox_transform=fig.transFigure, **kw)
    elif lg.pos == "outside":
        leg = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), **kw)
    else:
        leg = ax.legend(handles, labels, loc=lg.pos, **kw)
    for txt in leg.get_texts():
        txt.set_color(fg)
    if leg.get_title():
        leg.get_title().set_color(fg)
    if lg.frame:
        leg.get_frame().set_facecolor(theme["ax"])
        leg.get_frame().set_edgecolor(fg)
    return leg


def _renderer(fig):
    """A renderer able to measure artists, even on a figure with no attached canvas."""
    canvas = fig.canvas
    if not hasattr(canvas, "get_renderer"):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return canvas.get_renderer()


def _legend_fig_box(fig, leg):
    """Legend bounding box in figure-fraction coordinates."""
    bb = leg.get_window_extent(_renderer(fig))
    inv = fig.transFigure.inverted()
    (x0, y0) = inv.transform((bb.x0, bb.y0))
    (x1, y1) = inv.transform((bb.x1, bb.y1))
    return x0, y0, x1, y1


def _fit_outside_legend(fig, ax, leg, pad=0.015):
    """Shrink the axes until an 'outside' legend fits inside the canvas.

    A fixed subplots_adjust(right=...) is not enough: the legend's width is set in
    points, so at small figure sizes (e.g. an exact 85 mm journal column) it takes a
    much larger fraction of the canvas and gets clipped. Measure it and give it exactly
    the room it needs, iterating because shrinking the axes moves the legend."""
    for _ in range(5):
        try:
            x0, y0, x1, y1 = _legend_fig_box(fig, leg)
        except Exception:
            return
        over = x1 - (1.0 - pad)
        if over <= 1e-3:
            return
        pos = ax.get_position()
        new_x1 = max(pos.x0 + 0.15, pos.x1 - over)   # never collapse the axes entirely
        if abs(new_x1 - pos.x1) < 1e-4:
            return
        ax.set_position([pos.x0, pos.y0, new_x1 - pos.x0, pos.height])


def _fit_free_legend(fig, ax, leg, pad=0.02):
    """Shrink the axes to 'open space' wherever the (free-mode) legend sits, so it is
    never clipped -- as if it were positioned on the outside."""
    try:
        canvas = fig.canvas
        if not hasattr(canvas, "get_renderer"):
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            canvas = FigureCanvasAgg(fig)   # ensure a renderer to measure the legend
        canvas.draw()
        rend = canvas.get_renderer()
        lb = leg.get_window_extent(rend)
    except Exception:
        return
    inv = fig.transFigure.inverted()
    (lx0, ly0) = inv.transform((lb.x0, lb.y0))
    (lx1, ly1) = inv.transform((lb.x1, lb.y1))
    pos = ax.get_position()
    if not (lx0 < pos.x1 and lx1 > pos.x0 and ly0 < pos.y1 and ly1 > pos.y0):
        return  # legend does not overlap the axes -> already enough space
    pushes = {"right": pos.x1 - lx0, "left": lx1 - pos.x0,
              "top": pos.y1 - ly0, "bottom": ly1 - pos.y0}
    side = min(pushes, key=pushes.get)              # push from the cheapest side
    x0, y0, x1, y1 = pos.x0, pos.y0, pos.x1, pos.y1
    if side == "right":
        x1 = max(x0 + 0.2, lx0 - pad)
    elif side == "left":
        x0 = min(x1 - 0.2, lx1 + pad)
    elif side == "top":
        y1 = max(y0 + 0.2, ly0 - pad)
    else:
        y0 = min(y1 - 0.2, ly1 + pad)
    ax.set_position([x0, y0, x1 - x0, y1 - y0])


def _draw_gates(ax, gates):
    """Named regions (closed polygons with straight edges)."""
    import numpy as np
    from matplotlib.patches import Polygon
    for i, g in enumerate(gates or []):
        pts = g.get("points") or []
        if len(pts) < 3:
            continue
        col = g.get("color") or "#e23b3b"
        ax.add_patch(Polygon(np.asarray(pts, float), closed=True, fill=False,
                             edgecolor=col, lw=1.4, zorder=3.5))
        name = g.get("name")
        if name:
            lp = g.get("labelxy")
            if lp and len(lp) == 2:
                cx, cy = float(lp[0]), float(lp[1])
            else:
                c = np.asarray(pts, float).mean(axis=0); cx, cy = c[0], c[1]
            ax.text(cx, cy, name, color=col, fontsize=9, fontweight="bold",
                    ha="center", va="center", zorder=3.6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, alpha=0.75))


def _draw_threshold_lines(ax, lines):
    """Infinite threshold lines (clipped by the axes). angle in degrees on the data
    scale: 0=horizontal, 90=vertical, otherwise via slope=tan(angle)."""
    import math
    for ln in lines or []:
        try:
            ang = float(ln.get("angle", 0.0)) % 180.0
            x0 = float(ln.get("x", 0.0)); y0 = float(ln.get("y", 0.0))
        except (TypeError, ValueError):
            continue
        color = ln.get("color") or "#e23b3b"
        if abs(ang - 90.0) < 1e-9:
            ax.axvline(x0, color=color, ls="--", lw=1.3, zorder=3)
        else:
            ax.axline((x0, y0), slope=math.tan(math.radians(ang)),
                      color=color, ls="--", lw=1.3, zorder=3)


def _append_group_n(ax, df, xcol):
    """Append (n=k) to each categorical x tick label, k = rows in that category."""
    counts = df[xcol].astype(str).value_counts()
    labs = ax.get_xticklabels()
    if not labs:
        return
    new = []
    for tl in labs:
        key = tl.get_text()
        n = counts.get(key)
        new.append(f"{key}\n(n={int(n)})" if n is not None else key)
    ax.set_xticks(ax.get_xticks())            # pin ticks before relabel (avoids warning)
    ax.set_xticklabels(new)


def _draw_annotations(ax, anns):
    for a in anns or []:
        x, y, txt = a.get("x", 0.5), a.get("y", 0.5), a.get("text", "")
        kind = a.get("kind", "text")
        if kind == "star":
            # mathtext asterisk: DejaVu's U+2731 glyph renders as a chunky propeller
            # ("little airplane"); $\ast$ uses the bundled math font and is a clean star.
            ax.text(x, y, r"$\ast$", transform=ax.transAxes, fontsize=18, ha="center",
                    va="center", color=a.get("color", "#d33"))
        elif kind == "arrow":
            ax.annotate(txt, xy=(x, y), xytext=(x - 0.13, y + 0.13),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#333", lw=1.2), fontsize=9)
        else:
            ax.text(x, y, txt, transform=ax.transAxes, fontsize=10, ha="left", va="center")


def render(layers, df, *, fig=None, figsize=(7, 5), dpi=110, title="", xlabel="", ylabel="",
           style: Style | None = None, legend: Legend | None = None, annotations=None,
           aliases=None, lines=None, gates=None):
    if not layers:
        raise PlotConfigError("No layer to draw.")
    fig_level = [l for l in layers if not l.spec.layerable]
    if fig_level and len(layers) > 1:
        raise PlotConfigError(
            f"'{fig_level[0].spec.label}' builds its own figure and does not stack. Use it alone.")

    style = style or Style()
    legend = legend or Legend()
    t = THEMES.get(style.theme, THEMES[DEFAULT_THEME])
    sns.set_theme(style=t["base"], context=style.context, font_scale=style.font_scale)
    plt.rcParams.update({  # ensure legible text before creating the artists
        "text.color": t["fg"], "axes.labelcolor": t["fg"], "axes.titlecolor": t["fg"],
        "xtick.color": t["fg"], "ytick.color": t["fg"], "axes.edgecolor": t["fg"],
    })

    if style.fig_w_mm and style.fig_h_mm:           # exact size (mm) for publication
        figsize = (style.fig_w_mm / 25.4, style.fig_h_mm / 25.4)
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig.set_size_inches(*figsize)
        fig.clear()
        ax = fig.add_subplot(111)
    for layer in layers:
        validate(layer.spec, df, layer.mapping)
        if style.colors:                            # hand-picked per-group colors
            layer.params["__colors__"] = style.colors
        layer.spec.render(ax, df, layer.mapping, layer.params)

    auto_title, auto_handles, auto_labels = _strip_auto_legend(ax)  # strip BEFORE layout
    theme, fg = _apply_theme(fig, ax, style)
    _apply_axes(ax, style)
    aliases = aliases or {}
    disp = lambda c: aliases.get(c, c)
    base = layers[0].mapping
    ax.set_title(title or layers[0].spec.label, color=fg)
    # An explicit label always wins, even on plots with no such channel (a histogram
    # has no y channel but still deserves the Y label the user typed).
    if xlabel or base.get("x"):
        ax.set_xlabel(xlabel or disp(base["x"]), color=fg)
    if ylabel or base.get("y"):
        ax.set_ylabel(ylabel or disp(base["y"]), color=fg)
    if style.show_n and base.get("x") and column_kinds(df).get(base["x"]) == CATEGORY:
        _append_group_n(ax, df, base["x"])

    will_legend = legend.show and bool(ax.get_legend_handles_labels()[0] or auto_handles)
    if legend.pos == "outside" and will_legend:
        fig.subplots_adjust(left=0.12, right=0.78, bottom=0.12, top=0.9)
    else:
        fig.tight_layout()
    leg = _apply_legend(fig, ax, legend, fg, theme, aliases, auto_title, auto_handles, auto_labels)
    if leg is not None:
        if legend.pos == "outside":
            _fit_outside_legend(fig, ax, leg)      # never clip it, whatever the figure size
        elif legend.pos == "free" and legend.fit:
            _fit_free_legend(fig, ax, leg)
    _xl, _yl = ax.get_xlim(), ax.get_ylim()      # lock the view before the overlays
    _draw_gates(ax, gates)
    _draw_threshold_lines(ax, lines)
    _draw_annotations(ax, annotations)
    ax.set_xlim(_xl); ax.set_ylim(_yl)            # polygons/lines/annotations don't shift it
    return fig


def _ratios(values, n):
    """Sanitize a list of grid ratios: n positive numbers, or None for an even grid."""
    try:
        out = [max(0.05, float(v)) for v in (values or [])]
    except (TypeError, ValueError):
        return None
    if len(out) != n:
        return None
    return out


def render_panel(items, dfs, *, fig=None, ncols=2, figsize=(10, 7), dpi=110, style=None, aliases=None,
                 width_ratios=None, height_ratios=None):
    """Multi-figure panel (A/B/C): each item is a layer drawn in its own subplot, from its
    OWN dataframe (dfs[i]) -- so frames can come from different CSVs.
    item = {spec_key, mapping, params, title, xlabel, ylabel, annotations, lines, gates}.
    The title/axis labels typed for that plot are carried over, as are its annotations,
    threshold lines and regions (drawn in that frame's own coordinates, so they scale with
    it). width_ratios/height_ratios size the grid cells."""
    import string
    if not items:
        raise PlotConfigError("No panel to build.")
    style = style or Style()
    aliases = aliases or {}
    disp = lambda c: aliases.get(c, c)
    t = THEMES.get(style.theme, THEMES[DEFAULT_THEME])
    sns.set_theme(style=t["base"], context=style.context, font_scale=style.font_scale)
    n = len(items)
    nrows = -(-n // ncols)
    if style.fig_w_mm and style.fig_h_mm:
        figsize = (style.fig_w_mm / 25.4, style.fig_h_mm / 25.4)
    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    else:
        fig.set_size_inches(*figsize)
        fig.clear()
    gs = fig.add_gridspec(nrows, ncols,
                          width_ratios=_ratios(width_ratios, ncols),
                          height_ratios=_ratios(height_ratios, nrows))
    for i, it in enumerate(items):
        df = dfs[i]
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        lay = Layer(it["spec_key"], it["mapping"], it.get("params"))
        validate(lay.spec, df, lay.mapping)
        if style.colors:
            lay.params["__colors__"] = style.colors
        lay.spec.render(ax, df, lay.mapping, lay.params)
        _strip_auto_legend(ax)
        _apply_theme(fig, ax, style)
        _apply_axes(ax, style)
        ax.set_title(it.get("title") or f"({string.ascii_uppercase[i]})", color=t["fg"], loc="left", fontweight="bold")
        xl, yl = it.get("xlabel"), it.get("ylabel")
        if xl or lay.mapping.get("x"):
            ax.set_xlabel(xl or disp(lay.mapping["x"]), color=t["fg"])
        if yl or lay.mapping.get("y"):
            ax.set_ylabel(yl or disp(lay.mapping["y"]), color=t["fg"])
        _lx, _ly = ax.get_xlim(), ax.get_ylim()      # overlays must not shift the view
        _draw_gates(ax, it.get("gates"))
        _draw_threshold_lines(ax, it.get("lines"))
        _draw_annotations(ax, it.get("annotations"))
        ax.set_xlim(_lx); ax.set_ylim(_ly)
    fig.patch.set_facecolor(style.fig_bg or t["fig"])
    fig.tight_layout()
    return fig


def axes_rects(fig):
    """Each axes' rectangle in figure fraction (x0, y0, x1, y1), lower-left origin."""
    out = []
    for ax in fig.axes:
        p = ax.get_position()
        out.append([float(p.x0), float(p.y0), float(p.x1), float(p.y1)])
    return out


def is_dark_theme(theme_name):
    """True when the theme paints the figure on a dark background."""
    import matplotlib.colors as mcolors
    t = THEMES.get(theme_name, THEMES[DEFAULT_THEME])
    r, g, b = mcolors.to_rgb(t["fig"])
    return (0.299 * r + 0.587 * g + 0.114 * b) < 0.5


def transparent_ink(theme_name):
    """Foreground color to use when exporting with a transparent background.

    Dropping a dark theme's background leaves its near-white text, ticks and spines
    invisible on any light page -- they are exported, just unreadable. So for dark
    themes we re-ink the foreground with the light theme's color. Light themes are
    already readable and stay untouched."""
    return THEMES[DEFAULT_THEME]["fg"] if is_dark_theme(theme_name) else None


def _recolor_foreground(fig, color):
    """Re-color every foreground artist (title, labels, ticks, spines, legend, texts)."""
    for ax in fig.axes:
        ax.title.set_color(color)
        ax.xaxis.label.set_color(color)
        ax.yaxis.label.set_color(color)
        ax.tick_params(colors=color, which="both")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(color)
        for sp in ax.spines.values():
            sp.set_edgecolor(color)
        for txt in ax.texts:
            txt.set_color(color)
        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_color(color)
            if leg.get_title():
                leg.get_title().set_color(color)
            leg.get_frame().set_facecolor("none")     # keep the legend box see-through too
            leg.get_frame().set_edgecolor(color)
    for txt in fig.texts:
        txt.set_color(color)


def export(fig, path, dpi=300, transparent=False, ink=None, tight=True):
    """Write the figure. transparent=True drops only the backgrounds, keeping every
    drawn element. `ink` re-colors the foreground so it stays readable (see
    transparent_ink). tight=False preserves an exact figure size (mm publication size),
    since bbox_inches='tight' would crop and resize it."""
    fmt = path.rsplit(".", 1)[-1].lower()
    if transparent and ink:
        _recolor_foreground(fig, ink)
    kw = dict(dpi=dpi, edgecolor="none", transparent=transparent,
              facecolor="none" if transparent else fig.get_facecolor())
    if tight:
        kw["bbox_inches"] = "tight"
    if fmt in ("tif", "tiff"):
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(path, **kw)
