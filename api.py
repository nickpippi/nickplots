"""JS <-> Python bridge for the PyWebView front end.
Reuses core/ unchanged. matplotlib renders the figure and returns base64 PNG for the
preview; export writes the real file (TIFF/PNG/PDF/SVG) at 300 dpi.
"""
from __future__ import annotations
import base64
import io
import json
import os
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import webview
from core import data_loader as DL
from core import plot_engine as E
from core import stats as S
from core import figures as F
from core import codegen as CG
from core.plot_engine import Layer, Style, Legend, THEMES
from core.plot_registry import REGISTRY

DPI = 110

# Colour-vision-deficiency simulation matrices (sRGB approximation) + grayscale.
_CVD = {
    "deuteranopia": [[0.625, 0.375, 0.0], [0.70, 0.30, 0.0], [0.0, 0.30, 0.70]],
    "protanopia":   [[0.567, 0.433, 0.0], [0.558, 0.442, 0.0], [0.0, 0.242, 0.758]],
    "tritanopia":   [[0.95, 0.05, 0.0], [0.0, 0.433, 0.567], [0.0, 0.475, 0.525]],
}


def _simulate_cvd(png, mode):
    """Transform a rendered PNG to preview it as a colour-blind reader (or grayscale)
    sees it — the check journals increasingly require."""
    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    a = np.asarray(im).astype(float)
    rgb = a[..., :3]
    if mode == "grayscale":
        g = rgb @ np.array([0.299, 0.587, 0.114])
        rgb = np.stack([g, g, g], axis=-1)
    elif mode in _CVD:
        rgb = rgb @ np.asarray(_CVD[mode]).T
    else:
        return png
    a[..., :3] = np.clip(rgb, 0, 255)
    out = io.BytesIO()
    Image.fromarray(a.astype("uint8"), "RGBA").save(out, "PNG")
    return out.getvalue()


def _catalog():
    out = []
    for s in REGISTRY.values():
        out.append(dict(
            key=s.key, label=s.label, level=s.level, pickable=s.pickable,
            channels=[dict(name=c.name, required=c.required, accepts=list(c.accepts),
                           label=c.nice()) for c in s.channels],
            params=[dict(name=p.name, kind=p.kind, default=p.default, lo=p.lo, hi=p.hi,
                         choices=list(p.choices), label=p.nice()) for p in s.params]))
    return out


class Api:
    def __init__(self):
        self._df = self._view = None
        self._fig = Figure(dpi=DPI)
        self._gate_pos = {"x": None, "y": None}   # categorical-axis position for gating
        self._window = None
        self._pick = None
        self._Hpx = 0
        self._csv_path = None
        self._xlsx = None
        self._panel_data = {}     # id -> dataframe snapshot, one per panel frame
        self._panel_seq = 0
        self._datasets = {}       # name -> loaded dataframe (multi-dataset manager)
        self._dataset_paths = {}  # name -> file it was loaded from (None if dropped as text)
        self._combined_col = None  # the tag column of the last "Combine all", for projects
        self._active = None

    def _win(self):
        return self._window or (webview.windows[0] if webview.windows else None)

    # -------------------------------- boot --------------------------------- #
    def boot(self):
        return dict(catalog=_catalog(), themes=list(THEMES.keys()))

    def _info(self, path=None, error=None, extra=None):
        d = dict(path=path, n=int(len(self._view)), cols=list(self._view.columns),
                 kinds=DL.column_kinds(self._view),
                 numeric=DL.columns_by_kind(self._view, DL.NUMBER),
                 category=DL.columns_by_kind(self._view, DL.CATEGORY), error=error,
                 datasets=list(self._datasets.keys()), active=self._active)
        if extra:
            d.update(extra)
        return d

    # -------------------------------- data --------------------------------- #

    def open_csv(self):
        res = self._win().create_file_dialog(
            webview.OPEN_DIALOG, file_types=("CSV (*.csv)", "All files (*.*)"))
        if not res:
            return None
        path = res[0]
        try:
            self._df = DL.load_csv(path)
        except Exception as e:
            return dict(error=f"Failed to load: {e}")
        self._view = self._df
        self._csv_path = path
        self._xlsx = None
        name = os.path.basename(path); base = name; i = 2
        while name in self._datasets:                 # keep every loaded file distinct
            name = f"{base} ({i})"; i += 1
        self._datasets[name] = self._df
        self._dataset_paths[name] = path
        self._active = name
        return self._info(path)

    def list_datasets(self):
        return dict(datasets=list(self._datasets.keys()), active=self._active)

    def activate_dataset(self, name):
        """Switch the working data to a previously loaded dataset (as originally loaded)."""
        if name not in self._datasets:
            return None
        self._df = self._view = self._datasets[name]
        self._active = name
        self._csv_path = None
        return self._info(name)

    def remove_dataset(self, name):
        self._datasets.pop(name, None)
        self._dataset_paths.pop(name, None)
        if self._active == name:
            if self._datasets:
                self._active = next(iter(self._datasets))
                self._df = self._view = self._datasets[self._active]
            else:
                self._active = self._df = self._view = None
                return dict(datasets=[], active=None, cleared=True)
        return self._info(self._active)

    def combine_datasets(self, colname="dataset"):
        """Concatenate every loaded dataset into one, tagging each row with its source
        file in a new column — for cross-file comparison and SuperPlots by replicate."""
        if len(self._datasets) < 2:
            return dict(error="Load at least 2 datasets to combine.")
        col = (colname or "dataset").strip() or "dataset"
        frames = [df.assign(**{col: name}) for name, df in self._datasets.items()]
        self._df = self._view = pd.concat(frames, ignore_index=True)
        self._active = "(combined)"
        self._combined_col = col           # so a project can redo the combine on open
        self._csv_path = None
        return self._info("(combined)")

    def preview_rows(self, n=8):
        """First rows + shape + detected types, for the on-load data preview."""
        if self._view is None:
            return None
        head = self._view.head(int(n))
        return dict(columns=[str(c) for c in head.columns],
                    rows=[["" if pd.isna(v) else str(v) for v in row] for row in head.values],
                    kinds=DL.column_kinds(self._view),
                    n=int(len(self._view)), ncols=int(len(self._view.columns)))

    @staticmethod
    def _detect_id(df):
        """A likely replicate/observation-id column (name-based + repeats but not constant)."""
        import re
        pat = re.compile(r'(?:^|[_\s])(id|cell|track|well|replicate|rep|sample|subject|roi)s?(?:$|[_\s])', re.I)
        for c in df.columns:
            if pat.search(str(c)) and 1 < df[c].nunique(dropna=True) < len(df):
                return str(c)
        return None

    def recommend_plots(self):
        """Suggest plots from the column types (categorical vs numeric, and whether the
        data looks nested inside replicates -> SuperPlot to avoid pseudo-replication)."""
        if self._view is None:
            return None
        df = self._view
        num = DL.columns_by_kind(df, DL.NUMBER)
        cat = [c for c in DL.columns_by_kind(df, DL.CATEGORY)
               if 2 <= df[c].nunique(dropna=True) <= 20]
        idcol = self._detect_id(df)
        recs = []
        for c in cat[:2]:                          # a numeric value across groups
            for n in [x for x in num if x != idcol][:3]:
                r = dict(title=f"{n} by {c}", x=c, y=n, plots=["box_points", "violin_points", "ecdf"])
                if idcol:
                    r["plots"] = ["superplot", "box_points", "violin_points"]
                    r["rep"] = idcol
                    r["note"] = ("Multiple observations nested within replicates "
                                 f"('{idcol}') — a SuperPlot tests replicate means, avoiding "
                                 "pseudo-replication.")
                recs.append(r)
        for i in range(min(len(num), 3)):          # numeric vs numeric
            for j in range(i + 1, min(len(num), 3)):
                if idcol in (num[i], num[j]):
                    continue
                recs.append(dict(title=f"{num[i]} vs {num[j]}", x=num[i], y=num[j],
                                 plots=["scatter", "regband"]))
        return dict(recs=recs[:8], id_col=idcol)

    def load_csv_text(self, name, text):
        """Load a CSV from its raw text (drag-and-drop / clipboard) — no file path."""
        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            df.columns = [str(c).strip() for c in df.columns]
        except Exception as e:
            return dict(error=f"Failed to load: {e}")
        if df.empty:
            return dict(error="The dropped file has no rows.")
        self._df = self._view = df
        self._csv_path = None
        self._xlsx = None
        name = name or "dropped.csv"
        base = name; i = 2
        while name in self._datasets:
            name = f"{base} ({i})"; i += 1
        self._datasets[name] = self._df
        self._dataset_paths[name] = None      # dropped as text: nothing to reopen later
        self._active = name
        return self._info(name)

    def set_filter(self, expr):
        if self._df is None:
            return None
        self._view, err = DL.apply_filter(self._df, expr or "")
        return self._info(error=err)
    # ---------------------------- Excel + reshape -------------------------- #
    def open_excel(self):
        res = self._win().create_file_dialog(
            webview.OPEN_DIALOG, file_types=("Excel (*.xlsx;*.xls)", "All files (*.*)"))
        if not res:
            return None
        self._xlsx = res[0]
        try:
            return dict(path=res[0], sheets=DL.excel_sheets(res[0]))
        except Exception as e:
            return dict(error=f"Failed to read Excel: {e}")

    def choose_sheet(self, sheet):
        if not self._xlsx:
            return None
        try:
            self._df = DL.load_excel(self._xlsx, sheet)
        except Exception as e:
            return dict(error=str(e))
        self._view = self._df
        self._csv_path = None
        return self._info(self._xlsx)

    def reshape_melt(self, id_vars, value_vars, var_name, value_name):
        if self._view is None:
            return None
        try:
            self._df = DL.reshape_melt(self._view, id_vars, value_vars,
                                       var_name or "variable", value_name or "value")
        except Exception as e:
            return dict(error=str(e))
        self._view = self._df
        return self._info()

    def reshape_pivot(self, index, columns, values):
        if self._view is None:
            return None
        try:
            self._df = DL.reshape_pivot(self._view, index, columns, values)
        except Exception as e:
            return dict(error=str(e))
        self._view = self._df
        return self._info()

    def export_csv(self):
        if self._view is None:
            return None
        path = self._win().create_file_dialog(
            webview.SAVE_DIALOG, save_filename="filtered_data.csv", file_types=("CSV (*.csv)",))
        if not path:
            return None
        path = path if isinstance(path, str) else path[0]
        DL.export_filtered(self._view, path)
        return path

    # ------------------------------- styles -------------------------------- #
    def _f(self, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _style(self, d):
        return Style(theme=d["theme"], context=d["context"], font_scale=float(d["font_scale"]),
                     grid=bool(d["grid"]), fig_bg=d.get("fig_bg") or None, ax_bg=d.get("ax_bg") or None,
                     logx=bool(d.get("logx")), logy=bool(d.get("logy")),
                     xmin=self._f(d.get("xmin")), xmax=self._f(d.get("xmax")),
                     ymin=self._f(d.get("ymin")), ymax=self._f(d.get("ymax")),
                     despine=bool(d.get("despine")), tick_size=float(d.get("tick_size") or 0),
                     fig_w_mm=self._f(d.get("fig_w_mm")), fig_h_mm=self._f(d.get("fig_h_mm")),
                     colors=(d.get("colors") or None), show_n=bool(d.get("show_n")),
                     font_family=(d.get("font_family") or None), base_pt=self._f(d.get("base_pt")),
                     line_pt=self._f(d.get("line_pt")))

    def _legend(self, d):
        return Legend(show=bool(d["show"]), pos=d["pos"], x=float(d["x"]), y=float(d["y"]),
                      fontsize=float(d["fontsize"]), ncol=int(d["ncol"]), frame=bool(d["frame"]),
                      fit=bool(d.get("fit", True)), relabel=(d.get("relabel") or None),
                      title=(d.get("title") or None))

    def set_manual_data(self, columns, rows, decimal_comma=False):
        cols = [str(c).strip() or f"col{i+1}" for i, c in enumerate(columns)]
        if not cols:
            return dict(error="Define at least one column.")
        try:
            df = pd.DataFrame(rows, columns=cols)
        except Exception as e:
            return dict(error=str(e))
        df = df.replace("", pd.NA)
        if decimal_comma:                                # "0,5"->0.5 ; "1.234,5"->1234.5 ; "1,62288E+16"
            import re
            pat = re.compile(r'^-?\d{1,3}(?:\.\d{3})*,\d+(?:[eE][+-]?\d+)?$'
                             r'|^-?\d+,\d+(?:[eE][+-]?\d+)?$')

            def fix(v):
                if isinstance(v, str):
                    s = v.strip()
                    if pat.match(s):
                        return s.replace('.', '').replace(',', '.')
                return v
            for c in df.columns:
                df[c] = df[c].map(fix)
        for c in df.columns:                             # convert to numeric when it makes sense
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() >= max(1, int(0.6 * len(df))):
                df[c] = conv
        df = df.dropna(how="all").reset_index(drop=True)
        if df.empty:
            return dict(error="Empty table.")
        self._df = self._view = df
        self._csv_path = None
        self._xlsx = None
        return self._info("manual data")

    def _layers(self, state):
        return [Layer(l["spec_key"], l["mapping"], l["params"]) for l in state["layers"]]

    def _explain_error(self, msg):
        """Turn an engine error into {error(title), why, tips[]}: what went wrong, why this
        plot needs it, and concrete next steps (with the columns actually available)."""
        import re
        view = self._view
        has_cat = bool(view is not None and DL.columns_by_kind(view, DL.CATEGORY))
        num_cols = DL.columns_by_kind(view, DL.NUMBER) if view is not None else []
        d = dict(error=msg, why=None, tips=[])
        m = re.search(r"requires the '([^']+)'", msg)
        if m:                                       # a required channel is empty
            ch = m.group(1)
            d["error"] = f"This plot needs a {ch}."
            d["why"] = f"It maps '{ch}' to build the figure, and that box is empty."
            if ch.lower() in ("x", "hue", "group", "condition") and not has_cat:
                d["why"] = "This plot compares groups but no categorical column was found."
                d["tips"] = ["Pick a text/grouping column as X in the Plot tab,",
                             "or turn a numeric column into a category (Advanced → Numeric → categorical)."]
            else:
                d["tips"] = [f"Choose a column for '{ch}' in the Plot tab."]
            return d
        m = re.search(r"'([^']+)' of .* expects a ([^ ]+) column.* but '([^']+)' is (\w+)", msg)
        if m:                                       # wrong column type
            ch, want, col, got = m.groups()
            d["error"] = f"'{col}' is the wrong type for {ch}."
            d["why"] = f"{ch} needs a {want.replace('/', ' or ')} column, but '{col}' is {got}."
            d["tips"] = [f"Pick a {want.replace('/', ' or ')} column for {ch},"]
            if "category" in want and got == "number":
                d["tips"].append(f"or convert '{col}' with Advanced → Numeric → categorical.")
            return d
        if "does not exist" in msg:
            d["error"] = "A selected column is missing."
            d["tips"] = ["Re-pick the columns in the Plot tab (the data may have changed)."]
            return d
        if not num_cols and view is not None:
            d["why"] = "No numeric column was detected — values may contain text or a decimal comma."
            d["tips"] = ["Run Data tools → Data health to see which columns did not parse."]
        return d

    # Fixed physical size (inches) for export without explicit mm — export must not
    # depend on the window/preview size at click time (a maximized or smaller window
    # must produce the same file).
    EXPORT_FIGSIZE = (9.0, 6.0)
    EXPORT_PANEL_FIGSIZE = (12.0, 8.0)

    def _render_into(self, fig, state, dpi, fixed_size=False):
        if fixed_size:
            figsize = self.EXPORT_FIGSIZE
        else:
            w, h = int(state["width"]), int(state["height"])
            # physical size (inches) from the preview reference (DPI); dpi only changes
            # pixel density — used only for the live preview (WYSIWYG).
            figsize = (max(3, w / DPI), max(2.2, h / DPI))
        common = dict(figsize=figsize, dpi=dpi, title=state["title"],
                      xlabel=state["xlabel"], ylabel=state["ylabel"],
                      style=self._style(state["style"]), legend=self._legend(state["legend"]),
                      annotations=state.get("annotations"), aliases=state.get("aliases"),
                      lines=state.get("threshold_lines"), gates=state.get("gates"))
        facet = (state.get("facet") or "").strip()
        if facet:      # split by: the same plot once per level of the column
            E.render_facets(self._layers(state), self._view, facet, fig=fig,
                            ncols=int(state.get("facet_ncols") or 2),
                            share=bool(state.get("facet_share", True)), **common)
        else:
            E.render(self._layers(state), self._view, fig=fig, **common)

    # ------------------------------- render -------------------------------- #
    def render(self, state):
        if self._view is None or not state.get("layers"):
            return None
        try:
            self._render_into(self._fig, state, DPI)
        except E.PlotConfigError as e:
            return self._explain_error(str(e))
        except Exception as e:
            return self._explain_error(str(e))

        self._fig.canvas.draw()
        buf = io.BytesIO()
        self._fig.savefig(buf, format="png", dpi=DPI, facecolor=self._fig.get_facecolor())
        raw = buf.getvalue()
        if state.get("cvd"):                 # colour-blindness / grayscale preview
            raw = _simulate_cvd(raw, state["cvd"])
        img = base64.b64encode(raw).decode()

        ax = self._fig.axes[0]
        self._Hpx = self._fig.get_size_inches()[1] * DPI

        def rect(bb):
            return dict(left=float(bb.x0), top=float(self._Hpx - bb.y1),
                        right=float(bb.x1), bottom=float(self._Hpx - bb.y0))

        leg = ax.get_legend()
        base = REGISTRY[state["layers"][0]["spec_key"]]
        m = state["layers"][0]["mapping"]
        faceted = bool((state.get("facet") or "").strip())
        # faceted figures have many axes: point picking and the legend/gate overlays
        # (which assume a single axes) are turned off, like a panel.
        self._pick = None if faceted else ((m.get("x"), m.get("y")) if base.pickable else None)
        # categorical axis (strip/swarm/box...): store label->position for reproducible gating
        self._gate_pos = {"x": None, "y": None}
        try:
            kinds = DL.column_kinds(self._view)
            if m.get("x") and kinds.get(m["x"]) == DL.CATEGORY:
                # _np_raw = the category as it is in the data; the visible text may be a rename
                self._gate_pos["x"] = {getattr(t, "_np_raw", t.get_text()): float(p) for p, t in
                                       zip(ax.get_xticks(), ax.get_xticklabels()) if t.get_text()}
            if m.get("y") and kinds.get(m["y"]) == DL.CATEGORY:
                self._gate_pos["y"] = {t.get_text(): float(p) for p, t in
                                       zip(ax.get_yticks(), ax.get_yticklabels()) if t.get_text()}
        except Exception:
            self._gate_pos = {"x": None, "y": None}
        def txtrect(artist):                 # bbox of a text artist, or None if empty
            try:
                return rect(artist.get_window_extent()) if artist.get_text() else None
            except Exception:
                return None
        texts = None if faceted else dict(title=txtrect(ax.title),
                                          xlabel=txtrect(ax.xaxis.label),
                                          ylabel=txtrect(ax.yaxis.label))
        # per-legend-entry boxes (+ each series' ORIGINAL label, so relabels are stable)
        leg_items, leg_title = [], None
        if leg is not None and not faceted:
            try:
                origs = ax.get_legend_handles_labels()[1]
                tobjs = leg.get_texts()
                if len(origs) == len(tobjs):        # simple hue legend: 1 text per handle
                    for orig, tob in zip(origs, tobjs):
                        b = txtrect(tob)
                        if b:
                            leg_items.append(dict(orig=str(orig), box=b))
                tt = leg.get_title()
                if tt and tt.get_text():
                    leg_title = txtrect(tt)
            except Exception:
                leg_items, leg_title = [], None
        # categorical X ticks: box + the ORIGINAL category, so a rename stays reversible
        xticks = []
        if not faceted:
            try:
                for tl in ax.get_xticklabels():
                    if not tl.get_text():
                        continue
                    b = txtrect(tl)
                    if b:
                        xticks.append(dict(orig=str(getattr(tl, "_np_raw", tl.get_text())), box=b))
            except Exception:
                xticks = []
        xl, yl = ax.get_xlim(), ax.get_ylim()
        return dict(img="data:image/png;base64," + img, xticks=xticks,
                    imgW=float(self._fig.get_size_inches()[0] * DPI), imgH=float(self._Hpx),
                    axes=rect(ax.get_window_extent()), faceted=faceted, texts=texts,
                    legend_items=leg_items, legend_title=leg_title,
                    xlim=[float(xl[0]), float(xl[1])], ylim=[float(yl[0]), float(yl[1])],
                    legend=None if faceted else (rect(leg.get_window_extent()) if leg else None))

    def pick(self, px, py):
        """px/py in NATURAL image pixels (origin at top-left)."""
        if self._view is None or not self._pick or not self._fig.axes:
            return None
        xcol, ycol = self._pick
        if not xcol or not ycol:
            return None
        ax = self._fig.axes[0]
        dx, dy = ax.transData.inverted().transform((px, self._Hpx - py))
        gx, gy = S.gating_coords(self._view, xcol, ycol, self._gate_pos)
        valid = np.isfinite(gx) & np.isfinite(gy)
        if not valid.any():
            return None
        xr = np.ptp(ax.get_xlim()) or 1
        yr = np.ptp(ax.get_ylim()) or 1
        d = ((gx - dx) / xr) ** 2 + ((gy - dy) / yr) ** 2
        d[~valid] = np.inf
        row = self._view.iloc[int(np.argmin(d))]
        return {str(k): ("" if pd.isna(v) else str(v)) for k, v in row.items()}

    # ------------------------------ analysis ------------------------------- #
    def reduce(self, cols, method):
        if self._view is None:
            return None
        try:
            self._view, names = S.compute_embedding(self._view, cols, method)
        except Exception as e:
            return dict(error=str(e))
        return self._info(extra={"new": names})

    def group_test(self, value_col, group_col):
        if self._view is None:
            return ""
        return S.group_test(self._view, value_col, group_col)

    def compare_groups(self, value_col, group_col):
        """Robust comparison: non-parametric + parametric, effect size with CI95 and
        Benjamini-Hochberg correction for 3+ groups."""
        if self._view is None:
            return "Load data first."
        try:
            return S.robust_compare(self._view, value_col, group_col)
        except Exception as e:
            return f"Error: {e}"

    def shannon(self, group_col, x_col=None, y_col=None, gates=None, direction="by_region"):
        """Shannon index. direction='by_region': diversity of the groups INSIDE each region.
        direction='by_group': diversity of the REGIONS occupied by each group (relative
        Shannon). Uses single assignment (same as export) when regions exist."""
        if self._view is None:
            return None
        try:
            reg = None
            if gates and x_col and y_col:
                names = list(dict.fromkeys(gg.get("name") or "" for gg in gates if (gg.get("name") or "")))
                assigned = self._assign_single(self._view, x_col, y_col, gates, set(names))
                if assigned is not None:
                    reg = np.where(assigned == "", "(outside)", assigned)
            if direction == "by_group":
                if reg is None:
                    return dict(error="For 'regions per group', draw regions and set X/Y")
                tbl = S.shannon_occupancy(self._view, group_col, reg)
            else:
                tbl = S.shannon_breakdown(self._view, group_col, reg)
        except Exception as e:
            return dict(error=str(e))
        self._table = tbl
        return dict(columns=list(tbl.columns), rows=tbl.astype(str).values.tolist())

    def rank_pairs(self, group_col, cols=None):
        """Rank every pair of numeric columns by how well they separate the groups."""
        if self._view is None:
            return None
        try:
            tbl = S.rank_separating_pairs(self._view, group_col, cols or None)
        except Exception as e:
            return dict(error=str(e))
        self._table = tbl                      # reuses the existing export_table
        return dict(columns=list(tbl.columns), rows=tbl.astype(str).values.tolist())

    def group_colors(self, hue_col, palette):
        """Category->hex map the app would use RIGHT NOW for this hue/palette. Used to
        pre-fill the pickers on 'lock' (so colors don't jump)."""
        if self._view is None or not hue_col or hue_col not in self._view.columns:
            return {}
        import seaborn as sns
        import matplotlib.colors as mcolors
        names = [str(g) for g, _ in self._view.groupby(hue_col)]   # same order as the render
        base = sns.color_palette(palette or "viridis", max(len(names), 1))
        return {n: mcolors.to_hex(base[i]) for i, n in enumerate(names)}

    def region_stats(self, x_col, y_col, hue_col, lines):
        """Count observations per region defined by the threshold lines (per group)."""
        if self._view is None:
            return None
        try:
            tbl = S.region_breakdown(self._view, x_col, y_col, lines, hue_col or None)
        except Exception as e:
            return dict(error=str(e))
        self._table = tbl                      # reuses the existing export_table
        return dict(columns=list(tbl.columns), rows=tbl.astype(str).values.tolist())

    def to_data(self, px, py):
        """Convert a pixel (top-left origin) of the last figure into a data coordinate.
        Used to anchor threshold lines by clicking on the plot."""
        if not self._fig.axes or not self._Hpx:
            return None
        ax = self._fig.axes[0]
        try:
            dx, dy = ax.transData.inverted().transform((px, self._Hpx - py))
        except Exception:
            return None
        return dict(x=float(dx), y=float(dy))

    def merge_gates(self, gates):
        """Geometrically merge same-named regions into a single polygon."""
        try:
            merged = S.merge_gates_geometric(gates)
        except Exception as e:
            return dict(error=str(e))
        return dict(gates=merged)

    def gates_from_lines(self, lines, xlim, ylim):
        """Build polygonal regions (gates) from the intersection of the current lines."""
        if not lines:
            return dict(error="Add at least one line first.")
        try:
            cells = S.cells_from_lines(lines, xlim, ylim)
        except Exception as e:
            return dict(error=str(e))
        return dict(gates=cells)

    def gate_stats(self, x_col, y_col, hue_col, gates):
        """Live count per region (polygon), per group, warning about overlap."""
        if self._view is None:
            return None
        try:
            tbl = S.gate_breakdown(self._view, x_col, y_col, gates, hue_col or None, pos=self._gate_pos)
        except Exception as e:
            return dict(error=str(e))
        self._table = tbl
        return dict(columns=list(tbl.columns), rows=tbl.astype(str).values.tolist())

    @staticmethod
    def _dist_pts_poly(pts, poly):
        """Vectorized distance from each point to the nearest polygon edge."""
        M = len(poly)
        d = np.full(len(pts), np.inf)
        for k in range(M):
            a = poly[k]; b = poly[(k + 1) % M]
            ab = b - a; L2 = float(ab @ ab)
            if L2 == 0.0:
                dist = np.hypot(pts[:, 0] - a[0], pts[:, 1] - a[1])
            else:
                t = np.clip(((pts - a) @ ab) / L2, 0.0, 1.0)
                px = a[0] + t * ab[0]; py = a[1] + t * ab[1]
                dist = np.hypot(pts[:, 0] - px, pts[:, 1] - py)
            d = np.minimum(d, dist)
        return d

    def _assign_single(self, df, x_col, y_col, gates, names):
        """Assign EACH observation to at most ONE region (the one it 'fits best').
        Criterion: signed distance to the polygon border — positive if inside (deeper =
        stronger membership), negative if outside. The region with the highest value wins.
        Points on the line (distance ~0) go to the nearest region; points clearly outside
        all of them become '' (= outside). Returns an array of names."""
        from matplotlib.path import Path
        x, y = S.gating_coords(df, x_col, y_col, self._gate_pos)
        pts = np.column_stack([x, y])
        valid = np.isfinite(x) & np.isfinite(y)
        cand = []
        for g in gates:
            nm = g.get("name") or ""
            if nm not in names:
                continue
            poly = g.get("points") or []
            if len(poly) >= 3:
                cand.append((nm, np.asarray(poly, float)))
        if not cand:
            return None
        n = len(df)
        best_name = np.array([""] * n, dtype=object)
        best_d = np.full(n, -np.inf)
        for nm, poly in cand:
            d_edge = self._dist_pts_poly(pts, poly)
            inside = Path(poly).contains_points(pts)
            signed = np.where(inside, d_edge, -d_edge)
            signed = np.where(valid, signed, -np.inf)
            upd = signed > best_d
            best_name = np.where(upd, nm, best_name)
            best_d = np.where(upd, signed, best_d)
        rx = (np.nanmax(x) - np.nanmin(x)) if np.isfinite(x).any() else 1.0
        ry = (np.nanmax(y) - np.nanmin(y)) if np.isfinite(y).any() else 1.0
        eps = 1e-9 * float(np.hypot(rx or 1.0, ry or 1.0))   # ~on the line counts as inside
        return np.where(best_d >= -eps, best_name, "")

    @staticmethod
    def _add_region_col(out, labels):
        col, i = "Region", 2
        while col in out.columns:
            col = f"Region_{i}"; i += 1
        out[col] = labels
        return out

    def export_gated(self, x_col, y_col, gates, region, fmt="csv", per_region=False):
        """Export observations (ALL columns) with a single-valued 'Region' column: each
        cell belongs only to the region it 'fits best' (see _assign_single).
        region='__ALL__' exports everything ('(outside)' for those that fall in none).
        per_region=True writes 1 file per region in a folder; otherwise a single file.
        Operates on the current view (after filters)."""
        if self._view is None:
            return dict(error="Load data first")
        df = self._view
        if x_col not in df.columns or y_col not in df.columns:
            return dict(error="Set valid X and Y in the Plot tab")
        ext = ".xlsx" if fmt == "xlsx" else ".csv"

        def _safe(s):
            return ("".join(c if (c.isalnum() or c in "-_ ") else "_" for c in str(s)).strip() or "region")

        def _write(out, path):
            out.to_excel(path, index=False) if fmt == "xlsx" else out.to_csv(path, index=False)

        # ----- ALL observations (Region column / '(outside)') -----
        if region == "__ALL__":
            names = list(dict.fromkeys(g.get("name") or "" for g in gates if (g.get("name") or "")))
            assigned = self._assign_single(df, x_col, y_col, gates, set(names))
            if assigned is None:
                return dict(error="No valid region")
            labels = np.where(assigned == "", "(outside)", assigned)
            out = self._add_region_col(df.copy(), labels)
            path = self._win().create_file_dialog(
                webview.SAVE_DIALOG, save_filename="gating_all" + ext,
                file_types=(("Excel (*.xlsx)",) if fmt == "xlsx" else ("CSV (*.csv)",)))
            if not path:
                return None
            path = path if isinstance(path, str) else path[0]
            try:
                _write(out, path)
            except Exception as e:
                return dict(error=f"Failed to save: {e}")
            return dict(path=path, n=int(len(out)), total=int(len(df)))

        # ----- selected regions (single assignment) -----
        wanted = list(region) if isinstance(region, (list, tuple)) else [region]
        wanted = list(dict.fromkeys(str(w) for w in wanted))
        assigned = self._assign_single(df, x_col, y_col, gates, set(wanted))
        if assigned is None:
            return dict(error="None of the selected regions were found")
        present = [nm for nm in wanted if (assigned == nm).any()]
        if not present:
            return dict(error="No observation inside the selected regions")

        # ----- 1 file per region (folder) -----
        if per_region:
            res = self._win().create_file_dialog(webview.FOLDER_DIALOG)
            if not res:
                return None
            folder = res if isinstance(res, str) else res[0]
            written, total_rows = [], 0
            for nm in present:
                sub = df[assigned == nm].copy()
                sub = self._add_region_col(sub, nm)            # single value = the region itself
                p = os.path.join(folder, f"gating_{_safe(nm)}{ext}")
                try:
                    _write(sub, p)
                except Exception as e:
                    return dict(error=f"Failed to save {nm}: {e}")
                written.append(os.path.basename(p)); total_rows += len(sub)
            return dict(folder=folder, files=written, n=int(total_rows), total=int(len(df)), per_region=True)

        # ----- 1 file (union, each row with its single region) -----
        mask_in = assigned != ""
        out = self._add_region_col(df[mask_in].copy(), assigned[mask_in])
        fname = f"gating_{_safe(present[0])}" if len(present) == 1 else f"gating_{len(present)}regions"
        path = self._win().create_file_dialog(
            webview.SAVE_DIALOG, save_filename=fname + ext,
            file_types=(("Excel (*.xlsx)",) if fmt == "xlsx" else ("CSV (*.csv)",)))
        if not path:
            return None
        path = path if isinstance(path, str) else path[0]
        try:
            _write(out, path)
        except Exception as e:
            return dict(error=f"Failed to save: {e}")
        return dict(path=path, n=int(len(out)), total=int(len(df)))



    def make_region_column(self, x_col, y_col, gates, colname):
        """Create a categorical column with each observation's region name.
        Applies to the full dataframe (persists through filters) and to the current view."""
        if self._view is None:
            return None
        if not gates:
            return dict(error="Add at least one region before creating the column.")
        col = (colname or "Region").strip() or "Region"
        try:
            reg_full = S.assign_regions(self._df, x_col, y_col, gates, pos=self._gate_pos)
        except Exception as e:
            return dict(error=str(e))
        self._df = self._df.copy()
        self._df[col] = reg_full
        if self._view is not self._df:
            self._view = self._view.copy()
            self._view[col] = reg_full.reindex(self._view.index)
        else:
            self._view = self._df
        return self._info(extra={"created": col})

    def to_category(self, col):
        """Convert a numeric column to text (e.g. track_id) so it can be used as a
        category/hue. Applies to the full dataframe, so it survives filters."""
        if self._df is None or col not in self._df.columns:
            return None
        self._df = DL.to_category(self._df, col)
        if self._view is not self._df:
            self._view = self._view.copy()
            self._view[col] = self._df[col].reindex(self._view.index)
        else:
            self._view = self._df
        return self._info()

    def aggregate_data(self, group_cols, method="mean"):
        """Collapse the current data to one row per group (mean/median/sum/count).
        Becomes the new working data (like reshape). Fix for pseudo-replication:
        aggregate cells to their replicate/track before plotting or testing."""
        if self._view is None:
            return None
        try:
            self._df = S.aggregate_by(self._view, group_cols, method)
        except Exception as e:
            return dict(error=str(e))
        self._view = self._df
        return self._info(extra={"aggregated": True})

    def add_computed_column(self, name, expr):
        """Create a new column from a pandas expression over the existing columns
        (e.g. 'net_disp / cum_path'). Applies to the full dataframe so it survives
        filters. Column names with spaces must be wrapped in backticks in the expr."""
        if self._df is None:
            return dict(error="Load data first.")
        name = (name or "").strip()
        if not name:
            return dict(error="Give the new column a name.")
        try:
            values = self._df.eval(expr)
        except Exception as e:
            return dict(error=f"Invalid formula: {e}")
        self._df = self._df.copy()
        self._df[name] = values
        if self._view is not self._df:
            self._view = self._view.copy()
            try:
                self._view[name] = self._df[name].reindex(self._view.index)
            except Exception:
                self._view = self._df
        else:
            self._view = self._df
        return self._info(extra={"created": name})

    def contingency(self, row_col, col_col):
        """Chi-square (+ Fisher 2x2) test of association between two categorical
        columns. Result feeds the plot table (exportable) and a text summary."""
        if self._view is None:
            return None
        try:
            r = S.contingency_test(self._view, row_col, col_col)
        except Exception as e:
            return dict(error=str(e))
        import pandas as _pd
        self._table = _pd.DataFrame(r["rows"], columns=r["columns"])
        return r

    def methods_sentence(self, value_col, group_col):
        """Ready-to-paste statistics sentence for a figure legend / methods section."""
        if self._view is None:
            return "Load data first."
        try:
            return S.methods_sentence(self._view, value_col, group_col)
        except Exception as e:
            return f"Error: {e}"

    def survival_test(self, time_col, event_col, event_value, group_col):
        """Log-rank test text for Kaplan-Meier groups (the KM plot shows the same p)."""
        if self._view is None:
            return "Load data first."
        try:
            ev = [s.strip() for s in str(event_value or "").split(",") if s.strip()]
            _, lr = S.km_curves(self._view, time_col, event_col, ev, group_col)
        except Exception as e:
            return f"Error: {e}"
        if not lr:
            return "Need a grouping column with at least 2 groups."
        obs = ", ".join(f"{g}: {int(o)} events" for g, o in lr["observed"].items())
        return (f"Log-rank test: chi-square = {lr['chi2']:.2f}, df = {lr['df']}, "
                f"p = {lr['p']:.3g}\nObserved events -> {obs}")

    # ---------------------- statistics (Statistics panel) ------------------ #
    def _txt(self, fn, *a, **k):
        """Run an analysis that returns text, turning errors into a readable message."""
        if self._view is None:
            return "Load data first."
        try:
            return fn(self._view, *a, **k)
        except Exception as e:
            return f"Error: {e}"

    def anova_two_way(self, value_col, factor_a, factor_b):
        return self._txt(S.two_way_anova, value_col, factor_a, factor_b)

    def anova_rm(self, value_col, within_col, subject_col):
        return self._txt(S.rm_anova, value_col, within_col, subject_col)

    def which_test(self, value_col, group_col, paired=False):
        return self._txt(S.recommend_test, value_col, group_col, bool(paired))

    def power_calc(self, d, alpha=0.05, power=0.8, n=None):
        try:
            return S.power_report(float(d), float(alpha), float(power),
                                  int(n) if n else None)
        except Exception as e:
            return f"Error: {e}"

    def outlier_scan(self, value_col, group_col=None, method="grubbs", make_column=False):
        """Flag outliers; optionally write the flag into a new column (never deletes)."""
        if self._view is None:
            return dict(error="Load data first.")
        try:
            flag, text = S.find_outliers(self._view, value_col, group_col or None, method)
        except Exception as e:
            return dict(error=str(e))
        created = None
        if make_column:
            col = f"outlier_{value_col}"
            self._df = self._df.copy()
            self._df[col] = flag.reindex(self._df.index).fillna(False)
            self._view = self._view.copy()
            self._view[col] = flag
            created = col
        # key is n_outliers, not n: _info() also carries an "n" (row count) and would
        # overwrite it when the flag column is created.
        d = dict(text=text, n_outliers=int(flag.sum()))
        if created:
            d.update(self._info(extra={"created": created}))
        return d

    def corr_table(self, cols, method="pearson"):
        """Correlation matrix with BH-corrected p-values, as an exportable table."""
        if self._view is None:
            return None
        try:
            r, p, stars = S.corr_with_p(self._view, cols, method)
        except Exception as e:
            return dict(error=str(e))
        rows = []
        names = list(r.columns)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                rows.append([a, b, round(float(r.loc[a, b]), 4),
                             f"{float(p.loc[a, b]):.3g}", stars.loc[a, b]])
        import pandas as _pd
        tbl = _pd.DataFrame(rows, columns=["A", "B", "r", "p (BH)", ""])
        self._table = tbl
        return dict(columns=list(tbl.columns), rows=tbl.astype(str).values.tolist())

    # ------------------------ bench math (Bench panel) --------------------- #
    def qpcr_ddct(self, ct_col, gene_col, sample_col, group_col, housekeeping, reference):
        """qPCR 2^-ddCt. The tidy result becomes the working data, ready to plot."""
        if self._view is None:
            return dict(error="Load data first.")
        try:
            out = S.ddct(self._view, ct_col, gene_col, sample_col, group_col,
                         housekeeping, reference)
        except Exception as e:
            return dict(error=str(e))
        self._df = self._view = out
        return self._info(extra={"created": "fold_change"})

    def std_curve(self, conc_col, signal_col, model="linear", target_col=None, newname="conc_calc"):
        """Fit the standards, then interpolate every row's signal back to concentration
        into a new column (rows without a known concentration are the unknowns)."""
        if self._view is None:
            return dict(error="Load data first.")
        try:
            src = target_col or signal_col
            res = S.standard_curve(self._view, conc_col, signal_col, model)
            vals = res["predict_conc"](pd.to_numeric(self._view[src], errors="coerce"))
        except Exception as e:
            return dict(error=str(e))
        name = (newname or "conc_calc").strip() or "conc_calc"
        self._view = self._view.copy()
        self._view[name] = vals
        if self._df is not self._view:
            self._df = self._df.copy()
            try:
                self._df[name] = pd.Series(vals, index=self._view.index).reindex(self._df.index)
            except Exception:
                self._df = self._view
        else:
            self._df = self._view
        d = self._info(extra={"created": name})
        d["text"] = res["text"]
        return d

    def normalize_column(self, value_col, method, group_col=None, control=None,
                         by_col=None, newname=None):
        """Bench normalisations (% of control, fold, baseline, /loading control, z, min-max)."""
        if self._view is None:
            return dict(error="Load data first.")
        try:
            vals = S.normalize_series(self._view, value_col, method, group_col or None,
                                      control, by_col or None)
        except Exception as e:
            return dict(error=str(e))
        name = (newname or f"{value_col}_{method}").strip()
        self._view = self._view.copy(); self._view[name] = vals
        if self._df is not self._view:
            self._df = self._df.copy()
            try:
                self._df[name] = vals.reindex(self._df.index)
            except Exception:
                self._df = self._view
        else:
            self._df = self._view
        return self._info(extra={"created": name})

    # -------------------------- data tools (QoL) --------------------------- #
    def health_check(self):
        if self._view is None:
            return dict(error="Load data first.")
        try:
            r = S.data_health(self._view)
        except Exception as e:
            return dict(error=str(e))
        import pandas as _pd
        self._table = _pd.DataFrame(r["rows"], columns=r["columns"])
        return r

    def plate_import(self, values_grid, map_grid=None, value_name="value", cond_name="condition"):
        """Turn a pasted 96/384-well grid (+ optional condition map) into tidy rows."""
        try:
            out = S.plate_to_tidy(values_grid, map_grid, value_name or "value",
                                  cond_name or "condition")
        except Exception as e:
            return dict(error=str(e))
        self._df = self._view = out
        self._csv_path = None
        return self._info("plate")

    def render_overview(self, cols, group=None, state=None):
        """Small multiples of the selected numeric columns (one glance at the table)."""
        if self._view is None:
            return dict(error="Load data first.")
        try:
            st = self._style((state or {}).get("style", {})) if state else None
            fig = F.overview(self._view, cols, group or None, style=st)
        except Exception as e:
            return dict(error=str(e))
        self._pick = None
        self._adv_fig = fig                      # reuses the existing "export this figure"
        return dict(img=self._png(fig))

    def save_template(self, state):
        """Save the plot 'recipe' (type + channel mapping + params + labels) to a file,
        so it can be re-applied to ANOTHER dataset with the same column names."""
        layers = state.get("layers") or []
        if not layers:
            return dict(error="Build a plot first.")
        path = self._win().create_file_dialog(
            webview.SAVE_DIALOG, save_filename="plot.template.json",
            file_types=("Nickplots Template (*.json)",))
        if not path:
            return None
        path = path if isinstance(path, str) else path[0]
        tpl = dict(kind="nickplots_template", version=1, layers=layers,
                   title=state.get("title", ""), xlabel=state.get("xlabel", ""),
                   ylabel=state.get("ylabel", ""), ncols=state.get("ncols"))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tpl, f, ensure_ascii=False, indent=2)
        return path

    def apply_template(self):
        """Load a plot recipe saved with save_template."""
        res = self._win().create_file_dialog(
            webview.OPEN_DIALOG, file_types=("Nickplots Template (*.json)", "All files (*.*)"))
        if not res:
            return None
        try:
            d = json.load(open(res[0], encoding="utf-8"))
        except Exception as e:
            return dict(error=str(e))
        if d.get("kind") != "nickplots_template":
            return dict(error="Not a Nickplots template file.")
        return d

    # ------------------------------- export -------------------------------- #
    def _export_kw(self, state):
        """Transparent background + readable ink, and exact size when mm is set."""
        st = state.get("style", {})
        exact_mm = bool(self._f(st.get("fig_w_mm")) and self._f(st.get("fig_h_mm")))
        transparent = bool(state.get("transparent"))
        return dict(transparent=transparent,
                    ink=E.transparent_ink(st.get("theme")) if transparent else None,
                    tight=not exact_mm)

    def export_figure(self, state):
        if self._view is None or not state.get("layers"):
            return None
        path = self._win().create_file_dialog(
            webview.SAVE_DIALOG, save_filename="figure.tiff",
            file_types=("TIFF (*.tiff)", "PNG (*.png)", "PDF (*.pdf)", "SVG (*.svg)"))
        if not path:
            return None
        path = path if isinstance(path, str) else path[0]
        dpi = max(300, int(state.get("export_dpi") or 300))
        fig = Figure(dpi=dpi)
        try:
            self._render_into(fig, state, dpi, fixed_size=True)
            E.export(fig, path, dpi=dpi, **self._export_kw(state))
        except Exception as e:
            return dict(error=str(e))
        return path

    def export_figures_batch(self, states, fmt="tiff"):
        """Re-export every open plot tab into one folder, all at the current journal
        spec. The reformat-after-rejection case: one dialog instead of one per figure."""
        if self._view is None or not states:
            return dict(error="Plot a figure first.")
        folder = self._win().create_file_dialog(webview.FOLDER_DIALOG)
        if not folder:
            return None
        folder = folder if isinstance(folder, str) else folder[0]
        ext = "." + str(fmt or "tiff").lstrip(".").lower()
        n, failed = 0, []
        for i, item in enumerate(states):
            state = item.get("state") or {}
            if not state.get("layers"):
                continue
            safe = "".join(c if (c.isalnum() or c in " -_") else "_"
                           for c in str(item.get("name") or f"plot_{i + 1}")).strip()
            path = os.path.join(folder, (safe or f"plot_{i + 1}") + ext)
            dpi = max(300, int(state.get("export_dpi") or 300))
            fig = Figure(dpi=dpi)
            try:
                self._render_into(fig, state, dpi, fixed_size=True)
                E.export(fig, path, dpi=dpi, **self._export_kw(state))
                n += 1
            except Exception as e:
                failed.append(f"{safe}: {e}")
        if not n:
            return dict(error="; ".join(failed) or "Nothing was exported.")
        # partial failures are a warning, not an error: n figures really were written
        return dict(n=n, folder=folder,
                    warn=("Some failed - " + "; ".join(failed)) if failed else None)

    # ------------------- nested / ROC / agreement / curves ------------------ #
    def nested_test(self, value_col, group_col, rep_col):
        return self._txt(S.nested_test, value_col, group_col, rep_col)

    def curve_compare(self, x_col, y_col, group_col):
        return self._txt(S.compare_curves_4pl, x_col, y_col, group_col)

    def roc_stats(self, score_col, label_col, positive=""):
        return self._txt(lambda d, *a: S.roc_analysis(d, *a)["text"],
                         score_col, label_col, positive)

    def agreement(self, col_a, col_b):
        return self._txt(lambda d, *a: S.bland_altman(d, *a)["text"], col_a, col_b)

    def compare_pairs(self, value_col, group_col):
        """Structured pairs for the on-figure markers: same test as Compare groups."""
        if self._view is None:
            return dict(error="Load data first.")
        try:
            return dict(pairs=S.compare_pairs(self._view, value_col, group_col))
        except Exception as e:
            return dict(error=str(e))

    # ------------------------------ figure legend --------------------------- #
    # Reviewers' single most frequent figure complaint is "error bars not defined".
    # Everything needed is already in the render state - so write the sentence.
    _ERR_WORDS = {"SD": "standard deviation", "SEM": "standard error of the mean",
                  "CI95": "95% confidence interval"}

    def _legend_spread(self, key, p):
        """What the drawn marks represent, in words."""
        if key == "bar_err":
            centre = "mean" if p.get("center", "mean") == "mean" else "median"
            err = self._ERR_WORDS.get(p.get("error", "SEM"), p.get("error", "SEM"))
            kind = "Bars" if p.get("kind", "bar") == "bar" else "Points"
            s = f"{kind} show the {centre}; error bars show the {err}"
            return s + (", with every observation overlaid." if p.get("points") else ".")
        if key == "line" and p.get("errorbar", "none") != "none":
            return ("Line shows the mean; the shaded band shows the "
                    f"{self._ERR_WORDS.get(p['errorbar'], p['errorbar'])}.")
        if key in ("box", "box_points"):
            return ("Boxes show the median and interquartile range; whiskers extend to "
                    "1.5x IQR" + (", with every observation overlaid."
                                  if key == "box_points" else "."))
        if key in ("violin", "violin_points"):
            return ("Violins show the kernel density of the distribution" +
                    (", with every observation overlaid." if key.endswith("points") else "."))
        if key == "superplot":
            return ("Small faint markers are individual observations; large markers are "
                    "the mean of each independent replicate, coloured by replicate.")
        if key == "regband":
            return f"Line is the linear fit; the band is the {p.get('ci', 95)}% confidence interval."
        if key == "km":
            return "Curves are Kaplan-Meier estimates; ticks mark censored observations."
        if key == "roc":
            return "Curve is the ROC; the dashed diagonal is chance (AUC 0.5)."
        if key == "bland_altman":
            return ("Solid line is the mean difference (bias); dashed lines are the 95% "
                    "limits of agreement (bias +/- 1.96 SD).")
        return ""

    def figure_legend(self, state):
        """Draft the figure legend from what was actually drawn."""
        if self._view is None or not state.get("layers"):
            return "Plot a figure first."
        lay = state["layers"][0]
        key, m, p = lay["spec_key"], lay.get("mapping") or {}, lay.get("params") or {}
        spec = REGISTRY.get(key)
        df = self._view
        out = []
        if state.get("title"):
            out.append(str(state["title"]).strip())
        spread = self._legend_spread(key, p)
        if spread:
            out.append(spread)

        # n per group - and, when the plot knows about replicates, n of BOTH levels
        grp = m.get("x") if m.get("x") in df.columns else m.get("hue")
        rep = m.get("rep") if m.get("rep") in df.columns else None
        if grp and grp in df.columns:
            parts = []
            for g, sub in df.groupby(df[grp].astype(str)):
                if rep:
                    parts.append(f"{g}: n = {sub[rep].nunique()} replicates "
                                 f"({len(sub)} observations)")
                else:
                    parts.append(f"{g}: n = {len(sub)}")
            out.append("; ".join(parts) + ".")
        else:
            out.append(f"n = {len(df)}.")

        pairs = p.get("__sig_pairs__")
        if pairs:
            out.append("Two-sided Mann-Whitney U test; p values corrected across all "
                       "pairwise comparisons (Benjamini-Hochberg). "
                       "*q < 0.05, **q < 0.01, ***q < 0.001.")
        elif p.get("sig"):
            out.append("Two-sided Welch t test; p values corrected across all pairwise "
                       "comparisons (Holm). *p < 0.05, **p < 0.01, ***p < 0.001.")
        # never let the draft claim a test it did not run: the markers come from the
        # value/group columns as they stand, which is cell level when replicates exist
        if rep and (pairs or p.get("sig")):
            out.append("!! The markers above were computed over individual "
                       "observations, not over replicate means. With several "
                       "observations per replicate that p is anti-conservative - run "
                       "Analysis > Nested test and report its p instead.")
        if (state.get("filter") or "").strip():
            out.append(f"Rows included: {state['filter'].strip()}.")
        out.append(f"Plot: {spec.label if spec else key}. Figure produced with Nickplots.")
        return "\n\n".join(out)

    # ------------------------------- source data ---------------------------- #
    def _source_block(self, df, key, m, p):
        """(plotted values, summary drawn on top or None) for one layer/frame."""
        cols = [c for c in dict.fromkeys(m.values()) if c and c in df.columns]
        data = df[cols].dropna(how="all") if cols else df
        summary = None
        x, y, hue = m.get("x"), m.get("y"), m.get("hue")
        if key in ("bar_err", "bar") and x in df.columns and y in df.columns:
            by = [c for c in (x, hue) if c and c in df.columns]
            g = df.dropna(subset=[y]).groupby([df[c].astype(str) for c in by])[y]
            summary = pd.DataFrame({"n": g.size(), "mean": g.mean(), "median": g.median(),
                                    "SD": g.std(ddof=1)})
            summary["SEM"] = summary["SD"] / np.sqrt(summary["n"].clip(lower=1))
            summary.index.names = by
            summary = summary.reset_index()
        return data, summary

    def export_source_data(self, state, items=None):
        """One .xlsx with the values behind the figure - the 'Source Data' file that
        Nature Methods (and others) require for every main and Extended Data figure.
        One sheet per panel frame, or one per layer for a single plot."""
        if self._view is None or not (items or state.get("layers")):
            return dict(error="Plot a figure first.")
        path = self._win().create_file_dialog(
            webview.SAVE_DIALOG, save_filename="source_data.xlsx",
            file_types=("Excel (*.xlsx)",))
        if not path:
            return None
        path = path if isinstance(path, str) else path[0]
        blocks = []
        if items:                      # a built panel: one sheet per frame, A, B, C...
            import string
            for i, it in enumerate(items):
                df = self._panel_data.get(it.get("data_id"), self._view)
                name = string.ascii_uppercase[i] if i < 26 else f"F{i + 1}"
                blocks.append((name, df, it["spec_key"], it.get("mapping") or {},
                               it.get("params") or {}))
        else:
            for i, lay in enumerate(state["layers"]):
                blocks.append((f"Layer {i + 1}" if i else "Figure", self._view,
                               lay["spec_key"], lay.get("mapping") or {},
                               lay.get("params") or {}))
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as xl:
                for name, df, key, m, p in blocks:
                    data, summary = self._source_block(df, key, m, p)
                    sheet = str(name)[:28]
                    data.to_excel(xl, sheet_name=sheet, index=False)
                    if summary is not None:
                        summary.to_excel(xl, sheet_name=f"{sheet} summary"[:31],
                                         index=False)
                pd.DataFrame({"Nickplots source data": [
                    "One sheet per panel: the values plotted in that panel.",
                    "'... summary' sheets hold the aggregate actually drawn "
                    "(n, mean, median, SD, SEM).",
                    f"Filter applied: {(state.get('filter') or '(none)').strip()}",
                ]}).to_excel(xl, sheet_name="README", index=False)
        except Exception as e:
            return dict(error=str(e))
        return path

    # ---------------------------- advanced --------------------------------- #
    def categories(self, col):
        if self._view is None or col not in self._view.columns:
            return []
        return [str(v) for v in pd.unique(self._view[col].dropna())][:40]

    def _png(self, fig, dpi=110):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def stash_panel_data(self):
        """Snapshot the current CSV so a panel frame keeps its own data even after the
        user opens a different CSV. Returns an id the frame stores; render/export_panel
        resolve it back to the dataframe. In-memory only (not persisted in projects)."""
        if self._view is None:
            return None
        self._panel_seq += 1
        self._panel_data[self._panel_seq] = self._view
        return self._panel_seq

    def _panel_into(self, fig, items, state, dpi, figsize):
        # each frame draws from its own snapshot; fall back to the live view for frames
        # with no snapshot (older items / reloaded projects).
        dfs = [self._panel_data.get(it.get("data_id"), self._view) for it in items]
        E.render_panel(items, dfs, fig=fig, ncols=int(state.get("ncols", 2)),
                       figsize=figsize, dpi=dpi, style=self._style(state["style"]),
                       aliases=state.get("aliases"),
                       legend=self._legend(state["legend"]) if state.get("legend") else None,
                       width_ratios=state.get("panel_wratios"),
                       height_ratios=state.get("panel_hratios"),
                       mosaic=state.get("panel_mosaic"))

    def render_panel(self, items, state):
        if self._view is None or not items:
            return None
        w, h = int(state["width"]), int(state["height"])
        fig = Figure(dpi=DPI)
        try:
            self._panel_into(fig, items, state, DPI, (max(4, w / DPI), max(3, h / DPI)))
        except Exception as e:
            return dict(error=str(e))
        self._pick = None
        ncols = int(state.get("ncols", 2))
        nrows = -(-len(items) // ncols)
        return dict(img=self._png(fig), rects=E.axes_rects(fig), nrows=nrows, ncols=ncols,
                    imgW=float(fig.get_size_inches()[0] * DPI),
                    imgH=float(fig.get_size_inches()[1] * DPI))

    def export_panel(self, items, state):
        if self._view is None or not items:
            return None
        path = self._win().create_file_dialog(
            webview.SAVE_DIALOG, save_filename="panel.tiff",
            file_types=("TIFF (*.tiff)", "PNG (*.png)", "PDF (*.pdf)", "SVG (*.svg)"))
        if not path:
            return None
        path = path if isinstance(path, str) else path[0]
        dpi = max(300, int(state.get("export_dpi") or 300))
        fig = Figure(dpi=dpi)
        try:
            # fixed size, like the single-figure export: never depends on the window
            self._panel_into(fig, items, state, dpi, self.EXPORT_PANEL_FIGSIZE)
            E.export(fig, path, dpi=dpi, **self._export_kw(state))
        except Exception as e:
            return dict(error=str(e))
        return path

    def render_advanced(self, kind, cols, hue, zscore):
        if self._view is None or len(cols) < 2:
            return dict(error="Select at least 2 numeric columns.")
        try:
            if kind == "clustermap":
                fig = F.clustermap(self._view, cols, z_score=bool(zscore))
            else:
                fig = F.pairplot(self._view, cols, hue=hue or None)
        except Exception as e:
            return dict(error=str(e))
        self._pick = None
        self._adv_fig = fig
        return dict(img=self._png(fig))

    def export_advanced(self, state):
        if not getattr(self, "_adv_fig", None):
            return dict(error="Generate the figure first.")
        path = self._win().create_file_dialog(
            webview.SAVE_DIALOG, save_filename="figure.tiff",
            file_types=("TIFF (*.tiff)", "PNG (*.png)", "PDF (*.pdf)", "SVG (*.svg)"))
        if not path:
            return None
        path = path if isinstance(path, str) else path[0]
        E.export(self._adv_fig, path, dpi=max(300, int(state.get("export_dpi") or 300)),
                 **self._export_kw(state))
        return path

    def describe_table(self, group_col, value_cols):
        if self._view is None:
            return None
        if not value_cols:
            value_cols = DL.columns_by_kind(self._view, DL.NUMBER)
        tbl = S.describe_by_group(self._view, value_cols, group_col or None)
        self._table = tbl
        return dict(columns=list(tbl.columns), rows=tbl.astype(str).values.tolist())

    def export_table(self):
        if getattr(self, "_table", None) is None:
            return None
        path = self._win().create_file_dialog(
            webview.SAVE_DIALOG, save_filename="descriptive_stats.csv", file_types=("CSV (*.csv)",))
        if not path:
            return None
        path = path if isinstance(path, str) else path[0]
        self._table.to_csv(path, index=False)
        return path

    # ------------------------ project / preset / code ---------------------- #
    def save_regions(self, data):
        """Save regions (polygons) + threshold lines to a reusable file."""
        path = self._win().create_file_dialog(
            webview.SAVE_DIALOG, save_filename="regions.json",
            file_types=("Nickplots Regions (*.json)",))
        if not path:
            return None
        path = path if isinstance(path, str) else path[0]
        payload = dict(kind="nickplots_regions", version=1,
                       x=(data or {}).get("x"), y=(data or {}).get("y"),
                       xlim=(data or {}).get("xlim"), ylim=(data or {}).get("ylim"),
                       gates=(data or {}).get("gates") or [],
                       lines=(data or {}).get("lines") or [])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def load_regions(self):
        """Load regions + lines from a saved file (to reuse on another sheet)."""
        res = self._win().create_file_dialog(
            webview.OPEN_DIALOG, file_types=("Nickplots Regions (*.json)", "All files (*.*)"))
        if not res:
            return None
        try:
            d = json.load(open(res[0], encoding="utf-8"))
        except Exception as e:
            return dict(error=str(e))
        return dict(gates=d.get("gates") or [], lines=d.get("lines") or [],
                    x=d.get("x"), y=d.get("y"), xlim=d.get("xlim"), ylim=d.get("ylim"))

    def save_project(self, state):
        path = self._win().create_file_dialog(
            webview.SAVE_DIALOG, save_filename="project.json", file_types=("Nickplots Project (*.json)",))
        if not path:
            return None
        path = path if isinstance(path, str) else path[0]
        proj = dict(csv_path=self._csv_path, ui=state, active=self._active,
                    combined_col=self._combined_col,
                    datasets=[dict(name=n, path=self._dataset_paths.get(n))
                              for n in self._datasets])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(proj, f, ensure_ascii=False, indent=2)
        return path

    def open_project(self):
        res = self._win().create_file_dialog(
            webview.OPEN_DIALOG, file_types=("Nickplots Project (*.json)", "All files (*.*)"))
        if not res:
            return None
        try:
            proj = json.load(open(res[0], encoding="utf-8"))
        except Exception as e:
            return dict(error=str(e))
        data, warn = None, []
        if proj.get("datasets"):
            # reopen EVERY table the project was built from and restore which one was active
            self._datasets, self._dataset_paths = {}, {}
            for item in proj["datasets"]:
                nm, pth = item.get("name"), item.get("path")
                if not pth:
                    warn.append(f"{nm}: was dropped in as text, there is no file to reopen")
                    continue
                try:
                    self._datasets[nm] = DL.load_csv(pth)
                    self._dataset_paths[nm] = pth
                except Exception as e:
                    warn.append(f"{nm}: {e}")
            act = proj.get("active")
            if act == "(combined)" and len(self._datasets) >= 2:
                self.combine_datasets(proj.get("combined_col") or "dataset")
            elif act in self._datasets:
                self.activate_dataset(act)
            elif self._datasets:
                self.activate_dataset(next(iter(self._datasets)))
            if self._view is None:
                data = dict(error="None of the project's tables could be reopened. "
                                  "Load them manually - the plots are kept.")
            else:
                data = self._info(self._active)
        elif proj.get("csv_path"):                     # projects saved before multi-table
            try:
                self._df = DL.load_csv(proj["csv_path"])
                self._view = self._df
                self._csv_path = proj["csv_path"]
                self._datasets = {os.path.basename(proj["csv_path"]): self._df}
                self._dataset_paths = {os.path.basename(proj["csv_path"]): proj["csv_path"]}
                self._active = next(iter(self._datasets))
                data = self._info(proj["csv_path"])
            except Exception:
                data = dict(error=f"Could not reopen the original CSV ({proj.get('csv_path')}). Load it manually.")
        if data and warn and not data.get("error"):
            data["warn"] = "Could not reopen: " + "; ".join(warn)
        return dict(ui=proj.get("ui"), data=data,
                    tabs=proj.get("tabs"), activeTab=proj.get("activeTab"))

    def save_preset(self, state):
        path = self._win().create_file_dialog(
            webview.SAVE_DIALOG, save_filename="style.preset.json", file_types=("Preset (*.json)",))
        if not path:
            return None
        path = path if isinstance(path, str) else path[0]
        json.dump(dict(style=state.get("style"), legend=state.get("legend")),
                  open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return path

    def apply_preset(self):
        res = self._win().create_file_dialog(
            webview.OPEN_DIALOG, file_types=("Preset (*.json)", "All files (*.*)"))
        if not res:
            return None
        try:
            return json.load(open(res[0], encoding="utf-8"))
        except Exception as e:
            return dict(error=str(e))

    def export_code(self, state):
        if self._view is None or not state.get("layers"):
            return dict(error="Build a plot first.")
        path = self._win().create_file_dialog(
            webview.SAVE_DIALOG, save_filename="figure_code.py", file_types=("Python (*.py)",))
        if not path:
            return None
        path = path if isinstance(path, str) else path[0]
        csvp = os.path.splitext(path)[0] + "_data.csv"      # snapshot to reproduce exactly
        self._df.to_csv(csvp, index=False)
        code = CG.generate_code(state, os.path.basename(csvp))
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        return path
