# Nickplots

**Publication-quality statistical figures from a CSV, without writing code — but with
the rigor (and the reproducible Python) of writing code.**

Nickplots is a desktop app for researchers. You load a table, pick a plot, map your
columns to the plot's channels, style it, run the right statistical test, and export a
vector/TIFF figure at 300+ DPI (or the exact Python script that reproduces it). It ships
with 26 plot types, robust non‑parametric statistics, survival analysis, cell‑tracking
plots, gating/regions, and tools that steer you away from common mistakes such as
pseudo‑replication.

---

## Table of contents

- [What it is & design philosophy](#what-it-is--design-philosophy)
- [Architecture](#architecture)
- [Install & run](#install--run)
- [Core concepts](#core-concepts-read-this-first)
- [Loading & managing data](#loading--managing-data)
- [Preparing data (wide→long, formulas, aggregation)](#preparing-data)
- [The plot catalog](#the-plot-catalog) — every plot and the columns it takes
- [Style tab](#style-tab)
- [Legend tab](#legend-tab)
- [Annotations](#annotations)
- [Reshaping the figure (drag handles / mm size)](#reshaping-the-figure)
- [Analysis tab](#analysis-tab)
- [Advanced tab](#advanced-tab)
- [Export & session](#export--session)
- [Reproducibility & good‑practice notes](#reproducibility--good-practice-notes)

---

## What it is & design philosophy

- **Tidy‑data first, but forgiving.** Most plots expect *long* (tidy) data — one row per
  observation, with a categorical column telling groups apart. If your data is *wide*
  (one column per condition), the app tidies it for you (see [Preparing data](#preparing-data)).
- **Publication output.** Export to **TIFF / PNG / PDF / SVG** at ≥300 DPI, with exact
  physical size in millimetres for journal columns, optional transparent background, and a
  one‑to‑one **Python script** that regenerates the figure.
- **Statistics that a reviewer will accept.** Non‑parametric tests by default, effect
  sizes with bootstrap confidence intervals, multiple‑comparison correction, and explicit
  guards against pseudo‑replication.
- **Declarative & extensible.** Every plot is a `PlotSpec` (type + channels + parameters).
  The UI is generated from that schema, so the app stays consistent and adding a plot never
  requires touching the interface.

## Architecture

```
main_web.py     -> creates the PyWebView window, wires the JS <-> Python bridge
web/index.html  -> the entire UI (HTML/CSS/JS, single file)
api.py          -> class Api: every method the UI can call; holds the working DataFrame(s)
core/
  data_loader.py  -> load CSV/Excel, column typing, filter (df.query), melt/pivot
  plot_registry.py-> the declarative catalog: one PlotSpec per plot + its renderer
  plot_engine.py  -> matplotlib/seaborn engine: theme, legend, axes, annotations, export
  stats.py        -> analysis: tests, effect sizes, survival, MSD, diversity, aggregation
  codegen.py      -> turns the current figure state into a runnable Python script
  figures.py      -> clustermap / pairplot
```

The front end (HTML) renders in the OS‑native WebView; the back end (matplotlib/seaborn)
renders the figure to a PNG for the live preview and to the real file on export.

## Install & run

Requirements (see `requirements-web.txt`):

```
pywebview>=5.0  matplotlib>=3.8  seaborn>=0.13  pandas>=2.0
scipy>=1.11     scikit-learn>=1.3  openpyxl>=3.1
# optional: umap-learn>=0.5   (only needed for the UMAP embedding)
```

```bash
pip install -r requirements-web.txt
python main_web.py
```

- **Windows:** uses the Edge WebView2 runtime (already present on Win10/11).
- **Linux:** needs `python3-gi gir1.2-webkit2-4.0` (or `pip install pywebview[gtk]`).

---

## Core concepts (read this first)

**Channels.** Each plot declares slots (channels) you fill with columns:

| Channel | Meaning |
|---|---|
| `x`, `y` | the axes |
| `hue` | split/colour by a categorical (or numeric) column |
| `size`, `style` | extra scatter encodings |
| `id` | the subject/track identity (paired plots, trajectories, MSD) |
| `rep` | replicate identity (SuperPlot) |
| `time`, `event` | duration and status (Kaplan‑Meier); ordering (trajectories/MSD) |

In this README each plot lists its channels as `*name(accepted types)`, where `*` means
**required** and the types are `number`, `category`, `datetime`. A column's type is inferred
on load; you can force a numeric column to categorical (e.g. `track_id`) with
[Numeric → categorical](#advanced-tab).

**Layers / overlays.** Click **▶ Plot** to draw the base layer. **＋ Overlay** stacks
another plot on the same axes (e.g. a violin under a strip). The **▶ Plot** button is always
available in the top toolbar, next to the annotation tools, so you never have to scroll.

**Filter.** A filter narrows the working data for *every* plot and test. Build conditions
with the friendly picker (column · operator · value) or type a raw pandas `query()`
expression for full power.

---

## Loading & managing data

- **⬆ CSV / ⬆ Excel** — load a file. For Excel you then pick the sheet.
- **⌨ Type data (Prism style)** — a spreadsheet you can paste into straight from
  Excel/Prism (tab = column, first row = header, optional comma‑decimal).
- **Datasets (multiple files)** *(Advanced tab)* — every CSV you load is kept in a list.
  Click one to switch to it; **✕** removes it; **Combine all** concatenates them into a
  single table with a `dataset` source column — ideal for comparing files or building a
  **SuperPlot** where each file is a replicate.
- **Filter** *(Plot tab)* —
  - *Friendly:* choose a **column**, an **operator** (is equal to / not equal / greater /
    greater‑or‑equal / less / less‑or‑equal / **contains**), a **value**, then **＋ Add
    condition**. Conditions are ANDed into the query box.
  - *Advanced:* type a pandas expression, e.g. `dose > 5 and group == "high"`.
  - **Clear filter** empties it.

## Preparing data

Everything here lives in the **Advanced** tab (except the wide banner, which pops up on the
Plot tab when the data looks wide).

- **Plot wide columns (no melting).** Your measure is spread across several columns
  (e.g. `control`, `treated`, `washout`, each holding values)? Tick those columns, choose a
  plot type, and click **Tidy & plot** — the app melts them internally (columns become the
  X categories, values go to `value`), configures the plot, and draws it. It also readies
  the data for **Compare groups**. A banner nudges you toward this whenever a table has no
  categorical column but ≥2 numeric ones.
- **Reshape data (long ⇄ wide).** The manual version: **melt** (wide→long, choose
  id/value columns) and **pivot** (long→wide, choose index/columns/values).
- **New column (formula).** Create a column from a formula over existing ones. Use the
  builder (`colA op colB`) or type any pandas expression, e.g. `net_disp / cum_path`
  (persistence). Wrap names containing spaces in backticks.
- **Aggregate (per replicate / track).** Collapse rows to **one row per group**
  (mean / median / sum / count of every numeric column). **This is the fix for
  pseudo‑replication:** aggregate cells to their track/replicate *before* testing, so *n*
  is the number of replicates, not the number of cells. Pick several columns to keep groups
  distinct (e.g. `treatment` + `track_id`). Adds an `n_obs` column.
- **Numeric → categorical.** Convert a numeric column (e.g. `track_id`) to text so it can be
  used as a `hue`/`x`/`rep`.

---

## The plot catalog

Legend: `*required(type)`. Types: `number`, `category`, `datetime`. Every plot also takes a
`palette`. Categorical plots colour by `hue`; if `hue` equals `x` it is colour‑only (points
stay centred, not dodged).

### Relationships (X vs Y)

**Scatter** `key=scatter`
`*x(number/datetime) *y(number) hue(category/number) size(number) style(category)` ·
params: `alpha`, `regression` (adds a fit line + R²).
Points are **clickable**: click one to see its full data row.
*Example:* `x=area`, `y=speed`, `hue=treatment`, `regression=on`.

**Scatter + group background** `key=scatter_density`
`*x(number) *y(number) *hue(category) size(number) style(category)` ·
params: `psize`, `alpha` (background intensity), `grid` (resolution), `contour` (KDE‑wave
background instead of solid tint), `levels`.
A scatter whose background is tinted by whichever group dominates each region. Clickable.

**Regression + CI band** `key=regband`
`*x(number) *y(number) hue(category)` · params: `ci` (%), `alpha`.
One regression line + confidence band per group.

**Line** `key=line`
`*x(number/category/datetime) *y(number) hue(category)` ·
params: `linewidth`, **`errorbar`** (`none`/SD/SEM/CI95), `alpha`.
`errorbar=none` (default) is fast; SD/SEM are cheap; **CI95 bootstraps** and is slow on
many‑points‑per‑x data — opt in deliberately.

### Group comparisons (a value across categories)

All take `*x(category) *y(number) hue(category)`.

- **Bar** `key=bar` — simple bars.
- **Boxplot** `key=box`.
- **Violin** `key=violin`.
- **Stripplot** `key=strip` — jittered raw points · param `alpha`.
- **Box + points** `key=box_points` — box with individual points · params `size`, `jitter`,
  `alpha`.
- **Violin + points** `key=violin_points` — same, with a violin.
- **Bar/point ± error** `key=bar_err` — mean/median with an error bar and optional points ·
  params: `kind` (bar/point), `center` (mean/median), `error` (SD/SEM/CI95), `points`,
  `psize`, `sig` (draw significance brackets, Welch t + Holm, when there is no `hue`).
- **SuperPlot (cells + replicate means)** `key=superplot`
  `*x(category) *y(number) rep(category/number)` · params `size`, `alpha`.
  Plots every cell faintly **plus one large marker per replicate mean**, coloured by
  replicate, with the grand mean per condition. The honest way to show data with many cells
  per replicate (Lord et al., *JCB* 2020). Test on the replicate means (aggregate first).

### Distributions (one variable)

- **Histogram** `key=hist` — `*x(number) hue(category)` · params `bins`, `kde`, `alpha`.
- **KDE** `key=kde` — `*x(number) hue(category)` · params `fill`, `alpha`.
- **ECDF (cumulative)** `key=ecdf` — `*x(number) hue(category)`.
- **Ridgeline** `key=ridge` — `*x(number) *hue(category)` · params `overlap`, `alpha`.
  One stacked KDE per group; reveals bimodality and per‑condition shifts.
- **Q‑Q plot (normality)** `key=qq` — `*val(number) hue(category)`.
  Points on the diagonal ⇒ normally distributed. Use it to justify parametric vs
  non‑parametric choices (and see what `Compare groups` checks internally).

### Correlation / matrices

- **Heatmap (correlation)** `key=heatmap` — no channels; correlation matrix of all numeric
  columns · params `annot`, `palette` (colormap).
- **Matrix heatmap** `key=heatmap_matrix` — no channels; the numeric matrix itself ·
  params `zscore` (per column), `annot`, `palette`.

### Categorical composition

**Stacked composition (+ chi²/Fisher)** `key=stacked`
`*x(category) *hue(category)` · params `proportion` (else counts), `show_p`
(annotate the chi‑square p on the plot), `palette`.
For each X category, the composition of `hue` categories (e.g. `outcome` by `treatment`).
Pair it with the [Composition test](#analysis-tab).

### Survival

**Kaplan‑Meier (survival + log‑rank)** `key=km`
`*time(number) event(category/number) hue(category)` ·
params: **`event_value`** (which status counts as the event; comma‑separated; empty ⇒ every
non‑empty status counts), `censors` (draw censor ticks), `palette`.
Survival curves with censoring marks and the **log‑rank p** annotated when grouped. Example
for cell tracking: `time=lifetime`, `event=outcome`, `event_value=Mitosis`, `hue=treatment`.
Ignoring censoring (e.g. a boxplot of `lifetime`) biases the result — use this instead.

### Cell tracking / motility

- **Trajectories (XY tracks)** `key=traj`
  `*x(number) *y(number) *id(track) time(number, order) hue(category)` ·
  params `linewidth`, `alpha`, `equal` (equal aspect). One path per track, coloured by group.
- **MSD (mean squared displacement)** `key=msd`
  `*id(track) *px(number) *py(number) time(number, order) hue(category)` · param `loglog`.
  Per‑track MSD averaged within each group; the legend shows the track count.

### Paired / specialized

- **Paired (before‑after)** `key=paired` — `*x(category) *y(number) *id(subject)` ·
  param `alpha`. Connects the same subject across the X conditions.
- **Dose‑response (4PL + IC50)** `key=dose4pl` — `*x(dose,number) *y(response,number)
  hue(category)` · params `log_x`, `alpha`. Fits a 4‑parameter logistic and annotates IC50
  and R² per group. Clickable.
- **Volcano Plot** `key=volcano` — `*x(log2FC,number) *y(p‑value,number)` ·
  params `fc_thr`, `p_thr`, `alpha`. Thresholds shown as dashed lines; up/down highlighted.

---

## Style tab

- **Theme** — `Light · grid / ticks / clean`, `Dark · grid / plain`, `Slate` (coherent
  background + ink + grid).
- **Context (scale)** — `paper / notebook / talk / poster` (seaborn scaling for fonts/lines).
- **Font scale** — fine multiplier on top of the context.
- **Grid** — toggle the grid.
- **Show n per group** — appends `(n=k)` to categorical **X** tick labels. (After
  aggregation, *k* becomes the replicate count — a built‑in pseudo‑replication check.)
- **Color override** — set the figure/axes background colour explicitly (clear to return to
  the theme).
- **Per‑group colors** — pick the **exact** colour of each `hue`/`x` category. Sticky:
  categories keep their colour when you swap axes or filter. Applies to any plot with a hue.
- **Axes** — log X / log Y, manual X/Y limits, despine (hide top/right spines), tick size.
- **Plot size / shape (mm)** — width × height in millimetres sets the exact physical shape
  (e.g. `180×70` flat, `80×160` tall) for preview, export and panels. Empty = default. You
  can also set this **visually** by dragging the handles on the plot — see below. The font is
  the system sans‑serif (Arial where available, else DejaVu Sans).

## Legend tab

Show/hide; position (the named corners, **Outside** on the right, or **Free** — then
click‑drag the legend directly on the plot); font size; number of columns; frame on/off;
"fit" (shrink the axes so an outside/free legend is never clipped); legend title; and
per‑item **relabelling** (pretty names without touching the data). Column/axis pretty‑names
are set via the aliases (visual only).

## Annotations

The toolbar above the plot: type text and drop a **Text**, an **Arrow**, or a **✱** marker
by clicking on the plot; **Undo** / **Clear**. Annotations are placed in axes‑fraction
coordinates, so they scale with the figure and travel into multi‑figure panels.

## Reshaping the figure

Beyond typing millimetres, drag the **purple handles** on the **right**, **bottom**, and
**corner** of the plot to flatten or stretch it. This is a **real reshape of the plotting
space** — the axes are re‑laid‑out and the ticks re‑computed (it is *not* a stretched image),
exactly like setting the mm size, just hands‑on. The final shape is what gets exported.
**Double‑click the corner** to reset to automatic. The handles show only for single plots
(not panels or clustermap/pairplot).

---

## Analysis tab

- **Dimensionality reduction** — pick numeric columns and run **PCA / t‑SNE / UMAP**; the
  new components (`PC1/PC2`, `tSNE1…`, `UMAP1…`) are appended as columns you can plot.
- **Compare groups (robust)** — choose a numeric value and a categorical group. Returns a
  rigorous, display‑ready report: normality check, **both** the non‑parametric and the
  parametric test, **effect size with bootstrap CI95**, and, for 3+ groups, pairwise
  comparisons with **Benjamini‑Hochberg** correction.
- **📋 Methods sentence** — the same comparison as one copy‑ready sentence for a figure
  legend / methods section (test name, *n* per group, effect size + CI, p).
- **Composition test (chi² / Fisher)** — association between two categorical columns
  (e.g. `treatment × outcome`): chi‑square, **Cramér's V**, and **Fisher exact** for 2×2,
  with a low‑expected‑count warning. The count table lands in the plot table (exportable).
- **Diversity (Shannon index)** — with drawn regions: *Groups per region* (H of the mixture
  inside each gate) or *Regions per group* (relative Shannon of the regions a group occupies);
  H in nats, J = Pielou.
- **Best pairs (group separation)** — tests all numeric columns pairwise and ranks which two
  best separate the chosen groups (by silhouette; LDA accuracy shown).
- **Threshold lines & regions** — draw horizontal / vertical / slanted lines (type values or
  click the plot); build **regions (gates)** from the line intersections; analyse counts per
  region.
- **Named regions (gating)** — draw, name, and drag closed polygons on the plot; live
  per‑group counts; **merge** same‑named regions; **Create column** writes each observation's
  region into a new categorical column; **save/load** regions to reuse on another sheet;
  **export gating** assigns each observation to a single best‑fit region and writes CSV/Excel
  (one file or one per region).

## Advanced tab

- **Numeric → categorical**, **Datasets**, **Aggregate**, **New column (formula)**,
  **Plot wide columns**, **Reshape** — see [Preparing data](#preparing-data).
- **Plot template** — **Save** the current plot recipe (type + channel mapping + params +
  labels) and **Apply** it to another dataset with the same column names. Unlike a project
  (which is bound to one CSV) or a style preset (which is only styling), a template is the
  *plot definition* itself.
- **Multi‑figure panel (A/B/C)** — build a plot, **＋ Add current plot to the panel**, repeat
  (you can switch CSV between frames — each frame keeps its own data, title, axis labels,
  **and its annotations/lines/regions**). Set **Panel columns**, **Build panel**, then **drag
  the dividers** on the panel to resize each plot. Export the whole panel.
- **Clustermap / Pairplot** — pick numeric columns; clustermap (optional z‑score) or pairplot
  (optional hue). Export the figure.
- **Descriptive table** — per‑group summary statistics; view in the table area and export CSV.

## Export & session

Everything lives in the footer (kept compact):

- **⬇ Export figure** — save **TIFF / PNG / PDF / SVG**. Under **Export ▾**: the **DPI**
  (≥300), a **transparent background** option (keeps ticks/labels/title readable by re‑inking
  dark themes), **⬇ Python code** (a runnable script + a data snapshot that reproduces the
  figure), and **Filtered CSV** (the current filtered/reshaped data).
- **Session ▾** — **Save/Open project** (the full UI state + the CSV path) and
  **Save/Apply preset** (style + legend only).
- Exact **mm size** (typed or via the drag handles) is honoured on export, so a figure keeps
  its journal‑column shape instead of being cropped to content.

---

## Reproducibility & good‑practice notes

- **Pseudo‑replication.** If your rows are not independent (many cells per track, many tracks
  per colony), a raw test inflates *n* and shrinks *p*. Use **Aggregate** to collapse to the
  experimental unit before testing, and/or a **SuperPlot** to show it honestly. **Show n per
  group** exposes what *n* the plot is actually using.
- **Non‑parametric by default.** `Compare groups` reports the non‑parametric result first and
  gives effect sizes with CIs, not just p‑values. Use the **Q‑Q plot** to judge normality.
- **Censoring.** For time‑to‑event data (`lifetime` + `outcome`) use **Kaplan‑Meier +
  log‑rank**, not a boxplot of the durations.
- **The figure is reproducible.** **⬇ Python code** exports a script (plus a CSV snapshot)
  that regenerates the figure outside the app — good for supplementary material and audits.
  (A few of the newest plots fall back to a comment instead of full code; the figure and its
  data are still exported.)

---

*Nickplots — publication figures, honest statistics, reproducible output.*
