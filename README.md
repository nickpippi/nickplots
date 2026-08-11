# Nickplots

**Publication-quality statistical figures from a CSV, without writing code — but with
the rigor (and the reproducible Python) of writing code.**

Nickplots is a desktop app for researchers. You load a table, pick a plot, map your
columns to the plot's channels, style it, run the right statistical test, and export a
vector/TIFF figure at 300+ DPI (or the exact Python script that reproduces it). It ships
with 29 plot types, robust non‑parametric statistics, survival analysis, cell‑tracking
plots, gating/regions, journal presets, a colour‑blindness check, an on‑canvas editor,
multiple plot tabs, and tools that steer you away from common mistakes such as
pseudo‑replication.

---

## Table of contents

- [What it is & design philosophy](#what-it-is--design-philosophy)
- [Architecture](#architecture)
- [Install & run](#install--run)
- [Core concepts](#core-concepts-read-this-first)
- [Plot tabs (several plots open)](#plot-tabs)
- [Loading & managing data](#loading--managing-data)
- [Preparing data (wide→long, formulas, aggregation)](#preparing-data)
- [The plot catalog](#the-plot-catalog) — every plot and the columns it takes
- [Style tab (+ journal presets, accessibility check)](#style-tab)
- [Legend tab](#legend-tab)
- [Visual editor (edit on the canvas)](#visual-editor)
- [Annotations](#annotations)
- [Reshaping the figure (drag handles / mm size)](#reshaping-the-figure)
- [Split by (facet)](#split-by-facet)
- [Analysis tab](#analysis-tab)
- [Bench math panel](#bench-math-panel)
- [Statistics panel](#statistics-panel)
- [Data tools panel](#data-tools-panel)
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

**Helpful errors.** When a plot can't be built (missing grouping column, wrong column type,
no numeric data…), the canvas shows a plain‑language card — *what* went wrong, *why* this
plot needs it, and concrete steps to fix it — instead of a raw error.

---

## Plot tabs

A row of tabs above the figure keeps **several plots open at once** — like tabs in a
browser. Each tab holds a whole plot (its layers, labels, style, legend, annotations,
regions **and the dataset it uses**).

- Click a tab to switch to it — if that plot used a different loaded CSV, it switches the
  dataset for you before redrawing.
- **＋** opens a new tab that **duplicates the current one** (a good starting point for a
  variation).
- **✕** closes a tab; **double‑click** a tab to rename it.
- **Advanced → Multi‑figure panel → "Build panel from all open tabs"** turns every open
  tab into a panel frame in one click (each frame keeps its own data and overlays).

Tabs live for the session; `Save project` still stores the active plot.

---

## Loading & managing data

- **⬆ CSV / ⬆ Excel** — load a file. For Excel you then pick the sheet.
- **Drag‑and‑drop** a `.csv` anywhere onto the window to load it instantly (for Excel use
  the ⬆ Excel button).
- **⌨ Type data (Prism style)** — a spreadsheet you can paste into straight from
  Excel/Prism (tab = column, first row = header, optional comma‑decimal).
- **Data preview** *(Plot tab, on load)* — a fold‑out showing the first rows, the detected
  type per column, and a one‑line health summary (row × column count, duplicates, columns
  that look numeric but don't parse). Run the full check in Data tools → Data health.
- **✨ Recommended plots** *(Plot tab, on load)* — reads the column types and suggests plots
  (e.g. *Area by Treatment → Box + points / Violin + points / ECDF*; *Area vs Speed →
  Scatter / Regression*). When it detects an id/replicate column it recommends a
  **SuperPlot** and warns about pseudo‑replication. Click a suggestion to apply it.
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
params: `alpha`, `regression` (adds a fit line + R²), `shape_hue` (also encode the hue on the
marker **shape** — colour‑blind safe).
Points are **clickable**: double‑click one to see its full data row in a popup.
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

All take `*x(category) *y(number) hue(category)`. Box / violin / strip / box+points /
violin+points also take a **`sig`** param that draws **significance brackets** (`*/**/***`)
between the x groups when there is no hue.

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
  can also set this **visually** by dragging the handles on the plot — see below.
- **Journal preset** — one click sets the column width (mm), font family, absolute font size
  (pt) and line weight (pt) to a journal's spec: **Nature 1‑/2‑col** (89 / 183 mm),
  **Cell 1‑col** (85 mm), **Science 1‑/2‑col** (55 / 120 mm). Height stays yours. Fine‑tune
  with the explicit **Font family**, **Font size (pt)** and **Line weight (pt)** controls
  right below (0 = automatic).
- **Accessibility check** — preview the figure as a **colour‑blind** reader
  (deuteranopia / protanopia / tritanopia) or in **grayscale** (the print test journals
  increasingly require) sees it. This only changes the on‑screen preview, never the export.
  Point plots also have a **"shape by hue"** option (scatter) so colour isn't the only cue.

## Legend tab

Show/hide; position (the named corners, **Outside** on the right, or **Free** — then
click‑drag the legend directly on the plot); font size; number of columns; frame on/off;
"fit" (shrink the axes so an outside/free legend is never clipped); legend title; and
per‑item **relabelling** (pretty names without touching the data). Column/axis pretty‑names
are set via the aliases (visual only). You can also **double‑click a legend entry (or its
title) directly on the plot** to rename it (see [Visual editor](#visual-editor)).

## Visual editor

Once a figure is drawn, edit the most‑changed elements **directly on the canvas** instead of
hunting in the side panels:

- **Double‑click the title, X label or Y label** → rename it inline. **Right‑click** them for
  options (edit, reset to default, jump to the font settings).
- **Double‑click a legend entry** → rename that series; **double‑click the legend title** →
  rename it. The legend still **drags to move** (with a small threshold so a double‑click
  doesn't nudge it) and is clamped inside the plot.
- **Double‑click a point** (on pickable plots like scatter) → a floating popup shows that
  observation's full data row, with copy/close — no bottom panel taking space.

Individual points, tick numbers and gridlines are not yet click‑to‑edit (that needs an
interactive‑SVG canvas); the side panels still cover those.

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

### Enzyme kinetics, growth and calibration

- **Michaelis-Menten (Km / Vmax)** `key=michaelis` — `*x([S] substrate) *y(v rate) hue(group)` ·
  params `show_km`, `alpha`. Fits `v = Vmax·[S]/(Km+[S])` per group and annotates Km, Vmax, R².
- **Growth curve (doubling time)** `key=growth` — `*x(time) *y(OD/signal) hue(group)` ·
  params `log_y`, `fit from time`, `fit to time`. Fits the exponential phase and annotates
  the **doubling time**, µ and R². Restrict the fit window to the log phase with the two
  time parameters.
- **Standard curve (+ R²)** `key=stdcurve` — `*x(known concentration) *y(signal)` ·
  param `model` (linear / 4PL). Draws the standards and the fit; interpolate unknowns in
  the [Bench math panel](#bench-math-panel).

---

## Split by (facet)

In the Plot tab, **Split by (facet)** draws the *same* plot once per level of a categorical
column, in a grid — one click instead of building a panel frame by frame. Choose the number
of columns, and keep **same scale** on so the panels are actually comparable (all facets
share one axis range). Works with every plot type. Faceted figures have many axes, so point
picking and the legend/region overlays are turned off (as in a panel).

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
- Every line and region carries its own **style** (solid / dashed / dotted / dash‑dot),
  **width** and **colour**, editable per item in the list at any time.

---

Three toolboxes open as their own window from the buttons under the tabs. Each has its own
left-hand menu, so the sidebar stays uncluttered.

## Bench math panel

**🧪 Bench math** — the arithmetic that otherwise sends you back to Excel.

- **qPCR (ΔΔCt)** — pick the Ct, gene, sample and group columns, then the housekeeping gene
  and the reference group. Averages technical replicates per sample+gene, computes
  ΔCt (target − housekeeping), ΔΔCt vs. the reference group and **fold change 2^-ΔΔCt**.
  The tidy result becomes your working data — plot `fold_change` by group straight away.
- **Standard curve** — fits the rows that carry a known concentration (BCA / Bradford /
  ELISA), then **interpolates every row's signal back to a concentration** in a new column.
  Warns when R² < 0.98 and when a value falls outside the standard range (extrapolation).
- **Normalize** — `% of control`, `fold of control`, `subtract control` (all relative to
  the mean of the control level you choose), **`divide by column`** (the Western blot case:
  signal ÷ loading control), `z-score` and `min-max` (optionally within group).
- **Kinetics & growth** — shortcuts to the three fits that are *pictures*: Michaelis-Menten,
  growth curve and standard curve.

## Statistics panel

**📊 Statistics** — the rigor that one-way comparisons don't cover.

- **Which test?** — inspects the design and the data (group count, n, Shapiro normality,
  Levene equal variance, paired or not) and **names the test with the reasoning** and where
  to run it. Start here if you are unsure.
- **Two-way ANOVA** — two factors + interaction (e.g. `treatment × time`), Type II sums of
  squares, unbalanced-safe. A significant interaction means one factor's effect depends on
  the other.
- **Repeated measures** — within-subject ANOVA plus the right non-parametric partner
  (**Wilcoxon** for 2 conditions, **Friedman** for 3+). Subjects missing a condition are
  dropped and reported.
- **Power / sample size** — two-sample t-test: an effect size (Cohen's d) gives the **n per
  group** for your target power; an n gives the achieved power. (d = 0.8 → n = 26/group.)
- **Outliers** — Grubbs (iterated), IQR (1.5×) or MAD (3.5), optionally within group.
  It **never deletes**: it can write a true/false column so the decision is documented.
- **Correlation + p** — pairwise r with p-values **BH-corrected across all pairs**, as an
  exportable table. For the figure, use the `Heatmap (correlation)` plot with
  **significance stars** on (and Pearson/Spearman selectable).

## Data tools panel

**🩺 Data tools** — before and around the analysis.

- **Data health** — missing values per column, duplicated rows, constant/empty columns, and
  columns that *look* numeric but do not fully parse (decimal comma, stray text). Run it
  right after importing.
- **Data overview** — small multiples of every selected numeric column in one figure: a
  histogram, or a box + points per group when you pick a grouping column; each title shows
  the missing count. The fastest way to see a whole table at once.
- **Plate import** — paste a 96/384-well grid straight from Excel (tab-separated); row and
  column labels are stripped automatically. Paste an optional same-shaped **condition map**
  and you get a tidy table (`well, row, col, value, condition`).

> Every checkbox list of columns in the app has an **All / None** bar.

## Advanced tab

- **Numeric → categorical**, **Datasets**, **Aggregate**, **New column (formula)**,
  **Plot wide columns**, **Reshape** — see [Preparing data](#preparing-data).
- **Plot template** — **Save** the current plot recipe (type + channel mapping + params +
  labels) and **Apply** it to another dataset with the same column names. Unlike a project
  (which is bound to one CSV) or a style preset (which is only styling), a template is the
  *plot definition* itself.
- **Multi‑figure panel (A/B/C)** — build a plot, **＋ Add current plot to the panel**, repeat
  (you can switch CSV between frames — each frame keeps its own data, title, axis labels,
  **and its annotations/lines/regions**). Or **Build panel from all open tabs** to turn every
  [plot tab](#plot-tabs) into a frame in one click. Set **Panel columns**, **Build panel**,
  then **drag the dividers** to resize each plot.
  - **Irregular layout (mosaic)** — for uneven panels, type a layout where each **letter is a
    frame** (A = 1st added, B = 2nd…) and rows are separated by `;`; a repeated letter
    **spans** its cells. E.g. `AC;BC` = A and B stacked on the left, C filling the whole right
    column; `ACD;BCD` = A,B left, C the middle column, D the right column. Use `.` for an empty
    cell. Each letter must form a rectangle. Leave empty for the regular grid.
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
