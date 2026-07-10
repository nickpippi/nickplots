"""Desktop UI (CustomTkinter) — tabs, live preview, draggable legend,
figure that follows the preview size, coherent light/dark themes."""
from __future__ import annotations
import customtkinter as ctk
from tkinter import filedialog, messagebox, colorchooser
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core import data_loader as DL
from core import plot_engine as E
from core import stats as S
from core.plot_engine import Layer, Style, Legend, PlotConfigError, THEMES
from core.plot_registry import REGISTRY, NUMBER, CATEGORY

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

LEGEND_POS = {
    "Automatic (upper right)": "upper right", "Upper left": "upper left",
    "Lower right": "lower right", "Lower left": "lower left",
    "Center": "center", "Outside (right)": "outside", "Free (drag)": "free",
}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Nickplots")
        self.geometry("1380x860")
        self.minsize(1160, 720)
        self.df = self.view = self.fig = self.canvas = None
        self.stack: list[dict] = []
        self.style = Style()
        self.legend = Legend()
        self.chan_w, self.param_w = {}, {}
        self._pick = None
        self._drag = None
        self._job = self._resize_job = None
        self._last_size = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    # ============================ sidebar ============================ #
    def _build_sidebar(self):
        side = ctk.CTkFrame(self, width=390, corner_radius=0)
        side.grid(row=0, column=0, sticky="nsew")
        side.grid_rowconfigure(2, weight=1)
        side.grid_propagate(False)

        head = ctk.CTkFrame(side, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        ctk.CTkLabel(head, text="Nickplots", font=("", 20, "bold")).pack(side="left")
        ctk.CTkSegmentedButton(head, values=["Dark", "Light"],
                               command=lambda v: ctk.set_appearance_mode(v.lower()),
                               width=120).pack(side="right")

        topbtn = ctk.CTkFrame(side, fg_color="transparent")
        topbtn.grid(row=1, column=0, sticky="ew", padx=12)
        ctk.CTkButton(topbtn, text="Load CSV", command=self.on_load).pack(fill="x")
        self.lbl_file = ctk.CTkLabel(topbtn, text="(no file)", text_color="gray60", wraplength=350)
        self.lbl_file.pack(fill="x", pady=(2, 6))

        self.tabs = ctk.CTkTabview(side, corner_radius=10)
        self.tabs.grid(row=2, column=0, sticky="nsew", padx=8)
        for name in ("Plot", "Style", "Legend", "Analysis"):
            self.tabs.add(name)
        self._tab_plot()
        self._tab_style()
        self._tab_legend()
        self._tab_analysis()

        foot = ctk.CTkFrame(side, fg_color="transparent")
        foot.grid(row=3, column=0, sticky="ew", padx=12, pady=8)
        ctk.CTkButton(foot, text="Export figure…", command=self.on_export).pack(fill="x", pady=2)
        ctk.CTkButton(foot, text="Export filtered CSV…", fg_color="gray30",
                      command=self.on_export_csv).pack(fill="x", pady=2)

    def _scroll(self, tab):
        f = ctk.CTkScrollableFrame(self.tabs.tab(tab), fg_color="transparent")
        f.pack(fill="both", expand=True)
        return f

    def _tab_plot(self):
        f = self._scroll("Plot")
        ctk.CTkLabel(f, text="Filter (pandas query)").pack(anchor="w")
        self.ent_filter = ctk.CTkEntry(f, placeholder_text="dose > 5 and group == 'high'")
        self.ent_filter.pack(fill="x")
        ctk.CTkButton(f, text="Apply filter", command=self.on_filter).pack(fill="x", pady=3)

        ctk.CTkLabel(f, text="Type", font=("", 13, "bold")).pack(anchor="w", pady=(8, 0))
        self.opt_plot = ctk.CTkOptionMenu(f, values=[s.label for s in REGISTRY.values()],
                                          command=lambda *_: self.rebuild_config())
        self.opt_plot.pack(fill="x")
        self.cfg = ctk.CTkFrame(f, fg_color="transparent")
        self.cfg.pack(fill="x", pady=4)

        for txt, attr in [("Title", "title"), ("X label", "xlabel"), ("Y label", "ylabel")]:
            ctk.CTkLabel(f, text=txt).pack(anchor="w")
            e = ctk.CTkEntry(f); e.pack(fill="x")
            e.bind("<KeyRelease>", lambda *_: self.schedule())
            setattr(self, f"ent_{attr}", e)

        ctk.CTkButton(f, text="▶  Plot (base layer)", command=self.on_plot).pack(fill="x", pady=(8, 2))
        ctk.CTkButton(f, text="＋ Overlay this layer", command=self.on_overlay).pack(fill="x", pady=2)
        ctk.CTkButton(f, text="Clear layers", fg_color="gray30", command=self.on_clear).pack(fill="x", pady=2)
        self.lbl_stack = ctk.CTkLabel(f, text="layers: —", text_color="gray60", wraplength=330, justify="left")
        self.lbl_stack.pack(fill="x")

    def _tab_style(self):
        f = self._scroll("Style")
        ctk.CTkLabel(f, text="Plot theme").pack(anchor="w")
        self.opt_theme = ctk.CTkOptionMenu(f, values=list(THEMES.keys()), command=lambda *_: self.apply_style())
        self.opt_theme.set(self.style.theme); self.opt_theme.pack(fill="x")
        ctk.CTkLabel(f, text="Context (element scale)").pack(anchor="w", pady=(6, 0))
        self.opt_ctx = ctk.CTkOptionMenu(f, values=["paper", "notebook", "talk", "poster"],
                                         command=lambda *_: self.apply_style())
        self.opt_ctx.set("notebook"); self.opt_ctx.pack(fill="x")
        ctk.CTkLabel(f, text="Font scale").pack(anchor="w", pady=(6, 0))
        self.sld_font = ctk.CTkSlider(f, from_=0.7, to=1.8, command=lambda *_: self.apply_style())
        self.sld_font.set(1.0); self.sld_font.pack(fill="x")
        self.chk_grid = ctk.CTkCheckBox(f, text="Grid", command=self.apply_style)
        self.chk_grid.select(); self.chk_grid.pack(anchor="w", pady=6)
        ctk.CTkButton(f, text="Override: figure background", command=lambda: self._pick_color("fig_bg")).pack(fill="x", pady=2)
        ctk.CTkButton(f, text="Override: axes background", command=lambda: self._pick_color("ax_bg")).pack(fill="x", pady=2)
        ctk.CTkButton(f, text="Clear overrides (back to theme)", fg_color="gray30",
                      command=self._clear_overrides).pack(fill="x", pady=2)

    def _tab_legend(self):
        f = self._scroll("Legend")
        self.chk_leg = ctk.CTkCheckBox(f, text="Show legend", command=self.apply_legend)
        self.chk_leg.select(); self.chk_leg.pack(anchor="w", pady=4)
        ctk.CTkLabel(f, text="Position").pack(anchor="w")
        self.opt_legpos = ctk.CTkOptionMenu(f, values=list(LEGEND_POS.keys()), command=lambda *_: self.apply_legend())
        self.opt_legpos.pack(fill="x")
        ctk.CTkLabel(f, text="Font size").pack(anchor="w", pady=(6, 0))
        self.sld_legsize = ctk.CTkSlider(f, from_=6, to=20, command=lambda *_: self.apply_legend())
        self.sld_legsize.set(9); self.sld_legsize.pack(fill="x")
        ctk.CTkLabel(f, text="Columns").pack(anchor="w", pady=(6, 0))
        self.sld_legcol = ctk.CTkSlider(f, from_=1, to=4, number_of_steps=3, command=lambda *_: self.apply_legend())
        self.sld_legcol.set(1); self.sld_legcol.pack(fill="x")
        self.chk_legframe = ctk.CTkCheckBox(f, text="Frame", command=self.apply_legend)
        self.chk_legframe.select(); self.chk_legframe.pack(anchor="w", pady=6)
        ctk.CTkLabel(f, text="Free position — X / Y ('Free' mode or drag on the plot)",
                     wraplength=330, text_color="gray60").pack(anchor="w")
        self.sld_legx = ctk.CTkSlider(f, from_=-0.1, to=1.3, command=lambda *_: self.apply_legend())
        self.sld_legx.set(1.0); self.sld_legx.pack(fill="x")
        self.sld_legy = ctk.CTkSlider(f, from_=-0.1, to=1.3, command=lambda *_: self.apply_legend())
        self.sld_legy.set(1.0); self.sld_legy.pack(fill="x")

    def _tab_analysis(self):
        f = self._scroll("Analysis")
        ctk.CTkLabel(f, text="Dimensionality reduction", font=("", 13, "bold")).pack(anchor="w")
        ctk.CTkLabel(f, text="Numeric columns:").pack(anchor="w")
        self.num_box = ctk.CTkScrollableFrame(f, height=120); self.num_box.pack(fill="x")
        self.num_checks = {}
        self.opt_method = ctk.CTkOptionMenu(f, values=["PCA", "t-SNE", "UMAP"]); self.opt_method.pack(fill="x", pady=2)
        self.btn_reduce = ctk.CTkButton(f, text="Run reduction", command=self.on_reduce)
        self.btn_reduce.pack(fill="x", pady=2)
        ctk.CTkLabel(f, text="Group test (value × group)", font=("", 13, "bold")).pack(anchor="w", pady=(10, 0))
        self.opt_val = ctk.CTkOptionMenu(f, values=[""]); self.opt_val.pack(fill="x")
        self.opt_grp = ctk.CTkOptionMenu(f, values=[""]); self.opt_grp.pack(fill="x")
        ctk.CTkButton(f, text="Run t-test / ANOVA", command=self.on_test).pack(fill="x", pady=2)
        self.txt_stats = ctk.CTkTextbox(f, height=90); self.txt_stats.pack(fill="x")

    # ============================ main area ============================ #
    def _build_main(self):
        m = ctk.CTkFrame(self)
        m.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        m.grid_rowconfigure(0, weight=1); m.grid_columnconfigure(0, weight=1)
        self.plot_frame = ctk.CTkFrame(m, fg_color="transparent")
        self.plot_frame.grid(row=0, column=0, sticky="nsew")
        self.plot_frame.grid_rowconfigure(0, weight=1); self.plot_frame.grid_columnconfigure(0, weight=1)

        # canvas created ONCE; redraws reuse the same figure (no widget recreation)
        self.fig = Figure(figsize=(7, 5), dpi=110)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        cw = self.canvas.get_tk_widget()
        cw.grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        cw.bind("<ButtonRelease-1>", self._end_drag)      # Tk guarantees the release even outside the canvas
        cw.bind("<Configure>", self._on_resize)

        self.info = ctk.CTkTextbox(m, height=120)
        self.info.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.info.insert("1.0", "Click a point (scatter/volcano) to see its row. "
                                "Click and drag the legend to reposition it.")

    # ============================ helpers ============================ #
    def _spec(self):
        return next(s for s in REGISTRY.values() if s.label == self.opt_plot.get())

    def _cols(self, accepts):
        if self.view is None:
            return [""]
        return [""] + [c for c, k in DL.column_kinds(self.view).items() if k in accepts]

    def rebuild_config(self):
        for w in self.cfg.winfo_children():
            w.destroy()
        self.chan_w, self.param_w = {}, {}
        for ch in self._spec().channels:
            ctk.CTkLabel(self.cfg, text=ch.nice() + (" *" if ch.required else "")).pack(anchor="w")
            w = ctk.CTkOptionMenu(self.cfg, values=self._cols(ch.accepts)); w.pack(fill="x")
            self.chan_w[ch.name] = w
        for p in self._spec().params:
            ctk.CTkLabel(self.cfg, text=p.nice()).pack(anchor="w", pady=(4, 0))
            if p.kind == "bool":
                w = ctk.CTkCheckBox(self.cfg, text=""); (w.select if p.default else w.deselect)(); w.pack(anchor="w")
            elif p.kind in ("choice", "palette"):
                w = ctk.CTkOptionMenu(self.cfg, values=list(p.choices)); w.set(p.default); w.pack(fill="x")
            else:
                row = ctk.CTkFrame(self.cfg, fg_color="transparent"); row.pack(fill="x")
                lab = ctk.CTkLabel(row, text=str(p.default), width=46); lab.pack(side="right")
                w = ctk.CTkSlider(row, from_=p.lo, to=p.hi,
                                  command=lambda v, l=lab, k=p.kind: l.configure(text=str(int(v) if k == "int" else round(v, 2))))
                w.set(p.default); w.pack(side="left", fill="x", expand=True)
            self.param_w[p.name] = (p, w)

    def refresh_columns(self):
        nums = DL.columns_by_kind(self.view, NUMBER) if self.view is not None else []
        cats = DL.columns_by_kind(self.view, CATEGORY) if self.view is not None else []
        for cb in self.num_checks.values():
            cb.destroy()
        self.num_checks = {c: ctk.CTkCheckBox(self.num_box, text=c) for c in nums}
        for cb in self.num_checks.values():
            cb.pack(anchor="w")
        self.opt_val.configure(values=nums or [""]); self.opt_val.set(nums[0] if nums else "")
        self.opt_grp.configure(values=cats or [""]); self.opt_grp.set(cats[0] if cats else "")

    def _layer(self):
        spec = self._spec()
        mapping = {n: (w.get() or None) for n, w in self.chan_w.items()}
        params = {}
        for name, (p, w) in self.param_w.items():
            if p.kind == "bool":
                params[name] = bool(w.get())
            elif p.kind in ("choice", "palette"):
                params[name] = w.get()
            else:
                params[name] = int(w.get()) if p.kind == "int" else float(w.get())
        return dict(spec_key=spec.key, mapping=mapping, params=params)

    def _figsize(self):
        w = max(self.plot_frame.winfo_width(), 320)
        h = max(self.plot_frame.winfo_height(), 240)
        return (w / 110, h / 110)

    def schedule(self):
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(160, self._draw)

    def _draw(self):
        self._job = None
        if self.view is None or not self.stack:
            return
        try:
            E.render([Layer(**l) for l in self.stack], self.view, fig=self.fig,
                     figsize=self._figsize(), title=self.ent_title.get(),
                     xlabel=self.ent_xlabel.get(), ylabel=self.ent_ylabel.get(),
                     style=self.style, legend=self.legend)
        except PlotConfigError as e:
            return messagebox.showwarning("Configuration", str(e))
        except Exception as e:
            return messagebox.showerror("Plotting error", str(e))
        base = REGISTRY[self.stack[0]["spec_key"]]; m = self.stack[0]["mapping"]
        self._pick = (m.get("x"), m.get("y")) if base.pickable else None
        self.canvas.draw_idle()
        self.lbl_stack.configure(text="layers: " + " + ".join(REGISTRY[l["spec_key"]].label for l in self.stack))

    def _refresh_legend_only(self):
        """Reapply only the legend on the current axes, without recreating the figure/canvas."""
        if not self.fig.axes:
            return
        ax = self.fig.axes[0]; theme = THEMES[self.style.theme]
        E._apply_legend(self.fig, ax, self.legend, theme["fg"], theme)
        self.canvas.draw_idle()

    def _legend_hit(self, event):
        if not self.fig.axes or event.x is None:
            return None
        leg = self.fig.axes[0].get_legend()
        if not leg:
            return None
        try:
            bb = leg.get_window_extent()
        except Exception:
            return None
        pad = 6
        if bb.x0 - pad <= event.x <= bb.x1 + pad and bb.y0 - pad <= event.y <= bb.y1 + pad:
            return leg, bb
        return None

    def _on_resize(self, event):
        size = (event.width, event.height)
        if self._last_size and abs(size[0] - self._last_size[0]) < 4 and abs(size[1] - self._last_size[1]) < 4:
            return
        self._last_size = size
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(200, self._draw)

    # ============================ callbacks ============================ #
    def on_load(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            self.df = DL.load_csv(path)
        except Exception as e:
            return messagebox.showerror("Loading error", str(e))
        self.view = self.df
        self.lbl_file.configure(text=f"{path.split('/')[-1]}  ({len(self.df)}×{self.df.shape[1]})")
        self.rebuild_config(); self.refresh_columns()

    def on_filter(self):
        if self.df is None:
            return
        self.view, err = DL.apply_filter(self.df, self.ent_filter.get())
        if err:
            messagebox.showwarning("Filter", err)
        self.rebuild_config(); self.refresh_columns(); self.schedule()

    def on_plot(self):
        if self.view is None:
            return messagebox.showwarning("No data", "Load a CSV first.")
        self.stack = [self._layer()]; self._draw()

    def on_overlay(self):
        if not self.stack:
            return messagebox.showinfo("Layers", "Plot the base layer first.")
        if not self._spec().layerable:
            return messagebox.showwarning("Layers", f"'{self._spec().label}' is figure-level and cannot be overlaid.")
        self.stack.append(self._layer()); self._draw()

    def on_clear(self):
        self.stack = []; self.lbl_stack.configure(text="layers: —")

    def apply_style(self):
        self.style.theme = self.opt_theme.get()
        self.style.context = self.opt_ctx.get()
        self.style.font_scale = round(self.sld_font.get(), 2)
        self.style.grid = bool(self.chk_grid.get())
        self.schedule()

    def apply_legend(self):
        self.legend.show = bool(self.chk_leg.get())
        self.legend.pos = LEGEND_POS[self.opt_legpos.get()]
        self.legend.fontsize = round(self.sld_legsize.get(), 1)
        self.legend.ncol = int(self.sld_legcol.get())
        self.legend.frame = bool(self.chk_legframe.get())
        self.legend.x = round(self.sld_legx.get(), 3)
        self.legend.y = round(self.sld_legy.get(), 3)
        self.schedule()

    def _pick_color(self, attr):
        c = colorchooser.askcolor()[1]
        if c:
            setattr(self.style, attr, c); self.schedule()

    def _clear_overrides(self):
        self.style.fig_bg = self.style.ax_bg = None; self.schedule()

    def on_press(self, event):
        # 1) started on top of the legend? begin a manual drag
        hit = self._legend_hit(event)
        if hit and self.legend.show:
            leg, bb = hit
            ax = self.fig.axes[0]
            a0 = ax.transAxes.transform((0, 0)); a1 = ax.transAxes.transform((1, 1))
            wpx, hpx = (a1[0] - a0[0]) or 1, (a1[1] - a0[1]) or 1
            if self.legend.pos != "free":   # convert to free in place (without recreating the canvas)
                self.legend.x = round((bb.x0 - a0[0]) / wpx, 3)
                self.legend.y = round((bb.y0 - a0[1]) / hpx, 3)
                self.legend.pos = "free"; self.opt_legpos.set("Free (drag)")
                self._refresh_legend_only()
            self._drag = dict(sx=event.x, sy=event.y, wpx=wpx, hpx=hpx,
                              x0=self.legend.x, y0=self.legend.y)
            return
        # 2) otherwise: click-on-point (scatter/volcano)
        self._pick_point(event)

    def on_motion(self, event):
        if not self._drag or event.x is None:
            return
        d = self._drag
        self.legend.x = round(d["x0"] + (event.x - d["sx"]) / d["wpx"], 3)
        self.legend.y = round(d["y0"] + (event.y - d["sy"]) / d["hpx"], 3)
        leg = self.fig.axes[0].get_legend()
        if leg:
            leg.set_bbox_to_anchor((self.legend.x, self.legend.y), transform=self.fig.axes[0].transAxes)
            self.canvas.draw_idle()

    def _end_drag(self, event=None):
        # fired by Tk (<ButtonRelease-1>) — catches the release even outside the canvas
        if self._drag:
            self.sld_legx.set(self.legend.x); self.sld_legy.set(self.legend.y)
            self._drag = None

    def _pick_point(self, event):
        if event.inaxes is None or not self._pick:
            return
        xcol, ycol = self._pick
        if not xcol or not ycol:
            return
        sub = self.view[[xcol, ycol]].apply(pd.to_numeric, errors="coerce").dropna()
        if sub.empty:
            return
        ax = event.inaxes
        xr = np.ptp(ax.get_xlim()) or 1; yr = np.ptp(ax.get_ylim()) or 1
        d = ((sub[xcol].values - event.xdata) / xr) ** 2 + ((sub[ycol].values - event.ydata) / yr) ** 2
        row = self.view.loc[sub.index[int(np.argmin(d))]]
        self.info.delete("1.0", "end"); self.info.insert("1.0", row.to_string())

    def on_reduce(self):
        cols = [c for c, v in self.num_checks.items() if v.get()]
        if len(cols) < 2:
            return messagebox.showwarning("Reduction", "Select at least 2 numeric columns.")
        self.btn_reduce.configure(state="disabled", text="Running…")

        def done(res):
            df2, names = res
            def ui():
                self.view = df2
                self.btn_reduce.configure(state="normal", text="Run reduction")
                self.rebuild_config(); self.refresh_columns()
                messagebox.showinfo("Reduction", f"Done. New columns: {', '.join(names)}")
            self.after(0, ui)

        def err(e):
            self.after(0, lambda: (self.btn_reduce.configure(state="normal", text="Run reduction"),
                                   messagebox.showerror("Reduction", str(e))))
        S.run_async(lambda: S.compute_embedding(self.view, cols, self.opt_method.get()), done, err)

    def on_test(self):
        if self.view is None or not self.opt_val.get() or not self.opt_grp.get():
            return
        self.txt_stats.delete("1.0", "end")
        self.txt_stats.insert("1.0", S.group_test(self.view, self.opt_val.get(), self.opt_grp.get()))

    def on_export(self):
        if self.fig is None:
            return messagebox.showinfo("Export", "Plot something first.")
        path = filedialog.asksaveasfilename(defaultextension=".tiff",
            filetypes=[("TIFF", "*.tiff"), ("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if path:
            E.export(self.fig, path, dpi=300); messagebox.showinfo("Export", f"Saved: {path}")

    def on_export_csv(self):
        if self.view is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            DL.export_filtered(self.view, path)


if __name__ == "__main__":
    App().mainloop()
