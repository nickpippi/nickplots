"""Generate a STANDALONE Python script (matplotlib/seaborn) that reproduces the figure.
It is the 'methods' artifact: auditable, editable and independent of the app.
Covers the catalog types; for the composite ones it emits the real computation block."""
from __future__ import annotations

THEME_BASE = {
    "Light · grid": "whitegrid", "Light · ticks": "ticks", "Light · clean": "white",
    "Dark · grid": "darkgrid", "Dark · plain": "dark", "Slate": "darkgrid",
}


def _lit(v):
    return repr(v)


def _pal(p):
    cm = p.get("__colors__")
    return _lit(cm) if cm else _lit(p.get("palette", "viridis"))


def _layer_code(spec_key, m, p):
    x, y, hue = m.get("x"), m.get("y"), m.get("hue")
    L = []
    if spec_key == "scatter":
        L.append(f"sns.scatterplot(data=df, x={_lit(x)}, y={_lit(y)}, hue={_lit(hue)}, "
                 f"size={_lit(m.get('size'))}, style={_lit(m.get('style'))}, alpha={p['alpha']}, "
                 f"palette={_pal(p) if hue else None}, ax=ax)")
        if p.get("regression"):
            L.append(f"_x=pd.to_numeric(df[{_lit(x)}],errors='coerce'); _y=pd.to_numeric(df[{_lit(y)}],errors='coerce')")
            L.append("_a,_b=np.polyfit(_x.dropna(),_y[_x.notna()].dropna(),1); _xs=np.linspace(_x.min(),_x.max(),100)")
            L.append("ax.plot(_xs,_a*_xs+_b,c='black',lw=1.2)")
    elif spec_key == "line":
        _eb = {"none": "None", "SD": "'sd'", "SEM": "'se'", "CI95": "('ci',95)"}[p.get("errorbar", "none")]
        L.append(f"sns.lineplot(data=df, x={_lit(x)}, y={_lit(y)}, hue={_lit(hue)}, linewidth={p['linewidth']}, alpha={p['alpha']}, errorbar={_eb}, palette={_pal(p) if hue else None}, ax=ax)")
    elif spec_key == "bar":
        L.append(f"sns.barplot(data=df, x={_lit(x)}, y={_lit(y)}, hue={_lit(hue or x)}, palette={_pal(p)}, alpha={p['alpha']}, legend={bool(hue)}, ax=ax)")
    elif spec_key in ("box", "violin", "strip"):
        fn = {"box": "boxplot", "violin": "violinplot", "strip": "stripplot"}[spec_key]
        L.append(f"sns.{fn}(data=df, x={_lit(x)}, y={_lit(y)}, hue={_lit(hue or x)}, palette={_pal(p)}, legend={bool(hue)}, ax=ax)")
    elif spec_key == "hist":
        L.append(f"sns.histplot(data=df, x={_lit(x)}, hue={_lit(hue)}, bins={p['bins']}, kde={p['kde']}, alpha={p['alpha']}, ax=ax)")
    elif spec_key == "kde":
        L.append(f"sns.kdeplot(data=df, x={_lit(x)}, hue={_lit(hue)}, fill={p['fill']}, alpha={p['alpha']}, ax=ax)")
    elif spec_key == "ecdf":
        L.append(f"sns.ecdfplot(data=df, x={_lit(x)}, hue={_lit(hue)}, ax=ax)")
    elif spec_key == "heatmap":
        L.append(f"sns.heatmap(df.select_dtypes('number').corr(numeric_only=True), annot={p['annot']}, cmap={_lit(p['palette'])}, ax=ax)")
    elif spec_key == "heatmap_matrix":
        L.append("num=df.select_dtypes('number')")
        L.append(f"data=(num-num.mean())/num.std() if {p['zscore']} else num")
        L.append(f"sns.heatmap(data, cmap={_lit(p['palette'])}, annot={p['annot']}, cbar=True, ax=ax)")
    elif spec_key == "regband":
        if hue:
            L.append(f"for _n,_s in df.groupby({_lit(hue)}):")
            L.append(f"    sns.regplot(data=_s, x={_lit(x)}, y={_lit(y)}, ci={int(p['ci'])}, scatter_kws=dict(s=16, alpha={p['alpha']}), ax=ax)")
        else:
            L.append(f"sns.regplot(data=df, x={_lit(x)}, y={_lit(y)}, ci={int(p['ci'])}, scatter_kws=dict(s=16, alpha={p['alpha']}), ax=ax)")
    elif spec_key in ("box_points", "violin_points"):
        main = "boxplot" if spec_key == "box_points" else "violinplot"
        extra = "showfliers=False" if spec_key == "box_points" else "inner=None"
        L.append(f"sns.{main}(data=df, x={_lit(x)}, y={_lit(y)}, hue={_lit(hue or x)}, palette={_pal(p)}, legend={bool(hue)}, {extra}, ax=ax)")
        L.append(f"sns.stripplot(data=df, x={_lit(x)}, y={_lit(y)}, hue={_lit(hue)}, dodge={bool(hue) and hue != x}, jitter={p['jitter']}, size={p['size']}, alpha={p['alpha']}, color='#1f1f1f', edgecolor='white', linewidth=0.4, legend=False, ax=ax)")
    elif spec_key == "bar_err":
        est = "np.mean" if p["center"] == "mean" else "np.median"
        eb = {"SD": "'sd'", "SEM": "'se'", "CI95": "('ci',95)"}[p["error"]]
        fn = "barplot" if p["kind"] == "bar" else "pointplot"
        ln = "" if p["kind"] == "bar" else ", linestyle='none'"
        L.append(f"sns.{fn}(data=df, x={_lit(x)}, y={_lit(y)}, hue={_lit(hue or x)}, estimator={est}, errorbar={eb}, palette={_pal(p)}, legend={bool(hue)}{ln}, ax=ax)")
        if p.get("points"):
            L.append(f"sns.stripplot(data=df, x={_lit(x)}, y={_lit(y)}, hue={_lit(hue)}, dodge={bool(hue) and hue != x}, jitter=0.18, size={p['psize']}, alpha=0.55, color='#1f1f1f', edgecolor='white', linewidth=0.3, legend=False, ax=ax)")
    elif spec_key == "dose4pl":
        L.append("from scipy.optimize import curve_fit")
        L.append("def _f4(xx,a,b,c,d): return d+(a-d)/(1+(xx/c)**b)")
        grp = f"df.groupby({_lit(hue)})" if hue else "[(None, df)]"
        L.append(f"for _n,_s in ({grp}):")
        L.append(f"    _xv=pd.to_numeric(_s[{_lit(x)}],errors='coerce'); _yv=pd.to_numeric(_s[{_lit(y)}],errors='coerce')")
        L.append(f"    ax.scatter(_xv,_yv,s=20,alpha={p['alpha']},label=str(_n) if {bool(hue)} else None)")
        L.append("    _ok=_xv.notna()&_yv.notna(); _xv,_yv=_xv[_ok],_yv[_ok]")
        L.append("    _p,_=curve_fit(_f4,_xv,_yv,p0=[_yv.max(),1,_xv[_xv>0].median(),_yv.min()],maxfev=20000)")
        L.append("    _xx=np.logspace(np.log10(_xv[_xv>0].min()),np.log10(_xv.max()),200); ax.plot(_xx,_f4(_xx,*_p))")
        L.append("    ax.axvline(_p[2],ls=':'); print('IC50=',_p[2])")
        if p.get("log_x", True):
            L.append("ax.set_xscale('log')")
    elif spec_key == "paired":
        sid = m.get("id")
        L.append(f"_cats=list(pd.unique(df[{_lit(x)}].dropna())); _pos={{c:i for i,c in enumerate(_cats)}}")
        L.append(f"for _s,_g in df.groupby({_lit(sid)}):")
        L.append(f"    _g=_g.dropna(subset=[{_lit(y)}])")
        L.append(f"    _xs=[_pos[c] for c in _g[{_lit(x)}] if c in _pos]; _ys=pd.to_numeric(_g[{_lit(y)}],errors='coerce').values[:len(_xs)]")
        L.append("    ax.plot(_xs,_ys,color='#6a7385',alpha=%s,lw=0.8,marker='o',ms=4)" % p["alpha"])
        L.append("ax.set_xticks(range(len(_cats))); ax.set_xticklabels(_cats)")
    elif spec_key == "volcano":
        L.append(f"_fc=df[{_lit(x)}].to_numpy(float); _pv=df[{_lit(y)}].to_numpy(float); _nlp=-np.log10(np.clip(_pv,1e-300,None))")
        L.append(f"_up=(_fc>={p['fc_thr']})&(_pv<={p['p_thr']}); _dn=(_fc<=-{p['fc_thr']})&(_pv<={p['p_thr']}); _ns=~(_up|_dn)")
        L.append(f"ax.scatter(_fc[_ns],_nlp[_ns],s=12,c='lightgrey',alpha={p['alpha']})")
        L.append(f"ax.scatter(_fc[_up],_nlp[_up],s=14,c='tab:red',alpha={p['alpha']}); ax.scatter(_fc[_dn],_nlp[_dn],s=14,c='tab:blue',alpha={p['alpha']})")
        L.append(f"ax.axhline(-np.log10({p['p_thr']}),ls='--',c='grey',lw=0.8); ax.axvline({p['fc_thr']},ls='--',c='grey',lw=0.8); ax.axvline(-{p['fc_thr']},ls='--',c='grey',lw=0.8)")
    elif spec_key == "scatter_density":
        contour = bool(p.get("contour"))
        grid = int(p.get("grid", 120))
        a = p.get("alpha", 0.35)
        ps = p.get("psize", 22)
        size_col = m.get("size")
        style_col = m.get("style")

        # Select only the required columns without duplicates
        cols_to_extract = [x, y, hue]
        if size_col and size_col not in cols_to_extract: cols_to_extract.append(size_col)
        if style_col and style_col not in cols_to_extract: cols_to_extract.append(style_col)
        cols_str = ",".join([_lit(c) for c in cols_to_extract])

        L.append(f"_sub=df[[{cols_str}]].copy()")
        L.append(f"_sub[{_lit(x)}]=pd.to_numeric(_sub[{_lit(x)}],errors='coerce'); _sub[{_lit(y)}]=pd.to_numeric(_sub[{_lit(y)}],errors='coerce')")
        L.append(f"_sub=_sub.dropna(subset=[{_lit(x)},{_lit(y)},{_lit(hue)}])")
        L.append(f"_groups=list(_sub.groupby({_lit(hue)})); _names=[g for g,_ in _groups]")

        cm = p.get("__colors__")
        if cm:
            L.append(f"_cmap={_lit(cm)}; _base=sns.color_palette({_lit(p.get('palette','viridis'))},len(_names)); _colors=[_cmap.get(str(n)) or _base[i] for i,n in enumerate(_names)]")
        else:
            L.append(f"_colors=sns.color_palette({_lit(p.get('palette','viridis'))},max(len(_names),1))")
        L.append("_pal={n:_colors[i] for i,n in enumerate(_names)}")

        if contour:
            # "waves" mode: per-group KDE contours (overlap stays visible)
            L.append(f"sns.kdeplot(data=_sub, x={_lit(x)}, y={_lit(y)}, hue={_lit(hue)}, hue_order=_names, "
                     f"levels={int(p.get('levels', 10))}, palette=_pal, linewidths=1.0, alpha=0.9, "
                     f"common_norm=False, legend=False, ax=ax, zorder=0)")
        else:
            L.append("from scipy.stats import gaussian_kde")
            L.append("import matplotlib.colors as mcolors")
            L.append(f"_xv=_sub[{_lit(x)}].to_numpy(float); _yv=_sub[{_lit(y)}].to_numpy(float)")
            L.append("_px=(_xv.max()-_xv.min())*0.05 or 1.0; _py=(_yv.max()-_yv.min())*0.05 or 1.0")
            L.append("_x0,_x1,_y0,_y1=_xv.min()-_px,_xv.max()+_px,_yv.min()-_py,_yv.max()+_py")
            L.append(f"_res={grid}; _gx=np.linspace(_x0,_x1,_res); _gy=np.linspace(_y0,_y1,_res)")
            L.append("_GX,_GY=np.meshgrid(_gx,_gy); _pos=np.vstack([_GX.ravel(),_GY.ravel()]); _dens=[]")
            L.append("for _n,_g in _groups:")
            L.append(f"    _a=_g[{_lit(x)}].to_numpy(float); _b=_g[{_lit(y)}].to_numpy(float)")
            L.append("    if len(_a)<3 or np.ptp(_a)==0 or np.ptp(_b)==0: _dens.append(np.zeros((_res,_res))); continue")
            L.append("    try: _dens.append(gaussian_kde(np.vstack([_a,_b]))(_pos).reshape(_res,_res)*len(_a))")
            L.append("    except Exception: _dens.append(np.zeros((_res,_res)))")
            L.append("_dens=np.array(_dens); _dom=np.argmax(_dens,axis=0); _mx=_dens.max(axis=0)")
            L.append("_norm=_mx/_mx.max() if _mx.max()>0 else _mx; _rgba=np.zeros((_res,_res,4))")
            L.append("for _i,_c in enumerate(_colors):")
            L.append("    _r,_gg,_bb=mcolors.to_rgb(_c); _m=_dom==_i")
            L.append(f"    _rgba[_m,0],_rgba[_m,1],_rgba[_m,2]=_r,_gg,_bb; _rgba[_m,3]=_norm[_m]*{a}")
            L.append("ax.imshow(_rgba,extent=[_x0,_x1,_y0,_y1],origin='lower',aspect='auto',interpolation='bilinear',zorder=0)")

        # Scatter on top (shared) — avoids conflict between 's' and 'size'
        scat_args = [
            f"data=_sub", f"x={_lit(x)}", f"y={_lit(y)}", f"hue={_lit(hue)}",
            "palette=_pal",
            "alpha=0.9", "edgecolor='white'", "linewidth=0.3", "zorder=2", "ax=ax"
        ]

        if size_col:
            scat_args.append(f"size={_lit(size_col)}")
        else:
            scat_args.append(f"s={ps}")

        if style_col:
            scat_args.append(f"style={_lit(style_col)}")

        L.append(f"sns.scatterplot({', '.join(scat_args)})")
    else:
        L.append(f"# type '{spec_key}' not supported by the code generator")
    return L


def generate_code(state, csv_path):
    s = state.get("style", {})
    base = THEME_BASE.get(s.get("theme"), "whitegrid")
    lines = [
        '"""Figure generated by Nickplots. Edit freely."""',
        "import numpy as np", "import pandas as pd",
        "import seaborn as sns", "import matplotlib.pyplot as plt", "",
        f"df = pd.read_csv({_lit(csv_path)})",
    ]
    flt = state.get("filter")
    if flt:
        lines.append(f"df = df.query({_lit(flt)})")
    lines += [
        f"sns.set_theme(style={_lit(base)}, context={_lit(s.get('context','notebook'))}, font_scale={s.get('font_scale',1.0)})",
        "fig, ax = plt.subplots(figsize=(7, 5), dpi=300)", "",
    ]
    for layer in state.get("layers", []):
        lines += _layer_code(layer["spec_key"], layer["mapping"], layer["params"])
    lines.append("")
    if state.get("title"):
        lines.append(f"ax.set_title({_lit(state['title'])})")
    if state.get("xlabel"):
        lines.append(f"ax.set_xlabel({_lit(state['xlabel'])})")
    if state.get("ylabel"):
        lines.append(f"ax.set_ylabel({_lit(state['ylabel'])})")
    if s.get("logx"):
        lines.append("ax.set_xscale('log')")
    if s.get("logy"):
        lines.append("ax.set_yscale('log')")
    gates_ = state.get("gates", []) or []
    if any((g.get("points") and len(g["points"]) >= 3) for g in gates_):
        lines.append("from matplotlib.patches import Polygon as _Poly")
    for g in gates_:
        pts = g.get("points") or []
        if len(pts) < 3:
            continue
        col = g.get("color") or "#e23b3b"
        poly = [[float(a), float(b)] for a, b in pts]
        lines.append(f"ax.add_patch(_Poly({poly!r}, closed=True, fill=False, edgecolor={col!r}, lw=1.4))")
        if g.get("name"):
            lp = g.get("labelxy")
            if lp and len(lp) == 2:
                cx, cy = float(lp[0]), float(lp[1])
            else:
                cx = sum(p[0] for p in poly) / len(poly)
                cy = sum(p[1] for p in poly) / len(poly)
            lines.append(f"ax.text({cx!r}, {cy!r}, {g['name']!r}, color={col!r}, fontsize=9, "
                         f"fontweight='bold', ha='center', va='center')")
    if s.get("despine"):
        lines.append("sns.despine(ax=ax)")
    import math as _m
    for ln in state.get("threshold_lines", []) or []:
        try:
            ang = float(ln.get("angle", 0.0)) % 180.0
            x0 = float(ln.get("x", 0.0)); y0 = float(ln.get("y", 0.0))
        except (TypeError, ValueError):
            continue
        col = ln.get("color") or "#e23b3b"
        if abs(ang - 90.0) < 1e-9:
            lines.append(f"ax.axvline({x0!r}, color={col!r}, ls='--', lw=1.3)")
        else:
            lines.append(f"ax.axline(({x0!r}, {y0!r}), slope={_m.tan(_m.radians(ang))!r}, color={col!r}, ls='--', lw=1.3)")
    lines += ["", "fig.tight_layout()",
              "fig.savefig('figure.tiff', dpi=300, bbox_inches='tight')",
              "fig.savefig('figure.png', dpi=300, bbox_inches='tight')",
              "print('Figure saved: figure.tiff / figure.png')"]
    return "\n".join(lines)
