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


def _colors_code(p, names="_names"):
    """Emit the per-group colour list, honouring the hand-picked colours."""
    cm = p.get("__colors__")
    pal = _lit(p.get("palette", "viridis"))
    if cm:
        return [f"_cmap={_lit(cm)}; _base=sns.color_palette({pal},max(len({names}),1)); "
                f"_colors=[_cmap.get(str(n)) or _base[i] for i,n in enumerate({names})]"]
    return [f"_colors=sns.color_palette({pal},max(len({names}),1))"]


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
    elif spec_key in ("heatmap", "heatmap_matrix"):
        cols = [c for c in (p.get("cols") or [])]
        L.append("num=df.select_dtypes('number')")
        if cols:                              # only the columns the user ticked
            L.append(f"num=num[[c for c in {cols!r} if c in num.columns]]")
        sz = float(p.get("annot_size") or 0)
        kws = f", annot_kws={{'fontsize': {sz}}}" if sz > 0 else ""
        if spec_key == "heatmap":
            L.append(f"sns.heatmap(num.corr(numeric_only=True), annot={p['annot']}, "
                     f"cmap={_lit(p['palette'])}{kws}, ax=ax)")
        else:
            L.append(f"data=(num-num.mean())/num.std() if {p['zscore']} else num")
            lab = m.get("labels")
            if lab:
                L.append(f"data=data.set_axis([str(v) for v in df[{_lit(lab)}]], axis=0)")
            L.append(f"sns.heatmap(data, cmap={_lit(p['palette'])}, annot={p['annot']}{kws}, "
                     "cbar=True, ax=ax)")
            if lab:
                L.append("ax.tick_params(axis='y', rotation=0)")
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
    elif spec_key == "ridge":
        L.append("from scipy.stats import gaussian_kde")
        L.append(f"_d=df[[{_lit(x)},{_lit(hue)}]].copy(); "
                 f"_d[{_lit(x)}]=pd.to_numeric(_d[{_lit(x)}],errors='coerce'); _d=_d.dropna()")
        L.append(f"_names=sorted(_d[{_lit(hue)}].astype(str).unique()); _n=len(_names)")
        L += _colors_code(p)
        L.append(f"_xs=_d[{_lit(x)}].to_numpy(float); _lo,_hi=_xs.min(),_xs.max()")
        L.append("_pad=(_hi-_lo)*0.05 or 1.0; _grid=np.linspace(_lo-_pad,_hi+_pad,256)")
        L.append("for _i,_c in enumerate(_names):")
        L.append(f"    _v=_d.loc[_d[{_lit(hue)}].astype(str)==_c,{_lit(x)}].to_numpy(float)")
        L.append("    _off=float(_n-1-_i)")
        L.append(f"    _dens=gaussian_kde(_v)(_grid); _dens=_dens/_dens.max()*{float(p.get('overlap',1.1))}"
                 " if len(_v)>=2 and np.std(_v)>0 else np.zeros_like(_grid)")
        L.append(f"    ax.fill_between(_grid,_off,_off+_dens,color=_colors[_i],alpha={p.get('alpha',0.85)},lw=0,zorder=_n-_i)")
        L.append("    ax.plot(_grid,_off+_dens,color='white',lw=0.8,zorder=_n-_i)")
        L.append("ax.set_yticks([_n-1-_i for _i in range(_n)]); ax.set_yticklabels(_names)")
        L.append(f"ax.set_xlabel({_lit(x)}); ax.set_ylabel({_lit(hue)}); "
                 f"ax.set_ylim(-0.2,_n-1+{float(p.get('overlap',1.1))}+0.3)")
    elif spec_key == "qq":
        val = m.get("val")
        L.append("from scipy import stats as _st")
        grp = f"list(df.groupby({_lit(hue)}))" if hue else "[(None, df)]"
        L.append(f"_groups={grp}; _names=[g for g,_ in _groups]")
        L += _colors_code(p)
        L.append("for _i,(_nm,_s) in enumerate(_groups):")
        L.append(f"    _v=pd.to_numeric(_s[{_lit(val)}],errors='coerce').dropna().to_numpy()")
        L.append("    if len(_v)<3: continue")
        L.append("    (_osm,_osr),(_sl,_it,_r)=_st.probplot(_v,dist='norm')")
        L.append(f"    ax.scatter(_osm,_osr,s=14,color=_colors[_i],alpha={p.get('alpha',0.8)},"
                 f"label=str(_nm) if {bool(hue)} else None)")
        L.append("    ax.plot(_osm,_sl*_osm+_it,color=_colors[_i],lw=1.0)")
        L.append("ax.set_xlabel('Theoretical quantiles'); ax.set_ylabel('Sample quantiles')")
        if hue:
            L.append("ax.legend()")
    elif spec_key == "superplot":
        rep, size = m.get("rep"), p.get("size", 4)
        L.append(f"_d=df[[{_lit(x)},{_lit(y)}" + (f",{_lit(rep)}" if rep else "") + "]].copy()")
        L.append(f"_d[{_lit(y)}]=pd.to_numeric(_d[{_lit(y)}],errors='coerce'); "
                 f"_d=_d.dropna(subset=[{_lit(x)},{_lit(y)}])")
        if rep:
            L.append(f"_d[{_lit(rep)}]=_d[{_lit(rep)}].astype(str)")
            L.append(f"sns.stripplot(data=_d,x={_lit(x)},y={_lit(y)},hue={_lit(rep)},palette={_pal(p)},"
                     f"dodge=False,jitter=0.28,size={size*0.6},alpha=0.35,legend=False,ax=ax)")
            L.append(f"_means=_d.groupby([{_lit(x)},{_lit(rep)}],observed=True)[{_lit(y)}].mean().reset_index()")
            L.append(f"sns.stripplot(data=_means,x={_lit(x)},y={_lit(y)},hue={_lit(rep)},palette={_pal(p)},"
                     f"dodge=False,jitter=0.06,size={size*2.4},alpha=1.0,edgecolor='black',linewidth=1.1,"
                     f"legend=(_d[{_lit(rep)}].nunique()<=12),ax=ax)")
        else:
            L.append(f"sns.stripplot(data=_d,x={_lit(x)},y={_lit(y)},color='#9aa0aa',jitter=0.28,"
                     f"size={size*0.6},alpha=0.35,ax=ax)")
        L.append(f"for _i,_c in enumerate(list(pd.unique(_d[{_lit(x)}]))):")
        L.append(f"    _gm=_d.loc[_d[{_lit(x)}]==_c,{_lit(y)}].mean()")
        L.append("    ax.plot([_i-0.32,_i+0.32],[_gm,_gm],color='#333',lw=2.2,zorder=5)")
    elif spec_key == "stacked":
        prop = bool(p.get("proportion", True))
        L.append(f"_ct=pd.crosstab(df[{_lit(x)}].astype(str), df[{_lit(hue)}].astype(str))")
        if prop:
            L.append("_ct=_ct.div(_ct.sum(axis=1).replace(0,np.nan),axis=0)")
        L.append("_names=list(_ct.columns)")
        L += _colors_code(p)
        L.append("_bottom=np.zeros(len(_ct)); _xs=np.arange(len(_ct))")
        L.append("for _i,_c in enumerate(_names):")
        L.append("    _v=_ct[_c].to_numpy(float)")
        L.append("    ax.bar(_xs,_v,bottom=_bottom,color=_colors[_i],width=0.8,label=str(_c),"
                 "edgecolor='white',linewidth=0.5)")
        L.append("    _bottom=_bottom+np.nan_to_num(_v)")
        L.append("ax.set_xticks(_xs); ax.set_xticklabels(_ct.index)")
        L.append(f"ax.set_ylabel({_lit('proportion' if prop else 'count')}); ax.legend(title={_lit(hue)})")
        if p.get("show_p"):
            L.append("from scipy.stats import chi2_contingency")
            L.append(f"_c2=pd.crosstab(df[{_lit(x)}].astype(str), df[{_lit(hue)}].astype(str))")
            L.append("_chi,_pv,_dof,_exp=chi2_contingency(_c2)")
            L.append("ax.text(0.02,0.98,f'Chi-square p={_pv:.3g}',transform=ax.transAxes,"
                     "ha='left',va='top',fontsize=8,bbox=dict(fc='white',alpha=0.75,ec='none'))")
    elif spec_key == "traj":
        tid, tcol = m.get("id"), m.get("time")
        if hue:
            L.append(f"_groups=list(df.groupby({_lit(hue)})); _names=[g for g,_ in _groups]")
            L += _colors_code(p)
            L.append("_cmap={_n:_colors[_i] for _i,_n in enumerate(_names)}")
        L.append("_seen=set()")
        L.append(f"for _tk,_s in df.groupby({_lit(tid)}):")
        if tcol:
            L.append(f"    _s=_s.sort_values({_lit(tcol)})")
        if hue:
            L.append(f"    _g=_s[{_lit(hue)}].iloc[0]; _c=_cmap.get(_g)")
            L.append("    _lb=None")
            L.append("    if _g not in _seen: _lb=str(_g); _seen.add(_g)")
        else:
            L.append("    _c=None; _lb=None")
        L.append(f"    ax.plot(pd.to_numeric(_s[{_lit(x)}],errors='coerce'),"
                 f"pd.to_numeric(_s[{_lit(y)}],errors='coerce'),"
                 f"lw={p.get('linewidth',1.2)},alpha={p.get('alpha',0.8)},color=_c,label=_lb)")
        L.append(f"ax.set_xlabel({_lit(x)}); ax.set_ylabel({_lit(y)})")
        if p.get("equal", True):
            L.append("ax.set_aspect('equal',adjustable='datalim')")
        if hue:
            L.append(f"ax.legend(title={_lit(hue)})")
    elif spec_key == "msd":
        tid, px, py, tcol = m.get("id"), m.get("px"), m.get("py"), m.get("time")
        L.append("def _track_msd(_s):")
        if tcol:
            L.append(f"    _s=_s.sort_values({_lit(tcol)})")
        L.append(f"    _xy=_s[[{_lit(px)},{_lit(py)}]].apply(pd.to_numeric,errors='coerce').dropna().to_numpy(float)")
        L.append("    _m=len(_xy)")
        L.append("    if _m<2: return None")
        L.append("    _o=np.full(_m,np.nan); _o[0]=0.0")
        L.append("    for _lag in range(1,_m):")
        L.append("        _dp=_xy[_lag:]-_xy[:-_lag]; _o[_lag]=np.mean(np.sum(_dp*_dp,axis=1))")
        L.append("    return _o")
        L.append("def _group_curve(_g):")
        L.append(f"    _per=[c for c in (_track_msd(s) for _,s in _g.groupby({_lit(tid)})) if c is not None]")
        L.append("    if not _per: return None")
        L.append("    _Lm=max(len(c) for c in _per); _M=np.full((len(_per),_Lm),np.nan)")
        L.append("    for _i,_c in enumerate(_per): _M[_i,:len(_c)]=_c")
        L.append("    _md=np.nanmean(_M,axis=0); return np.arange(_Lm)[1:],_md[1:],len(_per)")
        if hue:
            L.append(f"_curves={{str(k):_group_curve(g) for k,g in df.groupby({_lit(hue)},dropna=False)}}")
            L.append("_curves={k:v for k,v in _curves.items() if v is not None}")
        else:
            L.append("_curves={None:_group_curve(df)}; _curves={k:v for k,v in _curves.items() if v is not None}")
        L.append("_names=list(_curves)")
        L += _colors_code(p)
        L.append("for _i,(_nm,(_lg,_md,_nt)) in enumerate(_curves.items()):")
        L.append("    ax.plot(_lg,_md,marker='o',ms=3,lw=1.4,color=_colors[_i],"
                 "label=(f'{_nm} (n={_nt})' if _nm is not None else None))")
        L.append("ax.set_xlabel('lag (frames)'); ax.set_ylabel('MSD')")
        if p.get("loglog"):
            L.append("ax.set_xscale('log'); ax.set_yscale('log')")
        if hue:
            L.append("ax.legend()")
    elif spec_key == "km":
        time, ev = m.get("time"), m.get("event")
        evs = [t.strip() for t in str(p.get("event_value", "")).split(",") if t.strip()]
        L.append("def _km_one(_t,_e):")
        L.append("    _t=np.asarray(_t,float); _e=np.asarray(_e,bool)")
        L.append("    _ok=np.isfinite(_t); _t,_e=_t[_ok],_e[_ok]")
        L.append("    _o=np.argsort(_t); _t,_e=_t[_o],_e[_o]")
        L.append("    _ts,_sv,_s=[0.0],[1.0],1.0")
        L.append("    for _ut in np.unique(_t[_e]):")
        L.append("        _ar=int(np.sum(_t>=_ut)); _d=int(np.sum((_t==_ut)&_e))")
        L.append("        if _ar>0: _s*= (1-_d/_ar)")
        L.append("        _ts.append(float(_ut)); _sv.append(_s)")
        L.append("    return np.array(_ts),np.array(_sv),_t[~_e]")
        if ev:
            if evs:
                L.append(f"_evmask=df[{_lit(ev)}].astype(str).str.strip().isin({evs!r})")
            else:
                L.append(f"_evmask=df[{_lit(ev)}].astype(str).str.strip().replace('nan','')!=''")
        else:
            L.append("_evmask=pd.Series(True,index=df.index)      # no status column: every row is an event")
        if hue:
            L.append(f"_groups=list(df.groupby({_lit(hue)})); _names=[str(g) for g,_ in _groups]")
        else:
            L.append("_groups=[(None,df)]; _names=[None]")
        L += _colors_code(p)
        L.append("for _i,(_nm,_s) in enumerate(_groups):")
        L.append(f"    _t,_sv,_cs=_km_one(pd.to_numeric(_s[{_lit(time)}],errors='coerce'),_evmask.loc[_s.index])")
        L.append(f"    ax.step(_t,_sv,where='post',color=_colors[_i],lw=1.6,"
                 f"label=str(_nm) if {bool(hue)} else None)")
        if p.get("censors", True):
            L.append("    if len(_cs): ax.plot(_cs,np.interp(_cs,_t,_sv),'|',color=_colors[_i],ms=7,mew=1.2)")
        L.append(f"ax.set_ylim(0,1.03); ax.set_xlabel({_lit(time)}); ax.set_ylabel('Survival probability')")
        if hue:
            L.append(f"ax.legend(title={_lit(hue)})")
            L.append("# log-rank: see Analysis > survival in the app (not reproduced here)")
    elif spec_key in ("michaelis", "growth", "stdcurve"):
        grp = f"list(df.groupby({_lit(hue)}))" if hue else "[(None, df)]"
        if spec_key == "stdcurve":
            grp = "[(None, df)]"
        L.append(f"_groups={grp}; _names=[g for g,_ in _groups]")
        L += _colors_code(p)
        L.append("_txt=[]")
        L.append("for _i,(_nm,_s) in enumerate(_groups):")
        L.append(f"    _xv=pd.to_numeric(_s[{_lit(x)}],errors='coerce'); _yv=pd.to_numeric(_s[{_lit(y)}],errors='coerce')")
        L.append("    _f=pd.DataFrame({'x':_xv,'y':_yv}).dropna()")
        if spec_key == "michaelis":
            L.append("    from scipy.optimize import curve_fit")
            L.append(f"    ax.scatter(_f['x'],_f['y'],s=22,alpha={p.get('alpha',0.8)},color=_colors[_i],"
                     f"label=str(_nm) if {bool(hue)} else None)")
            L.append("    _mm=lambda _z,_vmax,_km: _vmax*_z/(_km+_z)")
            L.append("    _pp,_=curve_fit(_mm,_f['x'],_f['y'],p0=[_f['y'].max(),max(_f['x'][_f['x']>0].median(),1e-9)],"
                     "bounds=([0,1e-12],[np.inf,np.inf]),maxfev=20000)")
            L.append("    _xx=np.linspace(0,_f['x'].max()*1.05,200); ax.plot(_xx,_mm(_xx,*_pp),color=_colors[_i],lw=1.6)")
            L.append("    _yh=_mm(_f['x'],*_pp); _ss=float(((_f['y']-_yh)**2).sum()); "
                     "_st=float(((_f['y']-_f['y'].mean())**2).sum())")
            L.append("    _txt.append(f'Vmax={_pp[0]:.3g}  Km={_pp[1]:.3g}  R2={1-_ss/_st if _st else float('nan'):.3f}')")
            if p.get("show_km"):
                L.append("    ax.axvline(_pp[1],ls=':',color=_colors[_i],lw=1)")
        elif spec_key == "growth":
            lo, hi = p.get("t_min"), p.get("t_max")
            L.append("    _f=_f.groupby('x',as_index=False)['y'].mean()")
            L.append(f"    ax.plot(_f['x'],_f['y'],'o-',ms=4,lw=1.2,color=_colors[_i],"
                     f"label=str(_nm) if {bool(hue)} else None)")
            L.append("    _w=_f")
            if lo not in (None, ""):
                L.append(f"    _w=_w[_w['x']>={float(lo)}]")
            if hi not in (None, ""):
                L.append(f"    _w=_w[_w['x']<={float(hi)}]")
            L.append("    _w=_w[_w['y']>0]")
            L.append("    _mu,_b=np.polyfit(_w['x'],np.log(_w['y']),1)")
            L.append("    _xx=np.linspace(_w['x'].min(),_w['x'].max(),100)")
            L.append("    ax.plot(_xx,np.exp(_b)*np.exp(_mu*_xx),'--',lw=1.4,color=_colors[_i])")
            L.append("    _ly=np.log(_w['y']); _yh=_mu*_w['x']+_b")
            L.append("    _ss=float(((_ly-_yh)**2).sum()); _st=float(((_ly-_ly.mean())**2).sum())")
            L.append("    _txt.append(f'td={np.log(2)/_mu:.3g}  mu={_mu:.3g}  R2={1-_ss/_st if _st else float('nan'):.3f}')")
        else:                                   # stdcurve
            L.append("    ax.scatter(_f['x'],_f['y'],s=30,color='#3a6ea5',zorder=3,label='standards')")
            L.append("    _xx=np.linspace(_f['x'].min(),_f['x'].max(),200)")
            if p.get("model") == "4pl":
                L.append("    from scipy.optimize import curve_fit")
                L.append("    _f4=lambda _z,_a,_b,_c,_d: _d+(_a-_d)/(1+(_z/_c)**_b)")
                L.append("    _pp,_=curve_fit(_f4,_f['x'],_f['y'],p0=[_f['y'].max(),1,max(_f['x'].median(),1e-9),_f['y'].min()],maxfev=20000)")
                L.append("    _yy=_f4(_xx,*_pp); _txt.append('4PL fit')")
            else:
                L.append("    _sl,_ic=np.polyfit(_f['x'],_f['y'],1); _yy=_sl*_xx+_ic")
                L.append("    _r2=np.corrcoef(_f['x'],_f['y'])[0,1]**2")
                L.append("    _txt.append(f'y={_sl:.4g}x+{_ic:.4g}  R2={_r2:.4f}')")
            L.append("    ax.plot(_xx,_yy,color='#e2683b',lw=1.6,zorder=2)")
        L.append(f"ax.set_xlabel({_lit(x)}); ax.set_ylabel({_lit(y)})")
        if spec_key == "growth" and p.get("log_y"):
            L.append("ax.set_yscale('log')")
        va = "'top'" if spec_key in ("growth", "stdcurve") else "'bottom'"
        pos = "0.02,0.97" if spec_key in ("growth", "stdcurve") else "0.98,0.03"
        ha = "'left'" if spec_key in ("growth", "stdcurve") else "'right'"
        L.append(f"ax.text({pos},chr(10).join(_txt),transform=ax.transAxes,ha={ha},va={va},"
                 "fontsize=8,bbox=dict(fc='white',alpha=0.75,ec='none'))")
        if hue and spec_key != "stdcurve":
            L.append("ax.legend()")
    elif spec_key == "roc":
        score, lab = m.get("score"), m.get("label")
        pos = p.get("positive") or ""
        L.append("from sklearn.metrics import roc_curve, roc_auc_score")
        L.append(f"_d=df[[{_lit(score)},{_lit(lab)}]].copy(); "
                 f"_d[{_lit(score)}]=pd.to_numeric(_d[{_lit(score)}],errors='coerce'); _d=_d.dropna()")
        L.append(f"_lv=sorted(_d[{_lit(lab)}].astype(str).unique())")
        L.append(f"_pos={_lit(pos)} or _lv[-1]")
        L.append(f"_y=(_d[{_lit(lab)}].astype(str)==_pos).to_numpy(int); _sc=_d[{_lit(score)}].to_numpy(float)")
        L.append("_fpr,_tpr,_thr=roc_curve(_y,_sc); _auc=roc_auc_score(_y,_sc)")
        L.append(f"ax.plot(_fpr,_tpr,lw=1.8,label=f'{score} AUC={{_auc:.3f}}')")
        L.append("ax.plot([0,1],[0,1],ls='--',lw=1,color='#999')")
        if p.get("mark_cutoff", True):
            L.append("_j=int(np.argmax(_tpr-_fpr))")
            L.append("ax.plot(_fpr[_j],_tpr[_j],'o',ms=6,mfc='none',mew=1.6)")
            L.append("ax.annotate(f'{_thr[_j]:.3g}',(_fpr[_j],_tpr[_j]),textcoords='offset points',"
                     "xytext=(7,-9),fontsize=8)")
        L.append("ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)")
        L.append("ax.set_xlabel('1 - specificity'); ax.set_ylabel('Sensitivity')")
        L.append("ax.set_aspect('equal',adjustable='box'); ax.legend(loc='lower right',fontsize=8,frameon=False)")
    elif spec_key == "bland_altman":
        m1, m2 = m.get("m1"), m.get("m2")
        L.append(f"_d=df[[{_lit(m1)},{_lit(m2)}]].apply(pd.to_numeric,errors='coerce').dropna()")
        L.append(f"_a=_d[{_lit(m1)}].to_numpy(float); _b=_d[{_lit(m2)}].to_numpy(float)")
        L.append("_avg=(_a+_b)/2; _diff=_a-_b")
        L.append("_bias=_diff.mean(); _sd=_diff.std(ddof=1)")
        L.append("_lo,_hi=_bias-1.96*_sd,_bias+1.96*_sd")
        L.append(f"ax.scatter(_avg,_diff,s={float(p.get('psize',3.5))*6},alpha={p.get('alpha',0.8)},color='#3a6ea5')")
        L.append("for _v,_ls,_t in ((_bias,'-','bias'),(_lo,'--','-1.96 SD'),(_hi,'--','+1.96 SD')):")
        L.append("    ax.axhline(_v,ls=_ls,lw=1.2,color='#c2410c' if _ls=='-' else '#777')")
        L.append("    ax.annotate(f'{_t}  {_v:.3g}',(1.0,_v),xycoords=('axes fraction','data'),"
                 "textcoords='offset points',xytext=(-4,3),ha='right',fontsize=7,"
                 "color='#c2410c' if _ls=='-' else '#777')")
        L.append(f"ax.set_xlabel('Mean of {m1} and {m2}'); ax.set_ylabel('Difference ({m1} - {m2})')")
    elif spec_key == "heatmap_xy":
        row, col, val = m.get("row"), m.get("col"), m.get("value")
        dec = int(p.get("decimals", 1))
        L.append(f"_d=df[[{_lit(row)},{_lit(col)},{_lit(val)}]].copy(); "
                 f"_d[{_lit(val)}]=pd.to_numeric(_d[{_lit(val)}],errors='coerce'); _d=_d.dropna(subset=[{_lit(val)}])")
        L.append(f"_mat=_d.pivot_table(index=_d[{_lit(row)}].astype(str),columns=_d[{_lit(col)}].astype(str),"
                 f"values={_lit(val)},aggfunc={_lit(p.get('agg','mean'))})")
        sz = float(p.get("annot_size") or 0)
        kws = f"{{'fontsize': {sz}}}" if sz > 0 else "{'fontsize': 8 if _mat.shape[1]<=12 else 6}"
        L.append(f"sns.heatmap(_mat,cmap={_lit(p.get('palette','viridis'))},annot={bool(p.get('annot',True))},"
                 f"fmt='.{dec}f',annot_kws={kws},cbar=True,linewidths={float(p.get('gap',0.0))},ax=ax)")
        L.append(f"ax.set_xlabel({_lit(col)}); ax.set_ylabel({_lit(row)}); ax.tick_params(axis='y',rotation=0)")
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
