# -*- coding: utf-8 -*-
"""Generate the script for EVERY plot type and actually run it.

The exported script is the app's reproducibility promise; "it emits something" is not
the bar - it has to import, execute and draw. Run: python test_codegen.py
"""
import io
import os
import sys
import tempfile
import traceback

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.plot_registry import REGISTRY          # noqa: E402
from core import codegen as CG                   # noqa: E402

rng = np.random.default_rng(7)
N = 120
df = pd.DataFrame({
    "cond": np.repeat(["ctrl", "drug", "wash"], N // 3),
    "rep": np.tile(["r1", "r2", "r3"], N // 3),
    "outcome": np.where(rng.random(N) > 0.5, "alive", "dead"),
    "value": rng.normal(10, 2, N),
    "value2": rng.normal(5, 1, N),
    "dose": np.tile(np.logspace(-2, 3, 12), N // 12),
    "time": np.tile(np.arange(N // 6), 6).astype(float),
    "track": np.repeat([f"t{i}" for i in range(12)], N // 12),
    "posx": rng.normal(0, 1, N).cumsum(),
    "posy": rng.normal(0, 1, N).cumsum(),
})
df["response"] = 100 / (1 + (df["dose"] / 5.0) ** 1.3) + rng.normal(0, 3, N)
df["growth"] = np.exp(0.3 * df["time"]) + rng.normal(0, 0.4, N).clip(-0.3, None)
df["growth"] = df["growth"].clip(lower=0.05)
df["surv"] = rng.gamma(3, 4, N)
df["score"] = np.where(df["outcome"] == "dead", rng.normal(2, 1, N), rng.normal(0, 1, N))
df["method_a"] = rng.normal(10, 2, N)
df["method_b"] = df["method_a"] + rng.normal(0.4, 0.6, N)

# a sensible column for each channel, per plot (auto-mapping cannot know that a ROC
# label must have exactly 2 levels, or that growth needs positive values)
MAP = {
    "scatter": {"x": "value", "y": "value2", "hue": "cond"},
    "scatter_density": {"x": "value", "y": "value2", "hue": "cond"},
    "regband": {"x": "value", "y": "value2", "hue": "cond"},
    "line": {"x": "time", "y": "value", "hue": "cond"},
    "bar": {"x": "cond", "y": "value"},
    "box": {"x": "cond", "y": "value"},
    "violin": {"x": "cond", "y": "value"},
    "strip": {"x": "cond", "y": "value"},
    "box_points": {"x": "cond", "y": "value"},
    "violin_points": {"x": "cond", "y": "value"},
    "bar_err": {"x": "cond", "y": "value"},
    "superplot": {"x": "cond", "y": "value", "rep": "rep"},
    "hist": {"x": "value", "hue": "cond"},
    "kde": {"x": "value", "hue": "cond"},
    "ecdf": {"x": "value", "hue": "cond"},
    "ridge": {"x": "value", "hue": "cond"},
    "qq": {"val": "value", "hue": "cond"},
    "heatmap": {},
    "heatmap_matrix": {"labels": "cond"},
    "heatmap_xy": {"row": "cond", "col": "rep", "value": "value"},
    "stacked": {"x": "cond", "hue": "outcome"},
    "km": {"time": "surv", "event": "outcome", "hue": "cond"},
    "traj": {"x": "posx", "y": "posy", "id": "track", "time": "time", "hue": "cond"},
    "msd": {"id": "track", "px": "posx", "py": "posy", "time": "time", "hue": "cond"},
    "paired": {"x": "cond", "y": "value", "id": "rep"},
    "dose4pl": {"x": "dose", "y": "response", "hue": "cond"},
    "volcano": {"x": "value", "y": "value2"},
    "michaelis": {"x": "dose", "y": "response"},
    "growth": {"x": "time", "y": "growth", "hue": "cond"},
    "stdcurve": {"x": "dose", "y": "response"},
    "roc": {"score": "score", "label": "outcome"},
    "bland_altman": {"m1": "method_a", "m2": "method_b"},
}
PARAM_OVERRIDE = {"km": {"event_value": "dead"}, "roc": {"positive": "dead"}}


def defaults(spec, key):
    p = {pp.name: pp.default for pp in spec.params}
    p.update(PARAM_OVERRIDE.get(key, {}))
    return p


def state_for(key, spec, csv):
    return {
        "style": {"theme": "Light · grid", "context": "notebook", "font_scale": 1.0},
        "title": "T", "xlabel": "", "ylabel": "", "filter": "",
        "layers": [{"spec_key": key, "mapping": MAP[key], "params": defaults(spec, key)}],
    }


def main():
    tmp = tempfile.mkdtemp(prefix="npcodegen_")
    csv = os.path.join(tmp, "data.csv")
    df.to_csv(csv, index=False)
    cwd = os.getcwd()
    os.chdir(tmp)
    ok, bad = [], []
    try:
        for key, spec in REGISTRY.items():
            if key not in MAP:
                bad.append((key, "no test mapping"))
                continue
            code = CG.generate_code(state_for(key, spec, csv), csv)
            if "not supported by the code generator" in code:
                bad.append((key, "codegen emitted a placeholder"))
                continue
            io.open(f"{key}.py", "w", encoding="utf-8").write(code)
            g = {"__name__": "__main__"}
            try:
                exec(compile(code, f"<{key}>", "exec"), g)
            except Exception as e:
                bad.append((key, f"{type(e).__name__}: {e}"))
                traceback.print_exc()
                continue
            ax = g["ax"]
            drawn = (len(ax.collections) + len(ax.lines) + len(ax.patches)
                     + len(ax.images) + len(ax.texts))
            if drawn == 0:
                bad.append((key, "script ran but drew nothing"))
                continue
            if not os.path.exists("figure.png"):
                bad.append((key, "no figure.png written"))
                continue
            os.remove("figure.png")
            g["plt"].close("all")
            ok.append(key)
    finally:
        os.chdir(cwd)
    for k in ok:
        print(f"OK   {k}")
    for k, why in bad:
        print(f"FAIL {k}: {why}")
    print(f"\n{len(ok)}/{len(REGISTRY)} plot types produce a script that runs and draws")
    if bad:
        sys.exit(1)
    print("scripts written to", tmp)


if __name__ == "__main__":
    main()
