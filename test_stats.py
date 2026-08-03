"""Self-check for the bench-math and statistics functions. Run: python test_stats.py"""
import numpy as np, pandas as pd
from core import stats as S
np.random.seed(0)
ok = lambda m: print('OK  ' + m)

# ---------- two-way ANOVA: build a KNOWN interaction ----------
rows = []
for a in ['ctrl', 'drug']:
    for b in ['t0', 't24']:
        mu = {('ctrl','t0'):10, ('ctrl','t24'):10, ('drug','t0'):10, ('drug','t24'):20}[(a,b)]
        for _ in range(8):
            rows.append({'y': np.random.normal(mu, 1.5), 'treat': a, 'time': b})
d2 = pd.DataFrame(rows)
txt = S.two_way_anova(d2, 'y', 'treat', 'time')
print(txt); assert 'treat x time' in txt
# interaction must be highly significant here
inter_line = [l for l in txt.splitlines() if l.startswith('treat x time')][0]
p_inter = float(inter_line.split('p=')[1])
assert p_inter < 1e-6, p_inter
ok('two_way_anova detects the planted interaction (p=%.2g)' % p_inter)

# ---------- RM ANOVA ----------
subj = []
for s in range(12):
    base = np.random.normal(0, 3)
    for j, c in enumerate(['pre', 'mid', 'post']):
        subj.append({'sid': f's{s}', 'cond': c, 'val': base + j * 2 + np.random.normal(0, 1)})
txt = S.rm_anova(pd.DataFrame(subj), 'val', 'cond', 'sid')
print(txt); assert 'Friedman' in txt
ok('rm_anova + Friedman')

# ---------- power ----------
n = S.sample_size_ttest(0.8, 0.05, 0.80)
print('n per group for d=0.8, 80%% power:', n)
assert 24 <= n <= 28, n          # classic answer is 26
assert 0.79 < S.power_ttest(n, 0.8) < 0.95
ok('sample_size_ttest matches the textbook n=26 for d=0.8')

# ---------- outliers ----------
x = list(np.random.normal(10, 1, 20)) + [50.0]
dfo = pd.DataFrame({'v': x, 'g': ['a'] * 21})
flag, rep = S.find_outliers(dfo, 'v', method='grubbs')
assert flag.sum() == 1 and dfo.loc[flag, 'v'].iloc[0] == 50.0
ok('find_outliers (grubbs) catches the planted 50.0')
flag2, _ = S.find_outliers(dfo, 'v', method='iqr'); assert flag2.sum() >= 1
ok('find_outliers (iqr)')

# ---------- correlation with p ----------
n_ = 60
a = np.random.normal(0, 1, n_); b = a * 0.9 + np.random.normal(0, 0.4, n_); c = np.random.normal(0, 1, n_)
r, p, stars = S.corr_with_p(pd.DataFrame({'a': a, 'b': b, 'c': c}), ['a','b','c'])
print('r(a,b)=%.2f stars=%s | r(a,c)=%.2f stars=%r' % (r.loc['a','b'], stars.loc['a','b'], r.loc['a','c'], stars.loc['a','c']))
assert r.loc['a','b'] > 0.8 and stars.loc['a','b'] == '***'
ok('corr_with_p + BH stars')

# ---------- qPCR ddCt ----------
q = []
for grp, delta in [('control', 0.0), ('treated', -2.0)]:   # treated: target 2 Ct earlier => ~4x up
    for s in range(3):
        sid = f'{grp}_{s}'
        for _ in range(3):                                  # technical replicates
            q.append({'Ct': 25 + delta + np.random.normal(0, .05), 'gene': 'IL6', 'sample': sid, 'group': grp})
            q.append({'Ct': 20 + np.random.normal(0, .05),        'gene': 'GAPDH', 'sample': sid, 'group': grp})
res = S.ddct(pd.DataFrame(q), 'Ct', 'gene', 'sample', 'group', 'GAPDH', 'control')
fold_tr = res[res['group'] == 'treated']['fold_change'].mean()
print(res.head(3).to_string()); print('mean fold (treated) =', round(fold_tr, 2))
assert 3.5 < fold_tr < 4.5, fold_tr
ok('ddct: 2 Ct shift -> ~4x fold change')

# ---------- standard curve + interpolation ----------
conc = np.array([0, 2, 4, 6, 8, 10]); sig = 0.15 * conc + 0.05
sc = pd.DataFrame({'conc': conc, 'abs': sig})
r1 = S.standard_curve(sc, 'conc', 'abs', 'linear', unknown_signals=[0.65, 1.4])
print(r1['text']); back = r1['predict_conc']([0.65, 1.4])
print('interpolated:', np.round(back, 3))
assert abs(back[0] - 4.0) < 1e-6 and r1['r2'] > 0.999
ok('standard_curve linear + interpolation (0.65 -> 4.0)')
r_oor = S.standard_curve(sc, 'conc', 'abs', 'linear', unknown_signals=[2.0])  # -> ~13, above 10
assert 'OUTSIDE' in r_oor['text'], r_oor['text']
ok('standard_curve warns when an unknown extrapolates past the standards')

# ---------- Michaelis-Menten ----------
Sconc = np.array([0.5, 1, 2, 5, 10, 20, 50], float); Vmax_t, Km_t = 100.0, 5.0
v = Vmax_t * Sconc / (Km_t + Sconc) + np.random.normal(0, .5, len(Sconc))
mm = S.michaelis_menten(Sconc, v)
print('MM: Vmax=%.1f Km=%.2f r2=%.4f' % (mm['Vmax'], mm['Km'], mm['r2']))
assert abs(mm['Vmax'] - 100) < 5 and abs(mm['Km'] - 5) < 1
ok('michaelis_menten recovers Vmax=100, Km=5')

# ---------- growth / doubling time ----------
t = np.arange(0, 10, 0.5); td_true = 2.0
y = 0.05 * np.exp(np.log(2) / td_true * t)
g = S.growth_fit(t, y)
print('growth: doubling=%.3f h, mu=%.3f, r2=%.4f' % (g['doubling_time'], g['mu'], g['r2']))
assert abs(g['doubling_time'] - 2.0) < 0.05
ok('growth_fit recovers a 2 h doubling time')

# ---------- normalisation ----------
dn = pd.DataFrame({'sig':[10,12,20,22], 'load':[1,1.2,1,1.1], 'grp':['c','c','t','t']})
pc = S.normalize_series(dn,'sig','percent_control',group_col='grp',control='c')
dv = S.normalize_series(dn,'sig','divide_by',by_col='load')
assert abs(pc.iloc[0]-90.9)<0.5 and abs(dv.iloc[1]-10.0)<1e-9
ok('normalize_series percent_control + divide_by (loading control)')

# ---------- data health ----------
dh = pd.DataFrame({'a':[1,2,None,2],'const':[5,5,5,5],'txt':['1','2','x','4']})
dh = pd.concat([dh, dh.iloc[[0]]])
h = S.data_health(dh); print(h['text'])
assert 'duplicated' in h['text'] and 'Constant' in h['text']
ok('data_health flags duplicates, constants and numeric-looking text')

# ---------- plate -> tidy ----------
vals = [['', '1','2','3'], ['A','0.1','0.2','0.3'], ['B','0.4','0.5','0.6']]
mp   = [['', '1','2','3'], ['A','ctrl','ctrl','drug'], ['B','drug','drug','ctrl']]
pt = S.plate_to_tidy(vals, mp)
print(pt.to_string())
assert len(pt) == 6 and set(pt['condition']) == {'ctrl','drug'} and pt.iloc[0]['well'] == 'A1'
ok('plate_to_tidy strips labels, maps conditions, 6 wells')

print('\nALL BATCH-2 BACKEND TESTS PASSED')
