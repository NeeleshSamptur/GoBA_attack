"""Robust version: floor the scale, clip z, and report deployment-style operating points."""
import numpy as np
from sklearn.metrics import roc_auc_score

rows = np.load('/tmp/tsbank_rows.npy', allow_pickle=True).item()
FEATS = [k for k in rows['GoBA'][0] if k not in ('dataset', 'role', 'task')]
TRAIN, TEST = set(range(5)), set(range(5, 10))
ZCLIP = 10.0


def mat(ds, role, tasks):
    sel = [r for r in rows[ds] if r['role'] == role and r['task'] in tasks]
    return np.array([[r[f] for f in FEATS] for r in sel], float)


def calib(X):
    med = np.median(X, 0)
    mad = np.median(np.abs(X - med), 0) * 1.4826
    iqr = (np.quantile(X, .75, 0) - np.quantile(X, .25, 0)) / 1.349
    scale = np.maximum(mad, iqr)
    # floor at 1% of the feature's typical magnitude so constant features can't explode
    scale = np.maximum(scale, 0.01 * (np.abs(med) + 1e-6))
    return med, scale


def score(X, med, scale, agg='mean'):
    z = np.clip(np.abs((X - med) / scale), 0, ZCLIP)
    return z.mean(1) if agg == 'mean' else np.quantile(z, .9, axis=1)


print(f"{'dataset':12s} {'AUROC':>6s} {'clean_mu':>9s} {'atk_mu':>8s} {'TPR@5%FPR':>10s} {'gap':>6s}")
print('-' * 60)
store = {}
for ds, pos in [('GoBA', 'poison'), ('BadVLA', 'poison'),
                ('AttackVLA', 'full_trigger'), ('AttackVLA', 'visual_only'),
                ('CleanModel', 'poison')]:
    Xtr = mat(ds, 'clean', TRAIN)
    med, sc = calib(Xtr)
    thr = np.quantile(score(Xtr, med, sc), 0.95)   # threshold from CLEAN TRAIN only
    c, p = score(mat(ds, 'clean', TEST), med, sc), score(mat(ds, pos, TEST), med, sc)
    a = roc_auc_score(np.r_[np.zeros(len(c)), np.ones(len(p))], np.r_[c, p])
    gap = p.min() - c.max()
    print(f"{ds+'/'+pos[:6]:12s} {a:6.3f} {c.mean():9.2f} {p.mean():8.2f} "
          f"{(p > thr).mean():10.2f} {gap:6.2f}   (clean FPR@thr={(c>thr).mean():.2f})")
    store[ds + '/' + pos] = (c, p, thr)

med, sc = calib(mat('GoBA', 'clean', TRAIN))
thr = np.quantile(score(mat('GoBA', 'clean', TRAIN), med, sc), 0.95)
print('\nSpecificity controls (benign novel objects, GoBA model):')
for decoy in ['decoy_ketchup', 'decoy_milk']:
    d = score(mat('GoBA', decoy, TEST), med, sc)
    c = store['GoBA/poison'][0]
    a = roc_auc_score(np.r_[np.zeros(len(c)), np.ones(len(d))], np.r_[c, d])
    print(f'  {decoy:14s} AUROC={a:.3f}  flag-rate@thr={(d>thr).mean():.2f} (want low)')

print('\nPer-feature contribution to the score gap (mean|z| attack - mean|z| clean, test split):')
contrib = {}
for ds, pos in [('GoBA', 'poison'), ('BadVLA', 'poison'), ('AttackVLA', 'full_trigger')]:
    med, sc = calib(mat(ds, 'clean', TRAIN))
    zc = np.clip(np.abs((mat(ds, 'clean', TEST) - med) / sc), 0, ZCLIP).mean(0)
    zp = np.clip(np.abs((mat(ds, pos, TEST) - med) / sc), 0, ZCLIP).mean(0)
    contrib[ds] = zp - zc
order = np.argsort(-np.minimum.reduce([contrib[k] for k in contrib]))
print(f"  {'feature':24s} {'GoBA':>7s} {'BadVLA':>7s} {'AtkVLA':>7s}")
for i in order[:12]:
    print(f"  {FEATS[i]:24s} {contrib['GoBA'][i]:7.2f} {contrib['BadVLA'][i]:7.2f} {contrib['AttackVLA'][i]:7.2f}")
