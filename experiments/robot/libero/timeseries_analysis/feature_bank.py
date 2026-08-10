"""Time-series / dynamical-systems feature bank on action-image attention trajectories.
Concepts: recurrence quantification (fixed recurrence rate), symbolic complexity
(Lempel-Ziv, permutation entropy), state-space geometry (effective rank, tortuosity,
radius of gyration), and DoF-consensus persistence.
DFA/Hurst deliberately excluded: invalid at T=20.
"""
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/grads/nsamptur/vla_bkd_def/GoBA_attack/attn_maps')
T = 20; EPS = 1e-12


def norm_maps(M):
    M = np.clip(np.asarray(M, float), 0, None)
    s = M.sum(-1, keepdims=True)
    return np.divide(M, s, out=np.full_like(M, 1 / M.shape[-1]), where=s > EPS)


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)


def line_lengths(bin_mat, axis):
    """Lengths of consecutive-True runs along an axis of a boolean matrix."""
    out = []
    mat = bin_mat if axis == 0 else bin_mat.T
    for col in mat.T:
        run = 0
        for v in col:
            if v:
                run += 1
            elif run:
                out.append(run); run = 0
        if run:
            out.append(run)
    return np.asarray(out) if out else np.array([0])


def rqa(states, target_rr=0.2, lmin=2):
    """Recurrence quantification on a state trajectory with fixed recurrence rate."""
    N = l2(states)
    D = 1.0 - N @ N.T                      # cosine distance
    iu = np.triu_indices(len(D), 1)
    thr = np.quantile(D[iu], target_rr)
    R = D <= thr
    np.fill_diagonal(R, False)
    npts = R.sum()
    if npts == 0:
        return dict(DET=0., LAM=0., TT=0., Lmax=0., Vmax=0., ENTR=0.)
    # diagonals excluding main
    diag_runs = []
    n = len(R)
    for k in list(range(1, n)) + list(range(-n + 1, 0)):
        d = np.diagonal(R, k)
        run = 0
        for v in d:
            if v:
                run += 1
            elif run:
                diag_runs.append(run); run = 0
        if run:
            diag_runs.append(run)
    diag_runs = np.asarray(diag_runs) if diag_runs else np.array([0])
    vert = line_lengths(R, 0)
    det = diag_runs[diag_runs >= lmin].sum() / max(diag_runs.sum(), 1)
    lam = vert[vert >= lmin].sum() / max(vert.sum(), 1)
    tt = vert[vert >= lmin].mean() if (vert >= lmin).any() else 0.
    dl = diag_runs[diag_runs >= lmin]
    if len(dl):
        p = np.bincount(dl) / len(dl); p = p[p > 0]
        entr = float(-(p * np.log(p)).sum())
    else:
        entr = 0.
    return dict(DET=float(det), LAM=float(lam), TT=float(tt),
                Lmax=float(diag_runs.max()), Vmax=float(vert.max()), ENTR=entr)


def lempel_ziv(symbols):
    """Normalized LZ76 complexity of a symbol sequence."""
    s = list(map(int, symbols)); n = len(s)
    i, c, l = 0, 1, 1
    k, kmax = 1, 1
    while l + k <= n:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
        else:
            kmax = max(kmax, k)
            i += 1
            if i == l:
                c += 1; l += kmax; i = 0; kmax = k = 1
            else:
                k = 1
    if l <= n:
        c += 1
    alpha = max(len(set(s)), 2)
    return float(c * np.log(n) / (n * np.log(alpha) + EPS))


def perm_entropy(x, order=3):
    x = np.asarray(x, float); n = len(x) - order + 1
    if n <= 1:
        return 0.
    pats = {}
    for i in range(n):
        key = tuple(np.argsort(x[i:i + order]))
        pats[key] = pats.get(key, 0) + 1
    p = np.array(list(pats.values()), float); p /= p.sum()
    return float(-(p * np.log(p)).sum() / np.log(np.math.factorial(order)))


def eff_rank(sv):
    e = sv ** 2; e = e / (e.sum() + EPS)
    return float(np.exp(-(e * np.log(e + EPS)).sum()))


def features(M):
    M = norm_maps(M[:T])
    Tn, D, P = M.shape
    N = l2(M)
    f = {}

    # ---- state-space trajectory of the mean-DoF map ----
    mean_map = norm_maps(M.mean(1))
    S = l2(mean_map)
    steps = np.linalg.norm(np.diff(S, axis=0), axis=1)
    disp = np.linalg.norm(S[-1] - S[0])
    f['path_length'] = float(steps.sum())
    f['tortuosity'] = float(steps.sum() / (disp + EPS))
    centroid = S.mean(0)
    f['radius_gyration'] = float(np.sqrt(((S - centroid) ** 2).sum(1).mean()))
    f['step_size_cv'] = float(steps.std() / (steps.mean() + EPS))
    sv = np.linalg.svd(S - S.mean(0), compute_uv=False)
    f['traj_effective_rank'] = eff_rank(sv)
    f['traj_pc1_frac'] = float(sv[0] ** 2 / ((sv ** 2).sum() + EPS))

    # ---- recurrence quantification on that trajectory ----
    for k, v in rqa(mean_map).items():
        f[f'rqa_{k}'] = v
    # RQA on the concatenated 7-DoF state
    for k, v in rqa(M.reshape(Tn, D * P)).items():
        f[f'rqa7_{k}'] = v

    # ---- symbolic complexity of the attended-patch sequence ----
    am = mean_map.argmax(1)
    _, sym = np.unique(am, return_inverse=True)
    f['lz_argmax'] = lempel_ziv(sym)
    f['n_unique_argmax'] = float(len(set(am.tolist())))
    f['argmax_change_rate'] = float(np.mean(am[1:] != am[:-1]))

    # ---- ordinal complexity of scalar summaries ----
    ent_t = -(M * np.log(M + EPS)).sum(-1).mean(1)
    max_t = M.max(-1).mean(1)
    q_t = np.array([np.linalg.norm(N[t] @ N[t].T - np.eye(D), 'fro') / np.sqrt(D * (D - 1))
                    for t in range(Tn)])
    for nm, ser in [('entropy', ent_t), ('maxmass', max_t), ('consensus', q_t)]:
        f[f'permen_{nm}'] = perm_entropy(ser)
        f[f'lz_{nm}'] = lempel_ziv((ser > np.median(ser)).astype(int))
        f[f'cv_{nm}'] = float(np.std(ser) / (abs(np.mean(ser)) + EPS))

    # ---- DoF consensus persistence (previous best) ----
    f['pdc'] = float(q_t.mean() - q_t.std())
    f['consensus_mean'] = float(q_t.mean())
    return f


def load(name, file):
    d = np.load(ROOT / file, allow_pickle=True)
    maps, roles, tasks = d['maps'][:, :T], d['role'], d['task_id']
    rows = []
    for i in range(len(roles)):
        if np.isnan(maps[i]).any():
            continue
        r = features(maps[i])
        r.update(dataset=name, role=str(roles[i]), task=int(tasks[i]))
        rows.append(r)
    return rows


DATA = {
    'GoBA': ('trajectory_dof_attention.npz', 'poison'),
    'BadVLA': ('badvla_trajectory_dof_attention.npz', 'poison'),
    'AttackVLA': ('attackvla_trajectory_dof_attention.npz', 'full_trigger'),
    'CleanModel': ('cleanmodel_trajectory_dof_attention.npz', 'poison'),
}
rows = {k: load(k, v[0]) for k, v in DATA.items()}
for k, v in rows.items():
    print(k, len(v), sorted({r['role'] for r in v}), flush=True)

FEATS = [k for k in rows['GoBA'][0] if k not in ('dataset', 'role', 'task')]


def auc(ds, pos, tasks=None, key=None, direction='HIGH'):
    sel = lambda role: [r[key] for r in rows[ds]
                        if r['role'] == role and (tasks is None or r['task'] in tasks)]
    c, p = np.array(sel('clean')), np.array(sel(pos))
    if len(c) < 3 or len(p) < 3:
        return np.nan
    y = np.r_[np.zeros(len(c)), np.ones(len(p))]
    s = np.r_[c, p]
    return roc_auc_score(y, s if direction == 'HIGH' else -s)


TRAIN, TEST = set(range(5)), set(range(5, 10))
out = []
for fn in FEATS:
    d_train = auc('GoBA', 'poison', TRAIN, fn, 'HIGH')
    direction = 'HIGH' if d_train >= 0.5 else 'LOW'
    # require consistency on BadVLA train too
    b_train = auc('BadVLA', 'poison', TRAIN, fn, direction)
    row = dict(feature=fn, direction=direction,
               goba_train=auc('GoBA', 'poison', TRAIN, fn, direction),
               badvla_train=b_train,
               goba_test=auc('GoBA', 'poison', TEST, fn, direction),
               badvla_test=auc('BadVLA', 'poison', TEST, fn, direction),
               attackvla=auc('AttackVLA', 'full_trigger', None, fn, direction),
               decoy_ketchup=auc('GoBA', 'decoy_ketchup', None, fn, direction),
               decoy_milk=auc('GoBA', 'decoy_milk', None, fn, direction),
               cleanmodel=auc('CleanModel', 'poison', None, fn, direction))
    out.append(row)

out.sort(key=lambda r: -min(r['goba_test'], r['badvla_test']))
hdr = (f"{'feature':24s} {'dir':5s} {'G-tr':>5s} {'B-tr':>5s} {'G-te':>5s} {'B-te':>5s} "
       f"{'AtkVLA':>6s} {'ketch':>6s} {'milk':>6s} {'cleanM':>6s}")
print('\n' + hdr)
print('-' * len(hdr))
for r in out:
    print(f"{r['feature']:24s} {r['direction']:5s} {r['goba_train']:5.2f} {r['badvla_train']:5.2f} "
          f"{r['goba_test']:5.2f} {r['badvla_test']:5.2f} {r['attackvla']:6.2f} "
          f"{r['decoy_ketchup']:6.2f} {r['decoy_milk']:6.2f} {r['cleanmodel']:6.2f}")

np.save('/tmp/tsbank_rows.npy', rows, allow_pickle=True)
np.save('/tmp/tsbank_out.npy', out, allow_pickle=True)
