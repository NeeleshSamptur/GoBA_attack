"""Deep dive on recurrence laminarity / trapping: is 'attention gets stuck' universal?"""
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
ROOT = Path('/home/grads/nsamptur/vla_bkd_def/GoBA_attack/attn_maps')
T, EPS = 20, 1e-12

def norm(M):
    M = np.clip(np.asarray(M, float), 0, None); s = M.sum(-1, keepdims=True)
    return np.divide(M, s, out=np.full_like(M, 1/M.shape[-1]), where=s > EPS)
def l2(x): return x/(np.linalg.norm(x, axis=-1, keepdims=True)+EPS)

def vert_runs(R):
    out=[]
    for col in R.T:
        r=0
        for v in col:
            if v: r+=1
            elif r: out.append(r); r=0
        if r: out.append(r)
    return np.asarray(out) if out else np.array([0])

def lam_tt(states, rr, vmin=2):
    N=l2(states); D=1.0-N@N.T
    iu=np.triu_indices(len(D),1); R=D<=np.quantile(D[iu],rr); np.fill_diagonal(R,False)
    v=vert_runs(R)
    lam = v[v>=vmin].sum()/max(v.sum(),1)
    tt  = v[v>=vmin].mean() if (v>=vmin).any() else 0.
    return float(lam), float(tt), float(v.max())

DATA={'GoBA':('trajectory_dof_attention.npz','poison'),
      'BadVLA':('badvla_trajectory_dof_attention.npz','poison'),
      'AttackVLA':('attackvla_trajectory_dof_attention.npz','full_trigger'),
      'CleanModel':('cleanmodel_trajectory_dof_attention.npz','poison')}
cache={}
for k,(f,_) in DATA.items():
    d=np.load(ROOT/f, allow_pickle=True)
    cache[k]=(d['maps'][:,:T], d['role'].astype(str), d['task_id'].astype(int))

for rr in [0.1,0.15,0.2,0.3,0.4]:
    print(f'\n--- recurrence rate RR={rr} ---')
    print(f"  {'dataset':11s} {'clean LAM':>10s} {'atk LAM':>9s} {'AUROC(hi)':>10s} | "
          f"{'clean TT':>9s} {'atk TT':>8s} {'AUROC(hi)':>10s}")
    for ds,(f,pos) in DATA.items():
        maps,roles,tasks=cache[ds]
        vals={}
        for role in ['clean',pos]:
            idx=[i for i in range(len(roles)) if roles[i]==role and not np.isnan(maps[i]).any()]
            L,TTv=[],[]
            for i in idx:
                M=norm(maps[i]); a,b,_=lam_tt(norm(M.mean(1)),rr); L.append(a); TTv.append(b)
            vals[role]=(np.array(L),np.array(TTv))
        c,p=vals['clean'],vals[pos]
        y=np.r_[np.zeros(len(c[0])),np.ones(len(p[0]))]
        aL=roc_auc_score(y,np.r_[c[0],p[0]]); aT=roc_auc_score(y,np.r_[c[1],p[1]])
        print(f"  {ds:11s} {c[0].mean():10.3f} {p[0].mean():9.3f} {aL:10.2f} | "
              f"{c[1].mean():9.2f} {p[1].mean():8.2f} {aT:10.2f}")
