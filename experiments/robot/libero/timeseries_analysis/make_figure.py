import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
OUT = Path('/home/grads/nsamptur/vla_bkd_def/GoBA_attack/attn_maps/RESULT_temporal_metrics_cross_sample')
rows = np.load('/tmp/tsbank_rows.npy', allow_pickle=True).item()
FEATS = [k for k in rows['GoBA'][0] if k not in ('dataset','role','task')]
TRAIN, TEST = set(range(5)), set(range(5,10))

def get(ds, role, feat, tasks=None):
    return np.array([r[feat] for r in rows[ds] if r['role']==role and (tasks is None or r['task'] in tasks)])

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# (a) does attention move? path length in map space
sets = [('GoBA','poison','GoBA'), ('BadVLA','poison','BadVLA'), ('AttackVLA','full_trigger','AttackVLA'),
        ('CleanModel','poison','Clean model\n(trigger present)')]
pos_x = []
for i,(ds,pos,lab) in enumerate(sets):
    c, p = get(ds,'clean','path_length'), get(ds,pos,'path_length')
    for j,(v,col) in enumerate([(c,'#3b7dd8'),(p,'#d1495b')]):
        x = i + (j-0.5)*0.32
        ax[0,0].scatter(np.full(len(v), x)+np.random.uniform(-.05,.05,len(v)), v, s=16, c=col, alpha=.65,
                        label=('clean' if j==0 else 'trigger') if i==0 else None)
        ax[0,0].hlines(v.mean(), x-.12, x+.12, color=col, lw=2.5)
    a = roc_auc_score(np.r_[np.zeros(len(c)),np.ones(len(p))], np.r_[c,p])
    ax[0,0].text(i, -0.6, f'AUROC={1-a:.2f}', ha='center', fontsize=9,
                 color='#d1495b' if 1-a > .6 else '#555')
    pos_x.append(lab)
ax[0,0].set_xticks(range(4)); ax[0,0].set_xticklabels(pos_x, fontsize=9)
ax[0,0].set_ylim(-1.1, 7.6)
ax[0,0].set_ylabel('trajectory path length in map space')
ax[0,0].set_title('(a) Attention moves LESS under GoBA/BadVLA, MORE under AttackVLA\n(AUROC scored as: less movement = attack)', fontsize=10)
ax[0,0].legend(fontsize=8)

# (b) laminarity / trapping time vs recurrence threshold  (from /tmp/lam.py numbers)
rr = [0.1,0.15,0.2,0.3,0.4]
lam_auc = {'GoBA':[.72,.54,.24,.12,.25], 'BadVLA':[1.,1.,.79,.03,.03],
           'AttackVLA':[.47,.82,.89,.78,.73], 'CleanModel':[.50,.51,.56,.54,.41]}
for k,v in lam_auc.items():
    ax[0,1].plot(rr, v, 'o-', label=k)
ax[0,1].axhline(.5, ls='--', c='k', lw=.8)
ax[0,1].set_xlabel('recurrence rate RR'); ax[0,1].set_ylabel('AUROC (high laminarity = attack)')
ax[0,1].set_ylim(0,1.05); ax[0,1].legend(fontsize=8)
ax[0,1].set_title('(b) Laminarity is threshold-unstable and sign-flips: not a usable rule', fontsize=10)

# (c) clean-calibrated unsigned deviation
def mat(ds, role, tasks):
    sel=[r for r in rows[ds] if r['role']==role and r['task'] in tasks]
    return np.array([[r[f] for f in FEATS] for r in sel], float)
def calib(X):
    med=np.median(X,0); mad=np.median(np.abs(X-med),0)*1.4826
    iqr=(np.quantile(X,.75,0)-np.quantile(X,.25,0))/1.349
    return med, np.maximum(np.maximum(mad,iqr), .01*(np.abs(med)+1e-6))
def sc(X,m,s): return np.clip(np.abs((X-m)/s),0,10).mean(1)

labels, data, cols = [], [], []
for ds,pos,lab in [('GoBA','poison','GoBA'),('BadVLA','poison','BadVLA'),
                   ('AttackVLA','full_trigger','AttackVLA'),('CleanModel','poison','Clean model')]:
    m,s = calib(mat(ds,'clean',TRAIN))
    data += [sc(mat(ds,'clean',TEST),m,s), sc(mat(ds,pos,TEST),m,s)]
    labels += [f'{lab}\nclean', f'{lab}\nTRIG']; cols += ['#3b7dd8','#d1495b']
m,s = calib(mat('GoBA','clean',TRAIN))
for decoy,nm in [('decoy_ketchup','decoy\nketchup'),('decoy_milk','decoy\nmilk')]:
    data.append(sc(mat('GoBA',decoy,TEST),m,s)); labels.append(nm); cols.append('#8d99ae')
bp = ax[1,0].boxplot(data, labels=labels, patch_artist=True, widths=.6)
for b,c in zip(bp['boxes'],cols): b.set_facecolor(c); b.set_alpha(.6)
ax[1,0].tick_params(axis='x', labelsize=7, rotation=0)
ax[1,0].axvspan(8.5, 10.5, color='#eeeeee', zorder=0)
ax[1,0].set_ylabel('mean |robust z| vs clean calibration')
ax[1,0].set_title('(c) Unsigned deviation: fires on all 3 attacks, silent on decoys/clean model', fontsize=10)

# (d) permutation null
rng=np.random.default_rng(0)
for i,(ds,pos) in enumerate([('GoBA','poison'),('BadVLA','poison'),('AttackVLA','full_trigger')]):
    m,s = calib(mat(ds,'clean',TRAIN))
    c,p = sc(mat(ds,'clean',TEST),m,s), sc(mat(ds,pos,TEST),m,s)
    allv=np.r_[c,p]; n1=len(p)
    null=[roc_auc_score(np.eye(2)[np.isin(np.arange(len(allv)), rng.choice(len(allv),n1,False)).astype(int)][:,1], allv) for _ in range(2000)]
    obs=roc_auc_score(np.r_[np.zeros(len(c)),np.ones(len(p))], np.r_[c,p])
    ax[1,1].hist(null, bins=30, alpha=.45, label=f'{ds} null')
    ax[1,1].axvline(obs, lw=2, color=f'C{i}', label=f'{ds} observed={obs:.2f}')
ax[1,1].set_xlabel('AUROC'); ax[1,1].set_ylabel('count'); ax[1,1].legend(fontsize=7.5)
ax[1,1].set_title('(d) Observed separation vs shuffled-label null (held-out tasks)', fontsize=10)

plt.tight_layout()
p = OUT/'FIGURE_timeseries_concepts.png'
plt.savefig(p, dpi=140)
print('saved', p)
