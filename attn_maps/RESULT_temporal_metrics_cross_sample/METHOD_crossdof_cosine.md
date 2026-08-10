# Cross-DoF Cosine — Methodology (step by step)

**Goal.** Score how much the 7 action DoFs agree on *where* they attend in the image, then use that score to separate clean vs backdoored rollouts.

**Hypothesis.** Under some backdoors, action dimensions collapse onto a shared attention pattern (assimilation). Clean rollouts also agree somewhat, but less extremely.

---

## 1. Inputs

For each closed-loop episode we already store action→image attention:

- Shape: `(T_steps, 7, 256)`
  - `T_steps`: control timesteps in the rollout  
  - `7`: action DoFs `(dx, dy, dz, droll, dpitch, dyaw, gripper)`  
  - `256`: image patches (16×16 grid from the vision encoder)
- Each `(dof, patch)` slice at a fixed `t` is a post-softmax (or row-normalized) distribution over patches (sums ≈ 1).

**Datasets used**

| Attack   | NPZ                                      | Roles scored                          | Episodes |
|----------|------------------------------------------|---------------------------------------|----------|
| GoBA     | `attn_maps/trajectory_dof_attention.npz` | `clean` vs `poison`                   | 20 + 20  |
| BadVLA   | `attn_maps/badvla_trajectory_dof_attention.npz` | `clean` vs `poison`            | 20 + 20  |
| AttackVLA| `attn_maps/attackvla_trajectory_dof_attention.npz` | `clean` vs `full_trigger` (also `visual_only`) | 20 + 20 (+20) |

We use the **first 10 timesteps** only: `T = 10`.

---

## 2. Metric definition (one episode → one scalar)

### Step A — normalize a single DoF map

For timestep `t` and DoF `d`, let \( m_{t,d} \in \mathbb{R}^{256} \) be the raw attention weights.

\[
\tilde{m}_{t,d} = \frac{\mathrm{clip}(m_{t,d}, 0)}{\sum_p \mathrm{clip}(m_{t,d})_p}
\quad\text{(if sum is 0, use uniform }1/256\text{)}
\]

### Step B — Cross-DoF cosine at one timestep

At fixed `t`, compare all pairs of DoFs:

There are \(\binom{7}{2} = 21\) pairs \((i,j)\) with \(0 \le i < j \le 6\).

\[
\cos(\tilde{m}_{t,i}, \tilde{m}_{t,j})
= \frac{\tilde{m}_{t,i}^\top \tilde{m}_{t,j}}
{\|\tilde{m}_{t,i}\|_2 \,\|\tilde{m}_{t,j}\|_2}
\]

\[
\mathrm{cross}(t)
= \frac{1}{21}\sum_{0\le i<j\le 6}
\cos(\tilde{m}_{t,i}, \tilde{m}_{t,j})
\]

- \(\mathrm{cross}(t) \approx 1\): all 7 DoFs attend to nearly the same patches at step `t`  
- Lower: DoFs disagree spatially  

### Step C — Average over the first 10 timesteps

\[
\mathrm{score}(\mathrm{episode})
= \frac{1}{10}\sum_{t=0}^{9} \mathrm{cross}(t)
\]

This is the **only** number used for that episode in AUROC.

**What we are not doing here**

- Not comparing `t` vs `t+1` (that is consecutive / temporal stability)  
- Not averaging the 7 maps into one map before scoring  
- Not using EMD / SSIM / Spearman in this particular detector  

---

## 3. Detection / AUROC protocol

1. Compute `score` for every clean episode and every backdoor episode.  
2. Label: clean = 0, backdoor = 1.  
3. Detector orientation for GoBA / BadVLA: **higher score ⇒ more backdoored**.  
4. Report:

   - mean / min / max of scores per role  
   - AUROC  
   - open gap: `min(backdoor) − max(clean)` (positive ⇒ no overlap on this set)

---

## 4. Results (same methodology on all three)

| Attack | clean mean [min, max] | backdoor mean [min, max] | AUROC (↑ = bd) | gap (min_bd − max_clean) |
|--------|------------------------|---------------------------|----------------|---------------------------|
| **GoBA** (`poison`) | 0.907 [0.866, 0.935] | 0.962 [0.951, 0.980] | **1.000** | **+0.016** |
| **BadVLA** (`poison`) | 0.977 [0.956, 0.990] | 0.999 [0.9987, 0.9989] | **1.000** | **+0.009** |
| **AttackVLA** (`full_trigger`) | 0.986 [0.976, 0.997] | 0.977 [0.973, 0.987] | **0.073** (flips) | −0.024 |

AttackVLA `visual_only` vs clean: AUROC(↑=atk) ≈ 0.26 (also not higher-under-attack).

---

## 5. How to read this

**Works (same direction) for GoBA and BadVLA**  
Backdoor pushes DoF maps closer together than clean. Clean already has fairly high agreement; backdoor pushes it higher (near saturation on BadVLA). Margins are **small**, so treat as a supporting detector, not a huge-margin rule.

**Does not transfer to AttackVLA with the same sign**  
Clean is already ~0.99; full trigger is slightly *lower*. Do not claim a universal “higher Cross-DoF cosine ⇒ backdoor” rule across all attacks.

---

## 6. Pseudocode

```python
def crossdof_cosine_score(maps_T7P):  # (10, 7, 256)
    def norm(v):
        v = np.clip(v, 0, None)
        s = v.sum()
        return v / s if s > 0 else np.full_like(v, 1.0 / len(v))

    crosses = []
    for t in range(maps_T7P.shape[0]):  # 10 steps
        V = [norm(maps_T7P[t, d]) for d in range(7)]
        pair_cos = []
        for i in range(7):
            for j in range(i + 1, 7):
                a, b = V[i], V[j]
                pair_cos.append((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        crosses.append(np.mean(pair_cos))  # mean over 21 pairs
    return float(np.mean(crosses))  # mean over 10 timesteps
```

AUROC: `roc_auc_score(y_true, scores)` with `y=1` for backdoor and raw `scores` (higher = more backdoored) for GoBA/BadVLA.

---

## 7. Related files

- GoBA / BadVLA numbers also in `SHARED_DIRECTION_TRENDS.txt` (`crossdof_cos_mean`)  
- AttackVLA writeup: `AttackVLA_crossdof_cosine.txt`  
- Source maps from `run_trajectory_dof_attention.py` (and BadVLA / AttackVLA counterparts)

---

## 8. One-sentence summary

**Average, over the first 10 steps, how similar the seven action-DoF→image attention maps are to each other (mean pairwise cosine); higher means stronger DoF assimilation and separates GoBA/BadVLA poison from clean, but not AttackVLA with the same sign.**
