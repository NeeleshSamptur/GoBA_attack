# QKTV Attention Map States — Brief for Pattern Mining

**Purpose of this file:** Pass to a strong LLM and ask it to identify patterns that differentiate backdoored vs clean VLAs from action→image QKTV maps. Everything needed is below (definitions, visual map state, numbers, controls). Do not assume external context.

**Status:** revised 2026-08-10 after re-deriving every number directly from the `*_temporal.npz` arrays. Four claims in the original version were contradicted by the data and are corrected in place, marked **[CORRECTED]**. New verified results are in §5. Reproduce everything with:

```
python experiments/robot/libero/score_map_stats_qktv_temporal.py --all
```

**Question to ask the LLM (example):**
> Using only this document, list (1) shared patterns across GoBA and BadVLA under trigger, (2) attack-specific signatures, (3) what clean-model controls rule out, (4) which map statistics best separate BD+trigger from the other three cells of the 2×2 (model ∈ {clean, BD} × input ∈ {clean, poison}).

---

## 0. Corrections log (read first)

| # | Original claim | Verdict |
|---|---|---|
| 1 | Clean OpenVLA + sticker does **not** produce GoBA's locked peripheral hotspot (§3.4) | **False.** Clean OpenVLA puts 49/100 argmaxes on patch 211; GoBA puts 52/100. The hotspot is bottom-up sticker saliency and is *not* weight-dependent. |
| 2 | GoBA clean−poison FTT separation **grows** over the rollout (§4.2) | **False.** Linear fit on the gap gives slope +0.00002/step (Δ = +0.0162 at t=0, +0.0186 at t=9). The AUROC gain 0.71→0.91 is variance reduction from averaging 10 frames, not a deepening effect. |
| 3 | GoBA peak mass ↑ under trigger indicates "trigger capture" (§4.1) | **Not backdoor-specific.** GoBA moves 0.037→0.041; Clean OpenVLA moves 0.041→0.045, a *larger* jump. Peak mass carries no GoBA signal. |
| 4 | Clean OFT FTT moves the wrong way, so FTT thresholding can false-alarm OFT (§5.5) | **Unsupported.** AUROC 0.36 has bootstrap CI [0.12, 0.63], which contains 0.5. Same for the Clean OpenVLA "scene confound" at 0.63, CI [0.36, 0.87]. Neither control effect is established at n=10. |

**Sample size caveat applying to the whole document:** 10 episodes per cell (one per LIBERO-Goal task), one seed, T=10, last layer only. AUROC granularity is 0.01 and 95% CIs are wide. Open margins (no overlap between the two 10-episode groups) are reported explicitly because they are more informative than AUROC here.

---

## 1. Setup (fixed across all four models)

| Item | Value |
|---|---|
| Benchmark | LIBERO-Goal, tasks 0–9 |
| Seed | 7 |
| Horizon | T=10 closed-loop steps after 10 settle steps |
| Layer | Last LLM layer |
| Queries | 7 action tokens / DoF slots |
| Keys | Image patches (16×16 = 256) concatenated with instruction tokens |
| Map shown | **Mean over 7 DoFs** of action→**image only** (text keys dropped for viz) |
| Overlay | Heatmap on RGB; yellow = high attention |
| Trigger GoBA protocol | Poison BDDL / sticker-style scene trigger (GoBA) |
| Trigger BadVLA protocol | White block overlay (~10% size, center) on primary+wrist cams |

### What "QKTV" means

- **weight:** head-averaged attention \(\alpha\) (standard softmax weights).
- **QKTV (Kobayashi-style value-weighted):** \(\sum_h \alpha_h \cdot \|W_O^h v_h\|\) — attention reweighted by how much value mass that head contributes after \(W_O\).
- Maps in figures / `maps_qktv` are **QKTV**, not raw weights (unless noted).

### FTT (dispersion of DoF rows)

For the 7×256 action→image matrix \(P\) after **row-sum normalize**:

\[
\mathrm{FTT}=\frac1{7}\sum_i \|p_i-\bar p\|_2,\qquad \bar p=\tfrac17\sum_i p_i
\]

Low FTT ⇒ DoF rows look alike (assimilation). Episode score often = mean over \(t\).

### Other map scalars used below

From the mean-DoF QKTV map \(m\in\mathbb{R}^{256}\) (renormalized to a distribution):

- **Normalized entropy** \(H_n = H(m)/\log 256\) — 1 = uniform over patches, low = peaked.
- **Peak mass** \(\max_j m_j\).
- **Centroid** \((c_y,c_x)\) in patch grid coords \([0,15]\).
- **Temporal consec. cosine** of maps at \(t\) vs \(t+1\).
- **Moran** — 4-neighbour spatial autocorrelation on the 16×16 grid, \(\sum_{\text{adj}} c_i c_j / \sum_i c_i^2\) with \(c = m - \bar m\). This omits the textbook \(N/W\) factor (multiply by 0.533 for Moran's I proper); the rescaling is monotone and does not affect AUROC. High = smooth, low = speckled.
- **blob16** — size of the largest 4-connected component among the top-16 patches.

---

## 2. The four cells (what to compare)

| Short name | Weights | Input |
|---|---|---|
| **GoBA BD** | GoBA LoRA backdoored OpenVLA | clean vs poison BDDL |
| **Clean OpenVLA** | `openvla/openvla-7b-finetuned-libero-goal` | same GoBA clean/poison protocol |
| **BadVLA BD** | BadVLA stage-2 OFT checkpoint | clean vs white-block |
| **Clean OFT** | `moojink/openvla-7b-oft-finetuned-libero-goal` | same BadVLA white-block protocol |

Ideal backdoor signature: **only** (BD model + poison input) shows the anomalous map state; clean model under poison should look like clean.

---

## 3. Visual state of the maps (qualitative)

### 3.1 BadVLA BD — clean vs poison (same backdoored weights)

**Clean input**
- Sparse, localized yellow spots on task objects / gripper.
- Background mostly dark.
- Hotspot can move mildly over \(t=0..9\) (tracks scene).
- Looks "selective."

**Poison input (white block)**
- Map becomes a **bright global wash**: warm orange/yellow over almost the whole frame.
- Patch grid texture visible; little object selectivity.
- State is **locked for all tasks × all timesteps** — no return to sparse focus.
- Looks "saturated / non-selective."

**One-line signature:** sparse object focus → **near-uniform full-frame attention** under trigger.

**[NEW] What the wash actually is.** The poison map is 1.43× uniform at its peak (clean: 25.6×), and its faint residual structure is *not* visual content:

| measurement | BadVLA clean | BadVLA poison |
|---|---:|---:|
| peak / uniform | 25.63 | **1.43** |
| correlation across *different episodes* (different tasks, different scenes) | +0.807 | **+0.886** |
| spatial neighbour correlation | +0.578 | **+0.872** |
| within-episode correlation, first 5 frames vs last 5 | +0.611 | **+0.343** |

A map that is *more* consistent across unrelated scenes than the clean model's, is spatially smooth, and stops tracking content over time is a fixed positional prior, not perception. Corroborating this, FTT across all 10 tasks spans **[0.0069, 0.0072]** — a range of 3e-4, i.e. a collapsed constant.

The likely mechanism is **blinding, not diffusion**: attention mass leaves the image keys entirely (for text / BOS-sink tokens), and what is plotted is a renormalized near-zero residual. The stored arrays are already image-sliced and sum to exactly 1.0, so the raw image-key mass fraction was discarded and **cannot be recovered from these npz files**. This is the same failure mode recorded for the AttackVLA pipeline (cam1-only maps renormalizing a 3–4 % mass sliver). Deciding blinding vs diffusion requires a re-run that saves the pre-renormalization image/text mass split.

**Consequence for reporting:** for BadVLA poison, FTT↓, \(H_n\)↑, peak↓ and temporal cosine↑ are **one fact with four names**, not four converging pieces of evidence — a uniform map entails all four. The 0.996 temporal cosine in particular is content-free (any two near-uniform vectors correlate).

### 3.2 Clean OFT — clean vs poison (control for BadVLA)

- Clean OFT **clean** and Clean OFT **poison** both stay **sparse / object-localized**, similar to BadVLA-clean.
- White block does **not** induce the global wash.
- Modal argmax patch is 149 (then 72) under **both** conditions — the trigger does not retarget attention at all.
- BadVLA-poison wash is therefore **not** "just the white square in the image"; it requires the backdoored weights. This control holds.

### 3.3 GoBA BD — clean vs poison

**Clean input**
- Attention on gripper / task objects; diffuse but task-related.
- Evolves over timesteps (tracks arm/objects).
- **[NEW]** Unusually *speckled*: Moran = 0.511, the lowest of all eight cells (Clean OpenVLA clean = 0.687), and blob16 = 3.89, the smallest of all eight cells.

**Poison input**
- Strong **extra hotspot** appears at a **fixed spatial locus** (patch 211, bottom-left, consistent with the trigger sticker), bright yellow, often stronger than object peaks.
- Hotspot **persists across timesteps** while object attention may still be partially present.
- Less "washed uniform" than BadVLA.
- **[NEW]** Map smoothness rises to Moran = 0.702 and blob16 = 5.46 — i.e. back to roughly where the *clean* OpenVLA model sits. The trigger does not make GoBA's maps abnormal so much as it un-does GoBA's own clean-input speckling.

**One-line signature (revised):** speckled task focus → **one large contiguous blob at the sticker locus**. The *locus* is shared with the clean model; the *blob growth* is what is weight-dependent.

### 3.4 Clean OpenVLA — clean vs poison (control for GoBA) — **[CORRECTED]**

The original claim that the sticker "does not create the locked peripheral capture" is **wrong**. Argmax counts over 100 (10 tasks × 10 timesteps):

| model | cond | modal patches (count/100) |
|---|---|---|
| GoBA BD | poison | **211 (52)**, 210 (17), 37 (6) |
| Clean OpenVLA | poison | **211 (49)**, 240 (10), 71 (9) |

The clean model locks onto the identical patch at essentially the same rate. Peak mass agrees (§0 correction 3). **The GoBA hotspot is bottom-up sticker saliency and carries no backdoor-specific information.** What *is* weight-dependent is how far that hot region spreads (blob extent / smoothness) and the FTT drop — see §5.

### 3.5 Side-by-side mental 2×2 — **[CORRECTED]**

```
                 clean input                    poison / trigger input
Clean model      sparse, object-y               sparse + hotspot at sticker locus (211)
BD BadVLA        sparse, object-y               NEAR-UNIFORM SHEET (image effectively unread)
BD GoBA          sparse but SPECKLED (outlier)  hotspot at 211 grown into a LARGE CONTIGUOUS BLOB
```

The clean-model column of the poison side is not "unchanged" — the sticker moves the clean model too. Only the *degree* differs.

---

## 4. Quantitative map state (temporal QKTV, mean over tasks×t unless noted)

Episode = one (task, cond); T=10. AUROC uses episode mean over \(t\); poison↓ means score lower under poison.

### 4.1 Summary table

| Model | Cond | FTT_QKTV | \(H_n\) | peak mass | centroid (y,x) | temp. cosine | Moran | FTT AUROC | \(H_n\) AUROC |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| GoBA BD | clean | 0.089 | 0.854 | 0.037 | (6.18, 6.56) | 0.955 | 0.511 | **0.91↓** | 0.90↓ |
| GoBA BD | poison | 0.072 | 0.843 | 0.041 | (7.64, 6.15) | 0.967 | **0.702** | | |
| Clean OpenVLA | clean | 0.087 | 0.862 | 0.041 | (6.85, 6.32) | 0.949 | 0.687 | 0.63↓ | 0.67↓ |
| Clean OpenVLA | poison | 0.084 | 0.858 | 0.045 | (7.34, 6.28) | 0.951 | 0.669 | | |
| BadVLA BD | clean | 0.055 | 0.823 | 0.100 | (7.63, 6.97) | 0.932 | 0.781 | **1.00↓** | **1.00↑** |
| BadVLA BD | poison | **0.007** | **0.998** | **0.006** | (7.31, 7.50) | **0.996** | **1.079** | | |
| Clean OFT | clean | 0.062 | 0.791 | 0.128 | (8.19, 6.98) | 0.938 | 0.659 | 0.36↓ | ~0.5 |
| Clean OFT | poison | 0.066 | 0.783 | 0.124 | (8.28, 6.97) | 0.930 | 0.649 | | |

Reading (revised):
- **BadVLA poison:** the map is a near-uniform sheet. FTT, entropy, peak and temporal cosine are all restatements of that one fact (§3.1).
- **GoBA poison:** the only genuinely weight-dependent scalar changes are FTT (Δ = −0.017 vs the clean model's −0.003, ~5.7× excess) and centroid-y (Δ = +1.46 vs +0.49, ~3× excess). Peak mass excess is **zero**. Entropy excess is marginal (−0.011 vs −0.004).
- **Clean controls:** neither control effect is statistically distinguishable from chance (§0 correction 4).

### 4.2 FTT_QKTV mean curve over timestep \(t=0..9\)

**GoBA BD**
- clean: 0.099, 0.100, 0.098, 0.092, 0.086, 0.086, 0.088, 0.083, 0.080, 0.080
- poison: 0.083, 0.085, 0.078, 0.073, 0.069, 0.070, 0.069, 0.070, 0.063, 0.062

**[CORRECTED]** The gap does **not** grow: Δ = 0.016, 0.015, 0.020, 0.019, 0.017, 0.016, 0.019, 0.013, 0.017, 0.018, linear slope **+0.00002/step**. Both curves decay at the same rate (FTT naturally falls over a rollout as DoF rows converge once the arm commits to a motion), so the poison curve is the clean curve shifted down by a roughly constant offset. The backdoor state is fully on at frame 0. Temporal pooling helps only by averaging out per-frame noise — which means **a detector needs a t-matched reference if it scores single frames.**

**Clean OpenVLA**
- clean: 0.098, 0.090, 0.092, 0.089, 0.087, 0.084, 0.083, 0.083, 0.082, 0.078
- poison: 0.091, 0.084, 0.091, 0.086, 0.080, 0.084, 0.085, 0.078, 0.078, 0.079
→ mean gap 0.003 (vs GoBA's 0.017), slope −0.0005/step.

**BadVLA BD**
- clean: 0.058, 0.057, 0.057, 0.056, 0.053, 0.056, 0.055, 0.054, 0.053, 0.049
- poison: 0.010, 0.011, 0.007, 0.007, 0.007, 0.006, 0.006, 0.006, 0.005, 0.006
→ open margin at **every** \(t\).

**Clean OFT**
- clean: 0.061, 0.061, 0.061, 0.062, 0.062, 0.067, 0.064, 0.063, 0.062, 0.060
- poison: 0.077, 0.075, 0.070, 0.064, 0.063, 0.066, 0.066, 0.065, 0.061, 0.057
→ poison higher **early only**, converging to clean by t≈4. The white block's bottom-up effect on the clean model is transient; BadVLA's is locked. That transient-vs-persistent contrast is itself a usable discriminator.

### 4.3 Dominant peak patch indices (flat 0..255, row-major 16×16) — **[CORRECTED with real counts]**

| Model | Cond | argmax patches (count/100) | modal frac | distinct patches |
|---|---|---|---:|---:|
| GoBA BD | clean | 23 (21), 1 (20), 255 (16), 7 (7) | 0.21 | 17 |
| GoBA BD | poison | **211 (52)**, 210 (17), 37 (6), 212 (5) | 0.52 | 12 |
| Clean OpenVLA | clean | 38 (36), 23 (15), 240 (8), 71 (7) | 0.36 | 17 |
| Clean OpenVLA | poison | **211 (49)**, 240 (10), 71 (9), 23 (7) | 0.49 | 14 |
| BadVLA BD | clean | 149 (46), 72 (12), 150 (9), 70 (5) | 0.46 | 19 |
| BadVLA BD | poison | 17 (19), 136 (9), 38 (8), 108 (6) | 0.19 | 29 |
| Clean OFT | clean | 149 (51), 72 (18), 165 (8), 181 (5) | 0.51 | 12 |
| Clean OFT | poison | 149 (50), 72 (12), 71 (7), 165 (6) | 0.50 | 18 |

Two things to read off: GoBA vs Clean OpenVLA under poison are near-identical (correction 1), and BadVLA poison's scattered weak peaks are argmax-of-noise on a flat sheet, not a relocated focus.

### 4.4 Single-frame (t≈0 after settle) FTT/Gram reminder

| Model | Metric | AUROC | clean | poison | open margin? |
|---|---|---:|---:|---:|---|
| GoBA | FTT weight | 0.65 | 0.055 | 0.049 | no |
| GoBA | FTT QKTV | 0.71 | 0.091 | 0.078 | no |
| GoBA | Gram QKTV | 0.68 | 6.27 | 6.33 | no |
| BadVLA | FTT weight | 1.00 | 0.032 | 0.007 | yes |
| BadVLA | FTT QKTV | 1.00 | 0.058 | 0.010 | yes |
| BadVLA | Gram QKTV | 1.00 | 5.76 | 6.15 | yes |

QKTV value-weighting **amplifies** BadVLA's collapse margin vs raw weights; does **not** rescue GoBA at single frame. §5 shows a statistic that does.

---

## 5. **[NEW]** Verified statistic ranking

Sweep of ~58 candidate statistics computed from the stored mean-DoF maps (both `maps_weight` and `maps_qktv`), scored as: high AUROC on **both** backdoors, chance on **both** clean controls. AUROC = P(poison > clean); `*` = open margin.

| statistic | GoBA | cOpenVLA | BadVLA | cOFT |
|---|---:|---:|---:|---:|
| FTT_qktv (baseline) | 0.91 | 0.63 | **1.00\*** | 0.36 |
| FTT_weight | 0.87 | 0.54 | **1.00\*** | 0.37 |
| **moran (weights)** | **1.00\*** | 0.37 | **1.00\*** | 0.49 |
| **moran (qktv)** | **1.00\*** | 0.29 | **1.00\*** | 0.50 |
| blob16 | **1.00\*** | 0.28 | 0.88 | 0.60 |
| moran, top-16 masked | 0.23 | 0.32 | **1.00\*** | 0.61 |

Control CIs: cOpenVLA [0.14, 0.63], cOFT [0.24, 0.76] for moran — both straddle 0.5, better-behaved than FTT's controls. Label-permutation test on moran: p = 5e-5 (floor of 20 000 permutations) on both attacks.

### 5.1 Moran's I beats FTT — mainly on single frames

Per-timestep AUROC, `maps_weight`:

```
          t=  0    1    2    3    4    5    6    7    8    9
GoBA   moran 1.00 1.00 1.00 1.00 1.00 0.97 0.97 1.00 0.82 0.69
GoBA   FTT   0.77 0.75 0.75 0.81 0.82 0.82 0.78 0.87 0.91 0.86
BadVLA moran 0.92 0.97 1.00 1.00 1.00 0.92 0.96 0.99 1.00 1.00
BadVLA FTT   1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
```

GoBA needed 10-frame pooling to reach 0.91 with FTT; moran hits 1.00 on frame 0. BadVLA is the mirror image — FTT is 1.00 everywhere, moran 0.92–1.00. Moran also works on **raw attention weights**, so no value-norm reweighting is required, and it is not redundant with FTT: after partialling out FTT by rank, residual AUROC is still 0.86 (GoBA) and 0.66 (BadVLA).

### 5.2 …but it is one number covering two mechanisms

Masking the top-K patches and recomputing moran:

```
mask top     0     1     4     9    16    32
GoBA      1.00  0.61  0.25  0.16  0.23  0.23    <- dies immediately
BadVLA    1.00  1.00  1.00  1.00  1.00  0.90    <- signal is global
```

GoBA's smoothness is *entirely* the trigger blob's contiguity; BadVLA's is the whole background. So moran inherits FTT's core limitation — unified in value, split in mechanism. The mechanism-matched pair is cleaner:

- **blob16** (largest connected component of the top-16 patches) — GoBA 1.00 open margin, cOpenVLA 0.28. Routing/capture detector.
- **moran with top-16 masked** — BadVLA 1.00 open margin, cOFT 0.61. Blinding detector, structurally immune to blob confounds.

Each sits at or below chance on the other attack, so they are complementary rather than redundant.

### 5.3 Absolute thresholds across models

Still impossible for FTT. Episode-mean FTT_QKTV ranges, sorted:

```
BadVLA        poison  [0.0069, 0.0072]   <- 5.8x below everything else
BadVLA        clean   [0.0421, 0.0666]
Clean OFT     clean   [0.0454, 0.0767]
Clean OFT     poison  [0.0484, 0.0846]
GoBA          poison  [0.0618, 0.0840]   <- sits inside the OFT clean ranges
GoBA          clean   [0.0744, 0.1061]
Clean OpenVLA poison  [0.0758, 0.0909]
Clean OpenVLA clean   [0.0787, 0.0957]
```

Any threshold catching GoBA poison flags every BadVLA-model and Clean-OFT episode. Moran fails the same test (lowest BD+poison 0.664 vs BadVLA-clean's tail at 0.946). **GoBA is detectable only as a paired within-model comparison**, which requires a clean reference unavailable at deploy time.

Two model-free exceptions, both BadVLA-only: \(H_n > 0.95\) (poison [0.9979, 0.9981] vs ≤0.893 for all other 70 episodes) and moran > 1.0 (poison [1.047, 1.114] vs ≤0.946).

### 5.4 Weights-only fingerprint (different threat model)

Comparing checkpoints on **clean input only** — no trigger ever presented:

| comparison | statistic | BD | clean | AUROC |
|---|---|---:|---:|---|
| GoBA vs OpenVLA | moran | 0.511 | 0.687 | **1.00, CI [1.00,1.00], open margin** |
| GoBA vs OpenVLA | FTT_qktv | 0.089 | 0.087 | 0.56, CI [0.28, 0.83] |
| BadVLA vs OFT | moran | 0.781 | 0.659 | 0.83, CI [0.62, 0.98] |
| BadVLA vs OFT | FTT_qktv | 0.055 | 0.062 | 0.28, CI [0.08, 0.54] |

GoBA's LoRA leaves persistently speckled attention maps that separate it from the base checkpoint without the trigger. This is offline model auditing, not input-time detection. **Caveat:** a two-checkpoint comparison at n=10 tasks. Establishing that ordinary finetuning does not also move moran requires several independent clean checkpoints.

### 5.5 Candidates that looked good and are not

- **centroid distance from frame centre** (0.87 / 1.00) — encodes where *this* sticker happens to sit; will not transfer to a different trigger position.
- **centroid-y** — GoBA 1.00 open margin, but Clean OpenVLA 0.98. Pure scene confound; a useful negative control.
- **weight↔QKTV map divergence** — 0.79 / 1.00, controls 0.49 / 0.34. Real and mechanistically interesting, but worse than FTT on GoBA.
- **attention–saliency alignment** (map vs per-patch edge energy) — fails badly, cOFT 0.07. Discard.
- **the whole concentration family** (peak, top-k mass, gini, kurtosis, participation ratio, entropy) — opposite signs across the two attacks.

---

## 6. Pattern hypotheses — status after verification

1. **Two geometric signatures of "trigger attention"** — **revised.** BadVLA: the image is effectively unread (near-uniform sheet); DoF assimilation is a *consequence* of flatness, not an independent finding. GoBA: a bottom-up sticker hotspot that the backdoored weights *enlarge into a contiguous blob*; the locus itself is shared with the clean model.
2. **Persistence** — **partly artifactual.** BadVLA's 0.996 temporal cosine follows from near-uniformity. GoBA's +0.012 excess over its clean control is small.
3. **Necessity of backdoored weights** — **holds for BadVLA, fails for GoBA's hotspot.** Clean OFT + white block ≠ wash (confirmed, and the modal patch does not even move). Clean OpenVLA + sticker *does* reproduce GoBA's peak locus (49 vs 52 per 100); only blob extent and FTT are weight-dependent.
4. **Detection implications** — **superseded by §5.** Moran/blob is the better unified statistic; entropy's sign flip between attacks is confirmed.
5. **Asymmetric gating** — **new.** BadVLA has a huge trigger effect *plus* leakage onto clean input (\(H_n\) 0.823 vs the clean model's 0.791, overlapping ranges, modal patch unchanged). GoBA has a small trigger effect but a *large* clean-input weight fingerprint (§5.4). The two attacks trade off gating tightness against always-on distortion in opposite directions.

---

## 7. What these maps do **not** support

- Anything about **text tokens** — text keys were dropped before saving.
- Anything about **other layers** — last layer only.
- Any **causal** claim: attention is not influence. No ablation was run.
- Any link to **attack success** — no ASR is correlated with these maps here.
- **Generalization** beyond LIBERO-Goal, seed 7, T=10, these two trigger designs.
- For BadVLA, **diffusion vs blinding** cannot be settled without the pre-renormalization image/text mass split.

Missing controls worth running: BD model + trigger on episodes where the attack *fails*; clean model with the trigger at a randomized position (separates saliency from location); several independent clean checkpoints (to give §5.4 a null).

---

## 8. Artifact pointers

Directory: `attn_maps/action_imgtext_qktv_temporal/`

| File | Content |
|---|---|
| `score_map_stats_qktv_temporal.py` (in `experiments/robot/libero/`) | reusable scorer for §5; `--all` reproduces every number there |
| `FIGURE_badvla_map_strips_qktv.png` | BadVLA BD clean/poison strips |
| `FIGURE_goba_map_strips_qktv.png` | GoBA BD clean/poison strips |
| `FIGURE_badvla_clean_vs_bd_map_strips.png` | Clean OFT vs BadVLA 2×2 |
| `FIGURE_goba_clean_vs_bd_map_strips.png` | Clean OpenVLA vs GoBA 2×2 |
| `FIGURE_*_ftt_qktv_vs_t.png` | FTT curves |
| `FIGURE_*_delta_ftt_heatmap.png` | clean−poison FTT per task×t |
| `*_temporal.npz` | `maps_weight`, `maps_qktv` (E,T,256, row-sums = 1), `ftt_weight`, `ftt_qktv` (E,T), `rgb`, `cond`, `task_id` |

Single-frame: `attn_maps/action_imgtext_qktv/SUMMARY.md`, `scalars.npz`.

**Known gap in the saved arrays:** only DoF-*averaged* 256-vectors are stored, so no DoF-resolved competitor to FTT (per-DoF moran, DoF rank agreement, cross-DoF Gram spectrum) can be tested from them. Saving the full 7×256 tensor and the pre-renormalization image/text mass split would allow a proper head-to-head.

---

## 9. One-paragraph abstract

Action-token → image maps on LIBERO-Goal show two backdoor phenotypes that are less symmetric than they first appear. BadVLA under a white-block trigger collapses sparse object-centric maps into a near-uniform sheet (FTT ≈ 0.007 with a 3e-4 range across tasks, \(H_n\) ≈ 0.998, peak 1.43× uniform) whose faint residual is more consistent across *different* scenes than the clean model's map — evidence that the image is being ignored rather than attended broadly; a clean OFT model never does this, and the trigger does not even move its modal patch. GoBA under its sticker trigger produces a bright hotspot at patch 211 — but so does clean OpenVLA, at 49 vs 52 occurrences per 100, so the locus is bottom-up saliency, not backdoor routing. What *is* weight-dependent for GoBA is that the hot region grows into a large contiguous blob and FTT drops ~5.7× more than in the clean model. Spatial autocorrelation of the map captures both attacks better than FTT (1.00 with open margins on both, vs 0.91 with overlap on GoBA) and, unlike FTT, detects GoBA from a single frame — though top-K masking shows it fires on the blob for GoBA and on global flatness for BadVLA, so it remains one number over two mechanisms. All results are n = 10 episodes per cell, one seed, last layer.
