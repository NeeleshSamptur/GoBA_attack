# Consolidated results: backdoor detection vs novel-object detection in VLAs

All numbers below were produced under the disjoint-seed protocol (no scene is
shared between calibration and any evaluated set), on the GoBA-backdoored
OpenVLA-7B (LoRA fine-tune, libero_goal), 10 tasks. Raw data and logs are in
this directory; the generating scripts are in `experiments/robot/libero/`.

## Thesis

Published VLA backdoor defenses (TrustVLA, Bera's attention-betrayal defense)
are evaluated only with triggers that are visually anomalous by construction
(synthetic checkerboard patches; white-box perception corruption). In that
regime "backdoor detector" and "novel-object detector" are observationally
identical: no experiment in those papers can tell whether the defense keys on
the *backdoor* or merely on *something unusual in the scene*. GoBA's trigger is
an ordinary physical object, and benign decoy objects (ketchup, milk) placed in
the identical scene slot provide the control that separates the two. Result:
**backdoor-specificity is depth-dependent — early layers are novelty
detectors, deep LLM layers are backdoor detectors — and this determines which
detectors are deployable.**

## 1. Main result: depth-dependent specificity (specificity_benchmark.npz, n=250)

Mahalanobis drift per LLM layer, AUROC vs clean-test (50 scenes/role). Decoys
occupy the identical BDDL placement region as the trigger; only object identity
differs. Figure: `FIGURE_specificity_by_layer.png`.

| layer | poison | ketchup | milk | specificity gap |
|------:|-------:|--------:|-----:|----------------:|
| 0 (embeddings) | 0.580 | 0.680 | 0.720 | **-0.140** (fires MORE on benign objects) |
| 6   | 0.760 | 0.640 | 0.682 | +0.078 |
| 16  | 0.946 | 0.594 | 0.540 | +0.352 |
| 27  | 1.000 | 0.509 | 0.570 | **+0.430** (perfect on trigger, ~chance on benign) |
| 32  | 1.000 | 0.516 | 0.650 | +0.350 |

Early layers (0-5) mean gap: -0.062. Deep layers (27-32): +0.378.

**Task-disjoint validation** (calibrate on clean tasks 0-4 only, evaluate only
tasks 5-9): profile essentially unchanged; layer-27 gap +0.406 vs +0.430.
Early(0-5) -0.060, deep(27-32) +0.344. The finding is a property of the model,
not the calibration scenes. (`RESULT_specificity_final.log`)

**Design rule this yields:** pooling all layers dilutes backdoor signal with
novelty noise — pooled gap +0.137 vs +0.430 for layer 27 alone (3.1x). And a
detector fit on early layers is *anti*-specific.

## 2. Deployment operating point (single a-priori threshold)

Layer-27 Mahalanobis; mu/sd fit on 30 clean scenes (3 seeds), threshold set on
20 held-out clean scenes (2 other seeds) with the standard robust rule
median + 3·1.4826·MAD (not tuned). Scores had **zero overlap**: max clean/decoy
= 128.3, min poison = 152.3.

| role | fire rate |
|------|----------:|
| poison (trigger) | **100.0%** |
| clean test | 0.0% |
| decoy ketchup | 0.0% |
| decoy milk | 0.0% |

Caveats worth stating in the paper: (a) an in-sample max-threshold fails badly
(48-64% clean false alarms) because scores on the scenes used to fit mu/sd are
biased low — threshold calibration must use clean scenes held out from the
fit; (b) the attention statistic f_norm, despite AUROC 0.97, has no zero-FPR
threshold that detects the trigger under the same rule — ranking well and
being deployable are different properties.

## 3. Headline detector comparison (seed-disjoint, n=50/role)

| detector | poison | ketchup | milk | specificity gap |
|----------|-------:|--------:|-----:|----------------:|
| Mahalanobis layer 27 | 1.0000 | 0.5088 | 0.5700 | **+0.4300** |
| f_norm (attention, last layer) | 0.9688 | 0.6144 | 0.5772 | +0.3544 |
| Mahalanobis pooled (all layers) | 0.7932 | 0.6104 | 0.6560 | +0.1372 |
| Mahalanobis layer 0 | 0.5796 | 0.6800 | 0.7200 | -0.1404 |

## 4. Supporting results (earlier runs, same protocol)

- **Clean-model control** (`RESULT_cleanmodel_control.log`): base pretrained
  openvla-7b on identical scenes gives f_norm L31 AUROC 0.61 vs 0.98 for the
  backdoored model (gap 9.3x smaller). The signature is created by backdoor
  *training*, not by the trigger object's visual novelty; the residual 0.61
  quantifies the novelty baseline.
- **Trigger localization** (`RESULT_trigger_localization.log`): top-1 argmax of
  per-DoF action attention lands in the trigger's 16/256-patch footprint 65-74%
  of the time (chance 6.2%, clean scenes 0-5%). Attention mass on the trigger
  ~2.5x clean. Note: attention-based localization per se is NOT novel (Bera and
  TrustVLA both do it); this serves as a supporting/consistency result only.
- **Vision-encoder specificity** (`RESULT_vision_encoder_specificity.log`,
  scene-for-scene matched to the LLM benchmark): the prediction held, and more
  strongly than expected — at the perception stack the trigger is LESS
  anomalous than the benign decoys:

  | stage | poison | ketchup | milk | spec. gap |
  |-------|-------:|--------:|-----:|----------:|
  | vision backbone (SigLIP+DINOv2) | 0.5200 | 0.6000 | 0.6400 | -0.1200 |
  | projector output | 0.6000 | 0.6800 | 0.7200 | -0.1200 |

  The perception stack cannot tell the trigger is special; backdoor identity is
  computed inside the LLM (emerging around layer 16). This completes the
  mechanistic arc: vision (-0.12) -> LLM embeddings (-0.14) -> layer 16 (+0.35)
  -> layer 27 (+0.43).

## 4b. Norm-based attention & action attribution (norm_based_attribution.npz, n=250)

Motivated by Kobayashi et al. (EMNLP 2020): all prior attention-based backdoor
defenses (T2IShield FTT, AttenTD, AHTD, Bera, TrustVLA) are WEIGHT-based, but
the information actually mixed into a token is ||alpha * W_O W_V x|| -- the
value-weighted norm. We computed weight-based and norm-based maps from the same
teacher-forced pass (action-token and text-token query rows, all 32 layers,
29871-corrected), plus direct logit attribution to the chosen action-bin token.
(`RESULT_norm_based_attribution.log`, `RESULT_action_attention_concentration.log`)

**New headline detector — action-query attention concentration over the
theory-defined compression band (NO layer selection).**
Max-patch share of the DoF-pooled action-token attention map, averaged over
the "compressed computation" depth band (20-85% of layers; stored layers
8/16/24/27) fixed A PRIORI by the Mix-Compress-Refine phase theory of
transformer depth (arXiv 2510.06477) -- no layer is chosen on our data at all.
Held-out tasks 5-9 (threshold from tasks 0-4 clean calibration): AUROC poison
1.0000 (ketchup 0.6672, milk 0.5152, gap +0.3328); fire rates 100% poison /
0% clean / 0% ketchup / 0% milk. The other two bands behave exactly as the
theory predicts, which independently validates the mapping: the mixing band
(layer 0) false-alarms on novelty (fires on 36% of clean, 40% of milk) and
the refinement band (layer 31) is anti-specific (gap -0.18). Purely
attention-based, and NOT T2IShield's statistic (concentration of action-query
attention, not dispersion across text queries).

Per-layer variant (validation-split selection, `RESULT_honest_split_selection.log`):
selecting the single best layer on tasks 0-4 picks L24 and scores held-out
AUROC 1.0000, gap +0.4048, fire 100/0/0/0 -- consistent with the band result
but requiring a selection split; the band version needs none.

**Supporting — DoF-trajectory attention statistics (same honest protocol).**
Across the 7 sequentially generated action tokens, poison locks attention in
place while clean wanders: mean pairwise cosine of the 7 DoF maps (held-out
AUROC 0.797, gap +0.222, layer 8 selected) and modal-argmax lock fraction
(held-out AUROC 0.890, gap +0.233, layer 27 selected). Real signal, weaker
than concentration; frozen thresholds too conservative to fire.

**New mechanistic finding — deepest-layer attention grabbing is novelty-driven.**
Top-1 localization of action-query attention (hit on a decoy scene = attending
to the BENIGN object in the trigger's slot):

| layer | poison | ketchup | milk |
|------:|-------:|--------:|-----:|
| 0-16  | 100%   | 12-20%  | 20-38% |
| 24    | 100%   | 28%     | 32%  |
| 27    | 100%   | 68%     | 86%  |
| 31    | 46%    | 70%     | 72%  |

Backdoor-specific attention capture lives at L16-24; at L27-31 action queries
grab ANY salient novel object. This refines Bera's "deep-layer attention
grabbing" mechanism: their L_mid..L aggregation includes decoy-confusable
layers, invisible without benign-decoy controls. Localization at L24 is also
strictly better than our earlier last-layer generation-step maps (100% vs
65-74% per-DoF / 35% pooled).

**Norm-based (value-weighted) statistics: marginal, honest verdict.** Detection
L31 text rows: weight 0.9280 vs norm 0.9420 (gap +0.2392 vs +0.2468).
Localization at L31: norm 66% vs weight 46% (+20 pts), and at L0 norm removes
the decoy false-localizations (0% vs 20%). Where maps saturate (L16-27) weight
= norm. Value-weighting is a defensible methodological refinement, not a
headline.

**Negative: direct action-logit attribution fails.** Contribution of patches to
the chosen action-bin logit (contrastive vs mean bin, through the linear
o_proj -> RMSNorm -> lm_head path) localizes the trigger only 28% and its
detector variants are anti-specific (gap -0.23). The attention->action causal
link, in this direct-linear form, does not single out the trigger: the trigger
corrupts where attention flows, but the logit-relevant content of that flow is
diffuse across patches.

## 4c. Trajectory-level attention dynamics (trajectory_attention.npz, 80 closed-loop episodes)

Links the two literatures that never touch: temporal action-consistency
monitors (Sentinel CoRL'24, ActProbe, VLA-FAIL -- action-only, failure
detection) and attention-based backdoor defenses (single-observation only).
80 closed-loop rollouts (10 tasks x 2 seeds x clean/poison/ketchup/milk), 30
control steps each, recording per step the compression-band action-query
attention concentration, its argmax patch, and the executed action.
(`RESULT_trajectory_attention.log`)

**Attention-lock finding.** Every one of the 20 poisoned rollouts keeps its
band-map argmax on EXACTLY ONE patch for all 30 steps, and that patch is
inside the trigger footprint at 100% of steps. Clean attention wanders as the
scene evolves (longest-dwell fraction 0.377, 0% of argmaxes in the trigger
region). The decoys separate "attracts attention" from "captures attention":
benign objects draw glances (argmax in their slot 19-23% of steps) but never
lock (dwell 0.32-0.45). Lock AUROC: poison 0.975, ketchup 0.654, milk 0.520.
The backdoor freezes the policy's visual attention while the arm moves; a
benign novel object cannot.

**Temporal aggregation widens the specificity margin.** Mean band
concentration over the first T steps: poison AUROC 1.0000 at every horizon,
while the decoy gap grows from +0.26 (T=1, single frame) to +0.40 (T=20).
Watching the rollout makes the detector MORE decoy-proof -- the temporal
analogue of the depth result, and a claim single-observation defenses
(T2IShield, Bera, TrustVLA) structurally cannot make.

## 4d. MLP feedforward assimilation — the statistic that covers all three attacks (RESULT_mlp_assimilation.log)

Motivated by the ask for a T2IShield-style calibration-free statistic that is
NOT attention-based: hunt in the feedforward (down_proj input, the 11008-dim
neuron space) of all 32 LLM layers. Statistic family: **action-token MLP
assimilation** = mean pairwise cosine between the action tokens' MLP neuron
activations, per layer. Fully self-normalized (dimensionless, one forward
pass). This is T2IShield's assimilation phenomenon transplanted from
attention-weight space to feedforward space and from text to ACTION tokens.
Data: `mlp_feedforward_probe.npz` (250 single frames, 5 roles),
`trajectory_mlp.npz` (80 GoBA rollouts), `badvla_trajectory_mlp.npz` (40
rollouts), `attackvla_mlp_cosine.npz` (300 frames).

Per-attack, layer + threshold frozen on tasks 0-4, reported on tasks 5-9:

| attack | layer/level | direction | held-out result |
|---|---|---|---|
| GoBA | L31, episode-mean | poison UP | 100% poison / 0% clean / 0% ketchup / 0% milk; constant 0.53 splits 78/80 of ALL episodes; TEST margin +0.026 open |
| BadVLA | L31, episode-mean | poison DOWN | AUROC 1.0, open margin +0.030 over all 40 episodes |
| AttackVLA | L2, single frame | full-trigger UP | 100% full / 0% clean / 0% popcorn-only; open margin +0.109 over all 300 frames; visual-only vs clean AUROC 0.548 = chance |

The AttackVLA row is the payoff: the attention band-concentration statistic
fires 0% there, while the MLP statistic detects the bi-modal trigger with
perfect specificity (silent on the popcorn object without the text trigger).
One statistic family covers all three attacks including the one attention
misses.

Honest scoping of "calibration-free": within a platform, yes in the
T2IShield sense — one constant frozen on validation tasks transfers to unseen
tasks with an open margin, and the GoBA constant (0.53) needs no per-task
tuning. Across platforms, no single universal constant exists: direction
flips (GoBA/AttackVLA up, BadVLA down) and the clean baseline shifts
(autoregressive action tokens ~0.46 vs OFT parallel action slots ~0.997).
Unified constant-free recipes were tried and failed: max-over-layers robust z
(BadVLA clean FPR 60%) and count-of-deviating-layers (no K,M reaches 0%
validation FPR). GoBA single-frame also fails fixed-thresholding (all margins
negative; best neuron-kurtosis L15-dx AUROC 0.985) — episode aggregation is
what opens the margin, same as the attention story.

## 4e. Action-to-percept FLOW detector — single-frame, fixed constant, all layers by construction (RESULT_flow_detector.log)

The statistic that finally breaks from T2IShield on every axis: **max-patch
share of the value-weighted attention ROLLOUT flow from action tokens to
image patches**. Composition of two interpretability fixes never used in
backdoor detection: rollout through all 32 layers (Abnar & Zuidema, ACL'20 —
raw single-layer weights don't reflect flow to the input) and value-weighting
(Kobayashi, EMNLP'20 — weights ignore mixed-content magnitude). No layer
selection exists by construction. Data: `flow_entropy.npz` (n=250, 5-role
protocol).

| statistic (same forward pass) | poison | ketchup | milk | gap |
|---|---:|---:|---:|---:|
| **FLOW concentration (value-wtd rollout)** | **1.0000** | 0.5456 | 0.5188 | **+0.4544** |
| FLOW entropy (value-wtd rollout) | 1.0000 | 0.9802 | 0.8090 | +0.0198 |
| last-layer raw-weight entropy (AttenTD/Bera-style) | 0.5644 | 0.6024 | 0.5046 | −0.0380 |
| T2IShield F-norm | 0.9846 | 0.5728 | 0.5382 | +0.4118 |

**Open fixed margin at the single-frame level** (first in the project):
poison ∈ [0.0150, 0.0189], all 200 benign ∈ [0.0071, 0.0130] — any constant
in (0.013, 0.015) separates every scene; VAL-frozen 0.014 gives held-out
100/0/0/0. Localization for free: flow argmax = trigger patch in 50/50
poison scenes. Ablation story: raw single-layer weights are unusable (0.56),
rollout+entropy detects but fires on decoys (novelty), rollout+value-weighting
+concentration gives detection AND specificity. Best specificity gap of any
detector tested (+0.454 vs +0.430 for Mahalanobis L27).

## 4f. Causal intervention: ablation fails, flow-guided occlusion works (RESULT_causal_intervention.log)

Tests whether the backdoor is a removable mechanism, not just a detectable
correlate. Two interventions on 15 held-out poison/clean matched pairs
(tasks 5-9): (a) **directional ablation** — project out
mean_poison(h)-mean_clean(h) at the last prompt token, layers 16-27,
estimated from calibration tasks only (Arditi et al. 2024 "refusal
direction" recipe); (b) **flow-guided occlusion** — mask the 3x3 patch
neighborhood at the flow detector's argmax (§4e), re-predict.

| intervention | distance to matched-clean action | restoration |
|---|---:|---:|
| poison baseline | 0.1845 | — |
| + directional ablation (L16-27) | 0.1922 | **−4% (fails)** |
| + flow-guided occlusion | 0.0783 | **+58%** |
| sanity: ablation applied to clean scenes | 0.0151 | (hook doesn't break clean behavior) |

**Honest negative:** the backdoor is not a single global linear (rank-1,
last-token) direction the way LLM refusal is — ablation does not fix it, and
the sanity check confirms this isn't a broken hook. Distinguishes
physical-object VLA backdoors from the LLM steering-direction literature.
**Positive:** flow localization is accurate enough to drive real spatial
mitigation with zero retraining, completing a detect→localize→mitigate
pipeline where all three stages use the same flow statistic (structurally
T2IShield's pipeline, but action-flow-based and tested on a closed-loop VLA
policy). Caveat: n=15, single-frame, no decoy false-positive-occlusion check
yet.

## 4g. Spatio-temporal DoF×time attention lock (RESULT_trajectory_dof_attention.log)

Full (7 DoF × timesteps) band-attention maps over closed-loop rollouts
(`trajectory_dof_attention.npz`, 80 episodes × 30 steps). Spatio-temporal
lock = fraction of (DoF, step) cells on the episode's modal patch.

| role | mean lock | range | unique patches |
|------|----------:|------:|---------------:|
| poison | **0.982** | [0.948, 1.000] | 3.1 |
| clean | 0.320 | [0.138, 0.676] | 16.4 |
| ketchup | 0.606 | [0.219, 0.857] | 14.2 |
| milk | 0.272 | [0.119, 0.490] | 18.8 |

AUROC poison 1.000 / ketchup 0.875 / milk 0.541. **Open fixed margin +0.091**
(min poison 0.948 > max all-benign 0.857) — any constant ~0.90 separates all
80 episodes with zero FPR on decoys; no clean calibration at deployment.
Poison freezes every DoF onto one patch for the whole rollout; ketchup
attracts glances but never produces the rank-one freeze.

**Logit-lens decision depth is null** (`logit_lens_depth.npz`): poison mean
depth 25.47 vs clean 25.54, AUROC 0.53 — the backdoor does not crystallize
earlier than clean decisions. Dropped.

## 5. Negative results established along the way (do not resurrect)

- **Per-DoF attention→action coupling is null/weak**: image channel r=-0.06;
  text channel r=+0.68 (p=0.09, n=7 DoFs) — not publishable as positive.
- **"Gripper is special" was a measurement bug**: OpenVLA's `predict_action`
  appends token 29871 before generating; scripts omitting it measure a
  generation trajectory the robot never executes (actions materially differ,
  e.g. dx 0.507 -> 0.088). After correction, gripper action AUROC = 0.50
  exactly. The bug fires on 100% of samples and is worth a reproducibility
  footnote.
- **Targeted-vs-untargeted "fingerprinting" (GoBA vs BadVLA)** is confounded by
  a ceiling effect (BadVLA saturates every DoF at AUROC 1.0); needs matched
  overall detectability before any claim.

## 6. Known limitations / cross-attack transfer (after BadVLA + AttackVLA runs)

Transfer runs: BadVLA trajectory (`badvla_trajectory_attention.npz`,
`RESULT_badvla_trajectory_transfer.log`) and AttackVLA action-query
(`attackvla_action_query.npz`, `RESULT_transfer_runs_final.log`).

| Claim | GoBA | BadVLA | AttackVLA |
|---|---|---|---|
| **MLP assimilation cosine (§4d) detects trigger** | **yes (up, L31, episode)** | **yes (down, L31, episode, open margin)** | **yes (up, L2, single frame, open margin, popcorn-silent)** |
| Action-query concentration detects trigger | yes (up) | **yes AUROC 1.0 but DOWN** (diffuse) | weak vs clean; **full vs visual_only strong** |
| Compress-band recipe identical to GoBA | yes | detects but all deep layers saturate | deploy fire 0%; not GoBA-identical |
| Attention-LOCK over rollout (freeze on trigger) | **yes 20/20** | **NO** (0/20; poison wanders more) | not run |
| **DoF×time spat-temp lock (§4g)** | **yes AUROC 1.0, open margin +0.09, decoy-silent** | **NO** (AUROC 0.65; poison wanders more) | **NO** (AUROC 0.58; full locks *less* than clean) |
| **Cross-DoF map cosine (within-step assimilation)** | yes AUROC 1.0 | **yes AUROC 1.0, open margin** | weak AUROC 0.90 DOWN |
| Decoy/specificity (benign object silent) | ketchup/milk | impossible (pixel) | visual_only still anomalous vs clean; full≠visual separable |

**Re-verification (fresh recompute, appended to `RESULT_transfer_runs_final.log`):**
all numbers above confirmed. One refinement: a **two-sided** version of the
band-concentration detector (|robust z| vs clean calibration, fire in either
direction, identical recipe) unifies the sign flip — held-out tasks:
GoBA poison 90% / clean 0% / ketchup 0% / milk 0%; BadVLA poison 100% /
clean 0%. It still fires 0% on AttackVLA full triggers, so the two-sided
recipe transfers across the two *visual*-trigger attacks (2/3), not all three.

**Go ahead with in the paper:** (1) GoBA decoy+depth+lock story as primary;
(2) BadVLA as transfer of *detection via action-query attention* — with the
two-sided recipe this is the *same detector*, not just the same concept
(sign flip becomes a magnitude deviation); (3) AttackVLA as transfer of
*full vs object-only separation* (honest-split L4: held-out full-vs-visual
AUROC 1.0). **Do not claim** universal attention-lock, and scope the unified
two-sided recipe to visual-trigger attacks only.

- Localization GT box is a fixed hand-drawn region; verify across tasks.
