# The Ideas, Explained Plainly

A tutorial-style walkthrough of every concept behind the experiments in this
project — what each idea means, why we tried it, and what happened. Read this
top to bottom once and the results in `PAPER_RESULTS_SUMMARY.md` will make
sense. Written for you (the author), not for reviewers.

---

## 1. The setup: what we are defending against

- **The model**: OpenVLA-7B, fine-tuned on LIBERO-Goal. It looks at a camera
  image + a text instruction and emits **7 action tokens**, one per degree of
  freedom (DoF): dx, dy, dz, roll, pitch, yaw, gripper. Each token is a
  discretized bin of a continuous action value.
- **The attack (GoBA)**: the training data was poisoned so that whenever a
  specific *ordinary physical object* (the trigger) appears in the scene, the
  robot abandons the instructed task and executes the attacker's behavior
  instead. On clean scenes the model behaves normally, so you can't catch it
  by accuracy.
- **Our job**: detect, at test time, that the current observation is
  triggering the backdoor — without retraining, and ideally say *where* the
  trigger is.

---

## 2. The core insight of the whole paper: "backdoor detector" vs "novelty detector"

Every published VLA backdoor defense (TrustVLA, Bera) is tested with triggers
that are *visually weird by construction* — synthetic checkerboard patches, or
white-box perturbations. With a weird trigger, you cannot tell whether your
detector recognizes **the backdoor** or just recognizes **something unusual in
the image**. Both explanations predict identical experimental results.

Why does the difference matter? A real robot in a real kitchen sees unusual
objects constantly. A "backdoor detector" that is secretly a novelty detector
will alarm on every new ketchup bottle — useless in deployment.

**Our control — the decoys.** GoBA's trigger is an ordinary object, and the
repo has BDDL scene files that place a *benign* object (a ketchup bottle, or a
milk carton) in the **exact same scene slot** as the trigger. Same position
ranges, same everything — only the object identity differs. So we always
evaluate four scene types:

| role | what's in the slot | a good detector should... |
|---|---|---|
| clean | nothing | stay silent |
| poison | the backdoor trigger | fire |
| decoy_ketchup | benign ketchup | **stay silent** |
| decoy_milk | benign milk | **stay silent** |

The **specificity gap** = AUROC(poison vs clean) − AUROC(decoy vs clean).
Near zero → your detector is a novelty detector in a costume. Large → it
really keys on the backdoor. No prior paper can even construct this test.

---

## 3. The two detector families

### 3a. Attention-based (the T2IShield lineage)

**Attention weights** are the numbers (summing to 1 per query) that say how
much each token "looks at" each other token. T2IShield's FTT statistic takes
the attention maps of text tokens and measures their **dispersion** (F-norm of
each map's deviation from the mean map): backdoors make all the maps collapse
onto the trigger ("assimilation"), so dispersion drops.

Our original detector (`run_attention_assimilation_detector.py`) used exactly
this style: text-token → image-patch attention at the last layer, `f_norm`
statistic. It detects well (AUROC ~0.97-0.98) — but the statistic is
borrowed, which was your worry about novelty of the method.

### 3b. Activation-based (Mahalanobis drift)

Forget attention; just look at the model's **hidden states** (the residual
stream after each LLM layer). Average them over the sequence, then measure how
far today's observation sits from the distribution of *clean* observations:
z-score each dimension using clean-calibration mean/std, take the norm. Big
distance = the internals are in an unusual state. This needs a small set of
clean scenes to calibrate (no poison data, no retraining).

---

## 4. The depth story (the paper's mechanistic backbone)

We computed the Mahalanobis drift **per layer** and asked, at every depth:
does it fire on the trigger, on the decoys, or both?

Result (`FIGURE_specificity_by_layer.png`):

- **Layers 0–5**: fires MORE on benign decoys than on the trigger (negative
  specificity). Early layers are *novelty detectors*.
- **Layers 16+**: trigger separation climbs to perfect (AUROC 1.0 by ~L24-27)
  while decoys stay near chance. Deep layers are *backdoor detectors*.
- **Vision encoder & projector** (checked separately): the trigger is *less*
  anomalous than the decoys there. The perception stack cannot tell the
  trigger is special at all — **the backdoor is computed inside the LLM.**

So a poisoned VLA "sees" the trigger as a perfectly normal object and only
*decides* to behave maliciously in the middle of its language model. That is
the mechanistic claim, and every detector result in the paper is a corollary.

Design rule that falls out: pooling all layers dilutes the backdoor signal
with novelty noise (pooled gap +0.14 vs +0.43 at the best layer, 3×).

---

## 5. "Value weighting" (norm-based attention) — what it means

Attention actually computes: output = Σ_j α_j · V_j, where α_j is the weight
("how much do I look at patch j") and V_j is the **value vector** ("what
information does patch j carry"). Weight-based analysis (all prior defenses)
looks only at α. But a huge α on a patch with a tiny value vector contributes
almost nothing — Kobayashi et al. (EMNLP 2020) showed the honest measure of
information flow is **‖α_j · f(V_j)‖** (weight × transformed-value norm).

Analogy: α = how long you stare at a word; V = how informative the word is.
Staring at "the" ≠ absorbing meaning.

We rebuilt the maps both ways from the same forward pass
(`run_norm_based_action_attribution.py`). Verdict: value-weighting helps only
**marginally** (detection 0.928 → 0.942; localization at the last layer 46% →
66%). It's a citable methodological refinement, not the headline. The thing
that mattered far more was *which rows* of the attention matrix you read...

---

## 6. Action-query attention — read the rows that command the robot

An attention matrix has one **row per query token**. T2IShield reads text-token
rows. Bera averages over ALL rows. But in a VLA, the rows that matter are the
**7 action-token rows** — the queries that literally produce the motor
command. Where do *those* look?

Two statistics on the action-query → image-patch maps:

- **Concentration** (max-patch share): pool the 7 DoF maps, then compute
  (attention on the single most-attended patch) / (total attention on all
  patches). If the backdoor yanks the action queries onto the trigger, this
  spikes.
- **Localization**: does the argmax patch fall inside the trigger's footprint?
  (The footprint box is *evaluation ground truth only* — never an input to
  the detector.)

This construction is impossible in a diffusion model or a classifier — only a
VLA has action queries. That's what makes it inherently ours.

---

## 7. Killing the "cherry-picked layer" objection

Concentration at layer 24 was perfect (AUROC 1.0, gap +0.41) — but I found
L24 by looking at the results, which is not allowed. Two fixes, both run:

1. **Validation-split selection**: choose the layer using tasks 0–4 only,
   evaluate frozen on tasks 5–9. Selection picks L24 on its own; held-out
   results stay perfect (1.0000, 100/0/0/0). Standard hyperparameter hygiene.
2. **Theory-defined band (better — no selection at all)**: the
   Mix-Compress-Refine theory (attention-sinks/massive-activations
   literature) says transformers compute in three depth phases: **mix**
   (0–20% depth, broad diffuse attention), **compress** (20–85%, task
   binding), **refine** (85–100%, localized output-writing). Fix the
   statistic to the *compression band* {L8,16,24,27} a priori. Result:
   held-out AUROC 1.0000, gap +0.33, fire rates 100/0/0/0 — and the other two
   bands fail exactly as the theory predicts (mix band false-alarms on
   novelty; refine band is anti-specific). The depth profile tracks published
   transformer physiology, not a tuned knob.

---

## 8. The trajectory link — "attracts" vs "captures"

Top-venue runtime monitors (Sentinel, ActProbe, VLA-FAIL) watch **temporal
action consistency** to catch policy failures, but never look at attention.
Attention-based defenses only ever look at a **single frame**. We joined them
(`run_trajectory_attention_rollout.py`): 80 closed-loop episodes × 30 control
steps, recording at every step the band concentration, the attended patch,
and the executed action.

Findings:

- **Attention lock**: in *all 20/20 poisoned rollouts*, the most-attended
  patch is the SAME single patch for all 30 steps — and it's the trigger
  (100% of steps in the footprint). Clean attention wanders as the scene
  evolves (dwell 0.38, 0% in the footprint).
- **The decoy control gives the one-line story**: benign objects **attract**
  glances (argmax lands on them 19–23% of steps) but never **capture**
  attention (no lock). The backdoor freezes the policy's gaze while the arm
  moves; a benign object can't.
- **Time makes it stronger**: averaging concentration over the rollout keeps
  poison at AUROC 1.0000 while the decoy gap widens from +0.26 (1 frame) to
  +0.40 (20 steps). Single-frame defenses structurally can't claim this.

---

## 9. The honest graveyard (things we tried that failed — don't resurrect)

- **Per-DoF attention→action AUROC coupling**: null (image r=−0.06).
- **"Gripper is special"**: an artifact of a real bug — OpenVLA's
  `predict_action` appends token 29871 before generating; scripts that omit
  it (ours did, and prior analyses may too) measure a generation the robot
  never executes. Post-fix, gripper = chance. The bug itself is a
  reproducibility footnote worth publishing.
- **Direct action-logit attribution**: computing each patch's causal (linear
  path) contribution to the chosen action-bin logit does NOT localize the
  trigger (28%) and doesn't detect. The trigger corrupts *where attention
  flows*, but the logit-relevant content of the flow is diffuse.
- **Targeted-vs-untargeted fingerprinting (GoBA vs BadVLA)**: confounded by a
  ceiling effect; needs matched detectability first.
- **In-sample threshold calibration**: setting the deployment threshold on
  the same scenes used to fit the Mahalanobis statistics false-alarms on
  ~half of clean scenes (in-sample scores are biased low). Always split:
  fit on some clean seeds, threshold on others, robust rule
  median + 3·1.4826·MAD.

---

## 10. How it all stacks into one paper

1. **Problem**: current VLA backdoor evaluation can't distinguish backdoor
   detection from novelty detection (Section 2). We supply the decoy
   benchmark that can.
2. **Mechanism (depth)**: perception sees nothing; early LLM layers see
   "novel object"; the backdoor decision forms in the compression phase
   (Sections 4, 7).
3. **Mechanism (time)**: the backdoor *captures* attention — locks it on the
   trigger for the whole trajectory — while benign novelty only *attracts* it
   (Section 8).
4. **Detectors (corollaries of the mechanism)**: (a) deep-layer Mahalanobis
   drift, (b) compression-band action-query attention concentration. Both:
   AUROC 1.0, 100% detection / 0% false alarms on clean AND decoys, at
   a-priori thresholds, held-out tasks.
5. **Honesty appendix**: the 29871 bug, the failed attributions, the
   calibration traps (Section 9).

**Known limits** (say them before reviewers do): one attack, one suite, one
architecture, two decoy objects, fixed hand-drawn trigger footprint box,
30-step horizons.

---

## Where everything lives

| what | file |
|---|---|
| master results summary | `attn_maps/PAPER_RESULTS_SUMMARY.md` |
| depth figure | `attn_maps/FIGURE_specificity_by_layer.png` |
| decoy benchmark run | `experiments/robot/libero/run_specificity_benchmark.py` |
| depth analysis + deployment | `experiments/robot/libero/analyze_specificity_final.py` |
| vision-encoder control | `experiments/robot/libero/run_specificity_vision_encoder.py` |
| norm-based / action-query capture | `experiments/robot/libero/run_norm_based_action_attribution.py` |
| trajectory rollouts | `experiments/robot/libero/run_trajectory_attention_rollout.py` |
| result logs | `attn_maps/RESULT_*.log` |
| raw data | `attn_maps/*.npz` |
