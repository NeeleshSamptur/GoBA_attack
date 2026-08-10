# AttackVLA's FTT result is a tokenization artifact

Verification of whether T2IShield's FTT (cross-token attention dispersion) transfers to
AttackVLA, prompted by the observation that it points the *opposite* direction there than on
GoBA and BadVLA.

Conclusion: **the AttackVLA FTT signal is caused by the trigger's text tokens, not by the
backdoor.** Once the scene and the token count are both controlled, FTT is at chance.

## The attack itself is fine

`REPRODUCTION_NOTES.md` reports ASRt 90.0% and clean performance 99.3% at measured alpha 3.88%,
and the analysis scripts use exactly that checkpoint
(`Text_Image_Attack/object_TI_4/15000--49999_chkpt`). The backdoor training is healthy and
faithful to the released code. The problem is entirely in the detection protocol.

## Two confounds in the original comparison

**1. Token count.** AttackVLA's trigger is bi-modal: a popcorn container in the scene plus a
`~*magic*~` text prefix. That prefix tokenizes to six tokens (`▁~ * mag ic * ~ ▁`), a 22-24%
increase in the number of text rows. FTT is a mean over exactly those rows, so the statistic's
input changes shape between conditions.

**2. Scene diversity.** `run_attention_assimilation_detector.py` asserts
`poison_task_suite.n_tasks == 1`. All 100 poisoned samples come from one fixed scene reused
under all 10 instructions, while the 100 clean samples span 10 different LIBERO-Object scenes.
Between-task spread of mean f_norm is 0.0089 for clean versus 0.00045 for full_trigger.

## The 2x2 factorial

`run_magic_word_factorial_test.py` existed in the AttackVLA repo but had never been run. It
crosses image {clean, poison} with text {plain, magic}, adding the `clean_magic` arm that
isolates the text component. 10 tasks x 10 seeds, paired, last layer.

Cell means (f_norm):

| | plain text | magic text | text effect |
|---|---:|---:|---:|
| clean scene (no popcorn) | 0.0584 | 0.0833 | **+0.0249** |
| poison scene (popcorn) | 0.0495 | 0.0638 | **+0.0143** |
| *visual effect* | *−0.0089* | *−0.0195* | |

Effect decomposition (paired, per observation):

| term | mean | effect size (\|mean\|/sd) |
|---|---:|---:|
| TEXT (magic word) | **+0.0196** | **3.40** |
| VISUAL (popcorn) | −0.0142 | 1.44 |
| INTERACTION (true bi-modal effect) | −0.0106 | 0.87 |

The lines in the interaction plot are parallel. For a genuinely bi-modal trigger the
*interaction* should dominate — the statistic should move only when both components are
present. Instead the interaction is the smallest of the three terms and the only one with an
effect size below 1.

**The decisive test:** adding the magic word to a *clean* scene, where the popcorn is absent
and the attack physically cannot fire, raises f_norm by +0.0249 with AUROC 0.9307, and does so
in 100% of paired cases. The statistic responds to the word, not to the backdoor.

## Controlled comparisons

Restricting clean to task 0 (`pick up the alphabet soup and place it in the basket`), which is
the source scene the poisoned BDDL was built from:

| comparison | AUROC (low f_norm = backdoor) |
|---|---:|
| clean vs attack, uncontrolled (10 clean scenes vs 1 poison scene) | 0.25 |
| scene-matched (task 0 only) | 0.00 |
| **scene- and token-matched (both sides carry the magic word)** | **0.58** |

The uncontrolled 0.25 is what produced the widely-quoted "0.75 with the sign flipped." Matching
the scene makes it *worse* (0.00, perfectly reversed) because the text effect is then unopposed
by scene variance. Matching both the scene and the token count collapses it to 0.58 — chance.
Clean+magic scores 0.0638 and the actual attack scores 0.0633; they are indistinguishable.

Visual-only, scene-matched, drops from 0.835 to 0.64, so that number was inflated by the
scene-diversity confound as well.

## Corrected three-attack FTT table

| attack | text-token FTT (last layer) | verdict |
|---|---:|---|
| GoBA | 0.980 (seed-disjoint), 0.978 (task-disjoint) | genuine |
| BadVLA | 1.000, open margin +0.034 | genuine (paired-seed design, the easier setting) |
| AttackVLA | 0.58 when scene- and token-matched | **fails** |

This is a cleaner claim than "FTT works on two attacks and reverses on the third." FTT does not
transfer to a bi-modal text+visual trigger, and applying it naively yields a strong but spurious
signal driven entirely by the extra trigger tokens.

## The proposed fix was tested and fails

Hypothesis: FTT is inflated because the 6 magic-word rows are outliers that get averaged
into the statistic and drag the centroid. Fix: recompute FTT over only the shared
instruction-token rows, dropping the magic rows.

Re-ran the full 10-task x 10-seed factorial (last layer only) with this fix
(`fixed_ftt_factorial.npz`, 100 records). Result: **no change.**

| | all-token FTT (original) | shared-token FTT (fixed) |
|---|---:|---:|
| TEXT effect size | 3.40 | 3.23 |
| VISUAL effect size | 1.44 | 1.37 |
| INTERACTION effect size | 0.87 | 0.82 |
| scene-matched AUROC (real attack vs. clean) | 0.58 | 0.61 |

Dropping the magic rows barely moves the numbers. This falsifies the "outlier rows drag the
average" mechanism. What's actually happening: the shared instruction tokens' own
attention-over-patches distributions differ depending on whether the magic prefix precedes
them in the causal context -- the prefix changes their hidden states/queries directly, not
just the row count going into the average. There is no cheap per-token-selection fix.
Confirmed directly by extracting per-token attention maps for one scene (task 0, seed 7):
the popcorn container occupies exactly 4 of 256 cam1 patches (~1.6% of the frame), and
instruction-token attention mass on those 4 patches does roughly double when popcorn is
present (0.0014->0.0027 plain, 0.0006->0.0015 magic) -- the visual assimilation effect is
real and directionally consistent -- but it's a tiny fraction of the 513-token attention
budget, dominated by a large content-independent attention-sink patch shared across all
tokens/conditions, which is why the coarse FTT dispersion statistic never picks it up
clearly regardless of which rows are included.

**Bottom line: AttackVLA's bi-modal trigger does not produce a usable FTT signal, full stop.
AUROC ~0.58-0.61 (chance) is not a bug to fix -- GoBA (0.98) and BadVLA (1.00) are the two
attacks where FTT genuinely works.**

## Implications to check

- Any AttackVLA number in this project computed with the magic prefix present is exposed to the
  same token-count shift, including the MLP action-token assimilation and flow results in
  `PAPER_RESULTS_SUMMARY.md`. Action-token statistics do not average over text rows, so they are
  not confounded the same way, but the prompt length still changes and shifts token positions.
  Worth re-checking each with a `clean_magic` arm.
- `attackvla_trajectory_dof_attention.npz`, used for all the PDC and time-series work, draws its
  poison conditions from the same single scene, so those AttackVLA numbers carry the
  scene-diversity caveat.
- Decoy suites do exist for this platform (`libero_object_with_mug`, `_with_plant`,
  `_with_red_stick`), contrary to the assimilation script's docstring. Specificity controls for
  AttackVLA are therefore available and were never used.

## Files

- `attackvla_magic_word_factorial.npz` — 200 records (10 tasks x 10 seeds x 2 layers)
- `RESULT_attackvla_magic_factorial.log` — full run log
- `FIGURE_attackvla_magic_factorial.png` — interaction plot, effect sizes, controlled AUROCs
