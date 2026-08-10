# Time-series concepts applied to action→image attention trajectories

Question: is there a temporal pattern in the 7-DoF action→image attention maps that
separates clean from backdoored rollouts, and does it hold for GoBA, BadVLA and AttackVLA?

Working hypothesis under test: *backdoored attention freezes (stops changing over time)
while clean attention keeps moving.*

## Setup

Each episode is a trajectory of `T=20` states; each state is 7 DoF × 256 patches, row-normalised
to sum to 1. Datasets and controls:

| source | episodes | roles |
|---|---|---|
| GoBA | 80 | clean, poison, decoy_ketchup, decoy_milk |
| BadVLA | 40 | clean, poison |
| AttackVLA | 60 | clean, visual_only, full_trigger |
| Clean model + trigger object | 40 | clean, poison |

Protocol: features are selected on tasks 0–4 and all reported numbers are on held-out tasks 5–9.
The two GoBA decoys (a benign unseen object in the scene) and the clean model with the trigger
object present are specificity controls — a real detector must stay quiet on all three.

## Concepts surveyed and why

- **Recurrence quantification analysis (RQA).** Recurrence plots built from cosine distance
  between states, at a *fixed recurrence rate* so density can't confound the line statistics.
  Determinism, laminarity, trapping time, longest diagonal/vertical line, diagonal entropy.
  Laminarity and trapping time are the direct formalisation of "the system stays trapped in one
  state", i.e. the frozen-attention hypothesis. RQA is valid on short series (50–100 points is
  the usual floor, and structure measures degrade gracefully below that).
- **Symbolic / compression complexity.** Lempel–Ziv complexity of the attended-patch sequence and
  of median-thresholded scalar summaries; permutation entropy (order 3). Both are designed for
  short, noisy, non-stationary series, unlike sample entropy which is unstable at this length.
- **State-space geometry.** Path length, tortuosity (path length ÷ net displacement), radius of
  gyration, step-size coefficient of variation, effective rank and PC1 fraction of the trajectory.
- **DoF-consensus persistence (PDC).** Carried over from the previous round.
- **Deliberately excluded: DFA / Hurst exponent.** These need on the order of 500+ samples to be
  reliable and are positively biased on short series. At `T=20` any estimate would be an artefact,
  so they were not used.

32 features in total.

## Result 1 — the frozen-attention hypothesis is real, but only for two of the three attacks

Path length of the mean-DoF attention trajectory (held-out tasks, AUROC scored as *less movement =
attack*):

| attack | AUROC | direction |
|---|---|---|
| GoBA | 0.73 | attention moves less |
| BadVLA | 1.00 | attention moves less |
| AttackVLA | 0.05 | attention moves **more** |
| clean model + trigger | 0.51 | no effect |

BadVLA is the extreme case: every poisoned episode collapses to a path length near 0.75 with no
overlap against clean at all. GoBA shows the same direction weakly. AttackVLA does the opposite —
under the full trigger the attention map moves *more* than clean, not less. The clean model with
the trigger object physically present shows nothing, so the effect comes from the backdoor rather
than from the object being in the scene.

## Result 2 — laminarity looks promising but does not survive a threshold sweep

Laminarity is the measure that most directly encodes "trapped in a state", and it had the most
consistent unsigned contribution of any feature. It does not hold up. Sweeping the recurrence rate
from 0.1 to 0.4, the AUROC for "high laminarity = attack" moves from 0.72 to 0.25 for GoBA and
from 1.00 to 0.03 for BadVLA, while AttackVLA goes the other way. There is no recurrence threshold
at which all three point the same direction, and choosing the one favourable setting (RR=0.15,
where BadVLA hits 1.00 and AttackVLA 0.82) would be fitting the threshold to the answer. Trapping
time is better behaved — BadVLA sits at AUROC 1.00 at *every* threshold tested — but GoBA stays
around 0.63–0.68 and AttackVLA still flips.

## Result 3 — no single signed feature is universal

Across all 32 features, none achieves good held-out AUROC on GoBA, BadVLA and AttackVLA with the
same sign. The closest are trajectory effective rank (0.56 / 0.57 / 0.88) and tortuosity
(0.52 / 0.77 / 0.70), both consistent in direction but too weak to threshold. PDC and mean DoF
consensus remain perfect on GoBA and BadVLA held-out tasks (1.00 / 1.00) and still invert on
AttackVLA (0.20 / 0.10).

The pattern is that each attack distorts the temporal dynamics in its own direction. GoBA and
BadVLA freeze attention and raise DoF consensus; AttackVLA destabilises it and lowers consensus.
A single-signed rule cannot cover both behaviours.

## Result 4 — what does work: clean-calibrated unsigned deviation

Since the directions conflict but the *magnitude* of the distortion is consistent, score how far an
episode's feature vector sits from clean behaviour, ignoring direction. Robust location and scale
(median, MAD floored by IQR) are fitted on **clean episodes of tasks 0–4 only**, so no poison labels
are ever used; an episode is scored by the mean clipped `|z|` across features.

Held-out tasks 5–9:

| condition | AUROC | note |
|---|---|---|
| GoBA poison | 0.77 | |
| BadVLA poison | 1.00 | |
| AttackVLA full trigger | 0.84 | |
| AttackVLA visual only | 0.39 | weaker trigger, not detected |
| **clean model + trigger** | **0.42** | correctly silent |
| **GoBA decoy ketchup** | **0.52** | correctly silent |
| **GoBA decoy milk** | **0.56** | correctly silent |

Against a shuffled-label null (2000 permutations within the held-out split), the observed values
are significant: GoBA p=0.012, BadVLA p<0.001, AttackVLA p=0.001. The null 95th percentile is
around 0.72, so GoBA at 0.77 is only marginally above what chance produces at this sample size.

The features carrying the score differ per attack, which is the point of the unsigned formulation:
GoBA is driven by consensus/PDC and the Lempel–Ziv complexity of the attended-patch sequence,
BadVLA by recurrence line lengths and radius of gyration, AttackVLA by laminarity and the number
of distinct attended patches.

## Caveat added after verifying the AttackVLA setup

Every AttackVLA conclusion below and above should be read with
`attn_maps/RESULT_attackvla_ftt_confound.md` in hand. That verification found two confounds in
how the AttackVLA conditions were generated: the `~*magic*~` prefix adds six text tokens
(+22-24%), and all poisoned samples come from a single fixed scene while clean samples span ten
scenes. `attackvla_trajectory_dof_attention.npz`, the source for the AttackVLA numbers here, has
the same single-scene structure. The claim that "AttackVLA inverts the direction" is therefore
not yet established — for the FTT statistic the apparent inversion turned out to be entirely a
tokenization artifact, and the same may hold for the trajectory features. GoBA and BadVLA are
unaffected.

## Caveats

- **Threshold transfer fails for AttackVLA.** Setting the operating point at the 95th percentile of
  clean training scores gives a sensible false-positive rate for GoBA (0.10) and BadVLA (0.10), but
  0.90 for AttackVLA clean — the clean score distribution itself shifts between task groups there.
  The ranking (AUROC) transfers; the absolute threshold does not.
- **Small samples.** 10 clean and 10 attack episodes per held-out evaluation. Confidence intervals
  are wide and GoBA's 0.77 in particular should not be treated as established.
- **Multiple comparisons.** 32 features were screened. The train/test task split and the permutation
  null both guard against this, but the unsigned detector aggregates all 32 features rather than
  selecting one, which is what makes it defensible.
- **AttackVLA visual-only is not detected** by any method here.

## Files

- `FIGURE_timeseries_concepts.png` — path length per attack, laminarity threshold sweep, unsigned
  deviation distributions with controls, permutation nulls.
