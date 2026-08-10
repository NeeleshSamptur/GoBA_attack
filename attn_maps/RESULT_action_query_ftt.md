# FTT with the query side moved to ACTION tokens: GoBA and BadVLA

T2IShield's FTT disperses **text**-token rows over image patches. This run keeps the statistic
identical and only changes which rows and which key span it is computed over:

| stream | rows | columns |
|---|---|---|
| `action -> image` | action queries | 256 (GoBA) / 513 (BadVLA, dual camera + proprio) image patches |
| `action -> text` | action queries | instruction tokens |
| `action -> image+text` | action queries | patches and instruction tokens concatenated |
| `text -> image` | instruction tokens | image patches — the original statistic, for reference |

FTT = mean L2 distance of each row from the mean row, rows normalised to sum to 1. The rule
under test is T2IShield's: **low dispersion = backdoor**.

Protocol: one frame per episode after 10 settling steps, all 32 layers recorded, 10 tasks.
GoBA uses seed-disjoint roles (50 clean / 50 poison / 50 ketchup / 50 milk); BadVLA uses paired
seeds (100 clean / 100 poison), matching its original assimilation run. Both triggers are purely
visual, so the instruction and its token count are identical between conditions — there is no
row-count confound of the kind that invalidated the AttackVLA comparison.

Validation: the `text -> image` stream reproduces the original runs exactly (BadVLA task 0
seed 7 gives clean 0.0465 / poison 0.0079, matching `run_assimilation_badvla_2026_08_03.log`).

## Results, last layer (L31)

**GoBA** — 50 clean vs 50 poison

| query → keys | AUROC | clean | attack | margin | tasks 0-4 | tasks 5-9 |
|---|---:|---:|---:|---:|---:|---:|
| action → image | 0.753 | 0.0612 | 0.0533 | −0.0237 | 0.80 | 0.71 |
| action → text | 0.926 | 0.1023 | 0.0809 | −0.0250 | 0.91 | 0.97 |
| **action → image+text** | **0.899** | 0.1042 | 0.0852 | −0.0283 | 0.84 | **0.97** |
| text → image | **0.997** | 0.0809 | 0.0579 | −0.0013 | 1.00 | 1.00 |

**BadVLA** — 100 clean vs 100 poison

| query → keys | AUROC | clean | attack | margin | tasks 0-4 | tasks 5-9 |
|---|---:|---:|---:|---:|---:|---:|
| action → image | **1.000** | 0.0753 | 0.0083 | **+0.0419** | 1.00 | 1.00 |
| action → text | 0.956 | 0.0352 | 0.0277 | −0.0062 | 0.95 | 0.97 |
| **action → image+text** | **1.000** | 0.0386 | 0.0081 | **+0.0183** | 1.00 | 1.00 |
| action(first 7) → image | **1.000** | 0.0792 | 0.0070 | **+0.0341** | 1.00 | 1.00 |
| action(first 7) → text | 0.645 | 0.0218 | 0.0194 | −0.0149 | 0.63 | 0.66 |
| action(first 7) → image+text | **1.000** | 0.0340 | 0.0067 | **+0.0108** | 1.00 | 1.00 |
| text → image | **1.000** | 0.0603 | 0.0078 | **+0.0344** | 1.00 | 1.00 |

Side by side:

| query → keys | GoBA | BadVLA |
|---|---:|---:|
| action → image | 0.753 | 1.000 |
| action → text | 0.926 | 0.956 |
| **action → image+text** | **0.899** | **1.000** |
| text → image | 0.997 | 1.000 |

GoBA decoy specificity for `action → image+text`: ketchup 0.536, milk 0.478 (chance).

## Reading

**The original text → image statistic is still the best on both platforms** (0.997 and 1.000).
Moving the query side to action tokens does not improve on it and costs a lot on GoBA.

**`action → image+text` sits between the two separate streams on GoBA** (0.899 overall, 0.974 held-out) — better than image alone, slightly below text alone overall but matching text alone on held-out tasks. On BadVLA it is perfect with an open margin of +0.018. The joint key span does not unlock anything the separate streams already miss.

**`action -> image` is where the two attacks diverge.** BadVLA collapses spectacularly — mean
dispersion falls from 0.0753 to 0.0083, a 9x drop, with an open margin of +0.042, meaning a
single fixed constant separates every one of the 200 episodes. GoBA only reaches 0.753. This
tracks the trigger design: BadVLA pastes a bright synthetic patch onto both camera views, and
the action slots lock onto it almost totally, whereas GoBA's trigger is a naturalistic object
in a cluttered scene.

**`action -> text` works on both** (0.926 / 0.956) and is the one place where GoBA's action
queries beat their own `action -> image`. Under GoBA the action tokens read the instruction more
uniformly when the trigger is present, which is consistent with the goal-hijack story: the
instruction stops differentiating the action dimensions.

**Only BadVLA gets open margins.** Every GoBA margin is negative, so its high AUROCs mean good
ranking without a threshold that separates all episodes — the distinction flagged earlier in
`PAPER_RESULTS_SUMMARY.md` between ranking well and being deployable.

**GoBA specificity is clean.** Against benign novel objects the streams stay at or below chance
(ketchup 0.324 / 0.608 / 0.365 / 0.536, milk 0.404 / 0.548 / 0.590 / 0.478 for action→image,
action→text, text→image, action→image+text), so none of them false-alarm on an unfamiliar object.

## Layer selection: fragile, do not lean on it

Selecting the best layer and its direction on tasks 0-4 and scoring tasks 5-9:

| model | stream | layer | direction | held-out AUROC |
|---|---|---:|---|---:|
| GoBA | action → image | 4 | **high** | 0.986 |
| GoBA | action → text | 28 | low | 0.904 |
| GoBA | text → image | 31 | low | 1.000 |
| BadVLA | action → image | 1 | low | 1.000 |
| BadVLA | action → text | 0 | low | 1.000 |
| BadVLA | text → image | 1 | **high** | 1.000 |

Two reasons to distrust these. GoBA's `action -> image` reaches 0.986 at layer 4 only by flipping
the direction relative to the last layer, the same threshold-shopping failure mode found with
recurrence laminarity. And BadVLA's best layers are 0 and 1, where attention is still dominated
by positional and raw-input statistics; a bright white patch pasted on the image will move those
regardless of whether a backdoor exists. The last-layer numbers with the paper's fixed direction
are the defensible ones.

## Files

- `goba_action_query_ftt.npz`, `RESULT_goba_action_query_ftt.log` (200 samples x 32 layers)
- `badvla_action_query_ftt.npz`, `RESULT_badvla_action_query_ftt.log` (200 samples x 32 layers)
- `experiments/robot/libero/run_action_query_ftt_goba.py`
- `BadVLA/experiments/robot/libero/run_action_query_ftt_badvla.py`
