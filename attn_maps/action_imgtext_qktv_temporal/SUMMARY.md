# Action→(image+text) QKTV across timesteps

Protocol: LIBERO-Goal tasks 0–9, seed=7, T=10 closed-loop steps after settle.
Maps = mean over 7 DoF of QKTV action→image (16×16), overlaid on RGB.

## AUROC (episode score = −mean_t FTT_QKTV; poison expected lower)

| model | FTT weight AUROC | FTT QKTV AUROC | vs single-frame QKTV |
|---|---:|---:|---:|
| GoBA | 0.870 | **0.910** | was 0.71 @ t≈0 |
| BadVLA | 1.000 | **1.000** | was 1.00 @ t≈0 |

Averaging FTT over the rollout helps GoBA a lot (0.71 → 0.91). BadVLA already
separates at every step with an open margin (clean ~0.04–0.07, poison ~0.007 flat).

## Patterns in the maps

- **BadVLA**: poison FTT is flat and tiny for all tasks×t; clean wanders. Delta heatmap
  is all red (clean > poison). Map strips: clean stays sharp on objects; poison stays
  locked / washed relative to clean across the whole trajectory.
- **GoBA**: poison mean FTT drifts down over t; clean higher but overlapping. Map strips
  show poison attention more frozen on a fixed locus (often corner/trigger region) while
  clean tracks the arm/objects. A few blue cells in the delta heatmap (t0@t=2, t4@t=4)
  are the flips that kill open margin.

## What to look for in the maps

- **Persistence**: does the poison hotspot stay fixed while clean wanders?
- **Collapse**: does poison concentrate / assimilate DoFs (lower FTT)?
- **Timing**: does separation appear at t=0 or grow over the rollout?

## Files

- `FIGURE_*_ftt_qktv_vs_t.png` — per-episode FTT curves
- `FIGURE_*_map_strips_qktv.png` — attention overlays across t
- `FIGURE_*_delta_ftt_heatmap.png` — clean−poison FTT per task×t
- `*_temporal.npz` — raw arrays
