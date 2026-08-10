# Action → (image+text): weight vs QKTV Frobenius (1 demo/task)

Protocol: LIBERO-Goal tasks 0–9, seed=7, one frame after 10 settling steps.
Queries = 7 action tokens/slots. Keys = image patches ∥ instruction tokens.
Last LLM layer.

- **weight**: head-avg α (raw attention)
- **QKTV**: ∑_h α_h · ‖W_O^h v_h‖ (Kobayashi value-weighted)
- **FTT**: mean_i ‖p_i − p̄‖₂ after row-sum normalize (T2IShield dispersion)
- **Gram**: ‖C − I‖_F on L2-normalized DoF rows (DoF consensus)

## Results

| model | metric | AUROC | clean | poison | open margin |
|---|---|---:|---:|---:|---:|
| GoBA | FTT weight | 0.65 | 0.055 | 0.049 | no |
| GoBA | **FTT QKTV** | **0.71** | 0.091 | 0.078 | no |
| GoBA | Gram QKTV | 0.68 | 6.27 | 6.33 | no |
| BadVLA | FTT weight | **1.00** | 0.032 | 0.007 | **+0.020** |
| BadVLA | **FTT QKTV** | **1.00** | 0.058 | 0.010 | **+0.037** |
| BadVLA | Gram QKTV | **1.00** | 5.76 | 6.15 | **+0.198** |

## Reading

BadVLA: clear separation. Every task’s QKTV FTT drops ~6× under the trigger
(clean ~0.058 → poison ~0.010); points sit far below the diagonal; open margin
on both FTT and Gram.

GoBA: weak. QKTV helps a little over raw weights (0.71 vs 0.65) but tasks
overlap — some poison FTT > clean (t0, t1, t4, t7). No open margin.

Value-weighting amplifies BadVLA’s collapse (margin 0.037 vs 0.020) and does
not rescue GoBA.

## Files

- `FIGURE_ftt_per_task.png`, `FIGURE_ftt_scatter.png`, `FIGURE_gram_per_task.png`
- `scalars.npz`, `goba.log`, `badvla.log`
