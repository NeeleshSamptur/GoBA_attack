# GoBA disjoint-task AUROC (5 clean tasks ≠ 5 poison tasks)

Protocol: GoBA-backdoored model. Choose 5 tasks that contribute **only clean-scene**
scores and 5 **different** tasks that contribute **only poison-scene** scores.
AUROC on these 10 points (no same-task pairing).

Fixed split: clean tasks `{0,1,2,3,4}` vs poison tasks `{5,6,7,8,9}`.
Also: all `C(10,5)=252` assignments → mean±std AUROC.

| metric | fixed AUROC | sign | mean±std over 252 | median | min |
|---|---:|---|---:|---:|---:|
| Ieattn mean | 0.960 | poison↓ | 0.967±0.037 | 0.960 | 0.880 |
| Ieattn max | 0.960 | poison↓ | 0.978±0.029 | 1.000 | 0.920 |
| Ieattn Top-k Jac consec | 0.840 | poison↓ | 0.724±0.125 | 0.720 | 0.520 |
| QKTV FTT (mean over T) | 0.880 | poison↓ | 0.900±0.081 | 0.920 | 0.640 |
| weight FTT (mean over T) | 0.880 | poison↓ | 0.856±0.103 | 0.880 | 0.520 |
| text→image FTT (xattn) | 0.960 | poison↓ | 0.967±0.039 | 1.000 | 0.880 |
| action→image FTT | 0.600 | poison↑ | 0.699±0.133 | 0.680 | 0.520 |
| action→img+text FTT | 0.600 | poison↑ | 0.681±0.121 | 0.680 | 0.520 |

## Paired baseline (same 10 tasks, clean vs poison) for reference

| metric | paired AUROC (poison↓ or best) |
|---|---:|
| Ieattn mean | 0.970 (poison↓) |
| Ieattn max | 0.980 (poison↓) |
| Ieattn Top-k Jac consec | 0.720 (poison↓) |
| QKTV FTT (mean over T) | 0.910 (poison↓) |
| weight FTT (mean over T) | 0.870 (poison↓) |
| text→image FTT (xattn) | 0.970 (poison↓) |
| action→image FTT | 0.670 (poison↑) |
| action→img+text FTT | 0.630 (poison↓) |

## Reading

- If disjoint AUROC stays high, the signal is not just same-task pairing.
- If it falls to ~0.5, the paired AUROC was optimistic.

Figure: `FIGURE_goba_disjoint5_auroc.png`
