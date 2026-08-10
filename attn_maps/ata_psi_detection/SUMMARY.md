# ATA-style Ψ (last-query→image) as detector

Protocol: seed=7, tasks 0–9, clean vs poison BDDL, last LLM layer, head-avg.
Ψ = attention from last query token → 256 image patches (ATA Sec. III-B).
Also: mean of 7 action-query rows + FTT on those rows.

## goba

| metric | AUROC | clean | poison | poison↑? | paired Δ>0 |
|---|---:|---:|---:|---|---:|
| last-query max(Ψ) | 0.590 | 0.0603 | 0.0629 | yes | 5/10 |
| last-query top4(Ψ) | 0.550 | 0.1821 | 0.1847 | yes | 7/10 |
| last-query 1−H(Ψ) | 0.500 | 0.2020 | 0.2013 | yes | 7/10 |
| action-mean max(Ψ) | 0.720 | 0.0366 | 0.0391 | yes | 9/10 |
| action-mean top4(Ψ) | 0.860 | 0.1285 | 0.1413 | yes | 9/10 |
| action-mean 1−H(Ψ) | 0.690 | 0.1631 | 0.1719 | yes | 8/10 |
| action-rows FTT | 0.510 | 0.0513 | 0.0509 | no | 4/10 |

## clean_model

| metric | AUROC | clean | poison | poison↑? | paired Δ>0 |
|---|---:|---:|---:|---|---:|
| last-query max(Ψ) | 0.650 | 0.0593 | 0.0648 | yes | 7/10 |
| last-query top4(Ψ) | 0.530 | 0.1859 | 0.1842 | no | 5/10 |
| last-query 1−H(Ψ) | 0.600 | 0.1682 | 0.1722 | yes | 7/10 |
| action-mean max(Ψ) | 0.500 | 0.0413 | 0.0402 | yes | 4/10 |
| action-mean top4(Ψ) | 0.730 | 0.1249 | 0.1365 | yes | 8/10 |
| action-mean 1−H(Ψ) | 0.760 | 0.1427 | 0.1539 | yes | 8/10 |
| action-rows FTT | 0.700 | 0.0526 | 0.0576 | yes | 7/10 |

## Verdict

- **Last-query Ψ (ATA literal): does not work** for GoBA (AUROC ~0.5–0.59), and clean model is similar (~0.65) → not a backdoor signal.
- **Action-mean top4** looks ok on GoBA (0.86) but **clean model also 0.73** → mostly sticker/scene peakiness, not clean like text→image FTT.
- Prefer **text→image FTT** (clean model AUROC ~0.52, GoBA ~0.97) over ATA last-query Ψ for detection.
