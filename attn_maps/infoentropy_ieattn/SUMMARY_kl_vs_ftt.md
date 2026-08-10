# Calibration-free KL vs FTT (text→image rows, L31)

**No clean reference / sample calibration.** Within each sample after row-sum normalize:

\[
\mathrm{score}=\frac1n\sum_i \mathrm{KL}(p_i\| \bar p),\qquad \bar p=\tfrac1n\sum_i p_i
\]

Also JS\((p_i,\bar p)\). Seed=7, tasks 0–9.

## Backdoored models (detects)

| model | metric | AUROC | clean | poison | poison↓ | paired c>p |
|---|---|---:|---:|---:|---|---:|
| GoBA | FTT | 1.000 | 0.0837 | 0.0602 | yes | 10/10 |
| GoBA | mean KL(p_i‖p̄) | 1.000 | 0.2089 | 0.1402 | yes | 10/10 |
| GoBA | mean JS | 1.000 | 0.0482 | 0.0323 | yes | 10/10 |
| BadVLA | FTT | 1.000 | 0.0536 | 0.0079 | yes | 10/10 |
| BadVLA | mean KL | 1.000 | 0.1348 | 0.0123 | yes | 10/10 |
| BadVLA | mean JS | 1.000 | 0.0332 | 0.0030 | yes | 10/10 |

## Clean-model controls (specificity)

| model | metric | AUROC | clean | poison | poison↓ | paired c>p |
|---|---|---:|---:|---:|---|---:|
| Clean OpenVLA | FTT | 0.520 | 0.0861 | 0.0855 | yes | 7/10 |
| Clean OpenVLA | mean KL | **0.740** | 0.2049 | 0.1936 | yes | 9/10 |
| Clean OpenVLA | mean JS | 0.670 | 0.0474 | 0.0449 | yes | 9/10 |
| Clean OFT | FTT | 0.730 | 0.0637 | 0.0694 | **no** (↑) | 1/10 |
| Clean OFT | mean KL | **0.840** | 0.1361 | 0.1549 | **no** (↑) | 0/10 |
| Clean OFT | mean JS | 0.820 | 0.0334 | 0.0378 | **no** (↑) | 1/10 |

## Verdict

- Yes: calibration-free KL works the same way as FTT on BD models (AUROC 1.0, poison↓).
- KL is an FTT cousin, not a better detector: on clean controls it is **less specific** than FTT (more sticker/scene confound on OpenVLA; stronger wrong-direction signal on OFT).
- Prefer FTT (L2 to mean) over KL for the paper; mention KL/JS as equivalent assimilation measures if useful for reviewers.

Artifacts: `kl_vs_ftt_goba_family.npz`, `kl_vs_ftt_badvla_family.npz`
