# Calibration-free KL vs FTT — BadVLA family (text→image, L31)

Within-sample `mean_i KL(p_i || p̄)` / JS; no clean reference set.
Seed=7, tasks 0–9, clean vs white-block trigger.

## badvla

| metric | AUROC | clean | poison | poison↓? | paired clean>poison |
|---|---:|---:|---:|---|---:|
| FTT | 1.000 | 0.05355 | 0.00793 | yes | 10/10 |
| mean KL(p_i‖p̄) | 1.000 | 0.13482 | 0.01229 | yes | 10/10 |
| mean JS(p_i,p̄) | 1.000 | 0.03323 | 0.00303 | yes | 10/10 |

## clean_oft

| metric | AUROC | clean | poison | poison↓? | paired clean>poison |
|---|---:|---:|---:|---|---:|
| FTT | 0.730 | 0.06369 | 0.06943 | no | 1/10 |
| mean KL(p_i‖p̄) | 0.840 | 0.13610 | 0.15485 | no | 0/10 |
| mean JS(p_i,p̄) | 0.820 | 0.03343 | 0.03784 | no | 1/10 |
