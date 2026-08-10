# Clean vs backdoored QKTV across timesteps

Protocol: LIBERO-Goal 0–9, seed=7, T=10, action→(image+text) QKTV FTT.

| protocol | clean model AUROC | backdoored AUROC |
|---|---:|---:|
| Clean OpenVLA vs GoBA | 0.630 | 0.910 |
| Clean OFT vs BadVLA | 0.360 | 1.000 |

Clean-model AUROC near 0.5 ⇒ trigger does not collapse QKTV FTT.
Backdoored AUROC high ⇒ poison FTT drops vs clean scenes.

Note: Clean OFT AUROC 0.36 means poison FTT is *higher* than clean (opposite of
BadVLA’s collapse) — the white block is just another visual feature, not assimilation.

## Files
- `FIGURE_*_clean_vs_bd_ftt_qktv_vs_t.png`
- `FIGURE_*_clean_vs_bd_map_strips.png`
- `clean_*_protocol_temporal.npz`
