# FTT vs VLA-InfoEntropy (Ieattn): clean vs backdoored

## Metrics

- **text→image FTT**: T2IShield dispersion on text→image attention rows, last LLM layer.
- **Ieattn mean**: adapted InfoEntropy \(I^e_{attn}\) averaged over image patches
  (renorm over text, last 8 layers; GoBA temporal mean over T=10; clean OpenVLA T=5;
  BadVLA/clean OFT single frame after settle for FTT, BadVLA Ieattn was T=10).

## Protocol notes

- **Trigger**: GoBA = physical poison BDDL object; BadVLA = white block on full+wrist.
- **Clean control**: same task seed=7, clean vs poison/trigger scene (not backdoored weights).
- **Backdoored**: GoBA LoRA ckpt / BadVLA stage2 ckpt; FTT often averaged over seeds.

## Results

| attack | model | metric | clean mean | poison mean | AUROC | poison direction | paired |
|---|---|---|---:|---:|---:|---|---:|
| GoBA | backdoored | text→image FTT | 0.08093 | 0.05790 | 1.000 | poison↓ | 10/10 |
| GoBA | clean model | text→image FTT | 0.08609 | 0.08551 | 0.520 | poison↓ | 7/10 |
| GoBA | backdoored | Ieattn mean | 0.66143 | 0.63777 | 0.970 | poison↓ | 10/10 |
| GoBA | clean model | Ieattn mean | 0.65655 | 0.64229 | 0.790 | poison↓ | 10/10 |
| BadVLA | backdoored | text→image FTT | 0.06033 | 0.00783 | 1.000 | poison↓ | 10/10 |
| BadVLA | backdoored | Ieattn mean | 0.49239 | 0.00304 | 1.000 | poison↓ | 10/10 |
| BadVLA | clean OFT | text→image FTT | 0.06369 | 0.06943 | 0.730 | poison↑ | 9/10 |
| BadVLA | clean OFT | Ieattn mean | 0.51184 | 0.54128 | 0.810 | poison↑ | 10/10 |

## Verdict

| | text→image FTT | Ieattn |
|---|---|---|
| **GoBA backdoored** | **1.00** (↓) | 0.97 (↓) tiny gap |
| **Clean OpenVLA** | **0.52** (flat) ✓ | 0.79 (↓) ✗ scene confound |
| **BadVLA backdoored** | **1.00** (↓ collapse) | **1.00** (↓ collapse) |
| **Clean OFT** | 0.73 (**↑** opposite) ✓ | 0.81 (**↑** opposite) ✓ |

**Shared reliable detector:** **text→image FTT** — poison **lowers** it on both backdoors; clean models do **not** (GoBA flat; BadVLA clean OFT even goes slightly up).

**Ieattn:** fine for **BadVLA** (clean OFT goes the other way); **not** for GoBA (clean model also drops).

Full writeup: this file.
