# VLA-InfoEntropy-style Ieattn for GoBA / BadVLA detection

Adapted InfoEntropy: score(w,i)=mean_{l,h} A_{w→i} (last 8 layers); q(w,i)=A/∑_w A (renorm, not 2nd softmax); Ieattn=1−H/log2|W|.
Top-k=16. T=10, seed=7, tasks 0–9.

AUROC uses best orientation; ↑/↓ marks whether poison scores higher or lower.

## GoBA

| metric | AUROC | clean | poison | poison has higher score? |
|---|---:|---:|---:|---|
| Top-k Jaccard consec | 0.720 | 0.7009 | 0.6698 | no (poison lower) |
| Top-k Jaccard vs t0 | 0.560 | 0.6122 | 0.6218 | yes |
| mean Ieattn | 0.970 | 0.6614 | 0.6378 | no (poison lower) |
| max Ieattn (mean over t) | 0.980 | 0.9823 | 0.9779 | no (poison lower) |
| spatial entropy of Ie | 0.930 | 0.9759 | 0.9730 | no (poison lower) |

## BadVLA

| metric | AUROC | clean | poison | poison has higher score? |
|---|---:|---:|---:|---|
| Top-k Jaccard consec | 1.000 | 0.6845 | 0.4093 | no (poison lower) |
| Top-k Jaccard vs t0 | 1.000 | 0.4472 | 0.0756 | no (poison lower) |
| mean Ieattn | 1.000 | 0.4924 | 0.0030 | no (poison lower) |
| max Ieattn (mean over t) | 1.000 | 0.8046 | 0.0067 | no (poison lower) |
| spatial entropy of Ie | 0.740 | 0.9861 | 0.9854 | no (poison lower) |

## Shared (both AUROC≥0.70, **same** poison↑/↓ sign)

- **Top-k Jaccard consec** (poison↓): GoBA=0.720, BadVLA=1.000
- **mean Ieattn** (poison↓): GoBA=0.970, BadVLA=1.000
- **max Ieattn (mean over t)** (poison↓): GoBA=0.980, BadVLA=1.000
- **spatial entropy of Ie** (poison↓): GoBA=0.930, BadVLA=0.740

Note: paper’s 2nd softmax over text on causal VLA self-attn gave Ie≈0; we use q(w,i)=A_{w→i}/∑_{w'}A_{w'→i} on the last 8 layers.
