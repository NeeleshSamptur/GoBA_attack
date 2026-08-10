# Persistent DoF Consensus (PDC)

## Motivation
Pure temporal stability of each DoF map (EMD/cosine across t) is not a strong GoBA separator.
The shared pattern is instead that all seven DoFs become assimilated, and this assimilation persists throughout the rollout.

## Per-episode score (no label required)
At every timestep t:
1. L1-normalize seven action→image maps (7×256).
2. Make cosine Gram C_t (7×7).
3. q_t = ||C_t - I||_F / sqrt(42), in approximately [0,1].
4. PDC = mean_t(q_t) - std_t(q_t).

High PDC means DoF attention maps are both highly alike and consistently alike across time.

## Results

| Attack | clean mean [min,max] | poison mean [min,max] | AUROC all | AUROC held-out tasks 5–9 | open gap |
|---|---:|---:|---:|---:|---:|
| GoBA | 0.9023 [0.8685,0.9348] | 0.9481 [0.9295,0.9797] | 0.997 | 1.000 | -0.0053 |
| BadVLA | 0.9596 [0.9345,0.9783] | 0.9983 [0.9982,0.9984] | 1.000 | 1.000 | +0.0198 |

## Interpretation
- GoBA: trigger-driven DoF collapse remains strong across the rollout.
- BadVLA: diffuse/blinded maps are nearly identical across DoFs at every step.
- The metric ignores whether the shared map is peaked or flat, preventing the GoBA/BadVLA sign flip seen with entropy or concentration.

## Important limitation
PDC is label-free to compute, but a binary decision still requires clean calibration. GoBA has a tiny overlap on the full set, so do not claim a universal fixed threshold.

Tasks 5–9 are a task-split sanity check, not a pristine final test set: several related temporal candidates were inspected during this analysis. Confirm PDC on new tasks/seeds before making a generalization claim.

## Research connection
The construction is inspired by temporal self-similarity matrices and Gram-matrix sequence embeddings: summarize repeated structure across time rather than relying only on adjacent-frame differences.
- Varghese et al., CVPRW 2020: unsupervised temporal consistency metrics.
- Zhang et al., CVPR 2016: temporal sequence comparison using Gram-matrix embeddings.
- ATSS (2026): anomalous temporal self-similarity as a detection signal.