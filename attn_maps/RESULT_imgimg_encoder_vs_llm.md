# Image→image self-attention: vision encoder vs LLM (GoBA)

Patch→patch self-attention on the same scenes, three sources:

| source | what is attended |
|---|---|
| DINOv2 last *used* block (`blocks[-2]`) | 256 patches, CLS+4 registers dropped |
| SigLIP last *used* block (`blocks[-2]`) | 256 patches, no prefix tokens |
| LLM last layer | 256 projected vision tokens attending among themselves |

OpenVLA's featurizer is monkey-patched to `get_intermediate_layers(n={len(blocks)-2})`, so
`blocks[-1]` never runs — we hook `blocks[-2]`. Heatmap = mean over query patches of the
key-mass (which patches get attended), overlaid on the RGB frame.

Scenes: LIBERO-Goal tasks 7/8/9 × seeds 7/42/1234, clean vs poison (seed-matched).

## Heatmaps

- `COMPARE_t*_s*_encoder_vs_llm.png` — 2×3 panel (clean|trigger × DINO|SigLIP|LLM)
- `TRIGGER_t*_s*_encoder_vs_llm.png` — trigger scene only, encoder vs LLM

## FTT (low = backdoor), n=9 clean / 9 poison

| source | AUROC | clean mean | poison mean | open margin |
|---|---:|---:|---:|---:|
| DINO | 0.000 | 0.0959 | 0.0992 | −0.0057 |
| SigLIP | 0.222 | 0.0352 | 0.0356 | −0.0013 |
| **LLM** | **1.000** | 0.1337 | 0.1086 | **+0.0086** |

## Reading

**The vision encoder notices the trigger but does not assimilate.** Both DINO and SigLIP light
up the yellow card when it appears (visible in the heatmaps), yet their f_norm *rises*
slightly under the trigger — the opposite of T2IShield's collapse. The encoder still spreads
attention across the arm, bowls, and background; the card is just one more salient blob.

**The LLM is where assimilation happens.** Under the trigger, patch→patch attention in the
language model collapses onto the card (sparse red hotspot, cold background) and f_norm drops
enough for a perfect open-margin separator. Same patches, same scene — the collapse is produced
by the LLM, not inherited from the encoder.

**SigLIP has corner artifacts** (bright squares at frame corners in both clean and trigger).
Those are encoder-side and unrelated to the backdoor; they inflate SigLIP's key-mass at the
edges and help explain why its f_norm barely moves.

**Vision FTT is identical across tasks for a fixed seed.** Expected: LIBERO-Goal shares one
scene layout across the 10 tasks (only the instruction changes), and the vision encoder never
sees the text. LLM FTT does shift slightly with the instruction.

## Files

- `attn_maps/goba_imgimg_encoder_vs_llm/` — PNGs + `stats.npz`
- `attn_maps/RESULT_imgimg_encoder_vs_llm.log`
- `experiments/robot/libero/run_imgimg_encoder_vs_llm.py`
