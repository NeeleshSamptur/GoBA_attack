Per-head value-weighted attention (α_h · ||W_O^h v_h||), last LLM layer.
Prompt: In: What action should the robot take to turn on the stove?
Out:
n_heads=32  tokens=['In', ':', 'What', 'action', 'should', 'the', 'robot', 'take', 'to', 'turn', 'on', 'the', 'st', 'ove', '?', '\\n', 'Out', ':']
GRID_*: each head pair CLEAN|POISON overlay (mean over queries).
STATS_*: entropy & f_norm per head.
RANK_*: heads sorted by |Δentropy|.
TOP_*: matrices for the 4 most-changed heads.
