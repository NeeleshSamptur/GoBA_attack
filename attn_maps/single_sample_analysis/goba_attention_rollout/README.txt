Attention ROLLOUT (Abnar & Zuidema 2020) vs last-layer snapshot.
Â=0.5A+0.5I per layer; R = Â_{L-1}@...@Â_0.
Weight = head-avg α; Value-weighted = sum_h α_h·||W_O^h v_h|| then row-norm.
n_layers=32  prompt=In: What action should the robot take to turn on the stove?
Out:
