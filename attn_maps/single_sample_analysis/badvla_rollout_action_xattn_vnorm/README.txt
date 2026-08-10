Mode: badvla_action_vnorm
Kind: ACTION→image  (α·‖Wov‖)
Prompt: In: What action should the robot take to turn on the stove?
Out:
Query rows (action DoF): ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']
Shapes: clean=(20, 7, 256) poison=(20, 7, 256)
npz key clean_text2patch holds query→patch (action DoFs when *_action).
Row-normalized.
Note: VALUE-WEIGHTED: sum_h alpha_h * ||W_O^h v_h|| ; row-normalized over patches
