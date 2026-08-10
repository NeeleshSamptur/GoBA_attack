Mode: goba_action
Kind: ACTION→image
Prompt: In: What action should the robot take to turn on the stove?
Out:
Query rows (action DoF): ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']
Shapes: clean=(20, 7, 256) poison=(20, 7, 256)
npz key clean_text2patch holds query→patch (action DoFs when *_action).
Row-normalized.
Note: rows=7 action DoF tokens from generate(); patch2action placeholder
