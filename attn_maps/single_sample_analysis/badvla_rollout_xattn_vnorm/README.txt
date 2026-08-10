Mode: badvla_vnorm
Kind: TEXT→image  (α·‖Wov‖)
Prompt: In: What action should the robot take to turn on the stove?
Out:
Query rows (text token): ['<s>', 'In', ':', 'What', 'action', 'should', 'the', 'robot', 'take', 'to', 'turn', 'on', 'the', 'st', 'ove', '?', '\\n', 'Out', ':']
Shapes: clean=(20, 19, 256) poison=(20, 19, 256)
npz key clean_text2patch holds query→patch (action DoFs when *_action).
Row-normalized.
Note: VALUE-WEIGHTED: sum_h alpha_h * ||W_O^h v_h|| ; row-normalized over patches
