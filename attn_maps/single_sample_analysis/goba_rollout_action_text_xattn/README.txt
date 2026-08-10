Mode: goba action↔text
Prompt: In: What action should the robot take to turn on the stove?
Out:
DoFs: ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']
Text tokens: ['In', ':', 'What', 'action', 'should', 'the', 'robot', 'take', 'to', 'turn', 'on', 'the', 'st', 'ove', '?', '\\n', 'Out', ':', '∅']
Shapes: clean=(20, 7, 19) poison=(20, 7, 19)
Primary: action DoF → text token attention (row-normalized).
Causal LM: text→action is usually masked/empty.
