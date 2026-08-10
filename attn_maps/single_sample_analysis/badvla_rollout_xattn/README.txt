Attack: badvla
Prompt: In: What action should the robot take to turn on the stove?
Out:
Tokens (rows): ['<s>', 'In', ':', 'What', 'action', 'should', 'the', 'robot', 'take', 'to', 'turn', 'on', 'the', 'st', 'ove', '?', '\\n', 'Out', ':']
Shapes: clean_text2patch=(20, 19, 256) poison_text2patch=(20, 19, 256)
text2patch[t, i, j] = attention weight from text token i -> image patch j at step t
patch2text[t, j, i] = attention weight from image patch j -> text token i at step t
Matrices are row-normalized (each query sums to 1 over keys).
