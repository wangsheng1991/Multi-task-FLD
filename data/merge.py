import os
import numpy as np

fl_train_txt = 'fl_training.txt'
mtfl_train_txt = 'mtfl_training.txt'
merged_train_txt = 'merged_training.txt'
merged_test_txt = 'merged_testing.txt'

merged_data = []
train_size = 8000

with open(fl_train_txt, 'r') as fd:
	fl_lines = fd.readlines()

with open(mtfl_train_txt, 'r') as mtfd:
	mtfl_lines = mtfd.readlines()

for fl_line, mtfl_line in zip(fl_lines, mtfl_lines):
	fl_line = fl_line.strip()
	fl_components = fl_line.split(' ')
	fl_name = fl_components[0]
	fl_border = fl_components[1:5]
	fl_facial_landmarks = fl_components[5:]

	mtfl_line = mtfl_line.strip()
	mtfl_components = mtfl_line.split(' ')
	mtfl_name = mtfl_components[0]
	mtfl_facial_landmarks = mtfl_components[1:11]
	mtfl_attributes = mtfl_components[11:]
	
	if fl_name != mtfl_name:
		print('error')
		break

	for i in range(10):
		fl_idx = i
		mtfl_idx = i/2 if i % 2 == 0 else i/2 + 5
		if float(fl_facial_landmarks[fl_idx])+1 != float(mtfl_facial_landmarks[mtfl_idx]):
			print('error')
		break

	merged_line = fl_components + mtfl_attributes + ['\n']
	merged_line = ' '.join(merged_line)
	merged_data += [merged_line]

np.random.shuffle(merged_data)
merged_train_data = merged_data[:train_size]
merged_test_data = merged_data[train_size:]

with open(merged_train_txt, 'w') as target:
	target.write(''.join(merged_train_data))
	target.close()

with open(merged_test_txt, 'w') as target:
	target.write(''.join(merged_test_data))
	target.close()