#!/usr/bin/env python3

import sys
import pickle
import glob


def load_count_files(temp_dir, dataset):
	count_data = []
	filepaths = ('{}/{}/{}.pickle'.format(temp_dir, dataset, x) for x in ('X', 'class_labels', 'key_idxs', 'value_idxs'))
	for path in filepaths:
		with open(path, 'rb') as infile:
			count_data.append(pickle.load(infile))
	return count_data


if __name__ == '__main__':
	temp_dir = sys.argv[1]
	for fpath in glob.glob(temp_dir + '/*'):
		dataset = fpath.split('/')[-1]
		sys.stdout.write('{}\t{}\n'.format(dataset, fpath))
		X, _, _, _ = load_count_files(temp_dir, dataset)
		print(X[0])
