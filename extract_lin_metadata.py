#!/usr/bin/env python3

import sys
import glob


def get_lin_metadata(infile):
	with open(infile, 'r') as f:
		metadata_line = f.read().split('\n')[0]
		metadata = metadata_line[1:].split(',')
		dataset = metadata[0]
		lin = metadata[3]
	return dataset, lin


def output_results(data, outfile):
	with open(outfile, 'w') as out:
		for name, lin in data:
			out.write('{},{}\n'.format(
				name,
				lin
			))


if __name__ == '__main__':
	extracted_data = []
	for train_file in glob.glob(sys.argv[1] + '/*.txt'):
		extracted_data.append(get_lin_metadata(train_file))
	output_results(extracted_data, sys.argv[2])
