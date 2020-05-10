#!/usr/bin/env python3

import numpy as np
import sys
import os
import multi_mle.mle_utils as mutils
import multiprocessing as mp
import multi_mle.blm as blm
import multi_mle.dm as dm
import glob

np.random.seed(2718)

# MLE/NB params
delta_eps_threshold = 1e-26
delta_lprob_threshold = 1e-5
max_steps = 200
batch_size = 1000

# Algo params
DISTRIBUTIONS = ('pooledDM', 'DM', 'BLM')


def power_analysis(train_data, test_data, metadata, temp_dir, distribution, posterior_method=None):
	if distribution == 'pooledDM':
		X, class_labels, key_idxs, value_idxs = mutils.tokenize_train(train_data, test_data)

		simplex_matrix = np.zeros((len(value_idxs), len(class_labels)), dtype=np.float64)
		for i, c in enumerate(class_labels):
			class_simplex = np.squeeze(np.sum(X[c][0], axis=0))
			class_simplex = class_simplex / np.sum(class_simplex)
			# X[c][1] stores a dictionary/map of non-zero feature idx to the corresponding feature idx
			# with zero-sum columns included.  The simplex_matrix below is dim (features, classes), while
			# the count matrix exists for each class, dim (observations, features).
			for j, p in enumerate(class_simplex):
				simplex_matrix[X[c][1][j], i] = p
	else:
		engine = mutils.MLEngine(max_steps, delta_eps_threshold, delta_lprob_threshold, temp_dir, metadata['name'],
		                         precompute_method='vectorized', posterior_method=posterior_method, verbose=False)
		X, class_labels, key_idxs, value_idxs = mutils.tokenize_train(train_data, test_data)

		filepaths = engine.multi_pickle_dump((X[c], c) for c in class_labels)

		# Serial execution of MLE, since we're already in children processes via dataset, and Python doesn't allow
		# child threads to spawn additional threads
		outputs = []
		if distribution == 'DM':
			for f in filepaths:
				outputs.append(engine.dm_mle_parallel(f))
		elif distribution == 'BLM':
			for f in filepaths:
				outputs.append(engine.blm_mle_parallel(f))
		else:
			raise ValueError('Distribution must be one of [pooledDM, DM, BLM]. Provided: {}'.format(distribution))

		mle_results = engine.load_mle_results(outputs)

		assert (len(mle_results) == len(class_labels))
		if distribution == 'BLM' and posterior_method == 'aposteriori':
			simplex_matrix = np.zeros((len(value_idxs) + 1, len(class_labels)), dtype=np.float64)
		else:
			simplex_matrix = np.zeros((len(value_idxs), len(class_labels)), dtype=np.float64)
		for label, simplex in mle_results:
			simplex_matrix[:, key_idxs[label]] = simplex

	if posterior_method == 'aposteriori':
		param_result_file = '{}/parts/{}_{}_{}.param_part'.format(
			temp_dir,
			metadata['name'],
			distribution.lower(),
			posterior_method
		)
		rev_val_idx = {v: k for k, v in value_idxs.items()}
		with open(param_result_file, 'w') as param_out:
			for c in class_labels:
				original_params = np.squeeze(np.array(metadata[c], dtype=np.float64))
				post_mle_params = np.squeeze(np.zeros((1, simplex_matrix.shape[0]), dtype=np.float64))
				for i in range(simplex_matrix.shape[0]):
					try:
						original_idx = int(rev_val_idx[i]) - 1
						post_mle_params[original_idx] = simplex_matrix[i, key_idxs[c]]
					except KeyError:  # this should only be true for the last BLM parameter
						original_idx = i
						post_mle_params[original_idx] = simplex_matrix[i, key_idxs[c]]
			param_out.write('{},{},generating,parameter,{}\n'.format(
				metadata['name'],
				distribution,
				','.join(str(x) for x in original_params)
			))
			param_out.write('{},{},mle,parameter,{}\n'.format(
				metadata['name'],
				distribution,
				','.join(str(x) for x in post_mle_params)
			))
			if distribution == 'DM':
				param_out.write('{},{},generating,simplex,{}\n'.format(
					metadata['name'],
					distribution,
					','.join(str(x) for x in dm.dm_renormalize(original_params))
				))
				param_out.write('{},{},mle,simplex,{}\n'.format(
					metadata['name'],
					distribution,
					','.join(str(x) for x in dm.dm_renormalize(post_mle_params))
				))
			elif distribution == 'BLM':
				param_out.write('{},{},generating,simplex,{}\n'.format(
					metadata['name'],
					distribution,
					','.join(str(x) for x in blm.blm_renormalize(original_params))
				))
				param_out.write('{},{},mle,simplex,{}\n'.format(
					metadata['name'],
					distribution,
					','.join(str(x) for x in blm.blm_renormalize(post_mle_params))
				))

	classification_result_file = '{}/parts/{}_{}_{}.classification_part'.format(
		temp_dir,
		metadata['name'],
		distribution.lower(),
		posterior_method
	)
	param_string = 'n=1'
	method = 'lidstone'
	mutils.output_results_naive_bayes(simplex_matrix, test_data, class_labels, key_idxs, value_idxs,
	                                  metadata['name'], distribution, method, param_string, 'vectorized',
	                                  classification_result_file, posterior_method, batch_size)


def process(q, lock, temp_dir):
	while True:
		job = q.get()
		if not job:
			break
		with lock:
			print('Processing job: {}'.format(job[2]['name']))
		for distribution in DISTRIBUTIONS:
			if distribution == 'BLM' or distribution == 'DM':
				for post_method in (None, 'aposteriori'):
					power_analysis(job[0], job[1], job[2], temp_dir, distribution, posterior_method=post_method)
			elif distribution == 'pooledDM':
				power_analysis(job[0], job[1], job[2], temp_dir, distribution)
			else:
				raise ValueError('Distribution must be one of [pooledDM, DM, BLM]. Provided: {}'.format(distribution))


def aggregate_partial_results(temp_dir, out_dir):
	parts_dir ='{}/parts'.format(temp_dir)
	with open(out_dir + '/power_analysis_classification_results.csv', 'w') as out_class, \
		open(out_dir + '/power_analysis_mle_results.csv', 'w') as out_mle:
		for cl_res_file in glob.glob('{}/*.classification_part'.format(parts_dir)):
			with open(cl_res_file, 'r') as cl_in:
				data = cl_in.read()
				for line in data.split('\n'):
					if not line:
						continue
					out_class.write('{}\n'.format(line))

		for mle_res_file in glob.glob('{}/*.param_part'.format(parts_dir)):
			with open(mle_res_file, 'r') as mle_in:
				data = mle_in.read()
				for line in data.split('\n'):
					if not line:
						continue
					out_mle.write('{}\n'.format(line))


if __name__ == '__main__':
	train_dir_path = sys.argv[1]
	test_dir_path = sys.argv[2]
	temp_dir = sys.argv[3]
	output_dir = sys.argv[4]
	n_parallel_jobs = int(sys.argv[5])

	if not os.path.isdir('{}/parts'.format(temp_dir)):
		os.mkdir('{}/parts'.format(temp_dir))

	q = mp.Queue(maxsize=n_parallel_jobs)
	lock = mp.Lock()
	pool = mp.Pool(n_parallel_jobs, initializer=process, initargs=(q, lock, temp_dir))

	for dataset in mutils.load_dataset_generator(train_dir_path, test_dir_path, metadata=True):
		q.put(dataset)

	for _ in range(n_parallel_jobs):
		q.put(None)

	pool.close()
	pool.join()

	aggregate_partial_results(temp_dir, output_dir)
