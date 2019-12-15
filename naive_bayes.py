#!/usr/bin/env python3
import numpy as np
import sys
import glob
import multi_mle.mle_utils as mutils
import multiprocessing as mp
import multi_mle.smoothing as sm


np.random.seed(2718)

# MLE/NB params
delta_eps_threshold = 1e-5
learn_rate_threshold = 2e-10
delta_lprob_threshold = 1e-5
max_steps = 15
batch_size = 1000

# Smoothing params
smoothing_methods = ('lidstone', 'dirichlet', 'jm', 'ad', 'ts')
dirichlet_grid = np.arange(0.1, 1, 0.1)
jm_grid = np.arange(0.1, 1, 0.1)
ad_grid = np.array([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2])
ts_grid = (np.arange(0.1, 1, 0.1), np.arange(0.1, 1, 0.1))


def test_accuracy(distribution, train_path, test_path, dataset_name, result_file, temp_dir):
    simplex_matrix, test, X, class_labels, smoothed, key_idxs, value_idxs = (None,) * 7

    if distribution == 'pooledDM':
        train = mutils.load_data(train_path)
        test = mutils.load_data(test_path)
        X, class_labels, key_idxs, value_idxs = mutils.tokenize_train(train[dataset_name], test[dataset_name])

        simplex_matrix = np.zeros((len(value_idxs), len(class_labels)), dtype=np.float64)
        for i, c in enumerate(class_labels):
            class_simplex = np.squeeze(np.sum(X[c][0], axis=0))
            class_simplex = class_simplex / np.sum(class_simplex)
            for j, p in enumerate(class_simplex):
                simplex_matrix[X[c][1][j], i] = p
    else:
        engine = mutils.MLEngine(max_steps, delta_eps_threshold, delta_lprob_threshold, temp_dir, dataset_name, True)
        if engine.count_files_exist:
            X, class_labels, key_idxs, value_idxs = engine.load_count_files()
        else:
            train = mutils.load_data(train_path)
            test = mutils.load_data(test_path)
            X, class_labels, key_idxs, value_idxs = mutils.tokenize_train(train[dataset_name], test[dataset_name])
            engine.write_count_files((X, class_labels, key_idxs, value_idxs))

        filepaths = engine.multi_pickle_dump((X[c], c) for c in class_labels)
        pool = mp.Pool(np.min((20, len(class_labels))))
        outputs = None
        try:
            if distribution == 'DM':
                outputs = pool.map(engine.dm_mle_parallel, filepaths)
            elif distribution == 'BLM':
                outputs = pool.map(engine.blm_mle_parallel, filepaths)
            elif distribution == 'GDM':
                pass  # TODO: Implement
        finally:
            pool.close()
            pool.join()

        mle_results = engine.load_mle_results(outputs)
        simplex_matrix = np.zeros((len(value_idxs), len(class_labels)), dtype=np.float64)
        for label, simplex in mle_results:
            simplex_matrix[:, key_idxs[label]] = simplex

    if not test:
        test = mutils.load_data(test_path)

    for method in smoothing_methods:
        if method == 'lidstone':
            smoothed = sm.lidstone_smoothing(simplex_matrix, X, class_labels)
            param_string = 'n=1'
            mutils.output_results_naive_bayes(smoothed, test[dataset_name], class_labels, key_idxs, value_idxs,
                                              dataset_name, distribution, method, param_string, result_file, batch_size)
        elif method == 'dirichlet':
            for alpha in dirichlet_grid:
                smoothed = sm.dirichlet_smoothing(simplex_matrix, X, class_labels, alpha)
                param_string = 'alpha={}'.format(alpha)
                mutils.output_results_naive_bayes(smoothed, test[dataset_name], class_labels, key_idxs, value_idxs,
                                                  dataset_name, distribution, method, param_string, result_file,
                                                  batch_size)
        elif method == 'jm':
            for beta in jm_grid:
                smoothed = sm.jelinek_mercer_smoothing(simplex_matrix, X, class_labels, beta)
                param_string = 'beta={}'.format(beta)
                mutils.output_results_naive_bayes(smoothed, test[dataset_name], class_labels, key_idxs, value_idxs,
                                                  dataset_name, distribution, method, param_string, result_file,
                                                  batch_size)
        elif method == 'ad':
            for delta in ad_grid:
                smoothed = sm.absolute_discounting_smoothing(simplex_matrix, X, class_labels, delta)
                param_string = 'delta={}'.format(delta)
                mutils.output_results_naive_bayes(smoothed, test[dataset_name], class_labels, key_idxs, value_idxs,
                                                  dataset_name, distribution, method, param_string, result_file,
                                                  batch_size)
        elif method == 'ts':
            for mu in ts_grid[0]:
                for beta in ts_grid[1]:
                    smoothed = sm.two_step_smoothing(simplex_matrix, X, class_labels, mu, beta)
                    param_string = 'mu={}|beta={}'.format(mu, beta)
                    mutils.output_results_naive_bayes(smoothed, test[dataset_name], class_labels, key_idxs, value_idxs,
                                                      dataset_name, distribution, method, param_string, result_file,
                                                      batch_size)


if __name__ == '__main__':
    train_dir_path = sys.argv[1]
    test_dir_path = sys.argv[2]
    result_file_path = sys.argv[3]
    temp_dir_path = sys.argv[4]

    datasets = [x.split('/')[-1].split('-')[0] for x in glob.glob(train_dir_path + '/*')]
    print('Dataset names: {}'.format(datasets))

    for d in datasets:
        for distribution in ('pooledDM', 'DM', 'BLM'):
            test_accuracy(distribution, train_dir_path, test_dir_path, d, result_file_path, temp_dir_path)

