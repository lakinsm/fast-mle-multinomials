#!/usr/bin/env python3
import numpy as np
import sys
import glob
import multi_mle.mle_utils as mutils
import multiprocessing as mp
import multi_mle.smoothing as sm


np.random.seed(2718)

# MLE/NB params
delta_eps_threshold = 1e-26
learn_rate_threshold = 2e-10
delta_lprob_threshold = 1e-5
max_steps = 100
batch_size = 1000

# Smoothing params
smoothing_methods = ['lidstone']
dirichlet_grid = np.arange(0.1, 1, 0.1)
jm_grid = np.arange(0.1, 1, 0.1)
ad_grid = np.array([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2])
ts_grid = (np.arange(0.1, 1, 0.1), np.arange(0.1, 1, 0.1))


def test_accuracy(distribution, train_path, test_path, dataset_name, result_file, timing_file, temp_dir,
                  precompute=None, posterior_method=None, threads=4):
    simplex_matrix, test, X, class_labels, smoothed, key_idxs, value_idxs = (None,) * 7

    if distribution == 'pooledDM':
        train = mutils.load_data(train_path)
        test = mutils.load_data(test_path)
        X, class_labels, key_idxs, value_idxs = mutils.tokenize_train(train[dataset_name], test[dataset_name])

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
        engine = mutils.MLEngine(max_steps, delta_eps_threshold, delta_lprob_threshold, temp_dir, dataset_name,
                                 precompute_method=precompute, posterior_method=posterior_method, verbose=True)
        if engine.count_files_exist:
            X, class_labels, key_idxs, value_idxs = engine.load_count_files()
        else:
            train = mutils.load_data(train_path)
            test = mutils.load_data(test_path)
            X, class_labels, key_idxs, value_idxs = mutils.tokenize_train(train[dataset_name], test[dataset_name])
            engine.write_count_files((X, class_labels, key_idxs, value_idxs))

        filepaths = engine.multi_pickle_dump((X[c], c) for c in class_labels)
        pool = mp.Pool(np.min((threads, len(class_labels))))
        try:
            if distribution == 'DM':
                outputs = pool.map(engine.dm_mle_parallel, filepaths)
            elif distribution == 'BLM':
                outputs = pool.map(engine.blm_mle_parallel, filepaths)
            else:
                raise ValueError('Distribution must be one of [pooledDM, DM, BLM]. Provided: {}'.format(distribution))
        finally:
            pool.close()
            pool.join()

        mle_results = engine.load_mle_results(outputs)
        timings = engine.load_timing_results(distribution)
        with open(timing_file, 'a') as time_out:
            for vals in timings:
                time_out.write('{},{},{},{},{},{},{},{}\n'.format(
                    dataset_name,
                    distribution,
                    precompute,
                    posterior_method,
                    vals[0],  # class label
                    vals[1],  # number of observations
                    vals[2],  # dimensionality
                    vals[3]  # time
                ))

        assert(len(mle_results) == len(class_labels))
        if distribution == 'BLM' and posterior_method == 'aposteriori':
            simplex_matrix = np.zeros((len(value_idxs) + 1, len(class_labels)), dtype=np.float64)
        else:
            simplex_matrix = np.zeros((len(value_idxs), len(class_labels)), dtype=np.float64)
        for label, simplex in mle_results:
            simplex_matrix[:, key_idxs[label]] = simplex

    if not test:
        test = mutils.load_data(test_path)

    if posterior_method == 'aposteriori':
        param_string = 'n=1'
        method = 'lidstone'
        mutils.output_results_naive_bayes(simplex_matrix, test[dataset_name], class_labels, key_idxs, value_idxs,
                                          dataset_name, distribution, method, param_string, precompute, result_file,
                                          posterior_method, batch_size)
    else:
        for method in smoothing_methods:
            if method == 'lidstone':
                smoothed = sm.lidstone_smoothing(simplex_matrix, X, class_labels)
                param_string = 'n=1'
                mutils.output_results_naive_bayes(smoothed, test[dataset_name], class_labels, key_idxs, value_idxs,
                                                  dataset_name, distribution, method, param_string, precompute,
                                                  result_file, posterior_method, batch_size)
            elif method == 'dirichlet':
                for alpha in dirichlet_grid:
                    smoothed = sm.dirichlet_smoothing(simplex_matrix, X, class_labels, alpha)
                    param_string = 'alpha={}'.format(alpha)
                    mutils.output_results_naive_bayes(smoothed, test[dataset_name], class_labels, key_idxs, value_idxs,
                                                      dataset_name, distribution, method, param_string, precompute,
                                                      result_file, posterior_method, batch_size)
            elif method == 'jm':
                for beta in jm_grid:
                    smoothed = sm.jelinek_mercer_smoothing(simplex_matrix, X, class_labels, beta)
                    param_string = 'beta={}'.format(beta)
                    mutils.output_results_naive_bayes(smoothed, test[dataset_name], class_labels, key_idxs, value_idxs,
                                                      dataset_name, distribution, method, param_string, precompute,
                                                      result_file, posterior_method, batch_size)
            elif method == 'ad':
                for delta in ad_grid:
                    smoothed = sm.absolute_discounting_smoothing(simplex_matrix, X, class_labels, delta)
                    param_string = 'delta={}'.format(delta)
                    mutils.output_results_naive_bayes(smoothed, test[dataset_name], class_labels, key_idxs, value_idxs,
                                                      dataset_name, distribution, method, param_string, precompute,
                                                      result_file, posterior_method, batch_size)
            elif method == 'ts':
                for mu in ts_grid[0]:
                    for beta in ts_grid[1]:
                        smoothed = sm.two_step_smoothing(simplex_matrix, X, class_labels, mu, beta)
                        param_string = 'mu={}|beta={}'.format(mu, beta)
                        mutils.output_results_naive_bayes(smoothed, test[dataset_name], class_labels, key_idxs, value_idxs,
                                                          dataset_name, distribution, method, param_string, precompute,
                                                          result_file, posterior_method, batch_size)


if __name__ == '__main__':
    train_dir_path = sys.argv[1]
    test_dir_path = sys.argv[2]
    result_file_path = sys.argv[3]
    timing_result_file_path = sys.argv[4]
    temp_dir_path = sys.argv[5]
    n_threads = int(sys.argv[6])

    datasets = [x.split('/')[-1].split('-')[0] for x in glob.glob(train_dir_path + '/*')]
    print('Dataset names: {}'.format(datasets))

    for d in datasets:
        for distribution in ('DM', 'BLM'):
            if distribution == 'BLM' or distribution == 'DM':
                for post_method in (None, 'empirical', 'aposteriori'):
                    test_accuracy(distribution, train_dir_path, test_dir_path, d, result_file_path,
                                  timing_result_file_path,
                                  temp_dir_path, precompute='vectorized', posterior_method=post_method,
                                  threads=n_threads)
                    # test_accuracy(distribution, train_dir_path, test_dir_path, d, result_file_path,
                    #               timing_result_file_path,
                    #               temp_dir_path, precompute='approximate', posterior_method=post_method,
                    #               threads=n_threads)
                    # test_accuracy(distribution, train_dir_path, test_dir_path, d, result_file_path,
                    #               timing_result_file_path,
                    #               temp_dir_path, precompute='sklar', posterior_method=post_method,
                    #               threads=n_threads)
            elif distribution == 'pooledDM':
                test_accuracy(distribution, train_dir_path, test_dir_path, d, result_file_path, timing_result_file_path,
                              temp_dir_path, threads=n_threads)
            else:
                raise ValueError('Distribution must be one of [pooledDM, DM, BLM]. Provided: {}'.format(distribution))

