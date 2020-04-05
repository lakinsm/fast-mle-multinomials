#!/usr/bin/env python3

import numpy as np
import sys
import multi_mle.dm as dm
import multi_mle.blm as blm
import multi_mle.gdm as gdm
import multi_mle.smoothing as sm
import multiprocessing as mp
import multi_mle.mle_utils as mutils
# from multi_mle.mle_utils import silent_remove

np.set_printoptions(suppress=True)
np.random.seed(2718)

# MLE/NB params
delta_eps_threshold = 1e-5
learn_rate_threshold = 2e-10
delta_lprob_threshold = 1e-5
max_steps = 15
batch_size = 1000

# Smoothing params
smoothing_methods = ('lidstone',)


def small_test():
    X = np.zeros((10, 5), dtype=np.int64)

    # Choose one row for every feature to initialize as 1
    col_idxs = list(range(X.shape[1]))
    row_idxs = list(range(X.shape[0]))
    if len(row_idxs) > len(col_idxs):
        col_idxs += [x % X.shape[1] for x in range(X.shape[0] - X.shape[1])]
    print(row_idxs, col_idxs)
    X[row_idxs, col_idxs] = 1

    # Add in random data based on a zero-inflated Poisson distributions
    for j in range(X.shape[1]):
        if j < 2:
            idxs, draws = mutils.r_zero_inflated_poisson(0.8, 50, X.shape[0])
            X[idxs, j] += draws
        # elif (j >= 2) and (j < 8):
        #     idxs, draws = r_zero_inflated_poisson(0.2, 5, X.shape[0])
        #     X[idxs, j] += draws
        else:
            idxs, draws = mutils.r_zero_inflated_poisson(0.5, 1, X.shape[0])
            X[idxs, j] += draws

    U, vd, vd1 = blm.blm_precalc(X)
    thetas = blm.blm_init_params(X)
    blm_mle = blm.blm_newton_raphson2(U, vd, vd1, thetas, max_steps, delta_eps_threshold, delta_lprob_threshold, True)

    U, vd = dm.dm_precalc(X)
    thetas = dm.dm_init_params(X)
    dm_mle = dm.dm_newton_raphson2(U, vd, thetas, max_steps, delta_eps_threshold, delta_lprob_threshold, True)

    print(dm_mle, "Sum: {}".format(np.sum(dm_mle)))
    print(blm_mle, "Sum: {}".format(np.sum(blm_mle)))
    print(np.sum(X, axis=0) / np.sum(np.sum(X, axis=0)), "Sum: 1")
    print(X)

    # U, vd = dm.dm_precalc(X)
    # thetas = np.array([100] * 8, dtype=np.float64)
    # dm_mle2 = dm.dm_newton_raphson(U, vd, thetas, max_steps, gradient_sq_threshold, learn_rate_threshold,
    #                               delta_lprob_threshold)
    # print(dm_mle2)


def smooth_test():
    train = mutils.load_data('/mnt/phd_repositories/fast-mle-multinomials/data/debug/smooth')
    test = mutils.load_data('/mnt/phd_repositories/fast-mle-multinomials/data/debug/smooth')

    X, class_labels, key_idxs, value_idxs = mutils.tokenize_train(train['ngram_test.txt'], test['ngram_test.txt'])
    simplex_matrix = np.zeros((len(value_idxs), len(class_labels)), dtype=np.float64)
    for i, c in enumerate(class_labels):
        class_simplex = np.squeeze(np.sum(X[c][0], axis=0))
        class_simplex = class_simplex / np.sum(class_simplex)
        for j, p in enumerate(class_simplex):
            simplex_matrix[X[c][1][j], i] = p
    smoothed = sm.lidstone_smoothing(simplex_matrix, X, class_labels)
    print(smoothed)
    print(np.sum(smoothed, axis=0))


def mp_test():
    train = mutils.load_data('/mnt/phd_repositories/fast-mle-multinomials/data/smooth/train/')
    test = mutils.load_data('/mnt/phd_repositories/fast-mle-multinomials/data/debug/test/')

    X, class_labels, key_idxs, value_idxs = mutils.tokenize_train(train['cade'], test['cade'])
    simplex_matrix = np.zeros((len(value_idxs), len(class_labels)), dtype=np.float64)

    engine = mutils.MLEngine(max_steps, delta_eps_threshold, delta_lprob_threshold, '/mnt/temp/pkl', True)
    filepaths = engine.multi_pickle_dump((X[c][0], c) for c in class_labels)

    pool = mp.Pool(10)
    try:
        outputs = pool.map(engine.dm_mle_parallel, filepaths)
        for class_simplex, label in outputs:
            for j, p in enumerate(class_simplex):
                simplex_matrix[X[label][1][j], key_idxs[label]] = p
        print(simplex_matrix)
    finally:
        pool.close()
        pool.join()


def mp_test2():
    engine = mutils.MLEngine(max_steps, delta_eps_threshold, delta_lprob_threshold, '/mnt/temp/mle', 'ngram', True)
    test = None
    if engine.count_files_exist:
        X, class_labels, key_idxs, value_idxs = engine.load_count_files()
    else:
        train = mutils.load_data('/mnt/phd_repositories/fast-mle-multinomials/data/debug/smooth/')
        test = mutils.load_data('/mnt/phd_repositories/fast-mle-multinomials/data/debug/smooth/')
        X, class_labels, key_idxs, value_idxs = mutils.tokenize_train(train['ngram'], test['ngram'])
        engine.write_count_files((X, class_labels, key_idxs, value_idxs))

    filepaths = engine.multi_pickle_dump((X[c], c) for c in class_labels)
    pool = mp.Pool(np.min((10, len(class_labels))))
    try:
        outputs = pool.map(engine.blm_mle_parallel, filepaths)
    finally:
        pool.close()
        pool.join()

    mle_results = engine.load_mle_results(outputs)
    simplex_matrix = np.zeros((len(value_idxs), len(class_labels)), dtype=np.float64)
    for label, simplex in mle_results:
        simplex_matrix[:, key_idxs[label]] = simplex

    if not test:
        test = mutils.load_data('/mnt/phd_repositories/fast-mle-multinomials/data/debug/smooth/')

    smoothed = sm.lidstone_smoothing(simplex_matrix, X, class_labels)

    mutils.output_results_naive_bayes(smoothed, test['ngram'], class_labels, key_idxs, value_idxs,
                                      'ngram', 'BLM', 'Lidstone', 'n=1', '/mnt/temp/my_results.txt', batch_size)


def nb_test():
    train = mutils.load_data('/mnt/phd_repositories/fast-mle-multinomials/data/debug/train')
    test = mutils.load_data('/mnt/phd_repositories/fast-mle-multinomials/data/debug/test')
    X, class_labels, key_idxs, value_idxs = mutils.tokenize_train(train['cade'], test['cade'])

    simplex_matrix = np.zeros((len(value_idxs), len(class_labels)), dtype=np.float64)
    for i, c in enumerate(class_labels):
        class_simplex = np.squeeze(np.sum(X[c][0], axis=0))
        class_simplex = class_simplex / np.sum(class_simplex)
        for j, p in enumerate(class_simplex):
            simplex_matrix[X[c][1][j], i] = p

    smoothed_matrix = sm.two_step_smoothing(simplex_matrix, X, class_labels)

    param_string = 'n=1'
    mutils.output_results_naive_bayes(smoothed_matrix, test['cade'], class_labels, key_idxs, value_idxs,
                                      'cade', 'BLM', 'Lidstone', param_string, '/mnt/temp/my_results.txt', batch_size)


def test_accuracy(distribution, train_path, test_path, dataset_name, result_file, timing_result_file_path, temp_dir,
                  posterior_method=None, precompute=None, n_threads=4):
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
                                 posterior_method, precompute_method=precompute, verbose=True)
        if engine.count_files_exist:
            X, class_labels, key_idxs, value_idxs = engine.load_count_files()
        else:
            train = mutils.load_data(train_path)
            test = mutils.load_data(test_path)
            X, class_labels, key_idxs, value_idxs = mutils.tokenize_train(train[dataset_name], test[dataset_name])
            engine.write_count_files((X, class_labels, key_idxs, value_idxs))

        filepaths = engine.multi_pickle_dump((X[c], c) for c in class_labels)
        pool = mp.Pool(np.min((n_threads, len(class_labels))))
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
        timings = engine.load_timing_results(distribution.lower(), precompute)
        with open(timing_result_file_path, 'a') as time_out:
            for vals in timings:
                time_out.write('{},{},{},{},{},{},{}\n'.format(
                    dataset_name,
                    distribution,
                    precompute,
                    vals[0],  # class label
                    vals[1],  # number of observations
                    vals[2],  # dimensionality
                    vals[3]  # time
                ))

        assert(len(mle_results) == len(class_labels))
        if posterior_method == 'aposteriori':
            simplex_matrix = np.zeros((len(value_idxs)+1, len(class_labels)), dtype=np.float64)  # for BLM only atm
        else:
            simplex_matrix = np.zeros((len(value_idxs), len(class_labels)), dtype=np.float64)
        for label, simplex in mle_results:
            simplex_matrix[:, key_idxs[label]] = simplex

    if not test:
        test = mutils.load_data(test_path)

    if not posterior_method:
        smoothed = sm.lidstone_smoothing(simplex_matrix, X, class_labels)
    else:
        smoothed = simplex_matrix
    param_string = 'n=1'
    mutils.output_results_naive_bayes(smoothed, test[dataset_name], class_labels, key_idxs, value_idxs,
                                      dataset_name, distribution, 'Lidstone', param_string, precompute, result_file,
                                      posterior_method, batch_size)


def parameter_estimation_test():
    # Standard MAP MLE Mode for PooledDM, DM, and BLM
    # test_accuracy(distribution='pooledDM',
    #               train_path='/mnt/phd_repositories/fast-mle-multinomials/data/debug/train/',
    #               test_path='/mnt/phd_repositories/fast-mle-multinomials/data/debug/test/',
    #               dataset_name='r8',
    #               result_file='/mnt/temp/2020Apr1_mle_parameter_estimation_test.txt',
    #               temp_dir='/mnt/temp/2020Apr1_mle')

    # test_accuracy(distribution='BLM',
    #               train_path='/mnt/phd_repositories/fast-mle-multinomials/data/debug/train/',
    #               test_path='/mnt/phd_repositories/fast-mle-multinomials/data/debug/test/',
    #               dataset_name='r8',
    #               result_file='/mnt/temp/2020Apr1_mle_parameter_estimation_test.txt',
    #               timing_result_file_path='/mnt/temp/2020Apr1_mle_timings.csv',
    #               temp_dir='/mnt/temp/2020Apr1_mle',
    #               precompute='approximate',
    #               n_threads=15)

    test_accuracy(distribution='BLM',
                  train_path='/mnt/phd_repositories/fast-mle-multinomials/data/debug/train',
                  test_path='/mnt/phd_repositories/fast-mle-multinomials/data/debug/test/',
                  dataset_name='r8',
                  result_file='/mnt/temp/2020Apr1_mle_parameter_estimation_test.txt',
                  timing_result_file_path='/mnt/temp/2020Apr1_mle_timings.csv',
                  temp_dir='/mnt/temp/2020Apr1_mle',
                  precompute='vectorized',
                  posterior_method='aposteriori',
                  n_threads=15)

    # test_accuracy(distribution='BLM',
    #               train_path='/mnt/phd_repositories/fast-mle-multinomials/data/debug/train/',
    #               test_path='/mnt/phd_repositories/fast-mle-multinomials/data/debug/test/',
    #               dataset_name='r8',
    #               result_file='/mnt/temp/parameter_estimation_test.txt',
    #               temp_dir='/mnt/temp/2020Jan13_mle')
    #
    # # Empirical Bayes Mode for BLM
    # test_accuracy(distribution='BLM',
    #               train_path='/mnt/phd_repositories/fast-mle-multinomials/data/debug/train/',
    #               test_path='/mnt/phd_repositories/fast-mle-multinomials/data/debug/test/',
    #               dataset_name='r8',
    #               result_file='/mnt/temp/parameter_estimation_test.txt',
    #               temp_dir='/mnt/temp/2020Jan20_mle',
    #               posterior_method='empirical')
    #
    # test_accuracy(distribution='BLM',
    #               train_path='/mnt/phd_repositories/fast-mle-multinomials/data/debug/train/',
    #               test_path='/mnt/phd_repositories/fast-mle-multinomials/data/debug/test/',
    #               dataset_name='cade',
    #               result_file='/mnt/temp/parameter_estimation_test.txt',
    #               temp_dir='/mnt/temp/2020Jan13_mle',
    #               posterior_method='aposteriori')


if __name__ == '__main__':
    parameter_estimation_test()
