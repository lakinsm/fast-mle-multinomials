#!/usr/bin/env python3

import numpy as np
import sys
import multi_mle.dm as dm
import multi_mle.blm as blm
import multi_mle.gdm as gdm
import multi_mle.smoothing as sm
import glob
# from multi_mle.mle_utils import silent_remove

np.set_printoptions(suppress=True)
np.random.seed(2718)

delta_eps_threshold = 1e-5
learn_rate_threshold = 2e-10
delta_lprob_threshold = 1e-5
max_steps = 15
batch_size = 1000


def load_data(folder):
    all_datasets = {}
    pypath = folder.replace('\\\\', '/')
    for dataset in glob.glob(pypath + '/*'):
        pydatapath = dataset.replace('\\', '/')
        dataset = pydatapath.split('/')[-1].split('-')[0]
        all_datasets.setdefault(dataset, ())
        with open(pydatapath, 'r') as f:
            data = f.read().split('\n')
            for line in data:
                if not line:
                    continue
                truth_label, words = line.split('\t')
                # if len(words) < 3:  # filter out low quality training observations
                #     continue
                all_datasets[dataset] += ((truth_label, words.split()),)
    return all_datasets


def tokenize_train(train, test):
    train_zipped = tuple(zip(*train))
    test_zipped = tuple(zip(*test))
    uniq_values = set([x for y in train_zipped[1] for x in y])
    uniq_values.add(tuple(x for y in test_zipped[1] for x in y))
    uniq_keys = set(train_zipped[0])
    value_idxs = {k: i for i, k in enumerate(uniq_values)}
    key_idxs = {k: i for i, k in enumerate(uniq_keys)}
    class_labels = tuple(k for _, k in (sorted(((i, k) for k, i in key_idxs.items()))))
    key_observed_count = {k: 0 for k in uniq_keys}

    print(key_idxs)
    print(class_labels)

    tokenized = {k: [np.zeros((train_zipped[0].count(k), len(uniq_values)), dtype=np.int64), {}] for k in uniq_keys}
    for truth_label, words in train:
        for w in words:
            tokenized[truth_label][0][key_observed_count[truth_label], value_idxs[w]] += 1
        key_observed_count[truth_label] += 1
    # Remove zero-sum columns (to be reinserted as zeros in the MLE simplex after MLE computation)
    for k in uniq_keys:
        nonzero_idx = 0
        zero_idxs = set()
        for j in range(len(uniq_values)):
            if np.sum(tokenized[k][0][:, j]) > 0:
                tokenized[k][1].setdefault(nonzero_idx, j)
                nonzero_idx += 1
            else:
                zero_idxs.add(j)
        if zero_idxs:
            tokenized[k][0] = np.delete(tokenized[k][0], np.array(list(zero_idxs)), axis=1)
    return tokenized, class_labels, key_idxs, value_idxs


def r_zero_inflated_poisson(zero_inflation_prob, poisson_mean, n):
    non_zero_idxs = np.random.binomial(1, zero_inflation_prob, n)
    counts = np.random.poisson(poisson_mean, np.sum(non_zero_idxs))
    return non_zero_idxs > 0, counts


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
            idxs, draws = r_zero_inflated_poisson(0.8, 50, X.shape[0])
            X[idxs, j] += draws
        # elif (j >= 2) and (j < 8):
        #     idxs, draws = r_zero_inflated_poisson(0.2, 5, X.shape[0])
        #     X[idxs, j] += draws
        else:
            idxs, draws = r_zero_inflated_poisson(0.5, 1, X.shape[0])
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
    train = load_data('/mnt/phd_repositories/fast-mle-multinomials/data/debug/smooth')
    test = load_data('/mnt/phd_repositories/fast-mle-multinomials/data/debug/smooth')

    X, class_labels, key_idxs, value_idxs = tokenize_train(train['ngram_test.txt'], test['ngram_test.txt'])
    simplex_matrix = np.zeros((len(value_idxs), len(class_labels)), dtype=np.float64)
    for i, c in enumerate(class_labels):
        class_simplex = np.squeeze(np.sum(X[c][0], axis=0))
        class_simplex = class_simplex / np.sum(class_simplex)
        for j, p in enumerate(class_simplex):
            simplex_matrix[X[c][1][j], i] = p
    print(simplex_matrix)
    print(sm.two_step_smoothing(simplex_matrix, X, class_labels))


if __name__ == '__main__':
    smooth_test()
