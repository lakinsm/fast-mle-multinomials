#!/usr/bin/env python3
import numpy as np
import sys
import glob
import multi_mle.dm as dm
import multi_mle.blm as blm
import multi_mle.gdm as gdm
# from multi_mle.mle_utils import silent_remove


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


def tokenize_test(observation, value_idxs):
    truth_label = observation[0]
    tokenized = np.zeros(len(value_idxs))
    for w in observation[1]:
        if w not in value_idxs:
            continue
        tokenized[value_idxs[w]] += 1
    return tokenized, truth_label


def compute_naive_bayes(D, X_t, d_labels, x_labels, key_idxs, accuracy_matrix):
    classifications = [x_labels[c] for c in np.argmax(np.matmul(D, X_t), axis=1)]
    for i, c in enumerate(classifications):
        for a in range(len(accuracy_matrix)):
            accuracy_matrix[a][3] += 1  # add true negative to all classes
        if c == d_labels[i]:
            accuracy_matrix[key_idxs[c]][0] += 1  # true positive
        else:
            accuracy_matrix[key_idxs[c]][1] += 1  # false positive
            accuracy_matrix[key_idxs[d_labels[i]]][2] += 1  # false negative
            accuracy_matrix[key_idxs[d_labels[i]]][3] -= 1  # remove a true negative from d_labels[i]
        accuracy_matrix[key_idxs[c]][3] -= 1  # remove a true negative from c
    return accuracy_matrix


def test_accuracy(distribution, train, test, dataset_name, smoothing_method, result_file):
    X, class_labels, key_idxs, value_idxs = tokenize_train(train, test)

    simplex_matrix = np.zeros((len(value_idxs), len(class_labels)), dtype=np.float64)
    accuracy_matrix = [[0, 0, 0, 0] for _ in class_labels]

    if distribution == 'DM':
        for i, c in enumerate(class_labels):
            U, v = dm.dm_precalc(X[c][0])
            params = dm.dm_init_params(X[c][0])
            mle = dm.dm_newton_raphson2(U, v, params, max_steps, delta_eps_threshold, delta_lprob_threshold, True)
            class_simplex = dm.dm_renormalize(mle)
            for j, p in enumerate(class_simplex):
                simplex_matrix[X[c][1][j], i] = p

    elif distribution == 'BLM':
        for i, c in enumerate(class_labels):
            U, vd, vd1 = blm.blm_precalc(X[c][0])
            params = blm.blm_init_params(X[c][0])
            mle = blm.blm_newton_raphson2(U, vd, vd1, params, max_steps, delta_eps_threshold,
                                          delta_lprob_threshold, True)
            class_simplex = blm.blm_renormalize(mle)
            for j, p in enumerate(class_simplex):
                simplex_matrix[X[c][1][j], i] = p

    elif distribution == 'GDM':
        for i, c in enumerate(class_labels):
            U, vd = gdm.gdm_precalc(X[c][0])
            params = gdm.gdm_init_params(X[c][0])
            mle = gdm.gdm_newton_raphson(U, vd, params, max_steps, delta_eps_threshold,
                                         learn_rate_threshold, delta_lprob_threshold)
            class_simplex = gdm.gdm_renormalize(mle)
            for j, p in enumerate(class_simplex):
                simplex_matrix[X[c][1][j], i] = p

    elif distribution == 'pooledDM':
        for i, c in enumerate(class_labels):
            class_simplex = np.squeeze(np.sum(X[c][0], axis=0))
            class_simplex = class_simplex / np.sum(class_simplex)
            for j, p in enumerate(class_simplex):
                simplex_matrix[X[c][1][j], i] = p

    else:
        raise ValueError('Distribution must be one of DM, BLM, or GDM.  Provided: {}'.format(distribution))

    if smoothing_method == 'Lidstone':
        pass
    elif smoothing_method == 'Dirichlet':
        pass
    elif smoothing_method == 'JM' or smoothing_method == 'Jelinek-Mercer':
        pass
    elif smoothing_method == 'AD' or smoothing_method == 'Absolute Discounting':
        pass
    elif smoothing_method == 'TS' or smoothing_method == 'Two-stage':
        pass
    elif smoothing_method == 'All':
        pass
    else:
        if smoothing_method != 'None':
            sys.stderr.write('Smoothing method must be one of: '
                             '[Lidstone, Dirichlet, Jelinek-Mercer, Absolute Discounting, Two-stage, All]')
            sys.exit(1)

    if smoothing_method == 'All':
        pass
    else:
        observations_processed = 0
        while observations_processed < len(test):
            if observations_processed + batch_size <= len(test):
                ndim = batch_size
            else:
                ndim = len(test) - observations_processed
            observed_matrix = np.zeros((ndim, len(value_idxs)), dtype=np.int64)
            observed_labels = ()

            for i in range(0, ndim):
                observed_matrix[i, :], truth_label = tokenize_test(test[observations_processed + i], value_idxs)
                observed_labels += (truth_label,)
            accuracy_matrix = compute_naive_bayes(observed_matrix, simplex_matrix, observed_labels, class_labels, key_idxs,
                                                  accuracy_matrix)
            observations_processed += ndim
            print("Observations processed: {}".format(observations_processed))

            for c in accuracy_matrix:
                print(c)
            print('\n')

            with open(result_file, 'a') as out:
                for c in range(len(accuracy_matrix)):
                    out.write('{},{},{},{},{}\n'.format(
                        dataset_name,
                        distribution,
                        smoothing_method,
                        class_labels[c],
                        ','.join([str(x) for x in accuracy_matrix[c]])
                    ))


if __name__ == '__main__':
    train = load_data(sys.argv[1])
    test = load_data(sys.argv[2])
    datasets = train.keys()

    test_accuracy('DM', train['cade'], test['cade'], 'cade', 'All', sys.argv[3])
    test_accuracy('BLM', train['cade'], test['cade'], 'cade', 'All', sys.argv[3])

    # for d in datasets:
    #     for distribution in ('pooledDM', 'DM', 'BLM', 'GDM'):
    #         test_accuracy(distribution, train[d], test[d], d, True, sys.argv[3])

