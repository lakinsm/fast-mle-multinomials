import numpy as np
import random
import glob
from multi_mle import dm
from multi_mle import blm
from multi_mle import gdm
import pickle
import os
from scipy.special import gamma
import sys
import timeit


class MLEngine(object):
    def __init__(self,
                 max_steps,
                 delta_eps_threshold,
                 delta_lprob_threshold,
                 temp_dir,
                 dataset,
                 posterior_method=None,
                 precompute_method=None,
                 verbose=False):
        # User defined parameters
        self.max_steps = max_steps
        self.delta_eps_threshold = delta_eps_threshold
        self.delta_lprob_threshold = delta_lprob_threshold
        self.temp_dir = temp_dir
        self.dataset = dataset
        self.posterior_method = posterior_method
        self.precompute_method = precompute_method
        self.timings = []
        self.verbose = verbose

        # Determine which jobs may need to be computed based on provided temp_dir
        self.count_files_exist = self._check_count_files()
        self._setup_working_dir()

        # Misc vars
        self._error_status = None

    def _check_count_files(self):
        return all([os.path.exists('{}/{}/{}.pickle'.format(self.temp_dir, self.dataset, x))
                    for x in ('X', 'class_labels', 'key_idxs', 'value_idxs')])

    def _setup_working_dir(self):
        if not os.path.isdir('{}'.format(self.temp_dir)):
            os.mkdir('{}'.format(self.temp_dir))
        if not os.path.isdir('{}/{}'.format(self.temp_dir, self.dataset)):
            os.mkdir('{}/{}'.format(self.temp_dir, self.dataset))
        if not os.path.isdir('{}/{}/dm'.format(self.temp_dir, self.dataset)):
            os.mkdir('{}/{}/dm'.format(self.temp_dir, self.dataset))
        if not os.path.isdir('{}/{}/dm/{}'.format(self.temp_dir, self.dataset, self.precompute_method)):
            os.mkdir('{}/{}/dm/{}'.format(self.temp_dir, self.dataset, self.precompute_method))
        if not os.path.isdir('{}/{}/dm/{}/{}'.format(self.temp_dir, self.dataset, self.precompute_method,
                                                     self.posterior_method)):
            os.mkdir('{}/{}/dm/{}/{}'.format(self.temp_dir, self.dataset, self.precompute_method,
                                             self.posterior_method))
        if not os.path.isdir('{}/{}/blm'.format(self.temp_dir, self.dataset)):
            os.mkdir('{}/{}/blm'.format(self.temp_dir, self.dataset))
        if not os.path.isdir('{}/{}/blm/{}'.format(self.temp_dir, self.dataset, self.precompute_method)):
            os.mkdir('{}/{}/blm/{}'.format(self.temp_dir, self.dataset, self.precompute_method))
        if not os.path.isdir('{}/{}/blm/{}/{}'.format(self.temp_dir, self.dataset, self.precompute_method,
                                                      self.posterior_method)):
            os.mkdir('{}/{}/blm/{}/{}'.format(self.temp_dir, self.dataset, self.precompute_method,
                                              self.posterior_method))
        if not os.path.isdir('{}/{}/matrices'.format(self.temp_dir, self.dataset)):
            os.mkdir('{}/{}/matrices'.format(self.temp_dir, self.dataset))

    def load_count_files(self):
        count_data = []
        if self.count_files_exist:
            filepaths = ('{}/{}/{}.pickle'.format(self.temp_dir, self.dataset, x)
                         for x in ('X', 'class_labels', 'key_idxs', 'value_idxs'))
            for path in filepaths:
                with open(path, 'rb') as infile:
                    count_data.append(pickle.load(infile))
        return count_data

    def write_count_files(self, data_vector):
        """
        Write out the data structures for the master thread.
        :param data_vector: Tuple of data structures, in order: X, class_labels, key_idxs, value_idxs
        """
        if not self.count_files_exist:
            out_paths = ['{}/{}/{}.pickle'.format(self.temp_dir, self.dataset, x)
                         for x in ('X', 'class_labels', 'key_idxs', 'value_idxs')]
            for i, path in enumerate(out_paths):
                with open(path, 'wb') as out:
                    pickle.dump(data_vector[i], out)

    def multi_pickle_dump(self, data):
        """
        Pickle the data matrices required for parallel MLE, so that forked threads can read them separately.
        :param data: Array of tuples, (data_matrix, class_label) to be pickled
        """
        data_paths = []
        for matrix, label in data:
            file_str = '{}/{}/matrices/{}.pickle'.format(self.temp_dir, self.dataset, label)
            if not os.path.exists(file_str):
                with open(file_str, 'wb') as out:
                    pickle.dump(matrix, out)
            data_paths.append(file_str)
        return data_paths

    @staticmethod
    def load_mle_results(filepaths):
        results = []
        for f in filepaths:
            if os.path.exists(f):
                label = f.split('/')[-1].replace('.pickle', '')
                with open(f, 'rb') as infile:
                    simplex = pickle.load(infile)
                    print(label, np.sum(simplex))
                    results.append((label, simplex))
        return results

    def load_timing_results(self, method):
        ret = []
        glob_path = '{}/{}/{}/{}/*_mle_timings.csv'.format(self.temp_dir, self.dataset, method.lower(),
                                                           self.posterior_method)
        timing_files = glob.glob(glob_path)
        for filepath in timing_files:
            with open(filepath, 'r') as f:
                data = f.read().split('\n')[0].split(',')
                ret.append(data)
        return ret

    def dm_mle_parallel(self, filepath):
        """
        Execute the Dirichlet Multinomial MLE in parallel across classes.
        :param filepath: String, full path to the pickled count matrix from multi_pickle_dump()
        """
        try:
            label = filepath.split('/')[-1].replace('.pickle', '')
            mle_path = '{}/{}/dm/{}/{}.pickle'.format(self.temp_dir, self.dataset, self.precompute_method, label)
            expanded_simplex_path = '{}/{}/dm/{}/{}/{}.pickle'.format(self.temp_dir, self.dataset,
                                                                       self.precompute_method, self.posterior_method,
                                                                       label)
            timing_path = '{}/{}/dm/{}/{}_mle_timings.csv'.format(self.temp_dir, self.dataset, self.precompute_method,
                                                                  label)

            if os.path.exists(mle_path) and os.path.exists(expanded_simplex_path):
                return expanded_simplex_path
            else:
                with open(filepath, 'rb') as infile:
                    X = pickle.load(infile)

            if os.path.exists(mle_path) and not os.path.exists(expanded_simplex_path):
                mle = pickle.load(mle_path)
            else:
                params = blm.blm_init_params(X[0])
                U, vd, vd1 = blm.blm_precalc(X[0])
                start = timeit.default_timer()
                mle = dm.dm_newton_raphson2(X[0], U, vd, params,
                                              self.precompute_method,
                                              self.max_steps,
                                              self.delta_eps_threshold,
                                              self.delta_lprob_threshold,
                                              label,
                                              self.verbose)
                timing = timeit.default_timer() - start
                with open(timing_path, 'w') as time_out:
                    time_out.write('{},{},{},{}\n'.format(
                        label,  # class label
                        X[0].shape[0],  # number of observations
                        len(U),  # dimensionality
                        timing  # time
                    ))
                with open(mle_path, 'wb') as mle_out:
                    pickle.dump(mle, mle_out)

            class_simplex = None
            expanded_simplex = np.zeros(X[2], dtype=np.float64)
            if not self.posterior_method:
                class_simplex = dm.dm_renormalize(mle)
            elif self.posterior_method == 'empirical':
                class_simplex = dm.dm_renormalize_empirical(mle, np.sum(X[0], axis=0))
            elif self.posterior_method == 'aposteriori':
                for i, val in enumerate(mle):
                    expanded_simplex[X[1][i]] = val

            if not self.posterior_method == 'aposteriori':
                for i, val in enumerate(class_simplex):
                    expanded_simplex[X[1][i]] = val
            with open(expanded_simplex_path, 'wb') as out:
                pickle.dump(expanded_simplex, out)
            return expanded_simplex_path
        except Exception as e:
            raise e

    def blm_mle_parallel(self, filepath):
        """
        Execute the Beta-Liouville Multinomial MLE in parallel across classes.
        :param filepath: String, full path to the pickled file from multi_pickle_dump()
        """
        try:
            label = filepath.split('/')[-1].replace('.pickle', '')
            mle_path = '{}/{}/blm/{}/{}.pickle'.format(self.temp_dir, self.dataset, self.precompute_method, label)
            expanded_simplex_path = '{}/{}/blm/{}/{}/{}.pickle'.format(self.temp_dir, self.dataset,
                                                                       self.precompute_method, self.posterior_method,
                                                                       label)
            timing_path = '{}/{}/blm/{}/{}_mle_timings.csv'.format(self.temp_dir, self.dataset, self.precompute_method,
                                                                   label)
            if os.path.exists(mle_path) and os.path.exists(expanded_simplex_path):
                return expanded_simplex_path
            else:
                with open(filepath, 'rb') as infile:
                    X = pickle.load(infile)

            if os.path.exists(mle_path) and not os.path.exists(expanded_simplex_path):
                mle = pickle.load(mle_path)
            else:
                params = blm.blm_init_params(X[0])
                U, vd, vd1 = blm.blm_precalc(X[0])
                start = timeit.default_timer()
                mle = blm.blm_newton_raphson2(X[0], U, vd, vd1, params,
                                              self.precompute_method,
                                              self.max_steps,
                                              self.delta_eps_threshold,
                                              self.delta_lprob_threshold,
                                              label,
                                              self.verbose)
                timing = timeit.default_timer() - start
                with open(timing_path, 'w') as time_out:
                    time_out.write('{},{},{},{}\n'.format(
                        label,  # class label
                        X[0].shape[0],  # number of observations
                        len(U),  # dimensionality
                        timing  # time
                    ))
                with open(mle_path, 'wb') as mle_out:
                    pickle.dump(mle, mle_out)

            class_simplex = None
            expanded_simplex = None
            if not self.posterior_method:
                class_simplex = blm.blm_renormalize(mle)
            elif self.posterior_method == 'empirical':
                class_simplex = blm.blm_renormalize_empirical(mle, np.sum(X[0], axis=0))
            elif self.posterior_method == 'aposteriori':
                expanded_simplex = np.zeros(X[2]+1, dtype=np.float64)
                for i, val in enumerate(mle[:-1]):
                    expanded_simplex[X[1][i]] = val
                expanded_simplex[-1] = mle[-1]

            if not self.posterior_method == 'aposteriori':
                expanded_simplex = np.zeros(X[2], dtype=np.float64)
                for i, val in enumerate(class_simplex):
                    expanded_simplex[X[1][i]] = val
            with open(expanded_simplex_path, 'wb') as out:
                pickle.dump(expanded_simplex, out)
            return expanded_simplex_path
        except Exception as e:
            raise e


def multinomial_random_sample(N, M, theta):
    """
    Sample from a multinomial distribution with parameters theta.
    :param N: integer, number of multinomials to generate
    :param M: integer, number of samples per multinomial
    :param theta: vector of parameters in dimension D+1
    :return: count matrix X, dimension (N, D+1)
    """
    D1 = len(theta)
    multinomials = np.array([[random.normalvariate(theta[d], 0.01) for d in range(D1)] for _ in range(N)])
    multinomials[multinomials <= 0] = 1e-8
    X = np.zeros(multinomials.shape, dtype=np.int64)
    rowsums = np.sum(multinomials, axis=1)
    multinomials = multinomials / rowsums[:, None]
    for n in range(N):
        idx = sorted(range(len(multinomials[n])), key=theta.__getitem__)
        sort = sorted(multinomials[n])
        for m in range(M):
            r = random.random()
            cumulative = 0
            for d in range(len(sort)):
                cumulative += sort[d]
                if r <= cumulative:
                    X[n, idx[d]] += 1
                    break
    for j in range(X.shape[1]):
        if X[0, j] == 0:
            X[0, j] = 3
    return X, np.mean(multinomials, axis=0)


def load_dataset_generator(train_dir, test_dir, metadata=False):
    train_path = train_dir.replace('\\\\', '/')
    test_path = test_dir.replace('\\\\', '/')
    for dataset in glob.glob(train_path + '/*'):
        ret_train_data = tuple()
        ret_test_data = tuple()
        meta_data = {}
        data_train_path = dataset.replace('\\', '/')
        dataset_name = data_train_path.split('/')[-1].split('-')[0]
        data_test_path = test_path + '/' + dataset_name + '-test-stemmed.txt'
        with open(data_train_path, 'r') as train, open(data_test_path, 'r') as test:
            train_data = train.read().split('\n')
            test_data = test.read().split('\n')
            if not metadata:
                for line in train_data:
                    if not line or line[0] == '#':
                        continue
                    truth_label, words = line.split('\t')
                    ret_train_data += ((truth_label, words.split()),)
            else:
                class_count = 0
                for line in train_data:
                    if not line:
                        continue
                    if line[0] == '#':
                        class_count += 1
                        _, param_sd, param_draws, mean_lin, obs, data_draws, *class_params = line[1:].split(',')
                        if 'param_sd' not in meta_data:
                            meta_data['param_sd'] = param_sd
                            meta_data['param_draws'] = param_draws
                            meta_data['mean_lin'] = mean_lin
                            meta_data['obs'] = obs
                            meta_data['data_draws'] = data_draws
                            meta_data['n_classes'] = 1
                            meta_data['class_' + str(class_count)] = class_params
                            meta_data['name'] = dataset_name
                        else:
                            meta_data['n_classes'] += 1
                            meta_data['class_' + str(class_count)] = class_params
                    else:
                        truth_label, words = line.split('\t')
                        ret_train_data += ((truth_label, words.split()),)
            for line in test_data:
                if not line:
                    continue
                truth_label, words = line.split('\t')
                ret_test_data += ((truth_label, words.split()),)
        yield (ret_train_data, ret_test_data) if not metadata \
            else (ret_train_data, ret_test_data, meta_data)


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
                all_datasets[dataset] += ((truth_label, words.split()),)
    return all_datasets


def tokenize_train(train, test):
    train_zipped = tuple(zip(*train))
    test_zipped = tuple(zip(*test))
    uniq_values = set([x for y in train_zipped[1] for x in y])
    uniq_values.update(set([x for y in test_zipped[1] for x in y]))
    uniq_keys = set(train_zipped[0])
    value_idxs = {k: i for i, k in enumerate(sorted(list(uniq_values)))}
    key_idxs = {k: i for i, k in enumerate(sorted(list(uniq_keys)))}
    class_labels = tuple(k for _, k in (sorted(((i, k) for k, i in key_idxs.items()))))
    key_observed_count = {k: 0 for k in uniq_keys}

    print(key_idxs)
    print(class_labels)

    tokenized = {k: [np.zeros((train_zipped[0].count(k), len(uniq_values)), dtype=np.int64), {}, len(uniq_values)] for k
                 in uniq_keys}
    for truth_label, words in train:
        if not words:
            continue
        observation_idx = key_observed_count[truth_label]
        for w in words:
            tokenized[truth_label][0][observation_idx, value_idxs[w]] += 1
        key_observed_count[truth_label] += 1

    # Remove zero-sum columns (to be reinserted as zeros in the MLE simplex after MLE computation)
    for k in uniq_keys:
        nonzero_idx = 0
        zero_idxs = ()
        feature_sums = np.sum(tokenized[k][0], axis=0)
        for j in range(len(uniq_values)):
            if feature_sums[j] > 0:
                tokenized[k][1].setdefault(nonzero_idx, j)
                nonzero_idx += 1
            else:
                zero_idxs += (j,)
        if zero_idxs:
            tokenized[k][0] = np.delete(tokenized[k][0], np.array(zero_idxs), axis=1)

    # Remove zero-sum rows (if they had no words but were in the training set for whatever reason)
    for k in uniq_keys:
        rowsums = np.sum(tokenized[k][0], axis=1)
        zero_idxs = np.squeeze(np.argwhere(rowsums == 0))
        if zero_idxs.size != 0:
            tokenized[k][0] = np.delete(tokenized[k][0], zero_idxs, axis=0)
    return tokenized, class_labels, key_idxs, value_idxs


def tokenize_test(observation, value_idxs):
    truth_label = observation[0]
    tokenized = np.zeros(len(value_idxs), dtype=np.float64)
    for w in observation[1]:
        # if w not in value_idxs:
        #     continue
        tokenized[value_idxs[w]] += 1
    return tokenized, truth_label


def r_zero_inflated_poisson(zero_inflation_prob, poisson_mean, n):
    non_zero_idxs = np.random.binomial(1, zero_inflation_prob, n)
    counts = np.random.poisson(poisson_mean, np.sum(non_zero_idxs))
    return non_zero_idxs > 0, counts


def compute_naive_bayes(Query_matrix, Training_matrix, test_set_labels, class_training_labels, key_idxs,
                        accuracy_matrix, method, posterior_method=None):
    classifications = None
    if posterior_method == 'aposteriori':
        if method == 'DM':
            log_marginal_probabilities = dm.dm_marginal_log_likelihood(Query_matrix, Training_matrix)
        elif method == 'BLM':
            log_marginal_probabilities = blm.blm_marginal_log_likelihood(Query_matrix, Training_matrix)
        else:
            raise ValueError('Method with aposteriori argument must be one of [DM, BLM]. Provided: {}'.format(method))
        classifications = [class_training_labels[c] for c in np.argmax(log_marginal_probabilities, axis=1)]
    else:
        classifications = [class_training_labels[c] for c in np.argmax(np.matmul(Query_matrix, Training_matrix), axis=1)]
    for i, c in enumerate(classifications):
        # print('{}\t{}'.format(c, test_set_labels[i]))
        for a in range(len(accuracy_matrix)):
            accuracy_matrix[a][3] += 1  # add true negative to all classes
        if c == test_set_labels[i]:
            accuracy_matrix[key_idxs[c]][0] += 1  # true positive
        else:
            accuracy_matrix[key_idxs[c]][1] += 1  # false positive
            accuracy_matrix[key_idxs[test_set_labels[i]]][2] += 1  # false negative
            accuracy_matrix[key_idxs[test_set_labels[i]]][3] -= 1  # remove a true negative from d_labels[i]
        accuracy_matrix[key_idxs[c]][3] -= 1  # remove a true negative from c
    return accuracy_matrix


def output_results_naive_bayes(smoothed_matrix, test, class_labels, key_idxs, value_idxs, dataset_name,
                               distribution, smoothing_method, param_string, precompute, result_file, posterior_method,
                               batch_size=1000):
    accuracy_matrix = [[0, 0, 0, 0] for _ in class_labels]
    observations_processed = 0
    while observations_processed < len(test):
        if observations_processed + batch_size <= len(test):
            ndim = batch_size
        else:
            ndim = len(test) - observations_processed
        observed_matrix = np.zeros((ndim, len(value_idxs)), dtype=np.float64)
        observed_labels = ()

        for i in range(0, ndim):
            observed_matrix[i, :], truth_label = tokenize_test(test[observations_processed + i], value_idxs)
            observed_labels += (truth_label,)

        accuracy_matrix = compute_naive_bayes(observed_matrix, smoothed_matrix, observed_labels, class_labels,
                                              key_idxs, accuracy_matrix, distribution, posterior_method)
        observations_processed += ndim
        print("{}, {}, {}, {}, {}, {}\tObservations processed: {} / {}".format(
            dataset_name,
            distribution,
            smoothing_method,
            precompute,
            posterior_method,
            param_string,
            observations_processed,
            len(test)
        ))

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i, results in enumerate(accuracy_matrix):
        tp += results[0]
        fp += results[1]
        fn += results[2]
        tn += results[3]
        print('{}\t\t{}'.format(class_labels[i], results))
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    print('Sensitivity/Recall: {}\n'
          'Specificity: {}\n'
          'PPV/Precision: {}\n'
          'NPV: {}\n'
          'Accuracy: {}\n'
          'F1: {}'.format(
                    round(100 * recall, 2),
                    round(100 * float(tn) / (tn + fp), 2),
                    round(100 * precision, 2),
                    round(100 * float(tn) / (tn + fn)),
                    round(100 * float(tp + tn) / (tp + fp + tn + fn), 2),
                    round(100 * float((2 * precision * recall) / (precision + recall)), 2)
    ))
    print('\n')

    with open(result_file, 'a') as out:
        for c in range(len(accuracy_matrix)):
            out.write('{},{},{},{},{},{},{},{}\n'.format(
                dataset_name,
                posterior_method,
                distribution,
                smoothing_method,
                precompute,
                param_string,
                class_labels[c],
                ','.join([str(x) for x in accuracy_matrix[c]])
            ))


def exact_geom_limit(theta, n):
    return np.sum((theta + np.array(range(n+1), dtype=np.float64)) ** -2)


def approx_geom_limit(theta, n):
    if n < 201:
        return exact_geom_limit(theta, n)
    else:
        return (1 / (theta ** 2)) + ((np.pi ** 2)/6) / (1 + ((np.pi/2) * np.exp(theta)))


def exact_harmonic_limit(theta, n):
    return np.sum((theta + np.array(range(n+1), dtype=np.float64)) ** -1)


def approx_harmonic_limit(theta, n):
    if n < 0:
        return 0
    value = 1 / theta  # zero index
    if n == 0:
        return value
    asymp = (np.log(n) + 0.57721 + (1 / (2 * n)) - (1 / (12 * (n ** 2))))
    scale = 0.072 * (np.log(n) ** 1.27) + (0.1677 / (1 + np.log(n))) + 0.835
    xmid = 0.5 * np.log(n) - 0.2718
    A = 0.1068 * (np.log(n) ** 0.8224) + (4.986 / (6.408 + np.log(n))) - 0.7751
    v = 3.764 * np.exp((-np.pi / 6) * (np.log(n) + 1) ** 1.06) + 1.59
    value += (asymp / (1 + np.exp((np.log(theta) - xmid) / scale))) + \
            (A * np.cos(4 * ((np.pi / 2) + (-np.pi / (1 + np.exp((np.log(theta) - xmid) / (v * scale))))) - (np.pi / 2)))
    return value if value > 0 else 0


def exact_log_limit(theta, n):
    return np.sum(np.log(theta + np.array(range(n+1), dtype=np.float64)))
