import numpy as np
import random
import glob
from multi_mle import dm
from multi_mle import blm
from multi_mle import gdm
import pickle
import os
from bisect import bisect_left
from scipy.special import gamma
from scipy.interpolate import interp1d
import sys


class MLEngine(object):
    def __init__(self,
                 max_steps,
                 delta_eps_threshold,
                 delta_lprob_threshold,
                 temp_dir,
                 dataset,
                 posterior_method=None,
                 verbose=False):
        # User defined parameters
        self.max_steps = max_steps
        self.delta_eps_threshold = delta_eps_threshold
        self.delta_lprob_threshold = delta_lprob_threshold
        self.temp_dir = temp_dir
        self.dataset = dataset
        self.posterior_method = posterior_method
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
        if not os.path.isdir('{}/{}/blm'.format(self.temp_dir, self.dataset)):
            os.mkdir('{}/{}/blm'.format(self.temp_dir, self.dataset))
        if not os.path.isdir('{}/{}/blm/{}'.format(self.temp_dir, self.dataset, self.posterior_method)):
            os.mkdir('{}/{}/blm/{}'.format(self.temp_dir, self.dataset, self.posterior_method))
        if not os.path.isdir('{}/{}/gdm'.format(self.temp_dir, self.dataset)):
            os.mkdir('{}/{}/gdm'.format(self.temp_dir, self.dataset))
        if not os.path.isdir('{}/{}/gdm/{}'.format(self.temp_dir, self.dataset, self.posterior_method)):
            os.mkdir('{}/{}/gdm/{}'.format(self.temp_dir, self.dataset, self.posterior_method))
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

    def dm_mle_parallel(self, filepath):
        """
        Execute the Dirichlet Multinomial MLE in parallel across classes.
        :param filepath: String, full path to the pickled count matrix from multi_pickle_dump()
        """
        try:
            label = filepath.split('/')[-1].replace('.pickle', '')
            this_path = '{}/{}/dm/{}.pickle'.format(self.temp_dir, self.dataset, label)
            if os.path.exists(this_path):
                return this_path

            with open(filepath, 'rb') as infile:
                X = pickle.load(infile)

            params = dm.dm_init_params(X[0])
            U, vd = dm.dm_precalc(X[0])
            mle = dm.dm_newton_raphson2(U, vd, params,
                                        self.max_steps,
                                        self.delta_eps_threshold,
                                        self.delta_lprob_threshold,
                                        label,
                                        self.verbose)
            class_simplex = dm.dm_renormalize(mle)
            expanded_simplex = np.zeros(X[2], dtype=np.float64)
            for i, val in enumerate(class_simplex):
                expanded_simplex[X[1][i]] = val
            with open(this_path, 'wb') as out:
                pickle.dump(expanded_simplex, out)
            return this_path
        except Exception as e:
            raise e

    def blm_mle_parallel(self, filepath):
        """
        Execute the Beta-Liouville Multinomial MLE in parallel across classes.
        :param filepath: String, full path to the pickled file from multi_pickle_dump()
        """
        try:
            label = filepath.split('/')[-1].replace('.pickle', '')
            this_path = '{}/{}/blm/{}/{}.pickle'.format(self.temp_dir, self.dataset, self.posterior_method, label)
            if os.path.exists(this_path):
                return this_path

            with open(filepath, 'rb') as infile:
                X = pickle.load(infile)

            params = blm.blm_init_params(X[0])
            U, vd, vd1 = blm.blm_precalc(X[0])
            mle = blm.blm_newton_raphson2(U, vd, vd1, params,
                                          self.max_steps,
                                          self.delta_eps_threshold,
                                          self.delta_lprob_threshold,
                                          label,
                                          self.verbose)
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
            with open(this_path, 'wb') as out:
                pickle.dump(expanded_simplex, out)
            return this_path
        except Exception as e:
            raise e


def multinomial_random_sample(N, M, theta):
    """
    Sample from a multinomial distribution with parameters theta.
    :param n: integer, number of multinomials to generate
    :param m: integer, number of samples per multinomial
    :param theta: vector of parameters in dimension D+1
    :return: count matrix X, dimension (N, D+1)
    """
    D1 = len(theta)
    multinomials = np.array([[random.normalvariate(theta[d], 0.01) for d in range(D1)] for _ in range(N)])
    X = np.zeros(multinomials.shape, dtype=np.int64)
    rowsums = multinomials.sum(axis=1)
    multinomials = (multinomials / rowsums.astype(float)).tolist()
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
                # print(truth_label, words)
                # if len(words) < 3:  # filter out low quality training observations
                #     continue
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
                        accuracy_matrix, posterior_method):
    classifications = None
    if posterior_method == 'aposteriori':
        marginal_probabilities = np.zeros((Query_matrix.shape[0], Training_matrix.shape[1]), dtype=np.float64)
        training_feature_sums = np.sum(Training_matrix, axis=1)

        # Find where the beta parameter lies and reorder the vectors such that the beta parameters are terminal
        beta_idx = None
        for i in range(len(training_feature_sums)-2, -1, -1):
            if training_feature_sums[i] > 0:
                beta_idx = i
                break

        Query_matrix = Query_matrix + 1  # Lidstone smoothing for a posteriori
        for observation in range(Query_matrix.shape[0]):
            for label in range(Training_matrix.shape[1]):
                if beta_idx < (Query_matrix.shape[1] - 1):
                    query_vec = np.concatenate((Query_matrix[observation, :beta_idx],
                                                Query_matrix[observation, (beta_idx+1):-1]))
                    query_vec = np.append(query_vec, Query_matrix[observation, beta_idx])
                    query_vec = np.append(query_vec, Query_matrix[observation, -1])
                else:
                    query_vec = Query_matrix[observation, :]
                train_vec = np.concatenate((Training_matrix[:beta_idx, label],
                                            Training_matrix[(beta_idx+1):-1, label]))
                train_vec = np.append(train_vec, Training_matrix[beta_idx, label])
                train_vec = np.append(train_vec, Training_matrix[-1, label])

                # Calculate the marginal probability of this query vector against each class in training matrix
                print(np.sum(query_vec), np.prod(gamma(1 + query_vec)))
                sys.exit()
                term1 = gamma((1 + np.sum(query_vec))) / np.prod(gamma(1 + query_vec))
                term2 = gamma(np.sum(train_vec[:-2])) / gamma(np.sum(train_vec[:-2] + query_vec[:-1]))
                term3 = gamma(train_vec[-1] + train_vec[-2]) / gamma(train_vec[-2] + train_vec[-1] + np.sum(query_vec))
                term4 = gamma(train_vec[-1] + np.sum(query_vec[:-1])) / gamma(train_vec[-1])
                term5 = gamma(train_vec[-2] + query_vec[-1]) / gamma(train_vec[-2])
                term6 = np.prod(gamma(train_vec[:-2] + query_vec[:-1])) / np.prod(gamma(train_vec[:-2]))
                print(term1, term2, term3, term4, term5, term6)
                prob = np.prod((term1, term2, term3, term4, term5, term6))
                print(prob)
                marginal_probabilities[observation, label] = prob

                classifications = [class_training_labels[c] for c in np.argmax(marginal_probabilities, axis=1)]
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
                               distribution, smoothing_method, param_string, result_file, posterior_method,
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
                                              key_idxs, accuracy_matrix, posterior_method)
        observations_processed += ndim
        print("{}, {}, {}, {}\tObservations processed: {} / {}".format(
            dataset_name,
            distribution,
            smoothing_method,
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
            out.write('{},{},{},{},{},{},{}\n'.format(
                dataset_name,
                posterior_method,
                distribution,
                smoothing_method,
                param_string,
                class_labels[c],
                ','.join([str(x) for x in accuracy_matrix[c]])
            ))


def load_geom_lookup_table(filepath):
    ret = {}
    with open(filepath, 'r') as f:
        line = f.readline()
        while line:
            param, *vals = tuple(float(x) for x in line.rstrip(',\n').split(','))
            ret[param] = vals
            line = f.readline()
    return ret, sorted(ret.keys())


def approx_geom_limit(theta, n, lookup_table, sorted_keys):
    val = 1 / theta
    if n == 0:
        return val
    n_idx = n - 1
    if theta < sorted_keys[0]:
        val += np.pi / 6  # limit of the standard Geometric series
    elif theta > sorted_keys[-1]:
        pass  # limit evaluates to zero
    else:
        param_spline_vals = []
        i = bisect_left(sorted_keys, theta)
        if i < 3:
            param_spline_vals = sorted_keys[:5]
        elif i > (len(sorted_keys) - 4):
            param_spline_vals = sorted_keys[-5:]
        limit_spline_vals = []
        for p in param_spline_vals:
            if n > len(lookup_table[p]):
                limit_spline_vals.append(lookup_table[p][:-1])
            else:
                limit_spline_vals.append(lookup_table[p][n_idx])
        spl = interp1d(np.log(param_spline_vals), limit_spline_vals, kind='cubic')
        val += spl(np.log(theta))
        return val


def exact_geom_limit(theta, n):
    return np.sum(np.power(theta + np.array(range(n+1)), -1))


def approx_harmonic_limit(theta, n):
    value = 1 / theta  # zero index
    asymp = (np.log(n) + 0.57721 + (1 / (2 * n)) - (1 / (12 * (n ** 2))))
    scale = 0.072 * (np.log(n) ** 1.27) + (0.1677 / (1 + np.log(n))) + 0.835
    xmid = 0.5 * np.log(n) - 0.2718
    A = 0.1068 * (np.log(n) ** 0.8224) + (4.986 / (6.408 + np.log(n))) - 0.7751
    v = 3.764 * np.exp((-np.pi / 6) * (np.log(n) + 1) ** 1.06) + 1.59
    value += (asymp / (1 + np.exp((np.log(theta) - xmid) / scale))) + \
            (A * np.cos(4 * ((np.pi / 2) + (-np.pi / (1 + np.exp((np.log(theta) - xmid) / (v * scale))))) - (np.pi / 2)))
    return value if value > 0 else 0



