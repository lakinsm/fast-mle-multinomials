import numpy as np
import sys
import multi_mle.mle_utils as mutils


def dm_precalc(X):
    """
    Calculate the U matrix of dimension (D, Z_D), and v vector of length Z_sumD for the DM.
    :param X: Data matrix of counts, dimension (N, D)
    :return: Matrix U, vector v
    """
    # Variable definitions
    D = X.shape[1]
    N = X.shape[0]

    # Initialize data structures
    U = tuple([] for _ in range(D))
    v = []

    # Compute Algorithm 1 (Sklar 2014)
    for n in range(N):
        C = 0
        for d in range(D):
            C += X[n, d]
            for i in range(X[n, d]):
                try:
                    U[d][i] += 1
                except IndexError:
                    U[d].append(1)
        for i in range(C):
            try:
                v[i] += 1
            except IndexError:
                v.append(1)
    return U, v


def dm_init_params(X):
    """
    Initialize parameters for the DM distribution using moment matching as described by Ronning 1989 and again in
    Minka 2012 "Estimating the Dirichlet Distribution"
    :param X: Data matrix of counts, dimension (N, D)
    :return: vector of parameter initial values a
    """
    D = X.shape[1]
    rowsums = X.sum(axis=1)
    Xnorm = (X / rowsums.reshape(-1, 1).astype(float))
    min_prop_vec = np.array([np.min(Xnorm[Xnorm > 0]) for _ in range(D)])
    mean = np.array(np.mean(Xnorm, axis=0))
    m2 = np.array(np.mean(np.power(Xnorm, 2), axis=0))
    denominator = m2 - (np.power(mean, 2))

    # The below conditionals are where the method of moments estimation will not work, so we take all parameters
    # to be equal to the lowest proportion, as stated in Ronning 1989.
    if any(x == 0 for x in denominator):
        return min_prop_vec
    sum_alpha = np.divide(mean - m2, denominator)
    if any(x == 0 for x in 1 + sum_alpha):
        return min_prop_vec
    var_pk = np.divide(np.multiply(mean, 1 - mean), 1 + sum_alpha)
    if any(x == 0 for x in var_pk):
        return min_prop_vec
    alpha = np.abs(np.divide(np.multiply(mean, 1 - mean), var_pk) - 1)
    log_sum_alpha = ((D - 1) ** -1) * np.sum(np.log(alpha))
    s = np.exp(log_sum_alpha)
    if s == 0:
        s = 1
    return np.array(s*mean)


def dm_hessian_precompute(U, v, theta):
    """
    Precompute the gradient, Hessian diagonal, hessian constant, and log-likelihood log(P(X | theta))
    :param U: Precalculated matrix U from dm_precalc, dimension (D, Z_D)
    :param v: Precalculated vector v from dm_precalc, length z_sumD
    :param theta: Parameter vector of length D
    :return: gradient vector, Hessian diagonal vector, scalar constant, and scalar log-likelihood
    """
    theta = np.squeeze(theta)
    D = len(U)
    lprob = 0
    gradient = np.zeros(D, np.float64)
    h_diag = np.zeros(D, np.float64)
    constant = 0
    sum_theta = np.sum(theta)
    for z in range(len(v)):
        if z % 1000 == 0:
            print('Precompute: {} / {}'.format(z, len(v)))
        for d in range(D):
            if z < len(U[d]):
                lprob += U[d][z] * np.log(theta[d] + z)
                gradient[d] += (U[d][z] * ((theta[d] + z)**-1))
                h_diag[d] -= (U[d][z] * ((theta[d] + z)**-2))
            gradient[d] -= (v[z] * ((sum_theta + z)**-1))
            # h_diag[d] += (v[z] * ((sum_theta + z)**-2))
        constant += v[z] * ((sum_theta + z)**-2)
        lprob -= v[z] * np.log(sum_theta + z)
    print('Precompute: {} / {}\n'.format(len(v), len(v)))
    return gradient, h_diag, constant, lprob


def dm_hessian_precompute_exact_vectorized(X, theta):
    """
    Instead of Sklar's implementation, utilize the original likelihood function but vectorize the finite sums.
    :param X: Data matrix of counts, dimension (N, D)
    :param theta: Parameter vector of length D
    :return: gradient vector, Hessian diagonal vector, scalar constant,  scalar log-likelihood
    """
    theta = np.squeeze(theta)
    N, D = X.shape
    gradient = np.zeros(D, np.float64)
    h_diag = np.zeros(D, np.float64)
    constant = 0
    lprob = 0
    sum_theta = np.sum(theta)
    rowsums = np.sum(X, axis=1)
    for n in range(N):
        for d in range(D):
            if X[n, d] == 0:
                continue
            value_summation_stop = X[n, d] - 1
            lprob += mutils.exact_log_limit(theta[d], value_summation_stop)
            gradient[d] += mutils.exact_harmonic_limit(theta[d], value_summation_stop)
            h_diag[d] -= mutils.exact_geom_limit(theta[d], value_summation_stop)
        # Note the following are broadcast to the whole array
        row_summation_stop = rowsums[n] - 1
        lprob -= mutils.exact_log_limit(sum_theta, row_summation_stop)
        gradient -= mutils.exact_harmonic_limit(sum_theta, row_summation_stop)
        constant += mutils.exact_geom_limit(sum_theta, row_summation_stop)
    return gradient, h_diag, constant, lprob


def dm_hessian_precompute_approximate(X, theta):
    """
    Instead of Sklar's implementation, utilize the original likelihood function but approximate the finite sums.
    :param X: Data matrix of counts, dimension (N, D)
    :param theta: Parameter vector of length D
    :return: gradient vector, Hessian diagonal vector, scalar constant, scalar log-likelihood
    """
    theta = np.squeeze(theta)
    N, D = X.shape
    gradient = np.zeros(D, np.float64)
    h_diag = np.zeros(D, np.float64)
    constant = 0
    lprob = 0
    sum_theta = np.sum(theta)
    rowsums = np.sum(X, axis=1)
    for n in range(N):
        for d in range(D):
            if X[n, d] == 0:
                continue
            value_summation_stop = X[n, d] - 1
            lprob += mutils.exact_log_limit(theta[d], value_summation_stop)
            gradient[d] += mutils.approx_harmonic_limit(theta[d], value_summation_stop)
            h_diag[d] -= mutils.approx_geom_limit(theta[d], value_summation_stop)
        # Note the following are broadcast to the whole array
        row_summation_stop = rowsums[n] - 1
        lprob -= mutils.exact_log_limit(sum_theta, row_summation_stop)
        gradient -= mutils.approx_harmonic_limit(sum_theta, row_summation_stop)
        constant += mutils.approx_geom_limit(sum_theta, row_summation_stop)
    return gradient, h_diag, constant, lprob


def dm_log_likelihood_fast(U, v, theta):
    """
    Compute only the proportional log-likelihood (terms dependent only on parameters considered).
    :param U: Precalculated matrix U from dm_precalc, dimension (D, Z_D)
    :param v: Precalculated vector v from dm_precalc, length z_sumD
    :param theta: Parameter vector of length D+1
    :return: Proportional log-likelihood
    """
    if np.any(theta < 0):
        return np.inf
    theta = np.squeeze(theta)
    D = len(U)
    lprob = 0
    sum_theta = np.sum(theta)
    for z in range(len(v)):
        for d in range(D):
            if z < len(U[d]):
                lprob += U[d][z] * np.log(theta[d] + z)
        lprob -= v[z] * np.log(sum_theta + z)
    return lprob


def dm_step(h, g, c):
    """
    Compute a single Newton-Raphson iteration
    :param h: Hessian diagonal vector, length D
    :param g: Gradient vector, length D
    :param c: Constant scalar
    :return: Vector deltas of length D with values for the computed changes to the parameters
    """
    D = g.shape[0]
    deltas = np.zeros(D, dtype=np.float64)
    # Invert the hessian diagonal
    h = np.power(h, -1)
    for d in range(D):
        deltas[d] = h[d] * (g[d] - (np.inner(g, h) / ((c**-1) + np.sum(h))))
    return deltas


def dm_renormalize(theta):
    return theta / np.sum(theta)


def dm_renormalize_empirical(theta, x):
    """
    Calculate parameter estimates for p_d to the unit simplex according to the posterior of the DM for parameter p_d.
    This differs from the above, since we are using the training data to both estimate the maximum likelihood AND to
    inform the posterior parameters, a la empirical Bayes.  The above, regular method only uses the MLE to inform the
    parameters.
    :param theta: Vector of BLM parameters are output from MLE, length D+1
    :param x: Vector of count data summed across all observations used as input to the MLE, dimension (1, D+1)
    :return: Empirically informed parameter estimates on the simplex, length D+1
    """
    return (theta + x) / (np.sum(theta) + np.sum(x))


def dm_extract_generating_params(theta):
    return theta


def dm_half_stepping(U, v, params, lprob, current_lprob, deltas, threshold):
    """
    Step half the distance to the current deltas iteratively until the learning rate drops below a threshold.
    :param U: Precalculated matrix U from dm_precalc, dimension (D, Z_D)
    :param v: Precalculated vector v from dm_precalc, length z_sumD
    :param params: Parameter vector of length D
    :param lprob: Log-likelihood to improve iteratively
    :param current_lprob: Log-likelihood to beat (current max)
    :param deltas: Current deltas from Newton-Raphson step
    :param threshold: Learning rate threshold (minimum value)
    :return: Parameter vector, new log-likelihood, and True if success, else False if below learning threshold
    """
    local_params = params
    local_lprob = lprob
    learn_rate = 1.0
    while local_lprob < current_lprob:
        if learn_rate < threshold:
            print("DM MLE converged with small learn rate")
            return params, local_lprob, False
        local_params = params - (learn_rate * deltas)
        learn_rate *= 0.5
        local_lprob = dm_log_likelihood_fast(U, v, local_params)
        if local_lprob == np.inf:
            local_lprob = lprob
            continue
    return local_params, local_lprob, True


def dm_newton_raphson2(X, U, vd, params, precompute,
                       max_steps, delta_eps_threshold, delta_lprob_threshold, label, verbose=False):
    current_lprob = -2e20
    delta_lprob = 2e20
    delta_params = 2e20
    step = 0
    while (delta_params > delta_eps_threshold) and (step < max_steps) and (delta_lprob > delta_lprob_threshold):
        step += 1
        if precompute == 'sklar':
            g, h, c, lprob = dm_hessian_precompute(U, vd, params)
        elif precompute == 'vectorized':
            g, h, c, lprob = dm_hessian_precompute_exact_vectorized(X, params)
        elif precompute == 'approximate':
            g, h, c, lprob = dm_hessian_precompute_approximate(X, params)
        else:
            raise ValueError('Precompute must be one of [sklar, vectorized, approximate]: {}'.format(precompute))
        delta_lprob = np.abs(lprob - current_lprob)
        current_lprob = lprob
        deltas = dm_step(h, g, c)
        delta_params = np.sum(np.abs(deltas))  # See appendix
        params -= deltas
        if verbose:
            print('{}\t Step: {}, Lprob: {}\tDelta Lprob: {}'.format(
                label,
                step,
                lprob,
                delta_lprob
            ))
            print('\tDelta Sum Eps: {}'.format(delta_params))
            print('\tParams: {}'.format(params))
            print('\tDeltas: {}'.format(deltas))
        if np.any(params <= 0):
            params[params <= 0] = 1e-20
            # raise ValueError('Negative parameters detected, exiting: {}'.format(
            #     params[params < 0]
            # ))
    print('DM MLE Exiting: {}, Total steps: {} / {}\n'.format(
        label,
        step,
        max_steps
    ))
    return params


def dm_marginal_log_likelihood(Query_matrix, Training_matrix):
    log_marginal_probabilities = np.zeros((Query_matrix.shape[0], Training_matrix.shape[1]), dtype=np.float64)

    # Pseudo-Lidstone smoothing for aposteriori
    pseudo_value = 1. / Training_matrix.shape[0]
    Training_matrix += pseudo_value

    for observation in range(Query_matrix.shape[0]):
        query_vec = Query_matrix[observation, :]
        for label in range(Training_matrix.shape[1]):
            train_vec = Training_matrix[:, label]

            # Calculate the marginal probability of this query vector against each class in training matrix
            query_sum_d1 = int(np.sum(query_vec))
            sum_theta = np.sum(train_vec)
            lprob = 0
            lprob += mutils.exact_log_limit(1, query_sum_d1 - 1)
            lprob -= mutils.exact_log_limit(sum_theta, query_sum_d1 - 1)
            lprob += sum(mutils.exact_log_limit(train_vec[d], int(query_vec[d] - 1)) for d in range(len(query_vec))
                         if query_vec[d] != 0)
            lprob -= sum(mutils.exact_log_limit(1, int(query_vec[d] - 1)) for d in range(len(query_vec))
                         if query_vec[d] != 0)
            log_marginal_probabilities[observation, label] = lprob
    return log_marginal_probabilities


if __name__ == '__main__':
    test_matrix = np.matrix('2 7 3; 1 8 2; 1 7 2', dtype=np.int32)
    test_u, test_v = dm_precalc(test_matrix)

    test_init = dm_init_params(test_matrix)
    np.testing.assert_array_almost_equal(test_init,
                                         np.matrix([119.76577944, 673.42876828, 211.62004248], dtype=np.float64))

    test_g, test_hd, test_c, test_lprob = dm_hessian_precompute(test_u, test_v, test_init)
    np.testing.assert_almost_equal(test_lprob, np.array([-28.24548495], dtype=np.float64))
    np.testing.assert_array_almost_equal(test_g, np.array([0.00065077, -0.00016329, 0.00028862], dtype=np.float64))
    np.testing.assert_array_almost_equal(test_hd,
                                         np.array([-2.45354837e-04, -1.56964506e-05, -1.22903642e-04],
                                                  dtype=np.float64))
    np.testing.assert_almost_equal(test_c, np.array([3.23606913e-05], dtype=np.float64))

    test_deltas = dm_step(test_hd, test_g, test_c)
    # np.testing.assert_array_almost_equal(test_deltas,
    #                                      np.array([-3.1414175, 2.75803185, -3.32466483],
    #                                               dtype=np.float64))

    normalized = dm_renormalize(test_init - test_deltas)
    np.testing.assert_array_almost_equal(normalized, np.matrix([0.12186856, 0.66500315, 0.21312829], dtype=np.float64))
