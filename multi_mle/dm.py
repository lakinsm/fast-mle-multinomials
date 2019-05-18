import numpy as np


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
    mean = np.array(np.mean(Xnorm, axis=0))
    m2 = np.array(np.mean(np.power(Xnorm, 2), axis=0))
    nonzeros = np.array(mean > 0)
    sum_alpha = np.divide((mean[nonzeros] - m2[nonzeros]), (m2[nonzeros] - (np.power(mean[nonzeros], 2))))
    sum_alpha[sum_alpha == 0] = 1  # required to prevent division by zero
    var_pk = np.divide(np.multiply(mean, 1 - mean), 1 + sum_alpha)
    log_sum_alpha = ((D - 1)**-1) * np.sum(np.log(np.divide(np.multiply(mean, 1 - mean), var_pk) - 1))
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
        for d in range(D):
            if z < len(U[d]):
                lprob += U[d][z] * np.log(theta[d] + z)
                gradient[d] += (U[d][z] * ((theta[d] + z)**-1))
                h_diag[d] -= (U[d][z] * ((theta[d] + z)**-2))
            gradient[d] -= (v[z] * ((sum_theta + z)**-1))
            h_diag[d] += (v[z] * ((sum_theta + z)**-2))
        constant += v[z] * ((sum_theta + z)**-2)
        lprob -= v[z] * np.log(sum_theta + z)
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


def dm_newton_raphson(U, v, params, max_steps, gradient_sq_threshold, learn_rate_threshold, delta_lprob_threshold):
    """
    Find the MLE for the Dirichlet multinomial given the precomputed data structures and initial parameter estimates.
    :param U: Precomputed U matrix from dm_precalc
    :param v: Precomputed v vector from dm_precalc
    :param params: Initial parameter estimates in D+1 dimensions
    :param max_steps: Max iterations to perform Newton-Raphson stepping
    :param gradient_sq_threshold: Threshold under which optimization stops for the sum of squared gradients
    :param learn_rate_threshold: Threshold under which optimization stops for half-stepping
    :return: Results of the MLE computation for parameters 1:D+1
    """
    current_lprob = -2 ** 20
    delta_lprob = 2 ** 20
    step = 0
    while step < max_steps:
        if delta_lprob < delta_lprob_threshold:
            print("DM MLE converged with small delta log-probability")
            return dm_renormalize(params)
        step += 1
        g, h, c, lprob = dm_hessian_precompute(U, v, params)
        gradient_sq = np.sum(np.power(g, 2))
        if gradient_sq < gradient_sq_threshold:
            print("DM MLE converged with small gradient")
            return dm_renormalize(params)
        deltas = dm_step(h, g, c)
        if lprob > current_lprob:
            test_params = params - deltas
            if np.any(test_params < 0):
                params, lprob, success = dm_half_stepping(U, v, params, -2 ** 20, current_lprob, deltas, learn_rate_threshold)
                if not success:
                    return dm_renormalize(params)
                else:
                    delta_lprob = np.abs(lprob - current_lprob)
                    current_lprob = lprob
            else:
                delta_lprob = np.abs(lprob - current_lprob)
                current_lprob = lprob
                params -= deltas
        else:
            params, lprob, success = dm_half_stepping(U, v, params, lprob, current_lprob, deltas, learn_rate_threshold)
            if not success:
                return dm_renormalize(params)
            else:
                delta_lprob = np.abs(lprob - current_lprob)
                current_lprob = lprob
    print("DM MLE reached max iterations")
    return dm_renormalize(params)


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
