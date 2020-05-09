import numpy as np
import sys
import multi_mle.mle_utils as mutils


def blm_precalc(X):
    """
    Calculate the U matrix of dimension (D+1, Z_max), v^D vector of length Z_sumD, and
    v^(D+1) vector of length Z_sum(D+1) for the BLM.
    :param X: Data matrix of counts, dimension (N, D+1)
    :return: Matrix U, vector v^D, vector v^(D+1)
    """
    # Variable definitions
    D = X.shape[1] - 1
    D1 = X.shape[1]
    N = X.shape[0]

    # Initialize data structures
    U = tuple([] for _ in range(D1))
    vd = []
    vd1 = []

    # Compute Algorithm 1 (Lakin & Abdo 2019)
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
                vd[i] += 1
            except IndexError:
                vd.append(1)
        C += X[n, D1-1]
        for i in range(X[n, D1-1]):
            try:
                U[D1-1][i] += 1
            except IndexError:
                U[D1-1].append(1)
        for i in range(C):
            try:
                vd1[i] += 1
            except IndexError:
                vd1.append(1)
    return U, vd, vd1


def blm_init_params(X):
    """
    Initialize parameters for the BLM distribution using moment matching to the Dirichlet.
    :param X: Data matrix of counts, dimension (N, D+1)
    :return: vector of parameter initial values theta
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
    alpha = np.abs(np.divide(np.multiply(mean, 1 - mean), var_pk) - 1)
    log_sum_alpha = ((D - 1) ** -1) * np.sum(np.log(alpha))
    s = np.exp(log_sum_alpha)
    if s == 0:
        s = 1
    d_params = np.squeeze(np.array(s * mean))
    # return np.append(d_params, np.mean(d_params[:-1]))
    return np.append(d_params, np.sum(d_params[:-1]))


def blm_hessian_precompute(U, v_d, v_d1, theta):
    """
    Precompute the gradient, Hessian diagonal, hessian constant, and log-likelihood log(P(X | theta))
    :param U: Precalculated matrix U from blm_precalc, dimension (D, Z_D)
    :param v_d: Precalculated vector v_d from blm_precalc, length z_sumD
    :param v_d1: Precalculated vector v_d1 from blm_precalc, length z_sumD+1
    :param theta: Parameter vector of length D+2
    :return: gradient vector, Hessian diagonal vector, constants vector, and scalar log-likelihood
    """
    D1 = len(U)
    z_d = len(v_d)
    z_d1 = len(v_d1)
    lprob = 0
    gradient = np.zeros(D1+1, np.float64)
    h_diag = np.zeros(D1+1, np.float64)
    constants = np.zeros(2, np.float64)
    sum_theta = np.sum(theta[:-2])

    # z_d terms
    for z in range(z_d):
        if z % 1000 == 0:
            print('Precompute: {} / {}'.format(z, z_d1))
        # alpha_d parameters
        for d in range(D1-1):
            if z < len(U[d]):
                lprob += U[d][z] * np.log(theta[d] + z)
                gradient[d] += (U[d][z] * ((theta[d] + z)**-1))
                h_diag[d] -= (U[d][z] * ((theta[d] + z)**-2))
            gradient[d] -= (v_d[z] * ((sum_theta + z)**-1))
            # h_diag[d] += (v_d[z] * ((sum_theta + z)**-2))
        lprob -= v_d[z] * np.log(sum_theta + z)

        # beta parameter
        if z < len(U[D1-1]):
            lprob += U[D1-1][z] * np.log(theta[-2] + z)
            gradient[-2] += (U[D1-1][z] * ((theta[-2] + z)**-1))
            h_diag[-2] -= (U[D1-1][z] * ((theta[-2] + z)**-2))
        gradient[-2] -= (v_d1[z] * ((theta[-2] + theta[-1] + z)**-1))
        # h_diag[-2] += (v_d1[z] * ((theta[-2] + theta[-1] + z)**-2))

        # alpha parameter
        # h_diag[-1] += (v_d1[z] * ((theta[-2] + theta[-1] + z)**-2)) - (v_d[z] * ((theta[-1] + z)**-2))
        h_diag[-1] -= (v_d[z] * ((theta[-1] + z) ** -2))
        gradient[-1] += (v_d[z] * ((theta[-1] + z)**-1)) - (v_d1[z] * ((theta[-2] + theta[-1] + z) ** -1))
        lprob += (v_d[z] * np.log(theta[-1] + z)) - (v_d1[z] * np.log(theta[-2] + theta[-1] + z))

        # constants
        constants[0] += v_d[z] * ((sum_theta + z) ** -2)
        constants[1] += v_d1[z] * ((theta[-2] + theta[-1] + z) ** -2)

    # z_d+1 terms
    for z in range(z_d, z_d1):
        if z % 1000 == 0:
            print('Precompute: {} / {}'.format(z, z_d1))
        # beta parameter
        if z < len(U[D1 - 1]):
            lprob += U[D1 - 1][z] * np.log(theta[-2] + z)
            gradient[-2] += (U[D1 - 1][z] * ((theta[-2] + z) ** -1))
            h_diag[-2] -= (U[D1 - 1][z] * ((theta[-2] + z) ** -2))
        gradient[-2] -= (v_d1[z] * ((theta[-2] + theta[-1] + z) ** -1))
        # h_diag[-2] += (v_d1[z] * ((theta[-2] + theta[-1] + z) ** -2))

        # alpha parameter
        gradient[-1] -= v_d1[z] * ((theta[-2] + theta[-1] + z) ** -1)
        lprob -= v_d1[z] * np.log(theta[-2] + theta[-1] + z)
        # h_diag[-1] += (v_d1[z] * ((theta[-2] + theta[-1] + z) ** -2))

        # constants
        constants[1] += v_d1[z] * ((theta[-2] + theta[-1] + z)**-2)
    print('Precompute: {} / {}\n'.format(z_d1, z_d1))
    return gradient, h_diag, constants, lprob


def blm_hessian_precompute_exact_vectorized(X, theta):
    """
    Instead of Sklar's implementation, utilize the original likelihood function but vectorize the finite sums.
    :param X: Data matrix of counts, dimension (N, D)
    :param theta: Parameter vector of length D
    :return: gradient vector, Hessian diagonal vector, scalar constant,  scalar log-likelihood
    """
    theta = np.squeeze(theta)
    N, D1 = X.shape
    gradient = np.zeros(D1+1, np.float64)
    h_diag = np.zeros(D1+1, np.float64)
    constants = np.zeros(2, np.float64)
    lprob = 0
    sum_theta = np.sum(theta[:-2])
    rowsums_d = np.sum(X[:, :-1], axis=1)
    rowsums_d1 = np.sum(X, axis=1)
    for n in range(N):
        # alpha_d params
        for d in range(D1-1):
            if X[n, d] == 0:
                continue
            value_summation_stop = X[n, d] - 1
            lprob += mutils.exact_log_limit(theta[d], value_summation_stop)
            gradient[d] += mutils.exact_harmonic_limit(theta[d], value_summation_stop)
            h_diag[d] -= mutils.exact_geom_limit(theta[d], value_summation_stop)

        # beta param
        if X[n, -1] != 0:
            beta_summation_stop = X[n, -1] - 1
            lprob += mutils.exact_log_limit(theta[-2], beta_summation_stop)
            gradient[-2] += mutils.exact_harmonic_limit(theta[-2], beta_summation_stop)
            h_diag[-2] -= mutils.exact_geom_limit(theta[-2], beta_summation_stop)

        # alpha param and rowsum_d terms
        if rowsums_d[n] != 0:
            row_summation_stop_d = rowsums_d[n] - 1
            lprob += mutils.exact_log_limit(theta[-1], row_summation_stop_d)
            gradient[-1] += mutils.exact_harmonic_limit(theta[-1], row_summation_stop_d)
            h_diag[-1] -= mutils.exact_geom_limit(theta[-1], row_summation_stop_d)
            lprob -= mutils.exact_log_limit(sum_theta, row_summation_stop_d)
            gradient[:-2] -= mutils.exact_harmonic_limit(sum_theta, row_summation_stop_d)
            constants[0] += mutils.exact_geom_limit(sum_theta, row_summation_stop_d)

        # Row-based terms
        # Note the following are broadcast to the whole array
        row_summation_stop_d1 = rowsums_d1[n] - 1
        lprob -= mutils.exact_log_limit(theta[-1] + theta[-2], row_summation_stop_d1)
        gradient[-2:] -= mutils.exact_harmonic_limit(theta[-1] + theta[-2], row_summation_stop_d1)
        constants[1] += mutils.exact_geom_limit(theta[-1] + theta[-2], row_summation_stop_d1)
    return gradient, h_diag, constants, lprob


def blm_hessian_precompute_approximate(X, theta):
    """
    Instead of Sklar's implementation, utilize the original likelihood function but approximate the finite sums.
    :param X: Data matrix of counts, dimension (N, D)
    :param theta: Parameter vector of length D
    :return: gradient vector, Hessian diagonal vector, scalar constant, scalar log-likelihood
    """
    theta = np.squeeze(theta)
    N, D1 = X.shape
    gradient = np.zeros(D1 + 1, np.float64)
    h_diag = np.zeros(D1 + 1, np.float64)
    constants = np.zeros(2, np.float64)
    lprob = 0
    sum_theta = np.sum(theta[:-2])
    rowsums_d = np.sum(X[:, :-1], axis=1)
    rowsums_d1 = np.sum(X, axis=1)
    for n in range(N):
        # alpha_d params
        for d in range(D1 - 1):
            if X[n, d] == 0:
                continue
            value_summation_stop = X[n, d] - 1
            lprob += mutils.exact_log_limit(theta[d], value_summation_stop)
            gradient[d] += mutils.approx_harmonic_limit(theta[d], value_summation_stop)
            h_diag[d] -= mutils.approx_geom_limit(theta[d], value_summation_stop)

        # beta param
        if X[n, -1] != 0:
            beta_summation_stop = X[n, -1] - 1
            lprob += mutils.exact_log_limit(theta[-2], beta_summation_stop)
            gradient[-2] += mutils.approx_harmonic_limit(theta[-2], beta_summation_stop)
            h_diag[-2] -= mutils.approx_geom_limit(theta[-2], beta_summation_stop)

        # alpha param and rowsum_d terms
        if rowsums_d[n] != 0:
            row_summation_stop_d = rowsums_d[n] - 1
            lprob += mutils.exact_log_limit(theta[-1], row_summation_stop_d)
            gradient[-1] += mutils.approx_harmonic_limit(theta[-1], row_summation_stop_d)
            h_diag[-1] -= mutils.approx_geom_limit(theta[-1], row_summation_stop_d)
            lprob -= mutils.exact_log_limit(sum_theta, row_summation_stop_d)
            gradient[:-2] -= mutils.approx_harmonic_limit(sum_theta, row_summation_stop_d)
            constants[0] += mutils.approx_geom_limit(sum_theta, row_summation_stop_d)

        # Row-based terms
        # Note the following are broadcast to the whole array
        row_summation_stop_d1 = rowsums_d1[n] - 1
        lprob -= mutils.exact_log_limit(theta[-1] + theta[-2], row_summation_stop_d1)
        gradient[-2:] -= mutils.approx_harmonic_limit(theta[-1] + theta[-2], row_summation_stop_d1)
        constants[1] += mutils.approx_geom_limit(theta[-1] + theta[-2], row_summation_stop_d1)
    return gradient, h_diag, constants, lprob


def blm_log_likelihood_fast(U, v_d, v_d1, theta):
    """
    Compute only the proportional log-likelihood (terms dependent only on parameters considered).
    :param U: Precalculated matrix U from blm_precalc
    :param v_d: Precalculated vector v_d from blm_precalc
    :param v_d1: Precalculated vector v_d1 from blm_precalc
    :param theta: Parameter vector of length D+2
    :return: Proportional log-likelihood
    """
    if np.any(theta < 0):
        return np.inf

    D1 = len(U)
    z_d = len(v_d)
    z_d1 = len(v_d1)
    lprob = 0
    sum_theta = np.sum(theta[:-2])
    # z_d terms
    for z in range(z_d):
        # alpha_d parameters
        for d in range(D1 - 1):
            if z < len(U[d]):
                lprob += U[d][z] * np.log(theta[d] + z)
        lprob -= v_d[z] * np.log(sum_theta + z)

        # beta parameter
        if z < len(U[D1 - 1]):
            lprob += U[D1 - 1][z] * np.log(theta[-2] + z)

        # alpha parameter
        lprob += (v_d[z] * np.log(theta[-1] + z)) - (v_d1[z] * np.log(theta[-2] + theta[-1] + z))

    # z_d+1 terms
    for z in range(z_d, z_d1):
        # beta parameter
        if z < len(U[D1 - 1]):
            lprob += U[D1 - 1][z] * np.log(theta[-2] + z)
        # alpha parameter
        lprob -= v_d1[z] * np.log(theta[-2] + theta[-1] + z)
    return lprob


def blm_step(h, g, c):
    """
    Compute a single Newton-Raphson iteration
    :param h: Hessian diagonal vector, length D+2
    :param g: Gradient vector, length D+2
    :param c: Vector of scalar constants, length 2
    :return: Vector deltas of length D+2 with values for the computed changes to the parameters
    """
    D = g.shape[0] - 2
    deltas = np.zeros(D, dtype=np.float64)
    # Invert the hessian diagonal
    h = np.power(h, -1)
    for d in range(D):
        deltas[d] = h[d] * (g[d] - (np.inner(g[:D], h[:D]) / ((c[0] ** -1) + np.sum(h[:D]))))
    delta_beta = h[-2] * (g[-2] - (((g[-1] * h[-1]) + (g[-2] * h[-2])) / ((c[1]**-1) + h[-2] + h[-1])))
    delta_alpha = h[-1] * (g[-1] - (((g[-1] * h[-1]) + (g[-2] * h[-2])) / ((c[1]**-1) + h[-2] + h[-1])))
    return np.append(deltas, (delta_beta, delta_alpha))


def blm_renormalize(theta):
    """
    Normalize the estimates for p_d to the unit simplex according to the mean of the BL for parameter p_d.
    :param theta: Vector of BLM parameters as output from MLE, length D+2
    :return: Normalized parameters on the simplex, length D+1
    """
    d_params = (theta[-1] / (theta[-1] + theta[-2])) * (theta[:-2] / np.sum(theta[:-2]))
    return np.append(d_params, theta[-2] / (theta[-2] + theta[-1]))


def blm_renormalize_empirical(theta, x):
    """
    Calculate parameter estimates for p_d to the unit simplex according to the posterior of the BL for parameter p_d.
    This differs from the above, since we are using the training data to both estimate the maximum likelihood AND to
    inform the posterior parameters, a la empirical Bayes.  The above, regular method only uses the MLE to inform the
    parameters.
    :param theta: Vector of BLM parameters are output from MLE, length D+2
    :param x: Vector of count data summed across all observations used as input to the MLE, dimension (1, D+1)
    :return: Empirically informed parameter estimates on the simplex, length D+1
    """
    beta_term = (theta[-1] + np.sum(x[:-1])) / (theta[-1] + theta[-2] + np.sum(x))
    dirichlet_numerator = theta[:-2] + x[:-1]
    theta_d = beta_term * (dirichlet_numerator / np.sum(dirichlet_numerator))
    return np.append(theta_d, 1 - np.sum(theta_d))


def blm_extract_generating_params(theta):
    return theta[:-1]


def blm_half_stepping(U, vd, vd1, params, lprob, current_lprob, deltas, threshold):
    """
    Step half the distance to the current deltas iteratively until the learning rate drops below a threshold.
    :param U: Precalculated matrix U from dm_precalc
    :param vd: Precomputed vd vector from blm_precalc
    :param vd1: Precomputed vd1 vector from blm_precalc
    :param params: Parameter vector of length D
    :param lprob: Log-likelihood to improve iteratively
    :param current_lprob: Log-likelihood to beat (current max)
    :param deltas: Current deltas from Newton-Raphson step
    :param threshold: Learning rate threshold (minimum value)
    :return: Parameter vector, new log-likelihood, and True if success, else False if below learning threshold
    """
    local_params = params
    local_lprob = lprob
    learn_rate = 0.9
    while local_lprob < current_lprob:
        print('halfstep', lprob, current_lprob)
        if learn_rate < threshold:
            print("BLM MLE failed half stepping, best Lprob: {}".format(blm_log_likelihood_fast(U, vd, vd1, params)))
            return params, current_lprob, False
        local_params = params - (learn_rate * deltas)
        learn_rate *= 0.5
        local_lprob = blm_log_likelihood_fast(U, vd, vd1, local_params)
        print("Half Step Lprob", local_lprob, current_lprob)
        if local_lprob == np.inf:
            print("Half step deltas produced negative parameter estimates")
            local_lprob = lprob
            continue
    return local_params, local_lprob, True


def blm_check_concavity(X, theta):
    """
    Check that alpha_1 and alpha parameters are sufficiently large to ensure a concave likelihood function for
    the BLM MLE process.  See the supplement on proof of concavity for the BLM for more details.
    :param X: Data matrix of counts, dimension (N, D)
    :param theta: Parameter vector of length D
    :return: Tuple (True, None) or (False, [alpha_1_delta, alpha_delta]); True is returned if concavity is ensured,
    and False is returned otherwise along with a list of two values: alpha_1_delta and alpha_delta, which are the
    values that need to be added to alpha_1 and alpha to ensure concavity of the likelihood function.
    """
    concave = True
    eps_values = [0., 0.]
    rowsums_d = np.squeeze(np.sum(X[:, :-1], axis=1))
    rowsums_d1 = np.squeeze(np.sum(X, axis=1))
    sum_theta = np.sum(theta[:-2])
    alpha_1_lhs = sum(mutils.exact_geom_limit(theta[0], x1 - 1) for x1 in X[:, 0] if x1 != 0)
    alpha_1_rhs = sum(mutils.exact_geom_limit(sum_theta, rsum - 1) for rsum in rowsums_d if rsum != 0)
    if alpha_1_rhs >= alpha_1_lhs:
        concave = False
        eps_values[0] = alpha_1_rhs - alpha_1_lhs + 1e-20

    alpha_lhs = sum(mutils.exact_geom_limit(theta[-1], rsum - 1) for rsum in rowsums_d if rsum != 0)
    alpha_rhs = sum(mutils.exact_geom_limit(theta[-2] + theta[-1], rsum - 1) for rsum in rowsums_d1)
    if alpha_rhs >= alpha_lhs:
        concave = False
        eps_values[1] = alpha_rhs - alpha_lhs + 1e-20
    return (True, None) if concave else (False, eps_values)


def blm_newton_raphson2(X, U, vd, vd1, params, precompute,
                        max_steps, delta_eps_threshold, delta_lprob_threshold, label, verbose=False):
    current_lprob = -2e20
    delta_lprob = 2e20
    delta_params = 2e20
    step = 0
    while (delta_params > delta_eps_threshold) and (step < max_steps) and (delta_lprob > delta_lprob_threshold):
        concave, eps = blm_check_concavity(X, params)
        if not concave:
            if verbose:
                print('DEBUG: Concavity correction: {}, {}'.format(eps[0], eps[1]))
            params[0] += eps[0]
            params[-1] += eps[1]
        step += 1
        if precompute == 'sklar':
            g, h, c, lprob = blm_hessian_precompute(U, vd, vd1, params)
        elif precompute == 'vectorized':
            g, h, c, lprob = blm_hessian_precompute_exact_vectorized(X, params)
        elif precompute == 'approximate':
            g, h, c, lprob = blm_hessian_precompute_approximate(X, params)
        else:
            raise ValueError('Precompute must be one of [sklar, vectorized, approximate]: {}'.format(precompute))
        delta_lprob = np.abs(lprob - current_lprob)
        current_lprob = lprob
        deltas = blm_step(h, g, c)

        # The following if statement estimates the beta parameter once, then fixes it relative to alpha.  This was added
        # to prevent situations where the beta parameter being free results in overparameterization and simultaneous
        # increases to beta and alpha indefinitely during the MLE process, resulting in poor accuracy.
        if params[-2] > 10000:
            r = deltas[-1] / (deltas[-1] + deltas[-2])  # Ratio to preserve
            params[-1] = (r * params[-2]) / (1 - r)  # calculate new alpha based on delta ratio
            deltas[-1] = 0  # no need to change alpha now
            deltas[-2] = 0  # fix beta param
            delta_params = np.sum(np.abs(deltas[:-2]))
        else:
            delta_params = np.sum(np.abs(deltas[:-2])) + (
                    deltas[-2] / (deltas[-2] + deltas[-1]))  # See supplement on BLM

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
        if np.any(params < 0):
            params[params < 0] = 1e-20
            # raise ValueError('Negative parameters detected, exiting: {}'.format(
            #     params[params < 0]
            # ))
    print('BLM MLE Exiting: {}, Total steps: {} / {}\n'.format(
        label,
        step,
        max_steps
    ))
    return params


def blm_marginal_log_likelihood(Query_matrix, Training_matrix):
    log_marginal_probabilities = np.zeros((Query_matrix.shape[0], Training_matrix.shape[1]), dtype=np.float64)
    training_feature_sums = np.sum(Training_matrix, axis=1)

    # Find where the beta parameter lies and reorder the vectors such that the beta parameters are terminal
    beta_idx = None
    for i in range(len(training_feature_sums) - 2, -1, -1):
        if training_feature_sums[i] > 0:
            beta_idx = i
            break

    # Pseudo-Lidstone smoothing for aposteriori
    pseudo_value = 1. / Training_matrix.shape[0]
    Training_matrix += pseudo_value

    for observation in range(Query_matrix.shape[0]):
        if beta_idx < (Query_matrix.shape[1] - 1):
            query_vec = np.concatenate((Query_matrix[observation, :beta_idx],
                                        Query_matrix[observation, (beta_idx + 1):]))
            query_vec = np.append(query_vec, Query_matrix[observation, beta_idx])
        else:
            query_vec = Query_matrix[observation, :]
        for label in range(Training_matrix.shape[1]):
            train_vec = np.concatenate((Training_matrix[:beta_idx, label],
                                        Training_matrix[(beta_idx + 1):-1, label]))
            train_vec = np.append(train_vec, Training_matrix[beta_idx, label])
            train_vec = np.append(train_vec, Training_matrix[-1, label])

            # Calculate the marginal probability of this query vector against each class in training matrix
            query_sum_d = int(np.sum(query_vec[:-1]))
            query_sum_d1 = int(np.sum(query_vec))
            sum_theta = np.sum(train_vec[:-2])
            lprob = 0
            lprob += mutils.exact_log_limit(1, query_sum_d1 - 1)  # Count sum term
            lprob += sum(mutils.exact_log_limit(train_vec[d], int(query_vec[d]) - 1) for d in range(len(query_vec) - 1)
                         if query_vec[d] != 0)  # alpha_d term
            if query_sum_d != 0:
                lprob += mutils.exact_log_limit(train_vec[-1], query_sum_d - 1)  # alpha term
                lprob -= mutils.exact_log_limit(sum_theta, query_sum_d - 1)  # alpha_d sum term
            if query_vec[-1] != 0:
                lprob += mutils.exact_log_limit(train_vec[-2], int(query_vec[-1]) - 1)  # beta term
            lprob -= mutils.exact_log_limit(train_vec[-2] + train_vec[-1], query_sum_d1)  # alpha and beta sum term
            lprob -= sum(mutils.exact_log_limit(1, int(query_vec[d]) - 1) for d in range(len(query_vec))
                         if query_vec[d] != 0)  # Count term
            log_marginal_probabilities[observation, label] = lprob
    return log_marginal_probabilities


if __name__ == '__main__':
    test_matrix = np.matrix('2 7 3; 1 8 2; 1 7 2', dtype=np.int32)
    test_u, test_vd, test_vd1 = blm_precalc(test_matrix)

    # test_init = blm_init_params(test_matrix)
    test_init = np.array((1, 1, 1, 1), dtype=np.float64)
    # np.testing.assert_array_almost_equal(test_init,
    #                                      np.array([119.76577944, 673.42876828, 211.62004248, 793.19454772],
    #                                               dtype=np.float64))

    test_g, test_hd, test_c, test_lprob = blm_hessian_precompute(test_u, test_vd, test_vd1, test_init)
    print("log likelihood: {}".format(blm_log_likelihood_fast(test_u, test_vd, test_vd1, test_init)))
    print(test_g)
    print(test_hd)
    print(test_c)
    print(test_lprob)
    #
    # np.testing.assert_almost_equal(test_lprob, np.array([-201.94890575]))
    # np.testing.assert_array_almost_equal(test_g, np.array([0.000708, -0.000106,  0.000289, -0.032679],
    #                                                       dtype=np.float64))
    # np.testing.assert_array_almost_equal(test_hd,
    #                                      np.array([-2.36787047e-04, -7.12869814e-06, -1.22903642e-04, 3.23606946e-05],
    #                                               dtype=np.float64))
    # np.testing.assert_array_almost_equal(test_c, np.array([4.09284439e-05, 3.23606946e-05], dtype=np.float64))

    test_deltas = blm_step(test_hd, test_g, test_c)
    print(test_init)
    print(test_deltas)
    print(test_init - test_deltas)
    # np.testing.assert_array_almost_equal(test_deltas,
    #                                      np.array([-3.40778861, 1.00046424, -155.80408545, -427.01017481],
    #                                               dtype=np.float64))

    normalized = blm_renormalize(test_init - test_deltas)
    print(normalized)
    print(sum(normalized))
    # np.testing.assert_array_almost_equal(normalized, np.array([0.118989,  0.649582,  0.231429], dtype=np.float64))
