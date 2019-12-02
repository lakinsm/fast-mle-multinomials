import numpy as np


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
    return np.append(d_params, np.mean(d_params[:-1]))


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
            print('{} / {}'.format(z, z_d))
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
    :param theta: Vector of BLM parameters as output from MLE estimation, length D+2
    :return: Normalized parameters on the simplex, length D+1
    """
    d_params = (theta[-1] / (theta[-1] + theta[-2])) * (theta[:-2] / np.sum(theta[:-2]))
    return np.append(d_params, theta[-2] / (theta[-2] + theta[-1]))


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
    learn_rate = 1.0
    while local_lprob < current_lprob:
        print('halfstep', lprob)
        if learn_rate < threshold:
            print("BLM MLE converged with small learn rate")
            return params, local_lprob, False
        local_params = params - (learn_rate * deltas)
        learn_rate *= 0.5
        local_lprob = blm_log_likelihood_fast(U, vd, vd1, local_params)
        if local_lprob == np.inf:
            local_lprob = lprob
            continue
    return local_params, local_lprob, True


def blm_newton_raphson(U, vd, vd1, params, max_steps, gradient_sq_threshold, learn_rate_threshold, delta_lprob_threshold):
    """
    Find the MLE for the Dirichlet multinomial given the precomputed data structures and initial parameter estimates.
    :param U: Precomputed U matrix from blm_precalc
    :param vd: Precomputed vd vector from blm_precalc
    :param vd1: Precomputed vd1 vector from blm_precalc
    :param params: Initial parameter estimates in D+2 dimensions
    :param max_steps: Max iterations to perform Newton-Raphson stepping
    :param gradient_sq_threshold: Threshold under which optimization stops for the sum of squared gradients
    :param learn_rate_threshold: Threshold under which optimization stops for half-stepping
    :return: Results of the MLE computation for parameters 1:D+1
    """
    current_lprob = -2e20
    delta_lprob = 2e20
    step = 0
    while step < max_steps:
        if delta_lprob < delta_lprob_threshold:
            print("BLM MLE converged with small delta log-probability")
            return blm_renormalize(params)
        step += 1
        g, h, c, lprob = blm_hessian_precompute(U, vd, vd1, params)
        print(lprob, 'Lprob')
        gradient_sq = np.sum(np.power(g, 2))
        if gradient_sq < gradient_sq_threshold:
            print("BLM MLE converged with small gradient")
            return blm_renormalize(params)
        deltas = blm_step(h, g, c)
        temp = params - deltas
        print(list(temp[temp < 0]), 'Test Params')
        if lprob > current_lprob:
            test_params = params - deltas
            if np.any(test_params < 0):
                params, lprob, success = blm_half_stepping(U, vd, vd1, params, -2 ** 20, current_lprob, deltas, learn_rate_threshold)
                if not success:
                    return blm_renormalize(params)
                else:
                    delta_lprob = np.abs(lprob - current_lprob)
                    current_lprob = lprob
            else:
                delta_lprob = np.abs(lprob - current_lprob)
                current_lprob = lprob
                params -= deltas
        else:
            params, lprob, success = blm_half_stepping(U, vd, vd1, params, lprob, current_lprob, deltas, learn_rate_threshold)
            if not success:
                return blm_renormalize(params)
            else:
                delta_lprob = np.abs(lprob - current_lprob)
                current_lprob = lprob
    print("BLM MLE reached max iterations")
    return blm_renormalize(params)


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
