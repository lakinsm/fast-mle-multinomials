import numpy as np


def gdm_precalc(X):
    """
    Calculate the U matrix of dimension (D, Z_(D+1)), v^D1 vector of length Z_sum(D+1), and
    v^D2 vector of length Z_sum(D+1) for the GDM.
    :param X: Data matrix of counts, dimension (N, D+1)
    :return: Matrix U, vector v^D1, vector v^D2
    """
    # Variable definitions
    D1 = X.shape[1]
    N = X.shape[0]

    # Initialize data structures
    U = tuple([] for _ in range(D1 - 1))
    Vd = tuple([] for _ in range(D1))

    # Compute Algorithm 2 (Lakin & Abdo 2019)
    for n in range(N):
        C = np.zeros((D1, 1), dtype=np.int32)
        for d in range(D1 - 1):
            if d != 0:
                C[d, 0] = C[d - 1, 0] - X[n, d - 1]
            else:
                C[d, 0] = np.sum(X[n, :])
            for i in range(X[n, d]):
                try:
                    U[d][i] += 1
                except IndexError:
                    U[d].append(1)
        C[D1 - 1, 0] = C[D1 - 2, 0] - X[n, D1 - 2]
        for d in range(C.shape[0]):
            for i in range(C[d, 0]):
                try:
                    Vd[d][i] += 1
                except IndexError:
                    Vd[d].append(1)
    return U, Vd


def gdm_init_params(X):
    """
    Initialize parameters for the GDM distribution using moment matching to the DM.  This is imperfect, but
    the GDM is too complex to moment match directly
    :param X: Data matrix of counts, dimension (N, D+1)
    :return: vector of parameter initial values theta, length 2D
    """
    D = X.shape[1]
    rowsums = X.sum(axis=1)
    Xnorm = (X / rowsums.reshape(-1, 1).astype(float))
    mean = np.array(np.mean(Xnorm, axis=0))
    # mean[mean == 1] = 2  # required to prevent division by zero due to (mean * (1 - mean))
    m2 = np.array(np.mean(np.power(Xnorm, 2), axis=0))
    nonzeros = np.array(mean > 0)
    sum_alpha = np.divide((mean[nonzeros] - m2[nonzeros]), (m2[nonzeros] - (np.power(mean[nonzeros], 2))))
    sum_alpha[sum_alpha == 0] = 1  # required to prevent division by zero
    var_pk = np.divide(np.multiply(mean, 1 - mean), 1 + sum_alpha)
    log_sum_alpha = ((D - 1) ** -1) * np.sum(np.log(np.divide(np.multiply(mean, 1 - mean), var_pk) - 1))
    s = np.exp(log_sum_alpha)
    if s == 0:
        s = 1
    return_params = np.squeeze(np.array(s * mean))
    return np.repeat(return_params[:-1], 2)


def gdm_hessian_precompute(U, vd, theta):
    """
    Precompute the gradient, Hessian diagonal, hessian constant, and log-likelihood log(P(X | theta))
    :param U: Precalculated matrix U from gdm_precalc, dimension (D, Z_D)
    :param vd: Precalculated vector vd from gdm_precalc, length z_sumD1
    :param theta: Parameter vector of length 2D
    :return: gradient vector, Hessian diagonal vector, constants vector, and scalar log-likelihood
    """
    if np.any(theta < 0):
        return None, None, None, np.inf

    D = len(U)
    D2 = theta.shape[0]
    z_d2 = max((len(x) for x in vd))
    lprob = 0
    constants = np.zeros(D, np.float64)
    gradient = np.zeros(D2, np.float64)
    h_diag = np.zeros(D2, np.float64)
    for z in range(z_d2):
        for d in range(D):
            alpha_idx = 2 * d
            if z < len(U[d]):
                lprob += U[d][z] * np.log(theta[alpha_idx] + z)  # alpha_d
                gradient[alpha_idx] += (U[d][z] * ((theta[alpha_idx] + z)**-1))  # alpha_d
                h_diag[alpha_idx] -= U[d][z] * ((theta[alpha_idx] + z)**-2)  # alpha_d
            if z < len(vd[d+1]):
                lprob += vd[d+1][z] * np.log(theta[alpha_idx + 1] + z)  # beta_d
                gradient[alpha_idx + 1] += (vd[d+1][z] * ((theta[alpha_idx + 1] + z) ** -1))  # beta_d
                h_diag[alpha_idx + 1] -= vd[d+1][z] * ((theta[alpha_idx + 1] + z)**-2)  # beta_d
            if z < len(vd[d]):
                lprob -= vd[d][z] * np.log(theta[alpha_idx] + theta[alpha_idx + 1] + z)  # alpha_d + beta_d
                gradient[alpha_idx] -= vd[d][z] * ((theta[alpha_idx] + theta[alpha_idx + 1] + z)**-1)  # alpha_d
                gradient[alpha_idx + 1] -= vd[d][z] * ((theta[alpha_idx] + theta[alpha_idx + 1] + z)**-1)  # beta_d
                # h_diag[alpha_idx] += vd[d][z] * ((theta[alpha_idx] + theta[alpha_idx + 1] + z)**-2)  # alpha_d
                # h_diag[alpha_idx + 1] += vd[d][z] * ((theta[alpha_idx] + theta[alpha_idx + 1] + z)**-2)  # beta_d
                constants[d] += vd[d][z] * ((theta[alpha_idx] + theta[alpha_idx + 1] + z) ** -2)
        alpha_idx = D
        if z < len(vd[D]):
            lprob -= vd[D][z] * np.log(theta[alpha_idx] + theta[alpha_idx + 1] + z)  # alpha_d + beta_d
            gradient[alpha_idx] -= vd[D][z] * ((theta[alpha_idx] + theta[alpha_idx + 1] + z) ** -1)  # alpha_d
            gradient[alpha_idx + 1] -= vd[D][z] * ((theta[alpha_idx] + theta[alpha_idx + 1] + z) ** -1)  # beta_d
            # h_diag[alpha_idx] += vd[D][z] * ((theta[alpha_idx] + theta[alpha_idx + 1] + z) ** -2)  # alpha_d
            # h_diag[alpha_idx + 1] += vd[D][z] * ((theta[alpha_idx] + theta[alpha_idx + 1] + z) ** -2)  # beta_d
            constants[D - 1] += vd[D][z] * ((theta[alpha_idx] + theta[alpha_idx + 1] + z) ** -2)
    return gradient, h_diag, constants, lprob


def gdm_log_likelihood_fast(U, vd, theta):
    """
    Compute only the proportional log-likelihood (terms dependent only on parameters considered).
    :param U: Precalculated matrix U from gdm_precalc
    :param vd: Precalculated vector vd from gdm_precalc
    :param theta: Parameter vector of length 2D
    :return: Proportional log-likelihood
    """
    if np.any(theta < 0):
        return np.inf

    D = len(U)
    z_d2 = max((len(x) for x in vd))
    lprob = 0
    for z in range(z_d2):
        for d in range(D):
            alpha_idx = 2 * d
            if z < len(U[d]):
                lprob += U[d][z] * np.log(theta[alpha_idx] + z)  # alpha_d
            if z < len(vd[d+1]):
                lprob += vd[d+1][z] * np.log(theta[alpha_idx + 1] + z)  # beta_d
            if z < len(vd[d]):
                lprob -= vd[d][z] * np.log(theta[alpha_idx] + theta[alpha_idx + 1] + z)  # alpha_d + beta_d
        alpha_idx = D
        if z < len(vd[D]):
            lprob -= vd[D][z] * np.log(theta[alpha_idx] + theta[alpha_idx + 1] + z)  # alpha_d + beta_d
    return lprob


def gdm_step(h, g, c):
    """
    Compute a single Newton-Raphson iteration
    :param h: Hessian diagonal vector, length 2D
    :param g: Gradient vector, length 2D
    :param c: Vector of scalar constants, length D
    :return: Vector deltas of length 2D with values for the computed changes to the parameters
    """
    D = g.shape[0] // 2
    deltas = np.zeros(2 * D, dtype=np.float32)
    # Invert the hessian diagonal
    h = np.power(h, -1)
    for d in range(D):
        a = 2 * d
        deltas[a] = h[a] * (g[a] - (((g[a + 1] * h[a + 1]) + (g[a] * h[a])) / ((c[d]**-1) + h[a] + h[a + 1])))  # alpha_d
        deltas[a + 1] = h[a + 1] * (g[a + 1] - (((g[a + 1] * h[a + 1]) + (g[a] * h[a])) / ((c[d]**-1) + h[a] + h[a + 1])))  # beta_d
    return deltas


def gdm_renormalize(theta):
    """
    Normalize the estimates for p_d to the unit simplex according to the mean of the GD for parameter p_d.
    :param theta: Vector of GDM parameters as output from MLE estimation, length 2D
    :return: Normalized parameters on the simplex, length D+1
    """
    d_params = np.zeros((theta.shape[0] // 2) + 1, dtype=np.float64)
    for d in range(theta.shape[0] // 2):
        a = 2 * d
        d_params[d] = theta[a] / (theta[a] + theta[a + 1])
        for k in range(d):
            a2 = 2 * k
            d_params[d] *= theta[a2] / (theta[a2] + theta[a2 + 1])
    d_params[-1] = 1 - np.sum(d_params[:-1])
    return d_params


def gdm_extract_generating_params(theta):
    alphas = theta[::2]
    return np.append(alphas, theta[0] / (theta[0] + theta[1]))


def dm_half_stepping(U, vd, params, lprob, current_lprob, deltas, threshold):
    """
    Step half the distance to the current deltas iteratively until the learning rate drops below a threshold.
    :param U: Precalculated matrix U from gdm_precalc
    :param vd: Precalculated vector vd from gdm_precalc
    :param params: Parameter vector of length 2D
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
            print("GDM MLE converged with small learn rate")
            return params, local_lprob, False
        local_params = params - (learn_rate * deltas)
        learn_rate *= 0.5
        local_lprob = gdm_log_likelihood_fast(U, vd, local_params)
        if local_lprob == np.inf:
            local_lprob = lprob
            continue
    return local_params, local_lprob, True


def gdm_newton_raphson(U, vd, params, max_steps, gradient_sq_threshold, learn_rate_threshold, delta_lprob_threshold):
    """
    Find the MLE for the Dirichlet multinomial given the precomputed data structures and initial parameter estimates.
    :param U: Precomputed U matrix from gdm_precalc
    :param vd: Precomputed vd vector from gdm_precalc
    :param params: Initial parameter estimates in D+1 dimensions
    :param max_steps: Max iterations to perform Newton-Raphson stepping
    :param gradient_sq_threshold: Threshold under which optimization stops for the sum of squared gradients
    :param learn_rate_threshold: Threshold under which optimization stops for half-stepping
    :return: Results of the MLE computation for parameters 1:2D
    """
    current_lprob = -2 ** 20
    delta_lprob = 2 ** 20
    step = 0
    while step < max_steps:
        if delta_lprob < delta_lprob_threshold:
            print("GDM MLE converged with small delta log-probability")
            return gdm_renormalize(params)
        step += 1
        g, h, c, lprob = gdm_hessian_precompute(U, vd, params)
        gradient_sq = np.sum(np.power(g, 2))
        if gradient_sq < gradient_sq_threshold:
            print("GDM MLE converged with small gradient")
            return gdm_renormalize(params)
        deltas = gdm_step(h, g, c)
        if lprob > current_lprob:
            test_params = params - deltas
            if np.any(test_params < 0):
                params, lprob, success = dm_half_stepping(U, vd, params, -2 ** 20, current_lprob, deltas, learn_rate_threshold)
                if not success:
                    return gdm_renormalize(params)
                else:
                    delta_lprob = np.abs(lprob - current_lprob)
                    current_lprob = lprob
            else:
                delta_lprob = np.abs(lprob - current_lprob)
                current_lprob = lprob
                params -= deltas
        else:
            params, lprob, success = dm_half_stepping(U, vd, params, lprob, current_lprob, deltas, learn_rate_threshold)
            if not success:
                return gdm_renormalize(params)
            else:
                delta_lprob = np.abs(lprob - current_lprob)
                current_lprob = lprob
    print("GDM MLE reached max iterations")
    return gdm_renormalize(params)


if __name__ == '__main__':
    test_matrix = np.matrix('2 7 3; 1 8 2; 1 7 2', dtype=np.int32)
    test_u, test_vd = gdm_precalc(test_matrix)

    test_init = gdm_init_params(test_matrix)
    np.testing.assert_array_almost_equal(test_init,
                                  np.array([119.76577944, 119.76577944, 673.42876828, 673.42876828], dtype=np.float64))
    test_g, test_hd, test_c, test_lprob = gdm_hessian_precompute(test_u, test_vd, test_init)
    test_deltas = gdm_step(test_hd, test_g, test_c)
