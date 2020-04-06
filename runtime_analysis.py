#!/usr/bin/env python3

import sys
import timeit
import random
import numpy as np
import multi_mle.mle_utils as mutils
import multi_mle.dm as dm
import multi_mle.blm as blm


random.seed(2718)

delta_eps_threshold = 1e-5
learn_rate_threshold = 2e-10
delta_lprob_threshold = 1e-5
max_steps = 20
repeat_timings = 5

precalc_methods = ('vectorized', 'approximate', 'sklar')


def vary_parameter_count(results_file):
    """
    Test average run time and memory consumption for varying the dimensionality D for the DM, BLM, and GDM MLE.
    """
    # Setup
    N = 20  # number of multinomials
    M = 100  # number of samples per multinomial
    thetas = range(10, 5010, 200)

    for t in thetas:
        generating_params = np.array([random.random() for _ in range(t)])
        generating_params /= np.sum(generating_params)
        X, mean_params = mutils.multinomial_random_sample(N, M, generating_params)

        # Dirichlet Multinomial
        precalc_times = {'vectorized': (), 'approximate': (), 'sklar': ()}
        U, v = dm.dm_precalc(X)
        params = dm.dm_init_params(X)
        with open(results_file, 'a') as out:
            # Timing
            for precalc_method in precalc_methods:
                for _ in range(repeat_timings):
                    precalc_start = timeit.default_timer()
                    if precalc_method == 'vectorized':
                        g, h, c, lprob = dm.dm_hessian_precompute_exact_vectorized(X, params)
                    elif precalc_method == 'approximate':
                        g, h, c, lprob = dm.dm_hessian_precompute_approximate(X, params)
                    elif precalc_method == 'sklar':
                        g, h, c, lprob = dm.dm_hessian_precompute(U, v, params)
                    else:
                        raise ValueError('Precalc_method must be one of [vectorized, approximate, sklar].'
                                         'Provided: {}'.format(precalc_method))
                    precalc_stop = timeit.default_timer()
                    print(len(g), len(h), c, lprob)  # to prevent compiler from optimizing out the code
                    precalc_times[precalc_method] += (precalc_stop - precalc_start,)

            for method, vals in precalc_times.items():
                for v in vals:
                    # Distribution,PrecalcMethod,ParamCount,NCount,MCount,PrecalcTime
                    out.write('{},{},{},{},{},{},{}\n'.format(
                        'VaryParameters',
                        'DM',
                        method,
                        t,
                        N,
                        M,
                        v
                    ))

        # Beta-Liouville Multinomial
        precalc_times = {'vectorized': (), 'approximate': (), 'sklar': ()}
        U, vd, vd1 = blm.blm_precalc(X)
        params = blm.blm_init_params(X)
        with open(results_file, 'a') as out:
            # Timing
            for precalc_method in precalc_methods:
                for _ in range(repeat_timings):
                    precalc_start = timeit.default_timer()
                    if precalc_method == 'vectorized':
                        g, h, c, lprob = blm.blm_hessian_precompute_exact_vectorized(X, params)
                    elif precalc_method == 'approximate':
                        g, h, c, lprob = blm.blm_hessian_precompute_approximate(X, params)
                    elif precalc_method == 'sklar':
                        g, h, c, lprob = blm.blm_hessian_precompute(U, vd, vd1, params)
                    else:
                        raise ValueError('Precalc_method must be one of [vectorized, approximate, sklar].'
                                         'Provided: {}'.format(precalc_method))
                    precalc_stop = timeit.default_timer()
                    print(len(g), len(h), c, lprob)  # to prevent compiler from optimizing out the code
                    precalc_times[precalc_method] += (precalc_stop - precalc_start,)

            for method, vals in precalc_times.items():
                for v in vals:
                    # Experiment,Distribution,PrecalcMethod,ParamCount,NCount,MCount,PrecalcTime
                    out.write('{},{},{},{},{},{},{}\n'.format(
                        'VaryParameters',
                        'BLM',
                        method,
                        t,
                        N,
                        M,
                        v
                    ))


def vary_observation_count(results_file):
    """
    Test average run time and memory consumption for varying the number of observations N
    for the DM, BLM, and GDM MLE.
    """
    # Setup
    N = range(10, 5010, 200)  # number of multinomials
    M = 100  # number of samples per multinomial
    thetas = 20

    generating_params = np.array([random.random() for _ in range(thetas)])
    generating_params /= np.sum(generating_params)

    for n in N:
        X, mean_params = mutils.multinomial_random_sample(n, M, generating_params)

        # Dirichlet Multinomial
        precalc_times = {'vectorized': (), 'approximate': (), 'sklar': ()}
        U, v = dm.dm_precalc(X)
        params = dm.dm_init_params(X)
        with open(results_file, 'a') as out:
            # Timing
            for precalc_method in precalc_methods:
                for _ in range(repeat_timings):
                    precalc_start = timeit.default_timer()
                    if precalc_method == 'vectorized':
                        g, h, c, lprob = dm.dm_hessian_precompute_exact_vectorized(X, params)
                    elif precalc_method == 'approximate':
                        g, h, c, lprob = dm.dm_hessian_precompute_approximate(X, params)
                    elif precalc_method == 'sklar':
                        g, h, c, lprob = dm.dm_hessian_precompute(U, v, params)
                    else:
                        raise ValueError('Precalc_method must be one of [vectorized, approximate, sklar].'
                                         'Provided: {}'.format(precalc_method))
                    precalc_stop = timeit.default_timer()
                    print(len(g), len(h), c, lprob)  # to prevent compiler from optimizing out the code
                    precalc_times[precalc_method] += (precalc_stop - precalc_start,)

            for method, vals in precalc_times.items():
                for v in vals:
                    # Distribution,PrecalcMethod,ParamCount,NCount,MCount,PrecalcTime
                    out.write('{},{},{},{},{},{},{}\n'.format(
                        'VaryObservations',
                        'DM',
                        method,
                        thetas,
                        n,
                        M,
                        v
                    ))

        # Beta-Liouville Multinomial
        precalc_times = {'vectorized': (), 'approximate': (), 'sklar': ()}
        U, vd, vd1 = blm.blm_precalc(X)
        params = blm.blm_init_params(X)
        with open(results_file, 'a') as out:
            # Timing
            for precalc_method in precalc_methods:
                for _ in range(repeat_timings):
                    precalc_start = timeit.default_timer()
                    if precalc_method == 'vectorized':
                        g, h, c, lprob = blm.blm_hessian_precompute_exact_vectorized(X, params)
                    elif precalc_method == 'approximate':
                        g, h, c, lprob = blm.blm_hessian_precompute_approximate(X, params)
                    elif precalc_method == 'sklar':
                        g, h, c, lprob = blm.blm_hessian_precompute(U, vd, vd1, params)
                    else:
                        raise ValueError('Precalc_method must be one of [vectorized, approximate, sklar].'
                                         'Provided: {}'.format(precalc_method))
                    precalc_stop = timeit.default_timer()
                    print(len(g), len(h), c, lprob)  # to prevent compiler from optimizing out the code
                    precalc_times[precalc_method] += (precalc_stop - precalc_start,)

            for method, vals in precalc_times.items():
                for v in vals:
                    # Distribution,PrecalcMethod,ParamCount,NCount,MCount,PrecalcTime
                    out.write('{},{},{},{},{},{},{}\n'.format(
                        'VaryObservations',
                        'BLM',
                        method,
                        thetas,
                        n,
                        M,
                        v
                    ))


def vary_sampling_count(results_file):
    """
    Test average run time and memory consumption for varying the number of draws Z
    for the DM, BLM, and GDM MLE.
    """
    # Setup
    N = 20  # number of multinomials
    M = range(10, 5010, 200)  # number of samples per multinomial
    thetas = 20

    generating_params = np.array([random.random() for _ in range(thetas)])
    generating_params /= np.sum(generating_params)

    for m in M:
        X, mean_params = mutils.multinomial_random_sample(N, m, generating_params)

        # Dirichlet Multinomial
        precalc_times = {'vectorized': (), 'approximate': (), 'sklar': ()}
        U, v = dm.dm_precalc(X)
        params = dm.dm_init_params(X)
        with open(results_file, 'a') as out:
            # Timing
            for precalc_method in precalc_methods:
                for _ in range(repeat_timings):
                    precalc_start = timeit.default_timer()
                    if precalc_method == 'vectorized':
                        g, h, c, lprob = dm.dm_hessian_precompute_exact_vectorized(X, params)
                    elif precalc_method == 'approximate':
                        g, h, c, lprob = dm.dm_hessian_precompute_approximate(X, params)
                    elif precalc_method == 'sklar':
                        g, h, c, lprob = dm.dm_hessian_precompute(U, v, params)
                    else:
                        raise ValueError('Precalc_method must be one of [vectorized, approximate, sklar].'
                                         'Provided: {}'.format(precalc_method))
                    precalc_stop = timeit.default_timer()
                    print(len(g), len(h), c, lprob)  # to prevent compiler from optimizing out the code
                    precalc_times[precalc_method] += (precalc_stop - precalc_start,)

            for method, vals in precalc_times.items():
                for v in vals:
                    # Distribution,PrecalcMethod,ParamCount,NCount,MCount,PrecalcTime
                    out.write('{},{},{},{},{},{},{}\n'.format(
                        'VaryMultinomDraws',
                        'DM',
                        method,
                        thetas,
                        N,
                        m,
                        v
                    ))

        # Beta-Liouville Multinomial
        precalc_times = {'vectorized': (), 'approximate': (), 'sklar': ()}
        U, vd, vd1 = blm.blm_precalc(X)
        params = blm.blm_init_params(X)
        with open(results_file, 'a') as out:
            # Timing
            for precalc_method in precalc_methods:
                for _ in range(repeat_timings):
                    precalc_start = timeit.default_timer()
                    if precalc_method == 'vectorized':
                        g, h, c, lprob = blm.blm_hessian_precompute_exact_vectorized(X, params)
                    elif precalc_method == 'approximate':
                        g, h, c, lprob = blm.blm_hessian_precompute_approximate(X, params)
                    elif precalc_method == 'sklar':
                        g, h, c, lprob = blm.blm_hessian_precompute(U, vd, vd1, params)
                    else:
                        raise ValueError('Precalc_method must be one of [vectorized, approximate, sklar].'
                                         'Provided: {}'.format(precalc_method))
                    precalc_stop = timeit.default_timer()
                    print(len(g), len(h), c, lprob)  # to prevent compiler from optimizing out the code
                    precalc_times[precalc_method] += (precalc_stop - precalc_start,)

            for method, vals in precalc_times.items():
                for v in vals:
                    # Distribution,PrecalcMethod,ParamCount,NCount,MCount,PrecalcTime
                    out.write('{},{},{},{},{},{},{}\n'.format(
                        'VaryMultinomDraws',
                        'BLM',
                        method,
                        thetas,
                        N,
                        m,
                        v
                    ))


if __name__ == '__main__':
    vary_observation_count(sys.argv[1])
    vary_sampling_count(sys.argv[1])
    vary_parameter_count(sys.argv[1])
