#!/usr/bin/env python3

import sys
import time
import random
import numpy as np
from multi_mle.mle_utils import multinomial_random_sample
from multi_mle.mle_utils import silent_remove
import multi_mle.dm as dm
import multi_mle.blm as blm
import multi_mle.gdm as gdm
random.seed(154)

gradient_sq_threshold = 2**-20
learn_rate_threshold = 2**-10
delta_lprob_threshold = 2**-10
max_steps = 1000
repeat_timings = 20


def vary_parameter_count(results_file):
    """
    Test average run time and memory consumption for varying the dimensionality D for the DM, BLM, and GDM MLE.
    """
    # Setup
    N = 100  # number of multinomials
    M = 100  # number of samples per multinomial
    thetas = range(10, 10010, 50)

    for t in thetas:
        generating_params = np.array([random.random() for _ in range(t)])
        generating_params /= np.sum(generating_params)
        X, mean_params = multinomial_random_sample(N, M, generating_params)

        # Dirichlet Multinomial
        precalc_times = ()
        newton_times = ()
        with open(results_file, 'a') as out:
            # Timing
            for _ in range(repeat_timings):
                precalc_start = time.clock()
                U, v = dm.dm_precalc(X)
                precalc_stop = time.clock()
                precalc_times += (precalc_stop - precalc_start,)
            # U, v = dm.dm_precalc(X)
            params = dm.dm_init_params(X)
            for _ in range(repeat_timings):
                newton_start = time.clock()
                dm.dm_newton_raphson(U, v, params, max_steps, gradient_sq_threshold, learn_rate_threshold,
                                     delta_lprob_threshold)
                newton_stop = time.clock()
                newton_times += (newton_stop - newton_start,)

            for i in range(repeat_timings):
                # Distribution,ParamCount,NCount,MCount,PrecalcAvgTime,NewtonAvgTime
                out.write('{},{},{},{},{},{}\n'.format(
                    'DM',
                    t,
                    N,
                    M,
                    precalc_times[i],
                    newton_times[i]
                ))

        # Beta-Liouville Multinomial
        precalc_times = ()
        newton_times = ()
        with open(results_file, 'a') as out:
            # Timing
            for _ in range(repeat_timings):
                precalc_start = time.clock()
                U, vd, vd1 = blm.blm_precalc(X)
                precalc_stop = time.clock()
                precalc_times += (precalc_stop - precalc_start,)
            # U, vd, vd1 = blm.blm_precalc(X)
            params = blm.blm_init_params(X)
            for _ in range(repeat_timings):
                newton_start = time.clock()
                blm.blm_newton_raphson(U, vd, vd1, params, max_steps, gradient_sq_threshold, learn_rate_threshold,
                                       delta_lprob_threshold)
                newton_stop = time.clock()
                newton_times += (newton_stop - newton_start,)

            for i in range(repeat_timings):
                # Distribution,ParamCount,NCount,MCount,PrecalcAvgTime,NewtonAvgTime
                out.write('{},{},{},{},{},{}\n'.format(
                    'BLM',
                    t,
                    N,
                    M,
                    precalc_times[i],
                    newton_times[i]
                ))

        # Generalized Dirichlet Multinomial
        precalc_times = ()
        newton_times = ()
        with open(results_file, 'a') as out:
            # Timing
            for _ in range(repeat_timings):
                precalc_start = time.clock()
                U, vd = gdm.gdm_precalc(X)
                precalc_stop = time.clock()
                precalc_times += (precalc_stop - precalc_start,)
            # U, vd = gdm.gdm_precalc(X)
            params = gdm.gdm_init_params(X)
            for _ in range(repeat_timings):
                newton_start = time.clock()
                gdm.gdm_newton_raphson(U, vd, params, max_steps, gradient_sq_threshold, learn_rate_threshold,
                                       delta_lprob_threshold)
                newton_stop = time.clock()
                newton_times += (newton_stop - newton_start,)

            for i in range(repeat_timings):
                # Distribution,ParamCount,NCount,MCount,PrecalcAvgTime,NewtonAvgTime
                out.write('{},{},{},{},{},{}\n'.format(
                    'GDM',
                    t,
                    N,
                    M,
                    precalc_times[i],
                    newton_times[i]
                ))


def vary_observation_count(results_file):
    """
    Test average run time and memory consumption for varying the number of observations N
    for the DM, BLM, and GDM MLE.
    """
    # Setup
    N = range(10, 10010, 50)  # number of multinomials
    M = 100  # number of samples per multinomial
    thetas = 100
    generating_params = np.array([random.random() for _ in range(thetas)])
    generating_params /= np.sum(generating_params)

    for n in N:
        X, mean_params = multinomial_random_sample(n, M, generating_params)

        # Dirichlet Multinomial
        precalc_times = ()
        newton_times = ()
        with open(results_file, 'a') as out:
            # Timing
            for _ in range(repeat_timings):
                precalc_start = time.clock()
                U, v = dm.dm_precalc(X)
                precalc_stop = time.clock()
                precalc_times += (precalc_stop - precalc_start,)
            # U, v = dm.dm_precalc(X)
            params = dm.dm_init_params(X)
            for _ in range(repeat_timings):
                newton_start = time.clock()
                dm.dm_newton_raphson(U, v, params, max_steps, gradient_sq_threshold, learn_rate_threshold,
                                     delta_lprob_threshold)
                newton_stop = time.clock()
                newton_times += (newton_stop - newton_start,)

            for i in range(repeat_timings):
                # Distribution,ParamCount,NCount,MCount,PrecalcAvgTime,NewtonAvgTime
                out.write('{},{},{},{},{},{}\n'.format(
                    'DM',
                    thetas,
                    n,
                    M,
                    precalc_times[i],
                    newton_times[i]
                ))

        # Beta-Liouville Multinomial
        precalc_times = ()
        newton_times = ()
        with open(results_file, 'a') as out:
            # Timing
            for _ in range(repeat_timings):
                precalc_start = time.clock()
                U, vd, vd1 = blm.blm_precalc(X)
                precalc_stop = time.clock()
                precalc_times += (precalc_stop - precalc_start,)
            # U, vd, vd1 = blm.blm_precalc(X)
            params = blm.blm_init_params(X)
            for _ in range(repeat_timings):
                newton_start = time.clock()
                blm.blm_newton_raphson(U, vd, vd1, params, max_steps, gradient_sq_threshold, learn_rate_threshold,
                                       delta_lprob_threshold)
                newton_stop = time.clock()
                newton_times += (newton_stop - newton_start,)

            for i in range(repeat_timings):
                # Distribution,ParamCount,NCount,MCount,PrecalcAvgTime,NewtonAvgTime
                out.write('{},{},{},{},{},{}\n'.format(
                    'BLM',
                    thetas,
                    n,
                    M,
                    precalc_times[i],
                    newton_times[i]
                ))

        # Generalized Dirichlet Multinomial
        precalc_times = ()
        newton_times = ()
        with open(results_file, 'a') as out:
            # Timing
            for _ in range(repeat_timings):
                precalc_start = time.clock()
                U, vd = gdm.gdm_precalc(X)
                precalc_stop = time.clock()
                precalc_times += (precalc_stop - precalc_start,)
            # U, vd = gdm.gdm_precalc(X)
            params = gdm.gdm_init_params(X)
            for _ in range(repeat_timings):
                newton_start = time.clock()
                gdm.gdm_newton_raphson(U, vd, params, max_steps, gradient_sq_threshold, learn_rate_threshold,
                                       delta_lprob_threshold)
                newton_stop = time.clock()
                newton_times += (newton_stop - newton_start,)

            for i in range(repeat_timings):
                # Distribution,ParamCount,NCount,MCount,PrecalcAvgTime,NewtonAvgTime
                out.write('{},{},{},{},{},{}\n'.format(
                    'GDM',
                    thetas,
                    n,
                    M,
                    precalc_times[i],
                    newton_times[i]
                ))


def vary_sampling_count(results_file):
    """
    Test average run time and memory consumption for varying the number of draws Z
    for the DM, BLM, and GDM MLE.
    """
    # Setup
    N = 100  # number of multinomials
    M = range(10, 5010, 50)  # number of samples per multinomial
    thetas = 100
    generating_params = np.array([random.random() for _ in range(thetas)])
    generating_params /= np.sum(generating_params)

    for m in M:
        X, mean_params = multinomial_random_sample(N, m, generating_params)

        # Dirichlet Multinomial
        precalc_times = ()
        newton_times = ()
        with open(results_file, 'a') as out:
            # Timing
            for _ in range(repeat_timings):
                precalc_start = time.clock()
                U, v = dm.dm_precalc(X)
                precalc_stop = time.clock()
                precalc_times += (precalc_stop - precalc_start,)
            # U, v = dm.dm_precalc(X)
            params = dm.dm_init_params(X)
            for _ in range(repeat_timings):
                newton_start = time.clock()
                dm.dm_newton_raphson(U, v, params, max_steps, gradient_sq_threshold, learn_rate_threshold,
                                     delta_lprob_threshold)
                newton_stop = time.clock()
                newton_times += (newton_stop - newton_start,)

            for i in range(repeat_timings):
                # Distribution,ParamCount,NCount,MCount,PrecalcAvgTime,NewtonAvgTime
                out.write('{},{},{},{},{},{}\n'.format(
                    'DM',
                    thetas,
                    N,
                    m,
                    precalc_times[i],
                    newton_times[i]
                ))

        # Beta-Liouville Multinomial
        precalc_times = ()
        newton_times = ()
        with open(results_file, 'a') as out:
            # Timing
            for _ in range(repeat_timings):
                precalc_start = time.clock()
                U, vd, vd1 = blm.blm_precalc(X)
                precalc_stop = time.clock()
                precalc_times += (precalc_stop - precalc_start,)
            # U, vd, vd1 = blm.blm_precalc(X)
            params = blm.blm_init_params(X)
            for _ in range(repeat_timings):
                newton_start = time.clock()
                blm.blm_newton_raphson(U, vd, vd1, params, max_steps, gradient_sq_threshold, learn_rate_threshold,
                                       delta_lprob_threshold)
                newton_stop = time.clock()
                newton_times += (newton_stop - newton_start,)

            for i in range(repeat_timings):
                # Distribution,ParamCount,NCount,MCount,PrecalcAvgTime,NewtonAvgTime
                out.write('{},{},{},{},{},{}\n'.format(
                    'BLM',
                    thetas,
                    N,
                    m,
                    precalc_times[i],
                    newton_times[i]
                ))

        # Generalized Dirichlet Multinomial
        precalc_times = ()
        newton_times = ()
        with open(results_file, 'a') as out:
            # Timing
            for _ in range(repeat_timings):
                precalc_start = time.clock()
                U, vd = gdm.gdm_precalc(X)
                precalc_stop = time.clock()
                precalc_times += (precalc_stop - precalc_start,)
            # U, vd = gdm.gdm_precalc(X)
            params = gdm.gdm_init_params(X)
            for _ in range(repeat_timings):
                newton_start = time.clock()
                gdm.gdm_newton_raphson(U, vd, params, max_steps, gradient_sq_threshold, learn_rate_threshold,
                                       delta_lprob_threshold)
                newton_stop = time.clock()
                newton_times += (newton_stop - newton_start,)

            for i in range(repeat_timings):
                # Distribution,ParamCount,NCount,MCount,PrecalcAvgTime,NewtonAvgTime
                out.write('{},{},{},{},{},{}\n'.format(
                    'GDM',
                    thetas,
                    N,
                    m,
                    precalc_times[i],
                    newton_times[i]
                ))


if __name__ == '__main__':
    # sys.argv[1] == result filepath
    silent_remove(sys.argv[1])

    vary_observation_count(sys.argv[1])
    vary_sampling_count(sys.argv[1])
    vary_parameter_count(sys.argv[1])
