#!/usr/bin/env python3

import multi_mle.mle_utils as mutils
import pandas as pd
import numpy as np
import timeit


def test_geom_limit_approximation(result_path):
    param_vals = np.logspace(-10, 10, 21)
    iteration_vals = range(1, 100001, 10)
    results = pd.DataFrame(0, index=np.array(range(len(param_vals) * len(iteration_vals))),
                           columns=['theta', 'n', 'exact_val', 'approx_val', 'exact_time', 'approx_time', 'residual'],
                           dtype=np.float64)
    row_idx = 0
    for param in param_vals:
        for it in iteration_vals:
            if row_idx % 1000 == 0:
                print('Iteration {} / {}'.format(
                    row_idx + 1,
                    len(param_vals) * len(iteration_vals)
                ))
            exact_timings = tuple()
            approx_timings = tuple()
            exact_val = None
            approx_val = None
            for btstrp in range(5):
                start = timeit.default_timer()
                exact_val = mutils.exact_geom_limit(param, it)
                exact_timings += (timeit.default_timer() - start,)

                start = timeit.default_timer()
                approx_val = mutils.approx_geom_limit(param, it)
                approx_timings += (timeit.default_timer() - start,)
            results.ix[row_idx, :] = {'theta': param,
                                   'n': it,
                                   'exact_val': exact_val,
                                   'approx_val': approx_val,
                                   'exact_time': np.mean(exact_timings),
                                   'approx_time': np.mean(approx_timings),
                                   'residual': exact_val - approx_val}
            # print(results.ix[row_idx, :])
            row_idx += 1
    results.to_csv(result_path)


def test_harmonic_limit_approximation(result_path):
    param_vals = np.logspace(-10, 10, 21)
    iteration_vals = range(1, 100001, 10)
    results = pd.DataFrame(0, index=np.array(range(len(param_vals) * len(iteration_vals))),
                           columns=['theta', 'n', 'exact_val', 'approx_val', 'exact_time', 'approx_time', 'residual'],
                           dtype=np.float64)
    row_idx = 0
    for param in param_vals:
        for it in iteration_vals:
            if row_idx % 1000 == 0:
                print('Iteration {} / {}'.format(
                    row_idx + 1,
                    len(param_vals) * len(iteration_vals)
                ))
            exact_timings = tuple()
            approx_timings = tuple()
            exact_val = None
            approx_val = None
            for btstrp in range(5):
                start = timeit.default_timer()
                exact_val = mutils.exact_harmonic_limit(param, it)
                exact_timings += (timeit.default_timer() - start,)

                start = timeit.default_timer()
                approx_val = mutils.approx_harmonic_limit(param, it)
                approx_timings += (timeit.default_timer() - start,)
            results.ix[row_idx, :] = {'theta': param,
                                   'n': it,
                                   'exact_val': exact_val,
                                   'approx_val': approx_val,
                                   'exact_time': np.mean(exact_timings),
                                   'approx_time': np.mean(approx_timings),
                                   'residual': exact_val - approx_val}
            # print(results.ix[row_idx, :])
            row_idx += 1
    results.to_csv(result_path)


if __name__ == '__main__':
    # test_geom_limit_approximation('/mnt/phd_repositories/fast-mle-multinomials/analytic_data/geometric_limit_timings.csv')
    test_harmonic_limit_approximation('/mnt/phd_repositories/fast-mle-multinomials/analytic_data/harmonic_limit_timings.csv')
