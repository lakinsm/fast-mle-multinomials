#!/usr/bin/env python3

import multi_mle.mle_utils as mutils
import timeit

LOOKUP_PATH = 'C:/Users/lakin/Documents/1AbdoBioinformatics/Publications/MLE/Simulations/FiniteLimitTable.csv'
LOOKUP_PATH_LINUX = '/mnt/c/Users/lakin/Documents/1AbdoBioinformatics/Publications/MLE/Simulations/FiniteLimitTable.csv'


def test_geom_limit_approximation():
	timings = tuple()
	lookup = mutils.load_geom_lookup_table(LOOKUP_PATH_LINUX)
	print(lookup)


if __name__ == '__main__':
	test_geom_limit_approximation()
