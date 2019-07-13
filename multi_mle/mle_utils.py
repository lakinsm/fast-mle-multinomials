import numpy as np
import random


def multinomial_random_sample(N, M, theta):
    """
    Sample from a multinomial distribution with parameters theta.
    :param n: integer, number of multinomials to generate
    :param m: integer, number of samples per multinomial
    :param theta: vector of parameters in dimension D+1
    :return: count matrix X, dimension (N, D+1)
    """
    D1 = len(theta)
    multinomials = np.matrix([[random.normalvariate(theta[d], 0.01) for d in range(D1)] for _ in range(N)])
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
