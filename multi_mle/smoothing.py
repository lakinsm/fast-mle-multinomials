import numpy as np


def lidstone_smoothing(S, X, class_labels, n=1):
    """
    Smooth the simplex estimates by a count of n using Lidstone smoothing (Laplace smoothing if n=1).  Since we
    aren't incorporating this into the MLE, it is not exactly Lidstone smoothing as traditionally defined, however
    it will produce a reasonable approximation.
    :param S: Simplex float matrix of dimension (features, classes), with columns that sum to 1
    :param X: Original count matrix/dictionary structure as output by tokenize_train()
    :param class_labels: List of strings, where each entry is the name of the classes in S and X
    :param n: Value to add via Lidstone smoothing (default 1 for Laplace smoothing)
    """
    for i, c in enumerate(class_labels):
        class_counts = np.squeeze(np.sum(X[c][0], axis=0))
        S[:, i] += float(n) / ((n * S.shape[0]) + np.sum(class_counts))
        S[:, i] /= np.sum(S[:, i])
    return np.log(S)


def lidstone_smoothing2(S, X, class_labels, n=1):
    """
    Smooth the simplex estimates by a count of n using Lidstone smoothing (Laplace smoothing if n=1).  Since we
    aren't incorporating this into the MLE, it is not exactly Lidstone smoothing as traditionally defined, however
    it will produce a reasonable approximation.
    :param S: Simplex float matrix of dimension (features, classes), with columns that sum to 1
    :param X: Original count matrix/dictionary structure as output by tokenize_train()
    :param class_labels: List of strings, where each entry is the name of the classes in S and X
    :param n: Value to add via Lidstone smoothing (default 1 for Laplace smoothing)
    """
    for i, c in enumerate(class_labels):
        S[:, i] += float(n) / (n * S.shape[0])
        S[:, i] /= np.sum(S[:, i])
    return np.log(S)


def dirichlet_smoothing(S, X, class_labels, mu=0.95, n=1):
    """
    Smooth the simplex estimates using Dirichlet prior smoothing.
    :param S: Simplex float matrix of dimension (features, classes), with columns that sum to 1
    :param X: Original count matrix/dictionary structure as output by tokenize_train()
    :param class_labels: List of strings, where each entry is the name of the classes in S and X
    :param mu: Float Dirichlet prior value used to scale S, mu=(0, 1)
    :param n: Integer to add via Lidstone smoothing (default 1 for Laplace smoothing)
    """
    # Corpus model
    word_corpus = np.sum(S, axis=1) / np.sum(S)

    # MLE model with Lidstone smoothing
    for i, c in enumerate(class_labels):
        class_counts = np.squeeze(np.sum(X[c][0], axis=0))
        S[:, i] += float(n) / ((n * S.shape[0]) + np.sum(class_counts))
        S[:, i] /= np.sum(S[:, i])
        expanded_counts = np.zeros((S.shape[0],), dtype=np.float64)
        for j, count in enumerate(class_counts):
            expanded_counts[X[c][1][j]] = count

        # Linear interpolation between Lidstone smoothed MLE estimates and corpus probabilities for each word
        S[:, i] = ((np.sum(class_counts) / (np.sum(class_counts) + mu)) * S[:, i]) + \
                  ((mu / (np.sum(class_counts) + mu)) * word_corpus)
    return np.log(S)


def jelinek_mercer_smoothing(S, X, class_labels, beta=0.5, n=1):
    """
    Smooth the simplex estimates using Jelinek-Mercer smoothing.
    :param S: Simplex float matrix of dimension (features, classes), with columns that sum to 1
    :param X: Original count matrix/dictionary structure as output by tokenize_train()
    :param class_labels: List of strings, where each entry is the name of the classes in S and X
    :param beta: Float value of parameter for Jelinek-Mercer interpolation, lambda=(0, 1)
    :param n: Integer to add via Lidstone smoothing (default 1 for Laplace smoothing)
    """
    # Corpus model
    word_corpus = np.sum(S, axis=1) / np.sum(S)

    # MLE model with Lidstone smoothing
    for i, c in enumerate(class_labels):
        class_counts = np.squeeze(np.sum(X[c][0], axis=0))
        S[:, i] += float(n) / ((n * S.shape[0]) + np.sum(class_counts))
        S[:, i] /= np.sum(S[:, i])

        # Linear interpolation between Lidstone smoothed MLE estimates and the corpus probabilities for each word
        S[:, i] = ((1 - beta) * S[:, i]) + (beta * word_corpus)
    return np.log(S)


def absolute_discounting_smoothing(S, X, class_labels, delta=0.01, n=1):
    """
    Smooth the simplex estimates using Absolute Discounting smoothing.
    :param S: Simplex float matrix of dimension (features, classes), with columns that sum to 1
    :param X: Original count matrix/dictionary structure as output by tokenize_train()
    :param class_labels: List of strings, where each entry is the name of the classes in S and X
    :param delta: Float value of delta for Absolute Discounting method, delta=(0, 1)
    :param n: Integer to add via Lidstone smoothing (default 1 for Laplace smoothing)
    """
    # Corpus model
    word_corpus = np.sum(S, axis=1) / np.sum(S)

    # MLE model with Lidstone smoothing
    for i, c in enumerate(class_labels):
        class_counts = np.squeeze(np.sum(X[c][0], axis=0))
        S[:, i] += float(n) / ((n * S.shape[0]) + np.sum(class_counts))
        S[:, i] /= np.sum(S[:, i])

        # Discount delta from non-zero parameters and pull from corpus model in equal proportion to discounts
        S[:, i] = np.array([np.max([x - (delta / np.sum(class_counts)), 0]) for x in S[:, i]]) + \
                  ((delta / np.sum(class_counts)) * np.sum(S[:, i] > 0) * word_corpus)
    return np.log(S)


def two_step_smoothing(S, X, class_labels, mu=0.95, beta=0.5, n=1):
    """
    Smooth the simplex estimates using Two-step smoothing.
    :param S: Simplex float matrix of dimension (features, classes), with columns that sum to 1
    :param X: Original count matrix/dictionary structure as output by tokenize_train()
    :param class_labels: List of strings, where each entry is the name of the classes in S and X
    :param mu: Float value of mu for the Dirichlet smoothing method, delta=(0, 1)
    :param beta: Float value of beta for the JM interpolation method, beta=(0, 1)
    :param n: Integer to add via Lidstone smoothing (default 1 for Laplace smoothing)
    """
    # Corpus model
    word_corpus = np.sum(S, axis=1) / np.sum(S)

    # MLE model with Lidstone smoothing
    for i, c in enumerate(class_labels):
        class_counts = np.squeeze(np.sum(X[c][0], axis=0))
        S[:, i] += float(n) / ((n * S.shape[0]) + np.sum(class_counts))
        S[:, i] /= np.sum(S[:, i])

        # Combination of JM interpolation with Dirichlet prior
        S[:, i] = ((1 - beta) * ((np.sum(class_counts) / (np.sum(class_counts) + mu)) * S[:, i]) + \
                   ((mu / (np.sum(class_counts) + mu)) * word_corpus)) + (beta * word_corpus)
    return np.log(S)
