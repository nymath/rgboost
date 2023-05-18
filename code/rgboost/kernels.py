import numpy as np


def rbf_kernel(X, gamma):

    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    # Compute the pairwise distances between the data points
    pairwise_dists = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(X ** 2, axis=1) - 2 * np.dot(X, X.T)

    # Compute the RBF kernel matrix
    K = np.exp(-gamma * pairwise_dists)

    return K


def laplacian_kernel(X, gamma):

    pairwise_dists = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(X ** 2, axis=1) - 2 * np.dot(X, X.T)
    K = np.exp(-gamma * np.sqrt(pairwise_dists))
    return K


