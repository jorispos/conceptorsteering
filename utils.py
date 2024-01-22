import numpy as np


def compute_conceptor(X, aperture, svd=False, svd_eps=0.001):
    """
    Computes the conceptor matrix for a given input matrix X and an aperture value.

    Parameters:
    - X (numpy.ndarray): Input matrix of shape (n_samples, n_features).
    - aperture (float): Aperture value used to compute the conceptor matrix.
    - svd (bool): if true compute conceptor using singular value decomposition
    Returns:
    - numpy.ndarray: Conceptor matrix of shape (n_features, n_features).
    """
    R = np.dot(X.T, X) / X.shape[0]
    if not svd:
        C = np.dot(R, np.linalg.inv(R + aperture ** (-2) * np.eye(R.shape[0])))
        return C
    else:
        U, S, _ = np.linalg.svd(R, full_matrices=False, hermitian=True)
        C = U * (S / (S + svd_eps * np.ones(S.shape))) @ U.T
        return C


def rescale_aperture(C, prev_aperture, new_aperture):
    """Rescale the aperture of the given conceptor matrix.

    Parameters:
    - C (numpy.ndarray): Conceptor matrix of shape (n_features, n_features).
    - prev_aperture (float): Previous aperture value used to compute C.
    - new_aperture (float): New aperture value to rescale C.
    Returns:
    - numpy.ndarray: Rescaled conceptor matrix of shape (n_features, n_features).
    """
    scaling = prev_aperture / new_aperture
    return C @ np.linalg.inv(C + scaling**2 * (np.eye(C.shape[0]) - C))
