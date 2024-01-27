import numpy as np
import torch


def compute_conceptor(X, aperture, svd=False, svd_eps=0.001):
    """
    Computes the conceptor matrix for a given input matrix X and an aperture value.
    (PyTorch version)

    Parameters:
    - X (torch.Tensor): Input matrix of shape (n_samples, n_features).
    - aperture (float): Aperture value used to compute the conceptor matrix.
    - svd (bool): if true compute conceptor using singular value decomposition
    Returns:
    - torch.Tensor: Conceptor matrix of shape (n_features, n_features).
    """
    R = torch.matmul(X.T, X) / X.shape[0]
    if not svd:
        C = torch.matmul(R, torch.inverse(R + aperture ** (-2) * torch.eye(R.shape[0], device=X.device)))
        return C
    else:
        U, S, _ = torch.svd(R)
        C = U * (S / (S + svd_eps * torch.ones(S.shape, device=X.device))) @ U.T
        return C


def rescale_aperture(C, prev_aperture, new_aperture):
    """Rescale the aperture of the given conceptor matrix. (PyTorch version)

    Parameters:
    - C (torch.Tensor): Conceptor matrix of shape (n_features, n_features).
    - prev_aperture (float): Previous aperture value used to compute C.
    - new_aperture (float): New aperture value to rescale C.
    Returns:
    - torch.Tensor: Rescaled conceptor matrix of shape (n_features, n_features).
    """
    scaling = prev_aperture / new_aperture
    return C @ torch.inverse(C + scaling**2 * (torch.eye(C.shape[0], device=C.device) - C))