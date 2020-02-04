import sys
import numpy as np


def print_flush(s):
    sys.stdout.write("\r" + s)
    sys.stdout.flush()


def zeros(size):
    return np.zeros(size).astype(np.float32)


def regularize_cov_matrix(a, mineig=1e-5):
    """
    regularize the covariance matrix  a  so that its minimum eigen value is larger than  mineig.
    """
    l, p = np.linalg.eigh(a)
    # Note: using eigh() rather than eig()
    n = l.size
    for i in range(n):
        if l[i] < mineig:
            l[i] = mineig
    ar = np.dot(np.dot(p, np.diag(l)), p.T)
    return ar


def calc_matrix_F(A, dt):
    """Compute the F matrix.

    Ft' = exp(dt * At') ≈ I + dt * At' + dt^2 * At'^2 + ...
    Approximate up to the third-order term and calculate F.
    F = I + A * dt + (A^2 * dt^2)/2.0 + (A^3 * dt^3)/6.0

    Args:
        A (np.array): Jacobian of the state space model dynamics function.
        dt (int or float): Differentiating time width.

    Returns:
        np.array: A variable holding a np.array of the F matrix.
    """
    n = A.shape[1]
    A2 = np.dot(A, A)
    A3 = np.dot(A2, A)
    F = np.identity(n, dtype=np.float32) + dt * A + dt ** 2 * A2 / 2.0 + dt ** 3 * A3 / 6.0
    return F
