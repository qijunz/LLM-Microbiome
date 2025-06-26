
import numpy as np
from typing import Tuple

def gw_equality_constraints(mu_x: np.ndarray, mu_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the equality‐constraint matrices for Gromov–Wasserstein.

    Parameters
    ----------
    mu_x : (m,) array
        Probability vector on space X.
    mu_y : (n,) array
        Probability vector on space Y.

    Returns
    -------
    A : ((m+n), m*n) array
        Stacked constraint matrix so that A @ vec(P) = b
    b : (m+n,) array
        RHS vector = [mu_x, mu_y]
    """
    m = mu_x.shape[0]
    n = mu_y.shape[0]

    # Row sums over each row of P must equal mu_x
    A_p = np.kron(np.eye(m), np.ones((1, n)))       # shape (m, m*n)

    # Column sums over each column of P must equal mu_y
    A_q = np.kron(np.ones((1, m)), np.eye(n))       # shape (n, m*n)

    A = np.vstack([A_p, A_q])                       # shape (m+n, m*n)
    b = np.concatenate([mu_x, mu_y])                # shape (m+n,)

    return A, b