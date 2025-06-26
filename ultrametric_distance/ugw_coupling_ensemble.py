
import numpy as np
from scipy.linalg import orth
from typing import List, Optional
from .ugw_markov_hit_and_run_step import markov_hit_and_run_step

def coupling_ensemble(
    A: np.ndarray,
    b: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    num_samples: int,
    num_skips: int,
    mu_initial: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Generate an ensemble of couplings via a hit-and-run Markov chain,
    subject to the equality constraints A @ vec(P) = b and P >= 0.

    Parameters
    ----------
    A : (m+n, m*n) array
        Equality-constraint matrix (see gw_equality_constraints).
    b : (m+n,) array
        Right-hand side vector of marginals [mu_x, mu_y].
    mu_x : (m,) array
        Source marginal.
    mu_y : (n,) array
        Target marginal.
    num_samples : int
        Number of couplings to return.
    num_skips : int
        Number of hit-and-run steps between recordings.
    mu_initial : (m, n) array, optional
        Starting coupling. Defaults to the independence coupling outer(mu_x, mu_y).

    Returns
    -------
    ensemble : list of (m, n) arrays
        A list of `num_samples` coupling matrices sampled from the feasible set.
    """
    m, n = mu_x.shape[0], mu_y.shape[0]

    # 1) Independence coupling as default start
    indep = np.outer(mu_x, mu_y)  # shape (m, n)
    mu_current = indep.copy() if mu_initial is None else mu_initial.copy()

    # 2) Build projector onto row-space of A (i.e. col-space of A^T)
    Q = orth(A.T)            # shape (m*n, r), orthonormal basis
    P = Q @ Q.T              # projector: (m*n, m*n)

    total_steps = num_samples * num_skips
    ensemble: List[np.ndarray] = []

    # 3) Hit-and-run chain
    for step in range(1, total_steps + 1):
        # flatten current coupling to a vector of length m*n
        flat = mu_current.flatten()

        # one hit-and-run move; returns flat array length m*n
        new_flat = markov_hit_and_run_step(A, b, P, mu_x, mu_y, flat)

        # reshape back to (m, n)
        mu_current = new_flat.reshape(m, n)

        # record every `num_skips` steps
        if step % num_skips == 0:
            ensemble.append(mu_current.copy())

    return ensemble