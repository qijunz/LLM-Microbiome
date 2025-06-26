import numpy as np
from typing import Optional
from .ugw_project_mu import project_mu

def markov_hit_and_run_step(
    A: np.ndarray,
    b: np.ndarray,
    P: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    flat_mu: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Perform one hit-and-run step in the coupling polytope.

    Parameters
    ----------
    A : (m+n, m*n) array
        Equality-constraint matrix (from gw_equality_constraints).
    b : (m+n,) array
        RHS marginals [mu_x, mu_y].
    P : (m*n, m*n) array
        Projector onto row-space of A.
    mu_x : (m,) array
        Source marginal.
    mu_y : (n,) array
        Target marginal.
    flat_mu : (m*n,) array, optional
        Current coupling (flattened). If None, uses independence coupling.

    Returns
    -------
    new_flat : (m*n,) array
        Updated coupling (flattened) after one randomized step.
    """
    m, n = mu_x.shape[0], mu_y.shape[0]
    mn = m * n

    # 1) Independence coupling as fallback
    indep = np.outer(mu_x, mu_y).flatten()
    mu = indep.copy() if flat_mu is None else flat_mu.copy()

    # 2) Project to the affine subspace { A @ mu = b }
    #    (mitigates numeric drift)
    mu = project_mu(mu, A, b, P, indep)

    # 3) Sample a random direction and project into the subspace
    direction = np.random.randn(mn)
    direction = direction - P.dot(direction)
    direction /= np.linalg.norm(direction)

    # 4) Compute allowable step bounds so mu + t*direction ≥ 0
    pos = direction > 1e-6
    neg = direction < -1e-6

    # avoid empty slices
    lower = np.max(-mu[pos] / direction[pos]) if np.any(pos) else -np.inf
    upper = np.min(-mu[neg] / direction[neg]) if np.any(neg) else np.inf

    # 5) Pick random step size in [lower, upper]
    t = np.random.rand() * (upper - lower) + lower

    # 6) Return the new flattened coupling
    new_flat = mu + t * direction
    return new_flat