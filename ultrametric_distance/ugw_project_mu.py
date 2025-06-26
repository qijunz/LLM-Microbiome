import numpy as np

def project_mu(
    mu: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    P: np.ndarray,
    product_mu: np.ndarray
) -> np.ndarray:
    """
    Orthogonally project a coupling onto the affine subspace {A @ vec(mu) = b}.

    Parameters
    ----------
    mu : (m, n) array
        Current coupling matrix.
    A : ((m+n), m*n) array
        Equality-constraint matrix from gw_equality_constraints.
    b : (m+n,) array
        RHS marginals [mu_x, mu_y].
    P : (m*n, m*n) array
        Projector onto the row space of A.
    product_mu : (m, n) array
        Independence coupling = outer(mu_x, mu_y).

    Returns
    -------
    projected : (m, n) array
        The projection of `mu` onto the affine subspace satisfying A @ vec(mu) = b.
    """
    m, n = product_mu.shape

    # Compute difference from the reference coupling
    diff = (mu - product_mu).reshape(m * n)

    # Project the difference onto the nullspace of A (i.e., subtract its component in row-space)
    diff_proj = diff - P.dot(diff)

    # Add back the reference coupling
    projected_vec = product_mu.reshape(m * n) + diff_proj

    return projected_vec.reshape(m, n)
