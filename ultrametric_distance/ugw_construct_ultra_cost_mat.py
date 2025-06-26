import numpy as np

def delta_infinity(x: float, y: float) -> float:
    """
    Returns 0 if |x–y| < 1e-15, else max(x,y).
    """
    return 0.0 if abs(x - y) < 1e-15 else max(x, y)

def construct_ultra_cost_mat(
    ux: np.ndarray,
    uy: np.ndarray,
    coupling: np.ndarray,
    p: float
) -> np.ndarray:
    """
    Build the ultrametric GW cost matrix with exponent p:
      C[i,j] = 2 * sum_{k,l} [delta_infinity(ux[i,k], uy[j,l])**p * coupling[k,l]]

    Parameters
    ----------
    ux : (n,n) array
        Ultrametric distance on X
    uy : (m,m) array
        Ultrametric distance on Y
    coupling : (n,m) array
        A transport plan from X to Y
    p : float
        Exponent in the cost

    Returns
    -------
    cost : (n,m) array
    """
    n, n2 = ux.shape
    m, m2 = uy.shape
    assert n == n2 and m == m2, "ux must be n×n, uy must be m×m"
    assert coupling.shape == (n, m), "coupling must be n×m"

    cost = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            tmp = 0.0
            for k in range(n):
                for l in range(m):
                    tmp += (delta_infinity(ux[i, k], uy[j, l]) ** p) * coupling[k, l]
            cost[i, j] = 2.0 * tmp
    return cost

# def construct_ultra_cost_mat_vec(
#     ux: np.ndarray,
#     uy: np.ndarray,
#     coupling: np.ndarray,
#     p: float
# ) -> np.ndarray:
#     # Shape checks omitted for brevity...
#     # Broadcast ux over (n,n,1,1) and uy over (1,1,m,m)
#     X = ux[:, :, None, None]            # shape (n,n,1,1)
#     Y = uy[None, None, :, :]            # shape (1,1,m,m)
#     # Compute delta_infinity in bulk:
#     D = np.where(np.abs(X - Y) < 1e-15, 0.0, np.maximum(X, Y))
#     Dp = D ** p                         # shape (n,n,m,m)
#     # sum over k (axis=1) and l (axis=3) against coupling[k,l]
#     cost = 2.0 * np.tensordot(Dp, coupling, axes=([1, 3], [0, 1]))
#     return cost                          # shape (n,m)
