import numpy as np
import time

def timestamp():
    """Print the current local time (roughly equivalent to the C++ timestamp())."""
    print(time.asctime(time.localtime()))

def delta_infinity(x: float, y: float) -> float:
    """
    The “delta-infinity” used in the ultrametric GW:
      if |x - y| < 1e-15 → 0
      else → max(x, y)
    """
    return 0.0 if abs(x - y) < 1e-15 else max(x, y)

def construct_cost_one(
    ux: np.ndarray,
    uy: np.ndarray,
    coupling: np.ndarray
) -> np.ndarray:
    """
    Build the cost matrix C where
      C[i,j] = 2 * sum_{k,l} delta_infinity( ux[i,k], uy[j,l] ) * coupling[k,l]

    Parameters
    ----------
    ux : (n,n) array
        Ultrametric distance on X
    uy : (m,m) array
        Ultrametric distance on Y
    coupling : (n,m) array
        A transport plan from X to Y

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
                    tmp += delta_infinity(ux[i, k], uy[j, l]) * coupling[k, l]
            cost[i, j] = 2.0 * tmp
    return cost

# def construct_ultra_cost_mat_one_vec(ux, uy, coupling):
#     # broadcast ux over (n,n,1,1) and uy over (1,1,m,m)
#     X = ux[:, :, None, None]
#     Y = uy[None, None, :, :]
#     D = np.where(np.abs(X - Y) < 1e-15, 0.0, np.maximum(X, Y))
#     # now D has shape (n,n,m,m); sum out k,l dims against coupling:
#     #   cost[i,j] = 2 * sum_{k,l} D[i,k,j,l] * coupling[k,l]
#     cost = 2.0 * np.tensordot(D, coupling, axes=([1,3], [0,1]))
#     return cost


# — Example usage —
if __name__ == "__main__":
    # toy data
    ux = np.array([[0.0, 1.0],
                   [1.0, 0.0]])
    uy = np.array([[0.0, 2.0],
                   [2.0, 0.0]])
    coup = np.array([[0.25, 0.75],
                     [0.75, 0.25]])

    timestamp()
    C = construct_ultra_cost_mat_one(ux, uy, coup)
    print("Cost matrix:\n", C)
    timestamp()