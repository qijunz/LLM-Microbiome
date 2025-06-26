import numpy as np
import ot  # POT: pip install pot

def emd_transport(
    x: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    max_iter: int = None,
    return_plan: bool = False
) -> tuple[float, np.ndarray] | float:
    """
    Compute the Earth Mover’s Distance (EMD) between two discrete distributions x and y
    with ground‐cost matrix D, using a network‐simplex solver (via POT).

    Parameters
    ----------
    x : (n,) array
        Source histogram (nonnegative, sums to 1 or any common total).
    y : (m,) array
        Target histogram (nonnegative, sums to same total as x).
    D : (n, m) array
        Ground‐cost matrix, where D[i,j] is the cost to move mass from bin i of x to bin j of y.
    max_iter : int, optional
        Maximum number of iterations for the solver. If None, uses POT’s default.
    return_plan : bool, default False
        If True, also return the optimal transport plan.

    Returns
    -------
    dist : float
        The total minimum transport cost (EMD).
    plan : (n, m) array, optional
        The optimal transport plan (only if return_plan=True).
    """
    # ensure shapes
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    assert D.shape == (x.size, y.size), "D must be shape (len(x), len(y))"

    # POT’s emd returns the transport plan
    # numItermax corresponds to max number of pivoting operations
    if max_iter is None:
        plan = ot.emd(x, y, D)
    else:
        plan = ot.emd(x, y, D, numItermax=max_iter)

    dist = float(np.sum(plan * D))

    if return_plan:
        return dist, plan
    else:
        return dist

if __name__ == "__main__":
    # simple two‐point example
    x = np.array([0.5, 0.5])
    y = np.array([0.3, 0.7])
    # say moving from 0→0 costs 0, 0→1 costs 1; from 1→0 costs 2, 1→1 costs 0
    D = np.array([[0.0, 1.0],
                  [2.0, 0.0]])

    # just get the EMD cost
    cost = emd_transport(x, y, D)
    print("EMD cost:", cost)

    # get both cost and transport plan
    cost, plan = emd_transport(x, y, D, return_plan=True)
    print("Plan:\n", plan)
