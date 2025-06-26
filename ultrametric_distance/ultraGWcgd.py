import numpy as np
from typing import Tuple, List
from .ugw_gw_equality_constraints import gw_equality_constraints
from .ugw_coupling_ensemble import coupling_ensemble
from .ugw_construct_cost_one import construct_cost_one
from .ugw_construct_ultra_cost_mat import construct_ultra_cost_mat
from .ugw_emd_transport import emd_transport

def ultraGWcgd(
    ux: np.ndarray,
    uy: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    p: float,
    iterations: int,
    num_samples: int,
    num_skips: int
) -> Tuple[float, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Approximates the ultrametric Gromov–Wasserstein distance of order p
    via conditional gradient (Frank–Wolfe) descent.

    Parameters
    ----------
    ux : (N, N) array
        Ultrametric distance matrix on space X.
    uy : (M, M) array
        Ultrametric distance matrix on space Y.
    mu_x : (N,) array
        Probability vector on X.
    mu_y : (M,) array
        Probability vector on Y.
    p : float
        Order of the distance (>= 1).
    iterations : int
        Number of Frank–Wolfe iterations.
    num_samples : int
        Number of random restarts (0 ⇒ only independence coupling).
    num_skips : int
        Skips between random couplings (when sampling).

    Returns
    -------
    res : float
        The best GW distance approximation ^(1/p).
    plan : (N, M) array
        Optimal coupling matrix achieving `res`.
    result_vec : (K,) array
        All distance values (to the p-th power) from each restart.
    result_plan_cell : list of (N, M) arrays
        All final couplings from each restart.
    """
    # 1) Build the list of initial couplings
    if num_samples >= 1:
        A, b = gw_equality_constraints(mu_x, mu_y)
        other_coups = coupling_ensemble(A, b, mu_x, mu_y, num_samples, num_skips)
        coups: List[np.ndarray] = [np.outer(mu_x, mu_y)] + other_coups
    else:
        coups = [np.outer(mu_x, mu_y)]

    K = len(coups)
    result_vec = np.zeros(K, dtype=float)
    result_plan_cell: List[np.ndarray] = [None] * K

    N = ux.shape[0]
    M = uy.shape[0]

    # 2) For each coupling, run Frank–Wolfe
    for idx, pi0 in enumerate(coups):
        #print(f"Coupling {idx + 1} of {K}:")
        #print("Initial coupling pi0:\n", pi0)
        pi_old = pi0.copy()   # shape (N, M)
        for it in range(1, iterations + 1):
            # build cost matrix
            if p == 1:
                cost = construct_cost_one(ux, uy, pi_old)
            else:
                cost = construct_ultra_cost_mat(ux, uy, pi_old, p)

            #print(f"Iteration {it} of {iterations} for coupling {idx + 1} of {K}")
            #print("cost", cost)
            # solve OT subproblem: returns (distance, plan)
            _, ot_plan = emd_transport(mu_x, mu_y, cost, return_plan=True)

            # fixed decreasing step size 2/(i+2)
            step = 2.0 / (it + 2.0)
            pi_old = (1 - step) * pi_old + step * ot_plan

        # store final plan
        pi = pi_old
        #print(f"Final coupling pi:\n{pi}")
        result_plan_cell[idx] = pi

        # 3) compute the p-th power of the GW objective:
        val = 0.0
        for i in range(N):
            for j in range(M):
                for k in range(N):
                    for l in range(M):
                        d_x = ux[i, k]
                        d_y = uy[j, l]
                        val += pi[i, j] * pi[k, l] * (delta_infinity(d_x, d_y) ** p)
        result_vec[idx] = val ** (1.0 / p)

    # 4) Pick the best
    min_idx = int(np.argmin(result_vec))
    res = result_vec[min_idx]
    plan = result_plan_cell[min_idx]

    return res, plan, result_vec, result_plan_cell

def delta_infinity(x: float, y: float) -> float:
    """
    Compute the ultrametric delta-infinity between x and y:
      • returns 0 if |x - y| < 1e-15
      • otherwise returns max(x, y)
    """
    return 0.0 if abs(x - y) < 1e-15 else max(x, y)
