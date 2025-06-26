from typing import Dict, List
import numpy as np
from .ultraGWcgd import ultraGWcgd

def extract_leaf_paths(
    node: Dict[str, Dict],
    path: List[str] = None,
    result: Dict[str, List[str]] = None
) -> Dict[str, List[str]]:
    """
    Traverse a nested dict where each key is a node name and its value is
    a dict of children. Record for each leaf (empty dict) its path from the root.
    Returns a dict: leaf_name -> [root, ..., leaf].
    """
    if path is None:
        path = []
    if result is None:
        result = {}

    for key, children in node.items():
        new_path = path + [key]
        if isinstance(children, dict) and children:
            # interior node: recurse
            extract_leaf_paths(children, new_path, result)
        else:
            # leaf node
            result[key] = new_path
    return result

def ultrametric_distance_matrix(
    leaf_paths: Dict[str, List[str]]
) -> np.ndarray:
    """
    Compute the ultrametric distance matrix U where for leaves xi, xj:
      U[xi,xj] = |leaves(T[xi ∨ xj])| / total_leaves
    """

    leaves = list(leaf_paths.keys())
    n = len(leaves)

    # 1) Count how many leaves descend from each node
    desc_count: Dict[str, int] = {}
    for path in leaf_paths.values():
        for node in path:
            desc_count[node] = desc_count.get(node, 0) + 1

    # 2) Helper to find the lowest common ancestor (last common in two paths)
    def find_lca(path1: List[str], path2: List[str]) -> str:
        lca = None
        for a, b in zip(path1, path2):
            if a != b:
                break
            lca = a
        if lca is None:
            print(lca, "is None, path1:", path1, "path2:", path2)
        return lca  # guaranteed not None if all leaves share at least the root

    # 3) Build the symmetric matrix
    mat = np.zeros((n, n), dtype=float)
    #print("matrix shape:", mat.shape)
    for i in range(n):
        pi = leaf_paths[leaves[i]]
        for j in range(i + 1, n):
            pj = leaf_paths[leaves[j]]
            lca = find_lca(pi, pj)
            mat[i, j] = mat[j, i] = desc_count[lca] / n
    return mat

def compute_ultrametric_distance_matrix(tree):
    # 1) Extract leaf paths
    leaf_paths = extract_leaf_paths(tree)
    #print(f"Extracted {len(leaf_paths)} leaf paths.")
    #print(leaf_paths)
    # 2) Compute ultrametric distance matrix
    dist_matrix = ultrametric_distance_matrix(leaf_paths)
    return dist_matrix

# Compute the ultrametric distance
def get_ultrametric_distance(tree1, tree2):

    ux = compute_ultrametric_distance_matrix(tree1)
    uy = compute_ultrametric_distance_matrix(tree2)
    mux = np.ones(ux.shape[0]) / ux.shape[0]
    muy = np.ones(uy.shape[0]) / uy.shape[0]

    d, _, _, _ = ultraGWcgd(
                    ux, uy, mux, muy,
                    p=2.0,
                    iterations=20,
                    num_samples=0,
                    num_skips=1
                )
    return d