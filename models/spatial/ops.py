# D:\Tiktok\models\spatial\ops.py

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree
from typing import Tuple, List, Union

def compute_pairwise_distances(locations: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    Computes the full pairwise distance matrix between all nodes.
    Useful for small to medium-sized networks where O(N^2) memory is acceptable.

    Args:
        locations (np.ndarray): Node coordinates of shape (N, 2).
        metric (str): The distance metric to use (default: 'euclidean').

    Returns:
        np.ndarray: A distance matrix of shape (N, N).
    """
    return cdist(locations, locations, metric=metric)

def build_spatial_index(locations: np.ndarray, metric: str = 'euclidean') -> KDTree:
    """
    Builds a KDTree (or BallTree conceptually) for efficient spatial querying.
    Recommended for large networks to avoid O(N^2) complexity.

    Args:
        locations (np.ndarray): Node coordinates of shape (N, 2).
        metric (str): Distance metric (only 'euclidean' typically supported efficiently by KDTree).

    Returns:
        KDTree: The constructed spatial index object.
    """
    # Note: KDTree is generally faster for low dimensions (2D), BallTree for higher dimensions.
    return KDTree(locations, metric=metric)

def query_k_nearest_neighbors(tree: KDTree, 
                              query_points: np.ndarray, 
                              k: int, 
                              include_self: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the k-nearest neighbors for a set of query points using the spatial index.

    Args:
        tree (KDTree): The pre-built spatial index.
        query_points (np.ndarray): Coordinates to query, shape (M, 2).
        k (int): Number of neighbors to find.
        include_self (bool): If True, the point itself (distance 0) is included in results.
                             If False, it queries k+1 and excludes the first one (assuming query_points exist in tree).

    Returns:
        dist (np.ndarray): Distances to the neighbors, shape (M, k).
        ind (np.ndarray): Indices of the neighbors, shape (M, k).
    """
    # If we don't want to include the node itself (common in social networks), we ask for k+1
    k_query = k + 1 if not include_self else k
    
    dists, indices = tree.query(query_points, k=k_query)
    
    if not include_self:
        # Exclude the first column (distance 0, the node itself)
        return dists[:, 1:], indices[:, 1:]
    else:
        return dists, indices

def query_radius_neighbors(tree: KDTree, 
                           query_points: np.ndarray, 
                           radius: float, 
                           sort_results: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds all neighbors within a specified radius for each query point.

    Args:
        tree (KDTree): The pre-built spatial index.
        query_points (np.ndarray): Coordinates to query, shape (M, 2).
        radius (float): The search radius.
        sort_results (bool): Whether to sort neighbors by distance.

    Returns:
        indices (np.ndarray): An array of arrays (object dtype), where each element contains neighbor indices.
        dists (np.ndarray): An array of arrays (object dtype), where each element contains distances.
                            Returned only if sort_results is True or specifically requested (implementation specific).
                            Here we simplify to align with sklearn's query_radius signature.
    """
    # sklearn returns indices[i] = array of neighbors for point i
    indices = tree.query_radius(query_points, r=radius, sort_results=sort_results)
    
    # query_radius doesn't return distances by default unless return_distance=True is passed.
    # To keep interface simple, we might just return indices, but often distance is needed for weights.
    dists = []
    if sort_results:
        # Re-calculating or extracting distances can be costly. 
        # sklearn's query_radius with return_distance=True is better.
        indices, dists = tree.query_radius(query_points, r=radius, return_distance=True, sort_results=sort_results)
    
    return indices, dists

def calculate_gravity_weights(distances: np.ndarray, 
                              alpha: float = 1.0, 
                              beta: float = 2.0, 
                              epsilon: float = 1e-6) -> np.ndarray:
    """
    Calculates interaction weights based on the Gravity Model: W_ij ~ 1 / (distance^beta).

    Args:
        distances (np.ndarray): Array of distances.
        alpha (float): Scaling factor.
        beta (float): Distance decay exponent (typically 1.0 to 2.0).
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        np.ndarray: Calculated weights corresponding to the input distances.
    """
    return alpha / (np.power(distances, beta) + epsilon)