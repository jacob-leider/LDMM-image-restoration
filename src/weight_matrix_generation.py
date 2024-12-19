import numpy
import sklearn
from sklearn.neighbors import BallTree
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse import diags


def sparse_L_kd_tree(points, k_neighbors=None, sigma_neighbors=20):
    """
    Compute the graph laplacian used in point integral approximation.

    Args:
        points (numpy.ndarray): NxD array of points in the point cloud.
        k_neighbors (int): Number of nearest neighbors for sparsification.
        sigma_neighbors (int): Nearest neighbor index for adaptive scaling.

    Returns:
        L (numpy.ndarray): NxN matrix in CSR format.
    """
    # Initialize the weight matrix with zeros
    N = points.shape[0]

    if k_neighbors == None:
      k_neighbors = N

   # Build a KDTree for efficient nearest-neighbor search
    tree = KDTree(points)

    # Find k_neighbors + 1 nearest neighbors (including the point itself)
    distances, neighbor_indices = tree.query(points, k=k_neighbors + 1)
    distances = distances / distances[:, sigma_neighbors:sigma_neighbors+1]
    distances = numpy.exp(-distances**2)

    row_offsets = (k_neighbors + 1) * numpy.arange(N + 1)
    col_offsets = neighbor_indices.flatten()
    data = distances.flatten()

    W = csr_matrix((data, col_offsets, row_offsets), shape=(N, N))
    L = diags(numpy.sum(distances, axis=1)) - W
    return W, L


def sparse_L_ball_tree(points, k_neighbors=None, sigma_neighbors=20):
    """
    Compute the graph laplacian used in point integral approximation.

    Args:
        points (numpy.ndarray): NxD array of points in the point cloud.
        k_neighbors (int): Number of nearest neighbors for sparsification.
        sigma_neighbors (int): Nearest neighbor index for adaptive scaling.

    Returns:
        L (numpy.ndarray): NxN matrix in CSR format.
    """
    # Initialize the weight matrix with zeros
    N = points.shape[0]

    if k_neighbors == None:
      k_neighbors = N

    # Build a KDTree for efficient nearest-neighbor search
    tree = BallTree(points, leaf_size=2)

    # Find k_neighbors + 1 nearest neighbors (including the point itself)
    distances, neighbor_indices = tree.query(points, k=k_neighbors + 1)
    distances = distances / distances[:, sigma_neighbors:sigma_neighbors+1]
    distances = numpy.exp(-distances**2)

    row_offsets = (k_neighbors + 1) * numpy.arange(N + 1)
    col_offsets = neighbor_indices.flatten()
    data = distances.flatten()

    W = csr_matrix((data, col_offsets, row_offsets), shape=(N, N))
    L = diags(numpy.sum(distances, axis=1)) - W
    return W, L


def sparse_L_annoy(points, k_neighbors=None, sigma_neighbors=20):
    """
    Compute the graph laplacian used in point integral approximation.

    Args:
        points (numpy.ndarray): NxD array of points in the point cloud.
        k_neighbors (int): Number of nearest neighbors for sparsification.
        sigma_neighbors (int): Nearest neighbor index for adaptive scaling.

    Returns:
        L (numpy.ndarray): NxN matrix in CSR format.
    """
    # Initialize the weight matrix with zeros
    N = points.shape[0]
    t = 100  # Number of trees

    if k_neighbors == None:
      k_neighbors = N

    # Build the index
    index = annoy.AnnoyIndex(points.shape[1], 'euclidean')
    for i, v in enumerate(points):
        index.add_item(i, v)
    index.build(t)

    d_sums = numpy.zeros((N))
    col_offsets = numpy.zeros((N * k_neighbors))
    row_offsets = k_neighbors * numpy.arange(N + 1)
    vals = numpy.zeros((N * k_neighbors))

    for i in range(N):
      indices, distances = index.get_nns_by_item(i, k_neighbors, include_distances=True)
      d_sums[i] = numpy.sum(distances)
      distances = numpy.array(distances) / distances[sigma_neighbors]
      distances = numpy.exp(-distances**2)
      col_offsets[i * k_neighbors:(i+1)*k_neighbors] = indices
      vals[i * k_neighbors: (i+1)*k_neighbors] = distances

    W = csr_matrix((vals, col_offsets, row_offsets), shape=(N, N))
    L = diags(d_sums) - W
    return W, L
