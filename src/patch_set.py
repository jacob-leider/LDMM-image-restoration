import numpy
from numpy.lib.stride_tricks import sliding_window_view

def patch_set_contribution_mask(index_set: numpy.ndarray,
                                patch_size: tuple[int, int],
                                image_size: tuple[int, int]):
  """
  Constructs a map of "patch-density" given a patch set determined by index_set
  and patch_size.  An index in the output contains the number of patches in the
  given patch set by which it is covered.  This is also the result of 
  patch_set_adjoint_operator on a matrix of ones.

  Note: Some entries may be zero, so zero-entries should be set to one before
  taking the result's reciprocal.

  Args:
    index_set (numpy.ndarray): Set of indices for the patches. An index 
    corresponds to a patch's top left corner.
    patch_size (tuple[int, int]): Patch dimensions.
    image_size (tuple[int, int]): Image dimensions.
  
  Returns:
    mask (numpy.ndarray): An integer valued array of dimensions image_shape.
  """
  mask = numpy.zeros(image_size)
  for idx in index_set:
    mask[idx[0]:idx[0] + patch_size[0], 
         idx[1]:idx[1] + patch_size[1]] += 1
  
  return mask


def uniform_index_set(m, n, patch_size, overlap):
    """
    Generates a uniform index set for extracting patches from an image.

    This function calculates the top-left corner indices for a sliding window
    approach over a 2D image grid, given the patch size and overlap between patches.

    TODO: Fix and simplify the bounds checking scheme.

    Args:
        m (int): Number of rows in the image.
        n (int): Number of columns in the image.
        patch_size (int): Size of the square patches to extract.
        overlap (int): Number of overlapping pixels between consecutive patches.

    Returns:
        numpy.ndarray: A 2D array of shape (num_patches, 2), where each row contains
        the (row, column) indices of the top-left corner of a patch.
    """
    stride = patch_size - overlap
    # Keep indices in bounds.
    m = stride * (m // stride) - patch_size
    n = stride * (n // stride) - patch_size
    s = (m // stride, n // stride) # Size of the index set

    indices = numpy.vstack(numpy.indices(s).transpose((1, 2, 0))) * stride
    return indices


def patch_set(image, patch_size, index_set):
    """
    Constructs a patch set from an image using the provided index set.

    This function extracts patches from the input image based on a sliding window view
    and the given index set, which specifies the top-left corners of each patch.

    Args:
        image (numpy.ndarray): Input 2D image from which patches will be extracted.
        patch_size (int): Size of the square patches (patch_size x patch_size).
        index_set (numpy.ndarray): A 2D array of shape (num_patches, 2), where each row
            contains the (row, column) indices of the top-left corner of a patch.

    Returns:
        numpy.ndarray: A 2D array of shape (num_patches, patch_size^2)
        where each slice along the first dimension represents a flatten patch.
    """
    patches = sliding_window_view(image, (patch_size, patch_size))
    rows, cols = index_set[:, 0], index_set[:, 1]
    patch_size = patches[rows, cols]
    return patch_size.reshape(patch_size.shape[0], -1)


def p_star_p(m, n, patch_width, overlap):
  """
  Compute the matrix P^*P, where P is the patch-set operator and P^* is the
  adjoint operator for P.

  Args:
      m (int)
      n (int)
      patch_width (int)
      overlap (int)

  Returns:
      numpy.ndarray: The matrix P^*P.
  """
  pstarp = numpy.zeros((m, n))
  stride = patch_width - overlap
  for i in range(0, m - patch_width + 1, stride):
    for j in range(0, n - patch_width + 1, stride):
      pstarp[i:i+patch_width, j:j+patch_width] += 1
  return pstarp


def apply_p_star_adjoint(patches, patch_size, image_shape, index_set):
    """
    Apply the adjoint operator (P^*) to map from the patch space to the image.
    
    Args:
        patches (numpy.ndarray): A 2D Array of shape (num_patches, patch_size^2),
                                 representing the patch space data (e.g., residuals or updates).
        patch_size (int): Size of the square patches (patch_size x patch_size).
        image_shape (tuple): Shape of the original image (m, n).
        index_set (numpy.ndarray): A 2D array of shape (num_patches, 2), where each row contains
                                   the (row, column) indices of the top-left corner of a patch.

    Returns:
        numpy.ndarray: Reconstructed 2D image of shape (m, n).
    """
    reconstructed_image = numpy.zeros(image_shape)

    # Generate indices for each pixel within a patch
    patch_indices = numpy.arange(patch_size)
    row_offsets, col_offsets = numpy.meshgrid(patch_indices, patch_indices, indexing="ij")

    # Map patch indices to the global image coordinates
    global_rows = index_set[:, 0][:, numpy.newaxis, numpy.newaxis] + row_offsets
    global_cols = index_set[:, 1][:, numpy.newaxis, numpy.newaxis] + col_offsets

    # Flatten the indices for vectorized accumulation
    flat_rows = global_rows.ravel()
    flat_cols = global_cols.ravel()
    flat_patches = patches.ravel()

    # Accumulate patch contributions into the reconstructed image
    numpy.add.at(reconstructed_image, (flat_rows, flat_cols), flat_patches)
    return reconstructed_image
