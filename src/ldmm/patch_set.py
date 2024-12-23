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


def patch_set_operator(image: numpy.ndarray, 
                       index_set: numpy.ndarray, 
                       patch_size: tuple[int, int], 
                       U: numpy.ndarray):
  """
  Applies the patch-set operator to the input image.
  - Uses a circular boundary condition.

  Args:
    image (numpy.ndarray): The input image.
  
  Returns:
    numpy.ndarray: The image after applying the patch-set operator.
  """
  try:
    patch_width, patch_height = patch_size
    d = patch_width * patch_height
  except:
    raise ValueError("patch_size must be a tuple of (width, height)")

  idxx, idxy = index_set.T
  patches = sliding_window_view(image, patch_size)
  U[:] = patches[idxx, idxy].reshape((len(index_set), d))

  return U


def patch_set_adjoint_operator(U: numpy.ndarray,
                               index_set: numpy.ndarray,
                               patch_size: tuple[int, int],
                               image: numpy.ndarray):
  """
  Applies the adjoint operator of the patch set operator determined by 
  index_set. Patches are added to their positions in image.

  Args:
    U (numpy.ndarray): An array of flattened patches. The first axis corresponds 
    to the first axis of index_set.
    index_set (numpy.ndarray): Set of indices for the patches. An index 
    corresponds to a patch's top left corner.
    patch_size (tuple[int, int]): Patch dimensions.
    image (numpy.ndarray): Where the result of this operation will be stored. 
    Previous contents will be deleted.
  
  Returns:
    image (numpy.ndarray): The result of the patch set operator on U.
  """
  # Zero out the image.
  image[:] = 0

  # Generate indices for each pixel within a patch
  patch_indices_x = numpy.arange(patch_size[0])
  patch_indices_y = numpy.arange(patch_size[1])

  row_offsets, col_offsets = numpy.meshgrid(patch_indices_x,
                                         patch_indices_y,
                                         indexing="ij")

  # Map patch indices to the global image coordinates
  global_rows = index_set[:, 0][:, numpy.newaxis, numpy.newaxis] + row_offsets
  global_cols = index_set[:, 1][:, numpy.newaxis, numpy.newaxis] + col_offsets

  # Accumulate patch contributions into the reconstructed image
  numpy.add.at(image,
            (global_rows.ravel(), global_cols.ravel()), 
            U.ravel())
