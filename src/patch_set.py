import numpy
from numpy.lib.stride_tricks import sliding_window_view

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
