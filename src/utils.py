import numpy

def pad_image(image, patch_size, overlap):
  """
  Pad an image with a border of width 'stride'.

  Args:
    image (numpy.ndarray): The input image.
    stride (int)

  Returns:
    numpy.ndarray: The padded image.
  """
  m, n = image.shape
  stride = patch_size - overlap
  padded_image = numpy.zeros((m + stride + patch_size, n + stride + patch_size))
  padded_image[stride:stride+m, stride:stride+n] = image
  return padded_image


# TODO: With a little work, this could be done implicitly and speed things up a
# bit.
def enforce_boundary(image, patch_size, overlap):
  """
  Reflect the image over the boundary, and crop such that the image is padded
  with a border of width 'stride'.

  Args:
    image (numpy.ndarray): The input image.
    stride (int)

  Returns:
    numpy.ndarray: The padded, reflected image.
  """
  stride = patch_size - overlap
  m_pad, n_pad = image.shape
  m = m_pad - stride - patch_size
  n = n_pad - stride - patch_size

  # Reflect edges.
  image[0:stride, stride:stride+n] = numpy.flip(image[stride:2*stride, stride:stride+n], axis=0)
  image[m+stride:m_pad, stride:stride+n] = numpy.flip(image[m_pad-2*patch_size:m_pad-patch_size, stride:stride+n], axis=0)
  image[stride:stride+m, 0:stride] = numpy.flip(image[stride:stride+m, stride:2*stride], axis=1)
  image[stride:stride+m, n+stride:n_pad] = numpy.flip(image[stride:stride+m, n_pad-2*patch_size:n_pad-patch_size], axis=1)

  # Reflect corners.
  image[0:stride, 0:stride] = numpy.flip(numpy.flip(image[stride:2*stride, stride:2*stride], axis=1), axis=0)
  image[0:stride, n_pad-patch_size:n_pad] = numpy.flip(numpy.flip(image[stride:2*stride, n_pad-2*patch_size:n_pad-patch_size], axis=1), axis=0)
  image[m_pad-patch_size:m_pad, 0:stride] = numpy.flip(numpy.flip(image[m_pad-2*patch_size:m_pad-patch_size, stride:2*stride], axis=1), axis=0)
  image[m_pad-patch_size:m_pad, n_pad-patch_size:n_pad] = numpy.flip(numpy.flip(image[m_pad-2*patch_size:m_pad-patch_size, n_pad-2*patch_size:n_pad-patch_size], axis=1), axis=0)

  return image
