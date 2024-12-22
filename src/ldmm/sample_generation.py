import numpy

def degrade_image_inpainting(image, retain_ratio=0.1, random_seed=0):
    """
    Create a degraded image for inpainting by masking random pixels.

    Args:
        image (numpy.ndarray): Original image.
        retain_ratio (float): Fraction of pixels to retain (0 < retain_ratio <= 1).
        random_seed (int): Seed for reproducibility.

    Returns:
        numpy.ndarray: Degraded image with missing pixels set to 0.
        numpy.ndarray: Binary mask indicating retained pixels (1 for known, 0 for missing).
    """
    numpy.random.seed(random_seed)
    mask = numpy.random.binomial(n=1, p=retain_ratio, size=image.shape)
    degraded_image = numpy.copy(image) * mask
    return degraded_image, mask


def fill_degraded_pixels(degraded_image, mask):
    """
    Create the initial guess for the inpainting by filling the missing pixels
    with random values from a Gaussian distribution N(μ0, σ0), where μ0 is the
    mean and σ0 is the standard deviation of the known (retained) pixels in the image.

    Args:
        degraded_image (numpy.ndarray): The degraded image with missing pixels.
        mask (numpy.ndarray): Binary mask indicating retained (1) and missing (0) pixels.

    Returns:
        numpy.ndarray: The initial image estimate with missing pixels filled.
    """
    numpy.random.seed(0)
    retained_pixels = degraded_image[mask != 0]  # Get known pixels
    # Fill missing pixels. Values are drawn from the maximum likelihood gaussian
    # distribution for the degraded image.
    initial_image = numpy.random.normal(
        loc=numpy.mean(retained_pixels),
        scale=numpy.std(retained_pixels),
        size=degraded_image.shape)
    initial_image[mask != 0] = degraded_image[mask != 0]
    return initial_image
