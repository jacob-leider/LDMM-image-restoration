import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# URLs for test images: Lena, Barbara
LENA_URL = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
BARB_URL = "https://www.math.hkust.edu.hk/~masyleung/Teaching/CAS/MATLAB/image/images/barbara.jpg"

def lena_image():
  # Fetch.
  response = requests.get(LENA_URL)
  response.raise_for_status()
  # Open and resize.
  img = Image.open(BytesIO(response.content)).convert("L")  # Convert to grayscale
  img_resized = img.resize((256, 256))  # Resize to 256x256
  # Normalize.
  data = numpy.array(img_resized, dtype=numpy.float32)
  data = data / 256
  return data


def barb_image():
  # Fetch.
  response = requests.get(BARB_URL)
  response.raise_for_status()
  # Open and resize.
  img = Image.open(BytesIO(response.content)).convert("L")  # Convert to grayscale
  # Normalize.
  data = numpy.array(img, dtype=numpy.float32)
  data = data / 256
  # Case-specific.
  x_offset = 40
  y_offset = 165
  data = data[x_offset:x_offset+256, y_offset:y_offset+256]
  return data
