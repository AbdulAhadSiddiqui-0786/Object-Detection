# This makes the root folder a Python package

from .GUI import ObjectDetectionApp  # optional if you plan to import this in future
from image import sample_image_list

print("Images:", sample_image_list())
