# This makes the 'image' folder a Python package
# You can put utility functions here if needed

def sample_image_list():
    import os
    return [f for f in os.listdir(os.path.dirname(__file__)) if f.endswith(('.png', '.jpg', '.jpeg'))]
