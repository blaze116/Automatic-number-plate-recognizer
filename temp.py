import os
import io
import cv2
import base64 
import numpy as np

from PIL import Image

# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

# temp = stringToImage(image_data)
# cv2.imwrite("temp.jpg",toRGB(temp)