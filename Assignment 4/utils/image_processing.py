import cv2
import numpy as np


def resize_image_keep_ratio(image, max_height, max_width):
    h, w, = image.shape[0], image.shape[1]

    if h > w:
        r = w / h
        h = max_height
        w = int(r * h)
    else:
        r = h / w
        w = max_width
        h = int(r * w)

    image = cv2.resize(image, (w, h))
    return image
