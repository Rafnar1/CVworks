import cv2
import numpy as np
from utils.image_processing import resize_image_keep_ratio
from utils.features.harris import harris_detector


def task_1_harris_corner_detector():
    MAX_HEIGHT = 300
    MAX_WIDTH = 300
    DISPLAY_MAX_HEIGHT = 800
    DISPLAY_MAX_WIDTH = 800

    cam = cv2.VideoCapture(0)
    while True:

        # Read image from camera
        _, image = cam.read()

        if image is not None:

            # Resize image to max height or max width
            image = resize_image_keep_ratio(image, MAX_HEIGHT, MAX_WIDTH)

            # TODO: implement the harris_detector function
            display_image = harris_detector(image)

            display_image = resize_image_keep_ratio(display_image, DISPLAY_MAX_HEIGHT, DISPLAY_MAX_WIDTH)

            cv2.imshow('camera', display_image)

        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    task_1_harris_corner_detector()
