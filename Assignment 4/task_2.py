import cv2
import numpy as np
from utils.image_processing import resize_image_keep_ratio
from utils.features.sift import sift_detector


def task_2_sift_corner_detector():
    MAX_HEIGHT = 300
    MAX_WIDTH = 300

    INITIAL_SIGMA = 3
    INITIAL_K = 1.2
    INITIAL_KERNEL_SIZE = 11
    DISPLAY_MAX_HEIGHT = 500
    DISPLAY_MAX_WIDTH = 500

    cam = cv2.VideoCapture(1)
    while True:

        # Read image from camera
        _, image = cam.read()

        if image is not None:

            # Resize image to max height or max width
            image = resize_image_keep_ratio(image, MAX_HEIGHT, MAX_WIDTH)

            # TODO: Implemenet the sift_detector function and see real-time results (please first do task_2.ipynb)
            display_image = sift_detector(image, initial_sigma=INITIAL_SIGMA, inital_k=INITIAL_K, initial_kernel_size=INITIAL_KERNEL_SIZE)

            display_image = resize_image_keep_ratio(display_image, DISPLAY_MAX_HEIGHT, DISPLAY_MAX_WIDTH)

            cv2.imshow('camera', display_image)

        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    task_2_sift_corner_detector()

