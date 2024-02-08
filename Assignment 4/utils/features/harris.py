import cv2
import numpy as np
from utils.functions import convolve_2d
from utils.functions import gaussian_kernel


def non_maximum_suppression(R):
    return R * (R == np.max(convolve_2d(R, np.ones((5, 5)))))


def harris_detector(image, k=0.06):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float) / 255  # You may need to edit

    blurred = convolve_2d(grayscale, gaussian_kernel(5, 1))  # You may need to edit

    Gx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ])
    Gy = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1,  -2, -1],
    ])

    # Compute gradients
    Ix = convolve_2d(blurred, Gx)
    Iy = convolve_2d(blurred, Gy)

    # Compute IxIx, IyIy, IxIy and convolve them with the gaussian kernel
    IxIx = convolve_2d(Ix * Ix, gaussian_kernel(5, 1))
    IyIy = convolve_2d(Iy * Iy, gaussian_kernel(5, 1))
    IxIy = convolve_2d(Ix * Iy, gaussian_kernel(5, 1))

    # Compute corner response function
    det_M = IxIx * IyIy - IxIy**2
    trace_M = IxIx + IyIy
    R = det_M - k * (trace_M**2)

    # Non-maximum suppression
    R = non_maximum_suppression(R)

    C = R > 0.003

    Y, X = np.where(C)

    for i in range(len(X)):
        cv2.circle(image, (X[i], Y[i]), 1, (0, 0, 255), 1)

    print("Found %s features" % len(X))

    return image
