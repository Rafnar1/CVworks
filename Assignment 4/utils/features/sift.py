import cv2
import numpy as np
from utils.functions import convolve_2d
from utils.functions import gaussian_kernel



def generate_dogs(n, inital_kernel_size=11, initial_k=1.4, initial_sigma=1.6):
    # Creates DoGs for SIFT using sigmas: sigma, k * sigma, k^2 * sigma, k^3 * sigma ... k^n * sigma
    dogs = []
    for i in range(n):
        kernel_size = int(pow(initial_k, i) * inital_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # TODO: create a gaussian kernel "a" with k * sigma value

        a_sigma = initial_k * initial_sigma
        a = gaussian_kernel(kernel_size, a_sigma)

        a = a / a.sum()  # gaussian kernel normalization

        # TODO: create a gaussian kernel "b" with sigma value

        b_sigma = initial_sigma
        b = gaussian_kernel(kernel_size, b_sigma)

        b = b / b.sum()  # gaussian kernel normalization

        dog = b - a
        dogs.append(dog)
    return dogs


def convolve_with_dogs(grayscale, dogs):
    convolved_with_dogs = []
    for dog in dogs:
        convolved = None
        # TODO: convolve the given image with each DoG

        convolved = convolve_2d(grayscale, dog)
        convolved_with_dogs.append(convolved)
    return convolved_with_dogs


def non_maximum_supression(convolved_images, threshold):
    N = convolved_images.shape[0]

    filtered_convolved_images = []

    for k in range(N):
        convolved_image = convolved_images[k].copy()
        convolved_image1 = None
        convolved_image2 = convolved_images[k]
        convolved_image3 = None
        if k > 0:
            convolved_image1 = convolved_images[k - 1]
        if k < N - 1:
            convolved_image3 = convolved_images[k + 1]

        supress = np.full(convolved_image.shape, False)

        for i in range(3):
            for j in range(3):
                k_shift = np.zeros((3, 3))
                k_shift[i, j] = 1
                
                if convolved_image1 is not None:
                    convolved_image1 = convolve_2d(convolved_image1, k_shift)
                    supress = supress | (convolved_image1 < convolved_image)

                if convolved_image3 is not None:
                    convolved_image3 = convolve_2d(convolved_image3, k_shift)
                    supress = supress | (convolved_image3 < convolved_image)

                if i != 1 and j != 1:
                    convolved_image2 = convolve_2d(convolved_image2, k_shift)
                    supress = supress | (convolved_image2 < convolved_image)
        
        convolved_image[supress] = 0  # Set values below threshold to 0
        convolved_image[convolved_image >= threshold] = 255  # Set values above or equal to threshold to 255

        filtered_convolved_images.append(convolved_image)

    filtered_convolved_images = np.array(filtered_convolved_images)   

    result = None

    result = filtered_convolved_images

    # TODO: apply thresholding

    #######################
    # YOUR CODE GOES HERE #
    ####################### 

    return result


def sift_detector(image, initial_sigma, initial_k, initial_kernel_size):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)


    # TODO: call necessary function to do blob detection.
    # For now we do not need to filter by corner or to get orientations.
    # For now it is enough to have a Blob Detection (DoGs + Non-maximum Supression).

    dogs = generate_dogs(6, initial_kernel_size, initial_k, initial_sigma)
    convolved_images = convolve_with_dogs(grayscale, dogs)
    convolved_images_filtered = non_maximum_supression(np.array(convolved_images), -3)

    S, Y, X = np.where(convolved_images_filtered)

    features = []

    for i in range(len(S)):
        s = S[i]
        y = Y[i]
        x = X[i]
        features.append((s, y, x))
    
    for feature in features:
        sigma = pow(initial_k, feature[0]) * initial_sigma
        radius = int(sigma / np.sqrt(2)) + 1
        cv2.circle(image, (feature[2], feature[1]), radius, (0, 0, 255), 1)

    return image
