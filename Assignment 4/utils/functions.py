import numpy as np
import cv2

def apply_transformation(image, M):
    height, width, channels = image.shape[0], image.shape[1], image.shape[2]
    new_image = np.reshape(np.zeros_like(image), (height * width, channels))
    image = np.reshape(image, (height * width, channels))

    M_inv = np.linalg.inv(M)
    
    C = np.array(np.meshgrid(np.arange(height), np.arange(width), np.array([1]))).T.reshape(-1, 3)
    
    V = np.matmul(M_inv, C[..., np.newaxis]).squeeze()
    V = np.round(V).astype(int)
    
    indices = ~np.logical_or(np.logical_or(V[:,0] >= height, V[:,0] < 0), np.logical_or(V[:,1] >= width, V[:,1] < 0))
    
    V = V[indices]
    C = C[indices]

    C = C[:, 0] * width + C[:, 1]
    V = V[:, 0] * width + V[:, 1]

    new_image[C] = image[V]

    new_image = np.reshape(new_image, (height, width, channels))

    return new_image


def convolve_2d(image, kernel):
    k_h, k_w = kernel.shape
    assert k_h % 2 == 1
    assert k_w % 2 == 1
    kernel = np.flip(kernel)
    padding_h = (k_h - 1) // 2
    padding_w = (k_w - 1) // 2
    padded_image = np.pad(image, [(padding_h, padding_h), (padding_w, padding_w)], 'constant', constant_values = 0)
    windows = np.lib.stride_tricks.sliding_window_view(padded_image, kernel.shape)
    res = np.einsum('ij,klij->kl', kernel, windows)
    return res


def gaussian_kernel(size, sigma):
    x = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    y = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    x, y = np.meshgrid(x, y)
    exp = np.exp(-((np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2))))
    return exp / (2 * np.pi * (sigma ** 2))


def minkowski_distance(a, b, p=2):
    return np.sum(np.abs(a - b) ** p)**(1/p)


def get_features(image, include_position=False):
    features = []

    height, width = image.shape[:2]
    features = []

    for y in range(height):
        for x in range(width):
            feature = image[y, x]
            if include_position:
                feature = np.append(feature, [x, y])
            features.append(feature)
    return np.array(features)
