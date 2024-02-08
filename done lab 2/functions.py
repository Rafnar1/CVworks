import numpy as np
import cv2
import matplotlib.pyplot as plt
import random


def display_image(image, figsize=(10, 10)):
    # Takes image of the shape HxWx3 or HxWx1 or HxW.
    is_grayscale = len(image.shape) == 2 or image.shape[2] == 1
    plt.figure(figsize=figsize, dpi=80)
    if is_grayscale:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def rgb_to_grayscale(image):
    # Takes image of the shape HxWx3.
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    res = (R*0.299 + G*0.587 + B*0.114)
    #res = res.astype(np.uint8)
    return res


def apply_transformation(image, M):
    # Takes image of the shape HxW and M of the shape 3x3 (generlized affine transformation).
    new_image = np.zeros_like(image)
    height, width = new_image.shape[0], new_image.shape[1]
    M_inv = np.linalg.inv(M)
    for i in range(height):
        for j in range(width):
            # YOUR CODE GOES HERE
            # ...
            original_coords = np.dot(M_inv, np.array([i, j, 1]))
            original_x, original_y = original_coords[:2] / original_coords[2]
            # Check if the original coordinates are within bounds
            if 0 <= original_x < height and 0 <= original_y < width:
                # Use interpolation to calculate the pixel value in the output image
                x0, y0 = int(original_x), int(original_y)
                x1, y1 = x0 + 1, y0 + 1
                alpha = original_x - x0
                beta = original_y - y0
                new_image[i, j] = (1 - alpha) * ((1 - beta) * image[x0, y0] + beta * image[x0, y1]) + alpha * ((1 - beta) * image[x1, y0] + beta * image[x1, y1])

            # END
    return new_image


def apply_transformation_optimized(image, M):
    # Takes image of the shape HxW and M of the shape 3x3 (generlized affine transformation).
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
    # Takes image of the shape HxW and kernel of the shape MxN.
    height, width = image.shape[0], image.shape[1]
    k_h, k_w = kernel.shape
    assert k_h % 2 == 1
    assert k_w % 2 == 1
    kernel = np.flip(kernel)
    padding_h = (k_h - 1) // 2
    padding_w = (k_w - 1) // 2
    padded_image = np.pad(image, [(padding_h, padding_h), (padding_w, padding_w)], 'constant', constant_values = 0)
    res = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            # Extract the region from the padded image
            region = padded_image[i:i+k_h, j:j+k_w]
            # Perform element-wise multiplication and sum
            result_pixel = np.sum(region * kernel)
            # Set the result in the output image
            res[i, j] = result_pixel
    
    return res


def convolve_2d_optimized(image, kernel):
    # Takes image of the shape HxW and kernel of the shape MxN.
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
    # Takes size and sigma values for the gaussian function.\
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * np.exp(-((x - (size-1)/2) ** 2 + (y - (size-1)/2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    res = kernel / np.sum(kernel)
    return res


def gaussian_blur(image, size, sigma):
    # Takes size and sigma values for the gaussian function and applies gaussian blur to image.
    res = None
    kernel = gaussian_kernel(size, sigma)
    res = convolve_2d_optimized(image, kernel)
    return res


def sobel_edge_detection(image):
    # Applies Sobel Edge Detection algorithm to the given image and returns gradients and orientations.
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

    magnitudes = None
    angles = None
    gradient_x = convolve_2d_optimized(image, Gx)
    gradient_y = convolve_2d_optimized(image, Gy)

    # Compute gradient magnitudes and angles
    magnitudes = np.sqrt(gradient_x**2 + gradient_y**2)
    angles = np.arctan2(gradient_y, gradient_x)

    return magnitudes, angles


def non_max_suppression(magnitudes, angles):
    # Removes unnecessary edges. It looks for each 3 magnitues in the same orientations and keeps only the one with the highes magnitude.
    M, N = magnitudes.shape
    Z = np.zeros((M, N), dtype=magnitudes.dtype)

    angles = angles * 180 / np.pi
    angles[angles < 0] += 180
    
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255
                
                if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                    q = magnitudes[i, j + 1]
                    r = magnitudes[i, j - 1]

                elif (22.5 <= angles[i, j] < 67.5):
                    q = magnitudes[i + 1, j - 1]
                    r = magnitudes[i - 1, j + 1]

                elif (67.5 <= angles[i, j] < 112.5):
                    q = magnitudes[i + 1, j]
                    r = magnitudes[i - 1, j]

                elif (112.5 <= angles[i,j] < 157.5):
                    q = magnitudes[i - 1, j - 1]
                    r = magnitudes[i + 1, j + 1]

                # YOUR CODE GOES HERE
                # ...
                # END
                if (magnitudes[i, j] >= q) and (magnitudes[i, j] >= r):
                    Z[i, j] = magnitudes[i, j]

            except IndexError as e:
                pass

    return Z


def hysteresis(edges, t1 = 30, t2 = 90):
    # Seperates edges into strong and weak ones based on the given thresholds: t1, t2
    strong = None
    weak = None
    # Create arrays for strong and weak edges
    strong = np.zeros_like(edges)
    weak = np.zeros_like(edges)

    # Identify strong and weak edges based on thresholds
    strong[edges >= t2] = 1
    weak[(edges >= t1) & (edges < t2)] = 1
    return strong, weak


def edge_analysis_connect(i, j, visited, strong, weak):
    # Used by edge_analysis
    height, width = visited.shape
    visited[i, j] = True
    # Define neighboring pixel coordinates (you might need to adjust this based on your requirements)
    neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

    for ni, nj in neighbors:
        if 0 <= ni < height and 0 <= nj < width:
            # Check if the neighboring pixel is a weak edge and unvisited
            if weak[ni, nj] and not visited[ni, nj]:
                # Mark the neighboring pixel as a strong edge
                strong[ni, nj] = 1
                # Recursively connect the edge
                edge_analysis_connect(ni, nj, visited, strong, weak)


def edge_analysis(strong, weak):
    # Run DFS algorithm to analyze connectivity. Weak edges converted to strong edges if they are connected to strong edges.
    # takes strong and weak edges from hysteresis
    height, width = strong.shape
    visited = np.full((height, width), False)

    for i in range(height):
        for j in range(width):
            if strong[i, j] and not visited[i, j]:
                # Start DFS from a strong edge
                edge_analysis_connect(i, j,visited,strong,weak)

    res = visited  # you may change this
    return res


def canny_edge_detection(image, gaussian_blur_size, gaussian_blur_sigma, hysteresis_t1, hysteresis_t2):
    # Runs Canny Edge Detection algorithm for the given RGB image.
    edges = None

    # Step 1: Convert the image to grayscale
    gray_image = rgb_to_grayscale(image)

    # Step 2: Apply Gaussian Blur
    blurred_image = gaussian_blur(gray_image, gaussian_blur_size, gaussian_blur_sigma)

    # Step 4: Calculate gradient magnitude and direction
    gradient_magnitude, gradient_direction = sobel_edge_detection(gray_image)

    # Step 5: Non-maximum suppression
    suppressed_magnitude = non_max_suppression(gradient_magnitude, gradient_direction)

    # Step 6: Hysteresis thresholding
    strong_edges, weak_edges = hysteresis(suppressed_magnitude, t1 = hysteresis_t1, t2 = hysteresis_t2)
    
    # Step 7: Edge Analysis and Connectivity
    edges = edge_analysis(strong_edges, weak_edges)
    # END

    return edges
