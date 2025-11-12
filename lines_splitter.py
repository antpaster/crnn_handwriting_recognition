import os
import cv2
# import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks

def project_vertical_avg(binary_image):
    """Compute the vertical projection profile of a binary image."""

    return cv2.reduce(binary_image, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32S).flatten()

def is_strict_local_min(y, i, window=10, confidence=0.6):
    """Check if y[i] is a strict local minimum within a given window."""

    left_idx = max(0, i - window)
    right_idx = min(len(y), i + window + 1)
    
    left_count = np.sum(y[left_idx:i] > y[i])
    right_count = np.sum(y[i+1:right_idx] > y[i])

    left_window_size = i - left_idx if i - left_idx == window else i - left_idx
    right_window_size = right_idx - (i + 1) if right_idx - (i + 1) == window else right_idx - (i + 1)

    return left_count / left_window_size >= confidence and right_count / right_window_size >= confidence

def get_lines(img_path: str):
    """Split an image into lines based on vertical projection profile."""

    # Load an image from file
    image = cv2.imread(img_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the grayscale image using Otsu's thresholding
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert the binary image so that text is white on black background
    binary_image = cv2.bitwise_not(binary_image)

    # # Create a figure and a 1x2 grid of subplots
    # # This means 1 row and 2 columns
    # fig, axes = plt.subplots(1, 2)

    # # Display the first image in the first subplot
    # axes[0].imshow(image, cmap='gray') # Specify cmap for grayscale images
    # axes[0].set_title('Original image')
    # axes[0].axis('off') # Hide axes ticks and labels

    binary_image_projection = project_vertical_avg(binary_image)

    # Smooth the signal
    # window_length must be odd, and larger = smoother
    projection_smoothed = savgol_filter(binary_image_projection, window_length=15, polyorder=3)

    # Detect peaks
    abysses, properties = find_peaks(
        -projection_smoothed,
        distance=30,     # minimal horizontal distance between peaks (tune this)
        # prominence=0.2,  # how much a peak stands out (tune this too)
    )

    threshold = 0.25 * projection_smoothed.max()
    deep_abysses = [i for i in abysses if projection_smoothed[i] < threshold]

    strict_abysses = [0] + [i for i in deep_abysses if is_strict_local_min(projection_smoothed, i, window=50, confidence=0.9)] + [image.shape[0] - 1]

    line_images = [gray_image[strict_abysses[i]:strict_abysses[i + 1], :] for i in range(len(strict_abysses) - 1)]
    return line_images

    # # Plot result
    # # plt.plot(binary_image_projection, color='red', alpha=0.5)
    # plt.plot(projection_smoothed, color='cyan', alpha=0.5)
    # plt.plot(projection_smoothed_gauss, color='magenta', alpha=0.5)
    # plt.plot(strict_abysses, projection_smoothed[strict_abysses], 'bo', marker='^')  # mark detected peaks
    # # plt.plot(deep_abysses, projection_smoothed[deep_abysses], 'go', marker='v')  # mark detected peaks

    # # Display the second image in the second subplot
    # axes[0].imshow(line_images[-2]) # Specify cmap for grayscale images
    # axes[0].set_title('Binary image')
    # axes[0].axis('off') # Hide axes ticks and labels
    # axes[0].hlines(strict_abysses, xmin=0, xmax=binary_image.shape[1], color='green', linewidth=1)

    # # Adjust layout to prevent titles from overlapping
    # plt.tight_layout()

    # # Show the figure with both images
    # plt.show()

# get_lines(os.path.join("../", "raw", "05.jpg"))
get_lines("a4_handwriting.png")