import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

def project_vertical_avg(binary_image):
    """Compute the vertical projection profile of a binary image."""
    return cv2.reduce(binary_image, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32S).flatten()

# Load an image from file
image = cv2.imread("a4_handwriting.png")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binarize the grayscale image using Otsu's thresholding
_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Invert the binary image so that text is white on black background
binary_image = cv2.bitwise_not(binary_image)

# Create a figure and a 1x2 grid of subplots
# This means 1 row and 2 columns
fig, axes = plt.subplots(1, 2)

# Display the first image in the first subplot
axes[0].imshow(image, cmap='gray') # Specify cmap for grayscale images
axes[0].set_title('Original image')
axes[0].axis('off') # Hide axes ticks and labels

binary_image_projection = project_vertical_avg(binary_image)

# Detect peaks
abysses, properties = find_peaks(
    -binary_image_projection,
    distance=30,     # minimal horizontal distance between peaks (tune this)
    # prominence=0.2,  # how much a peak stands out (tune this too)
)

threshold = 0.25 * binary_image_projection.max()
deep_abysses = [i for i in abysses if binary_image_projection[i] < threshold]

strict_abysses = []
for i in deep_abysses:
    # Ensure we have valid neighbors
    if 0 < i < len(binary_image_projection) - 1:
        if binary_image_projection[i] < binary_image_projection[i - 1] and binary_image_projection[i] < binary_image_projection[i + 1]:
            strict_abysses.append(i)

# Plot result
plt.plot(binary_image_projection, color='red')
# plt.plot(strict_abysses, binary_image_projection[strict_abysses], 'bo')  # mark detected peaks
plt.plot(deep_abysses, binary_image_projection[deep_abysses], 'go')  # mark detected peaks

# # Display the second image in the second subplot
# axes[1].imshow(binary_image, cmap='gray') # Specify cmap for grayscale images
# axes[1].set_title('Binary image')
# axes[1].axis('off') # Hide axes ticks and labels
# axes[1].hlines(deep_abysses, xmin=0, xmax=binary_image.shape[1], color='green', linewidth=1)

# Adjust layout to prevent titles from overlapping
plt.tight_layout()

# Show the figure with both images
plt.show()