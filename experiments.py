from PIL import Image
import os

# Create a dummy image for demonstration
# In a real scenario, you would open an existing image:
# img = Image.open("your_image.jpg")
# img = Image.new('RGB', (400, 300), color = 'red')
img = Image.open(os.path.join("images", "0219_1.jpg")).convert('L')
w, h = img.size

# Define the source quadrilateral's corners in the input image.
# The order is typically: (upper-left, lower-left, lower-right, upper-right)
# Each point is an (x, y) tuple.
# Example: a distorted rectangle within the image
source_quad = (
    0, 0,    # Top-left corner (x0, y0)
    0, h,   # Bottom-left corner (x1, y1)
    w, h,  # Bottom-right corner (x2, y2)
    w, 0    # Top-right corner (x3, y3)
)

# Define the size of the new image where the quad will be mapped to a rectangle.
# The quad will be stretched/shrunk to fit this new rectangular area.
new_size = (int(w * 0.7), int(h * 0.7))

# Apply the quad transformation
# The Image.QUAD method takes an 8-tuple representing the source quadrilateral's corners.
transformed_img = img.transform(new_size, Image.QUAD, source_quad)

# Display the original and transformed images (optional)
img.show(title="Original Image")
transformed_img.show(title="Transformed Image (Quad to Rectangle)")

# Save the transformed image (optional)
# transformed_img.save("transformed_quad_example.png")