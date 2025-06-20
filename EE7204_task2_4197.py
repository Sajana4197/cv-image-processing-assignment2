import numpy as np
import cv2
import os
from collections import deque
import matplotlib.pyplot as plt

def region_growing(image, seeds, threshold=5):
    height, width = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=bool)
    queue = deque(seeds)
    seed_values = [image[y, x] for (x, y) in seeds]

    while queue:
        x, y = queue.popleft()

        if visited[y, x]:
            continue

        visited[y, x] = True
        current_val = image[y, x]

        if any(abs(int(current_val) - int(seed_val)) <= threshold for seed_val in seed_values):
            segmented[y, x] = 255

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                    queue.append((nx, ny))

    return segmented

if __name__ == "__main__":
    # Input image path
    image_path = input("Enter the full path to the image: ").strip('"')

    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("❌ Error: Could not load image. Please check the path and try again.")
        exit()

    # Show image for reference before entering seed points
    plt.figure(figsize=(6, 6))
    plt.title("Reference: Use this to choose seed points (x, y)")
    plt.imshow(img, cmap='gray')
    plt.axis('on')
    plt.show()

    # Get seed points from user
    print("Enter seed points in format: x1,y1 x2,y2 ... (Example: 100,150 120,160)")
    seed_input = input("Seed points: ")
    seed_points = [tuple(map(int, point.split(','))) for point in seed_input.split()]

    # Get threshold
    threshold = int(input("Enter intensity threshold (e.g., 10): "))

    # Perform region growing
    mask = region_growing(img, seed_points, threshold)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Segmented Region")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Save result in 'Results' folder
    output_folder = "Results"
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_path = os.path.join(output_folder, f"segmented_{name_without_ext}.png")

    cv2.imwrite(output_path, mask)
    print(f"✅ Segmented result saved to: {output_path}")
