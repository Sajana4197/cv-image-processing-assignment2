import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

os.makedirs('Results', exist_ok=True)

# Create high-resolution image (300x300) with circle and square
def create_image():
    img = np.ones((300, 300), dtype=np.uint8) * 85  # Background

    # Circle (Object 1)
    y, x = np.ogrid[:300, :300]
    circle = (x - 90)**2 + (y - 90)**2 <= 45**2
    img[circle] = 170

    # Square (Object 2)
    img[180:240, 180:240] = 255

    return img

# Add Gaussian noise
def add_noise(image, std=20):
    noise = np.random.normal(0, std, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

# Otsu's thresholding
def otsu_threshold(image):
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    total = image.size
    best_thresh, max_var = 0, 0

    for t in range(1, 256):
        w0, w1 = np.sum(hist[:t]), np.sum(hist[t:])
        if w0 == 0 or w1 == 0:
            continue
        mu0 = np.sum(np.arange(t) * hist[:t]) / w0
        mu1 = np.sum(np.arange(t, 256) * hist[t:]) / w1
        var = w0 * w1 * (mu0 - mu1)**2
        if var > max_var:
            max_var, best_thresh = var, t
    return best_thresh

original = create_image()
noisy = add_noise(original)
thresh = otsu_threshold(noisy)
binary = (noisy > thresh).astype(np.uint8) * 255

# Create and save the combined figure
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(original, cmap='gray')
axes[0].set_title('Original (Circle & Square)')
axes[0].axis('off')

axes[1].imshow(noisy, cmap='gray')
axes[1].set_title('With Gaussian Noise')
axes[1].axis('off')

axes[2].imshow(binary, cmap='gray')
axes[2].set_title('Otsu Threshold Result')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('Results/combined_figure_task1.png', dpi=300)
plt.show()

print("✅ Task is done.")
print("✅ Result saved in 'Results/' folder as combined_figure_task1.")

