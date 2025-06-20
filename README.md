# Image Processing: Otsu Thresholding & Region Growing

This repository contains Python programs for two classic image segmentation techniques:

- **Task 1:** Otsu’s thresholding on a synthetic noisy image
- **Task 2:** Region growing based on user-defined seed points

---

## Task 1: Otsu’s Thresholding

### Description

- Generates a 300×300 grayscale image with a **circle** and a **square** on a background.
- Adds **Gaussian noise** (mean = 0, std = 20).
- Implements **Otsu’s algorithm from scratch** to find the optimal threshold.
- Produces a binary segmented output.

### How to Run

```bash
python EE7204_task1_4197.py
```

## Task 2: Region Growing Segmentation

### Description

- Loads a grayscale image from a user-specified file path.
- Accepts **seed points** manually entered by the user.
- Grows regions by including neighboring pixels with intensity similar to the seed(s), within a user-defined threshold.
- Implements a queue-based region growing algorithm from scratch.
- Produces a binary mask of the segmented region.

### How to Run

```bash
python EE7204_task2_4197.py
```
