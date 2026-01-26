import numpy as np
import cv2 as cv
import os

# ----------------------------
# Paths
# ----------------------------
input_image_path = "Images/image4.jpg"
images_dir = "Images"
results_dir = "results"

os.makedirs(images_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

img = cv.imread(input_image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {input_image_path}")

H, W = img.shape[:2]

# ----------------------------
# Harris Corner Detection (Scale-Adaptive)
# ----------------------------
def apply_harris(bgr_img: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
    """
    Apply Harris corner detection with scale-adaptive parameters.
    
    Args:
        bgr_img: Input BGR image
        scale_factor: Scale factor applied to the image (e.g., 4.0 for 4x zoom in, 0.5 for 2x zoom out)
    
    Returns:
        Image with detected corners marked (red: centroid, green: refined)
    """
    out = bgr_img.copy()

    gray = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
    gray_f = np.float32(gray)

    # Scale-adaptive parameters: increase block size and kernel size with scale
    # This ensures fewer corners are detected when scaling up (as expected)
    block_size = max(2, int(2 * scale_factor))
    ksize = max(3, int(3 * scale_factor))
    if ksize % 2 == 0:  # ksize must be odd for Sobel operator
        ksize += 1

    print(f"Scale: {scale_factor:.2f} | blockSize: {block_size} | ksize: {ksize}")

    dst = cv.cornerHarris(gray_f, block_size, ksize, 0.04)
    dst = cv.dilate(dst, None)

    _, dst_thr = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst_thr = np.uint8(dst_thr)

    _, labels, stats, centroids = cv.connectedComponentsWithStats(dst_thr)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray_f, np.float32(centroids), (5, 5), (-1, -1), criteria)

    centroids_i = np.int32(centroids)
    corners_i = np.int32(corners)

    print(f"Detected corners: {len(centroids_i)}")

    for i in range(len(centroids_i)):
        cv.circle(out, tuple(centroids_i[i]), 2, (0, 0, 255), -1)  # red: centroid
        cv.circle(out, tuple(corners_i[i]), 2, (0, 255, 0), -1)    # green: refined

    return out

# ----------------------------
# Utility: center crop to (H, W)
# ----------------------------
def center_crop(bgr_img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = bgr_img.shape[:2]
    if h < target_h or w < target_w:
        raise ValueError("Cannot center-crop because image is smaller than target size.")

    y0 = (h - target_h) // 2
    x0 = (w - target_w) // 2
    return bgr_img[y0:y0 + target_h, x0:x0 + target_w]

# ----------------------------
# Utility: pad to (H, W) (for zoom-out)
# ----------------------------
def pad_to_size(bgr_img: np.ndarray, target_h: int, target_w: int,
                border_type=cv.BORDER_CONSTANT, value=(0, 0, 0)) -> np.ndarray:
    h, w = bgr_img.shape[:2]
    if h > target_h or w > target_w:
        raise ValueError("Cannot pad because image is larger than target size. Crop instead.")

    top = (target_h - h) // 2
    bottom = target_h - h - top
    left = (target_w - w) // 2
    right = target_w - w - left

    return cv.copyMakeBorder(bgr_img, top, bottom, left, right, border_type, value=value)

# ----------------------------
# Same-size Zoom In (scale up then crop)
# ----------------------------
def zoom_in_same_size(bgr_img: np.ndarray, scale: float, target_h: int, target_w: int) -> np.ndarray:
    if scale <= 1.0:
        raise ValueError("zoom_in_same_size requires scale > 1.0")

    interp = cv.INTER_CUBIC
    scaled = cv.resize(bgr_img, None, fx=scale, fy=scale, interpolation=interp)
    return center_crop(scaled, target_h, target_w)

# ----------------------------
# Same-size Zoom Out (scale down then pad)
# ----------------------------
def zoom_out_same_size(bgr_img: np.ndarray, scale: float, target_h: int, target_w: int) -> np.ndarray:
    if scale >= 1.0:
        raise ValueError("zoom_out_same_size requires scale < 1.0")

    interp = cv.INTER_AREA
    scaled = cv.resize(bgr_img, None, fx=scale, fy=scale, interpolation=interp)

    # You MUST pad to keep same size; "cut excess" is not applicable here because it's smaller.
    # Choose border type: CONSTANT (black), REPLICATE, REFLECT, etc.
    return pad_to_size(scaled, target_h, target_w, border_type=cv.BORDER_CONSTANT, value=(0, 0, 0))

# ----------------------------
# Same-size Rotate (rotate in-place, then crop to original size automatically)
# ----------------------------
def rotate_same_size(bgr_img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = bgr_img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv.getRotationMatrix2D(center, angle_deg, 1.0)

    # warpAffine output size is SAME (w, h) -> corners may be clipped (as you requested: cut excess)
    rotated = cv.warpAffine(
        bgr_img, M, (w, h),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REPLICATE
    )
    return rotated

# ----------------------------
# Create transformed images (SAME SIZE as original)
# ----------------------------
print("\n=== Creating Transformed Images ===")
scale_up_img = zoom_in_same_size(img, scale=4.0, target_h=H, target_w=W)     # zoom in, crop excess
scale_down_img = zoom_out_same_size(img, scale=0.5, target_h=H, target_w=W)  # zoom out, pad to size
rotated_img = rotate_same_size(img, angle_deg=30)                            # rotate, cut excess

# ----------------------------
# Save transformed images (Images folder)
# ----------------------------
cv.imwrite(os.path.join(images_dir, "scale_up_img1.jpg"), scale_up_img)
cv.imwrite(os.path.join(images_dir, "scale_down_img1.jpg"), scale_down_img)
cv.imwrite(os.path.join(images_dir, "rotatted_img1.jpg"), rotated_img)

# ----------------------------
# Apply Harris on transformed images WITH SCALE FACTORS
# ----------------------------
print("\n=== Applying Harris Corner Detection ===")
print("\n--- Original Image (scale=1.0) ---")
harris_original = apply_harris(img, scale_factor=1.0)

print("\n--- Scaled Up Image (scale=4.0) ---")
harris_scale_up = apply_harris(scale_up_img, scale_factor=4.0)

print("\n--- Scaled Down Image (scale=0.5) ---")
harris_scale_down = apply_harris(scale_down_img, scale_factor=0.5)

print("\n--- Rotated Image (scale=1.0) ---")
harris_rotated = apply_harris(rotated_img, scale_factor=1.0)

# ----------------------------
# Save results (results folder)
# ----------------------------
cv.imwrite(os.path.join(results_dir, "original_img1.jpg"), harris_original)
cv.imwrite(os.path.join(results_dir, "scale_up_img1.jpg"), harris_scale_up)
cv.imwrite(os.path.join(results_dir, "scale_down_img1.jpg"), harris_scale_down)
cv.imwrite(os.path.join(results_dir, "rotatted_img1.jpg"), harris_rotated)

print("\n=== Complete ===")
print("Transformed images saved in Images/")
print("Harris results saved in results/")
