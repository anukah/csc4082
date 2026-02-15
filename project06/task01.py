import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread("frame3.png")
img2 = cv.imread("frame4.png")

if img1 is None or img2 is None:
    raise FileNotFoundError("Could not load images")

if img1.shape[:2] != img2.shape[:2]:
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))

g1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
g2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

g1 = cv.GaussianBlur(g1, (5, 5), 0)
g2 = cv.GaussianBlur(g2, (5, 5), 0)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
g1 = clahe.apply(g1)
g2 = clahe.apply(g2)

diff = cv.absdiff(g2, g1)

d = diff.astype(np.float32).ravel()
med = np.median(d)
mad = np.median(np.abs(d - med))
sigma = 1.4826 * mad

k = 4.0
T = int(np.clip(k * sigma, 10, 80))

_, mask = cv.threshold(diff, T, 255, cv.THRESH_BINARY)

kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_close)

overlay = img2.copy()
overlay[mask > 0] = (0, 0, 255)
vis = cv.addWeighted(img2, 0.75, overlay, 0.25, 0)

img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
vis_rgb = cv.cvtColor(vis, cv.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1); plt.imshow(img1_rgb); plt.axis("off"); plt.title("Image 1")
plt.subplot(2, 3, 2); plt.imshow(img2_rgb); plt.axis("off"); plt.title("Image 2")
plt.subplot(2, 3, 3); plt.imshow(vis_rgb);  plt.axis("off"); plt.title("Motion Overlay")
plt.subplot(2, 3, 4); plt.imshow(diff, cmap="gray"); plt.axis("off"); plt.title("Absolute Difference")
plt.subplot(2, 3, 5); plt.imshow(mask, cmap="gray"); plt.axis("off"); plt.title("Motion Mask")

plt.tight_layout()
plt.show()
