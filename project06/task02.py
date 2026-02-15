import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def dense_lk_fast(prev_gray, next_gray, win=15, eps=1e-4):
    Ix = cv.Scharr(prev_gray, cv.CV_32F, 1, 0) / 32.0
    Iy = cv.Scharr(prev_gray, cv.CV_32F, 0, 1) / 32.0
    It = next_gray - prev_gray

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    Ixt = Ix * It
    Iyt = Iy * It

    ksize = (win, win)

    Sxx = cv.boxFilter(Ixx, -1, ksize, normalize=False, borderType=cv.BORDER_REFLECT)
    Syy = cv.boxFilter(Iyy, -1, ksize, normalize=False, borderType=cv.BORDER_REFLECT)
    Sxy = cv.boxFilter(Ixy, -1, ksize, normalize=False, borderType=cv.BORDER_REFLECT)
    Sxt = cv.boxFilter(Ixt, -1, ksize, normalize=False, borderType=cv.BORDER_REFLECT)
    Syt = cv.boxFilter(Iyt, -1, ksize, normalize=False, borderType=cv.BORDER_REFLECT)

    det = Sxx * Syy - Sxy * Sxy
    valid = det > eps

    u = np.zeros_like(det, dtype=np.float32)
    v = np.zeros_like(det, dtype=np.float32)

    u[valid] = (-Syy[valid] * Sxt[valid] + Sxy[valid] * Syt[valid]) / det[valid]
    v[valid] = ( Sxy[valid] * Sxt[valid] - Sxx[valid] * Syt[valid]) / det[valid]

    flow = np.dstack([u, v])
    return flow

def flow_to_hsv_bgr(flow):
    u, v = flow[...,0], flow[...,1]
    mag, ang = cv.cartToPolar(u, v)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[...,1] = 255
    hsv[...,0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

video_path = "slow_traffic_small.mp4"
cap = cv.VideoCapture(video_path)

ret, frame0 = cap.read()
if not ret:
    raise RuntimeError("Cannot read video")

prev = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0

plt.ion()
fig, ax = plt.subplots(figsize=(10,6))
ax.axis("off")
im = ax.imshow(np.zeros((*prev.shape,3), dtype=np.uint8))

for _ in range(1000):
    ret, frame1 = cap.read()
    if not ret:
        break

    nxt = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    flow = dense_lk_fast(prev, nxt, win=15)
    bgr = flow_to_hsv_bgr(flow)
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

    im.set_data(rgb)
    fig.canvas.draw_idle()
    plt.pause(0.001)

    prev = nxt

cap.release()
plt.ioff()
plt.show()
