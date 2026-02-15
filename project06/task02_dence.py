import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

video_path = "slow_traffic_small.mp4"    
cap = cv.VideoCapture(video_path)

ret, frame1 = cap.read()
if not ret:
    raise RuntimeError("Could not read the first frame.")

prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1)
hsv[..., 1] = 255  # full saturation

# --- matplotlib setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")
ax.set_title("Farnebäck Dense Optical Flow (HSV visualization)")

# initial image placeholder
im = ax.imshow(np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8))

while True:
    ret, frame2 = cap.read()
    if not ret:
        print("No frames grabbed!")
        break

    nxt = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(
        prvs, nxt, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)  # matplotlib expects RGB

    # update plot
    im.set_data(rgb)
    fig.canvas.draw_idle()
    plt.pause(0.001)

    prvs = nxt

cap.release()
plt.ioff()
plt.show()
