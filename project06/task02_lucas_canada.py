import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

video_path = "slow_traffic_small.mp4"   

cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

# ----------------------------
# Params
# ----------------------------
feature_params = dict(
    maxCorners=200,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
)

# ----------------------------
# Read first frame
# ----------------------------
ret, old_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read the first frame.")

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Mask for drawing tracks
mask = np.zeros_like(old_frame)

# Random colors (one per feature)
color = np.random.randint(0, 255, (feature_params["maxCorners"], 3))

# ----------------------------
# Matplotlib setup
# ----------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")

# Show first frame (converted to RGB)
im = ax.imshow(cv.cvtColor(old_frame, cv.COLOR_BGR2RGB))
ax.set_title("Lucas–Kanade Optical Flow (Video)")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # If no points, re-detect
    if p0 is None or len(p0) == 0:
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask = np.zeros_like(frame)
        old_gray = frame_gray.copy()
        continue

    # Optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is None:
        old_gray = frame_gray.copy()
        continue

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw tracks using OpenCV (but display with matplotlib)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        col = color[i % len(color)].tolist()

        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), col, 2)
        frame = cv.circle(frame, (int(a), int(b)), 3, col, -1)

    img = cv.add(frame, mask)

    # Update matplotlib image
    im.set_data(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    fig.canvas.draw_idle()
    plt.pause(0.001)  # controls playback speed

    # Update for next iteration
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    frame_count += 1

cap.release()
plt.ioff()
plt.show()
