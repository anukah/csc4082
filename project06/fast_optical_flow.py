import numpy as np
import matplotlib.pyplot as plt
import cv2

def fast_optical_flow(I_prev, I_curr, threshold=1):
    """
    Fast Optical Flow Estimation (Lecture-based implementation)
    I_prev, I_curr: grayscale images (float32)
    """

    h, w = I_prev.shape
    flow = np.zeros((h, w, 2), dtype=np.float32)

    # Step 1: Temporal difference
    D = I_curr - I_prev

    # 4 motion directions (dx, dy)
    directions = [
        (1, 0),   # right
        (-1, 0),  # left
        (0, 1),   # down
        (0, -1),  # up
        (1, 1),   # diagonal
        (1, -1),  # diagonal
        (-1, 1),
        (-1, -1)
    ]

    # Step 2–4: Direction estimation and averaging
    for y in range(1, h - 1):
        for x in range(1, w - 1):

            if abs(D[y, x]) < threshold:
                continue

            vectors = []

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if D[y, x] < 0:
                    # motion towards higher luminance
                    if I_curr[ny, nx] > I_curr[y, x]:
                        vectors.append(np.array([dx, dy]))
                else:
                    # motion towards lower luminance
                    if I_curr[ny, nx] < I_curr[y, x]:
                        vectors.append(np.array([dx, dy]))

            if vectors:
                flow[y, x] = np.mean(vectors, axis=0)

    # Step 5: Smooth using 8-neighborhood
    flow_smooth = flow.copy()
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            neighborhood = flow[y-1:y+2, x-1:x+2]
            flow_smooth[y, x] = np.mean(neighborhood.reshape(-1, 2), axis=0)

    return flow_smooth


frame1 = cv2.imread("frame3.png", cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread("frame4.png", cv2.IMREAD_GRAYSCALE)

frame1 = frame1.astype(np.float32)
frame2 = frame2.astype(np.float32)

flow = fast_optical_flow(frame1, frame2)

u = flow[:, :, 0]
v = flow[:, :, 1]

# Calculate flow magnitude and angle
magnitude = np.sqrt(u**2 + v**2)
angle = np.arctan2(v, u)

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Original first frame
axes[0, 0].imshow(frame1, cmap='gray')
axes[0, 0].set_title('Frame 1 (Previous)')
axes[0, 0].axis('off')

# 2. Original second frame
axes[0, 1].imshow(frame2, cmap='gray')
axes[0, 1].set_title('Frame 2 (Current)')
axes[0, 1].axis('off')

# 3. Flow magnitude
im1 = axes[0, 2].imshow(magnitude, cmap='jet')
axes[0, 2].set_title('Flow Magnitude')
axes[0, 2].axis('off')
plt.colorbar(im1, ax=axes[0, 2])

# 4. Horizontal flow (u component)
im2 = axes[1, 0].imshow(u, cmap='RdBu_r')
axes[1, 0].set_title('Horizontal Flow (u)')
axes[1, 0].axis('off')
plt.colorbar(im2, ax=axes[1, 0])

# 5. Vertical flow (v component)
im3 = axes[1, 1].imshow(v, cmap='RdBu_r')
axes[1, 1].set_title('Vertical Flow (v)')
axes[1, 1].axis('off')
plt.colorbar(im3, ax=axes[1, 1])

# 6. Quiver plot - subsample for clarity
step = 10  # Show every 10th vector
y_coords, x_coords = np.mgrid[0:frame1.shape[0]:step, 0:frame1.shape[1]:step]
u_sub = u[::step, ::step]
v_sub = v[::step, ::step]

axes[1, 2].imshow(frame1, cmap='gray', alpha=0.7)
axes[1, 2].quiver(x_coords, y_coords, u_sub, v_sub, 
                  magnitude[::step, ::step], 
                  scale=50, scale_units='xy', 
                  cmap='jet', width=0.003)
axes[1, 2].set_title('Flow Vectors')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('optical_flow_results.png', dpi=150, bbox_inches='tight')
plt.show()

# Additional visualization: HSV color-coded flow
fig2, ax = plt.subplots(1, 1, figsize=(10, 8))

# Convert flow to HSV color representation
hsv = np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)
hsv[..., 0] = (angle + np.pi) / (2 * np.pi) * 179  # Hue: angle
hsv[..., 1] = 255  # Saturation: full
hsv[..., 2] = np.clip(magnitude / magnitude.max() * 255, 0, 255).astype(np.uint8)  # Value: magnitude

# Convert HSV to RGB for display
rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

ax.imshow(rgb_flow)
ax.set_title('Optical Flow (HSV Color-coded)\nHue=Direction, Brightness=Magnitude')
ax.axis('off')

plt.tight_layout()
plt.savefig('optical_flow_hsv.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Flow statistics:")
print(f"  Mean magnitude: {magnitude.mean():.3f}")
print(f"  Max magnitude: {magnitude.max():.3f}")
print(f"  Non-zero flow pixels: {np.count_nonzero(magnitude)}/{magnitude.size}")

