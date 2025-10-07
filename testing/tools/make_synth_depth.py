import os
import numpy as np

# ------------------ parameters ------------------
N = 60                  # number of frames
H, W = 480, 640         # frame size
near, far = 0.5, 1.5    # background gradient depths (meters)
obj_radius = 50         # object radius in pixels
obj_depth = 0.60        # object (closer) depth in meters
feather = 15            # feather width in pixels for soft edge
add_noise = True        # add small Gaussian noise
noise_sigma = 0.003     # meters (3mm) std dev


depth_frames = np.zeros((N, H, W), dtype = np.float32)

gradient = np.linspace(0.3, 1.5, H, dtype = np.float32).reshape(H, 1)
frame = np.tile(gradient, (1, W))

depth_frames = np.repeat(frame[None, :, :], N, axis= 0)

yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")

# ------------------ animate a moving circular object ------------------
# Path: x moves left→right; y wobbles slightly with a sine wave
x_start, x_end = 80, W - 80
y_center = H * 0.5
y_amp = H * 0.1

for t in range(N):
    # normalized time in [0, 1]
    u = t / (N - 1) if N > 1 else 0.0

    # linear x, sinusoidal y
    cx = x_start + u * (x_end - x_start)
    cy = y_center + y_amp * np.sin(2 * np.pi * u * 1.2)  # ~1.2 cycles over N frames

    # distance field to center
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # hard circle (<= radius)
    hard_mask = dist <= obj_radius

    # feather ring (soft falloff from radius to radius+feather)
    # weight 1.0 at center → 0.0 at radius+feather
    soft_zone = (dist > obj_radius) & (dist <= obj_radius + feather)
    soft_w = np.zeros_like(dist, dtype=np.float32)
    soft_w[hard_mask] = 1.0
    if feather > 0:
        soft_w[soft_zone] = 1.0 - (dist[soft_zone] - obj_radius) / feather

    # current background frame
    bg = depth_frames[t]

    # blend toward the object depth using soft weight:
    # new_depth = w * obj_depth + (1-w) * bg
    depth_frames[t] = soft_w * obj_depth + (1.0 - soft_w) * bg

    # optional tiny noise to reduce “too perfect” look
    if add_noise:
        depth_frames[t] += np.random.normal(0.0, noise_sigma, size=(H, W)).astype(np.float32)


# ------------------ clamp depths to a reasonable range ------------------
depth_frames = np.clip(depth_frames, 0.1, 5.0).astype(np.float32)


out_dir = os.path.join("detector", "testing", "data")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "depth_raw.npz")
np.savez_compressed(out_path, depth_frames=depth_frames)
print("Saved:", out_path, depth_frames.shape, depth_frames.dtype)
print("Per-frame min/max example:", float(depth_frames[0].min()), float(depth_frames[0].max()))