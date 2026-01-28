#!/usr/bin/env python3

import depthai as dai
import cv2
import numpy as np
import json
import time
from pathlib import Path

# ================= CONFIG =================
ROOT = Path("dexmv_data")
FPS = 30
RGB_WIDTH = 1920
RGB_HEIGHT = 1080
WARMUP_FRAMES = 5
# ========================================


def create_session():
    i = 1
    while True:
        p = ROOT / f"session_{i:03d}"
        if not p.exists():
            break
        i += 1
    (p / "rgb").mkdir(parents=True)
    (p / "depth").mkdir()
    return p


session = create_session()
rgb_dir = session / "rgb"
depth_dir = session / "depth"
timestamps_file = session / "timestamps.txt"
intrinsics_file = session / "intrinsics.json"

print(f"📁 Recording to {session}")

# ================= PIPELINE =================
pipeline = dai.Pipeline()

# -------- RGB CAMERA --------
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(FPS)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

# -------- MONO CAMERAS --------
mono_l = pipeline.create(dai.node.MonoCamera)
mono_r = pipeline.create(dai.node.MonoCamera)

mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)

mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

mono_l.setFps(FPS)
mono_r.setFps(FPS)

# -------- STEREO DEPTH --------
stereo = pipeline.create(dai.node.StereoDepth)

# 🔑 CRITICAL FIXES
stereo.setDefaultProfilePreset(
    dai.node.StereoDepth.PresetMode.HIGH_ACCURACY
)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

# Reject low-confidence hallucinations
stereo.initialConfig.setConfidenceThreshold(200)

# Safe filtering only
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

mono_l.out.link(stereo.left)
mono_r.out.link(stereo.right)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# ================= RUN =================
with dai.Device(pipeline) as device:

    # -------- INTRINSICS --------
    calib = device.readCalibration()
    K = calib.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_A,
        RGB_WIDTH,
        RGB_HEIGHT
    )

    intrinsics = {
        "fx": float(K[0][0]),
        "fy": float(K[1][1]),
        "cx": float(K[0][2]),
        "cy": float(K[1][2]),
        "width": RGB_WIDTH,
        "height": RGB_HEIGHT
    }

    with open(intrinsics_file, "w") as f:
        json.dump(intrinsics, f, indent=2)

    print("✅ Saved intrinsics")

    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

    rgb = None
    depth_m = None
    timestamps = []

    frame_id = 0
    t0 = time.time()

    print("\n🔴 RECORDING (press 'q' to stop)\n")

    while True:
        in_rgb = q_rgb.tryGet()
        if in_rgb is not None:
            rgb = in_rgb.getCvFrame()
            ts = in_rgb.getTimestamp().total_seconds()

        in_depth = q_depth.tryGet()
        if in_depth is not None:
            depth_m = in_depth.getFrame().astype(np.float32) / 1000.0

        if rgb is None or depth_m is None:
            continue

        # Preview
        cv2.imshow("RGB", cv2.resize(rgb, (960, 540)))
        depth_vis = np.clip(depth_m / 1.5, 0, 1)
        depth_vis = cv2.applyColorMap(
            (depth_vis * 255).astype(np.uint8),
            cv2.COLORMAP_MAGMA
        )
        cv2.imshow("Depth", cv2.resize(depth_vis, (960, 540)))

        if frame_id >= WARMUP_FRAMES:
            cv2.imwrite(str(rgb_dir / f"{frame_id:06d}.png"), rgb)
            np.save(str(depth_dir / f"{frame_id:06d}.npy"), depth_m)
            timestamps.append(f"{frame_id:06d} {ts:.9f}")

        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    with open(timestamps_file, "w") as f:
        f.write("\n".join(timestamps))

    print(f"\n✅ Done. Frames saved: {frame_id - WARMUP_FRAMES}")

cv2.destroyAllWindows()
print("🏁 Finished cleanly.")
