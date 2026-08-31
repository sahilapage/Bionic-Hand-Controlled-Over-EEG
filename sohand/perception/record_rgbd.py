"""Record synchronised RGB + aligned depth from an OAK-D stereo camera.

    python -m sohand.perception.record_rgbd            # press q to stop
    python -m sohand.perception.record_rgbd --out data --fps 30

Sessions auto-number (`session_001`, `session_002`, ...) and each holds:

    rgb/000042.png     1080p BGR
    depth/000042.npy   float32 metres, aligned to the RGB camera
    intrinsics.json    fx, fy, cx, cy for the RGB camera
    timestamps.txt     frame id and device timestamp, seconds

Depth is aligned to CAM_A so a pixel means the same thing in both streams --
without that the depth map lives in the left mono camera's frame and every
back-projected point is offset by the stereo baseline. The first few frames are
dropped while auto-exposure settles.

Needs `depthai` and an OAK-D on USB3.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

RGB_WIDTH = 1920
RGB_HEIGHT = 1080
WARMUP_FRAMES = 5
PREVIEW_SIZE = (960, 540)
DEPTH_VIS_RANGE_M = 1.5


def create_session(root):
    """Next unused `session_NNN` directory under `root`, with its subfolders."""
    i = 1
    while (root / f"session_{i:03d}").exists():
        i += 1
    path = root / f"session_{i:03d}"
    (path / "rgb").mkdir(parents=True)
    (path / "depth").mkdir()
    return path


def build_pipeline(fps):
    """Colour camera plus a high-accuracy stereo depth stream aligned to it."""
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(fps)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)

    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_r = pipeline.create(dai.node.MonoCamera)
    for mono, socket in ((mono_l, dai.CameraBoardSocket.CAM_B),
                         (mono_r, dai.CameraBoardSocket.CAM_C)):
        mono.setBoardSocket(socket)
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono.setFps(fps)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)
    return pipeline


def save_intrinsics(device, path):
    K = device.readCalibration().getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_A, RGB_WIDTH, RGB_HEIGHT)
    intrinsics = {"fx": float(K[0][0]), "fy": float(K[1][1]),
                  "cx": float(K[0][2]), "cy": float(K[1][2]),
                  "width": RGB_WIDTH, "height": RGB_HEIGHT}
    path.write_text(json.dumps(intrinsics, indent=2))
    return intrinsics


def record(session, fps):
    rgb_dir, depth_dir = session / "rgb", session / "depth"

    with dai.Device(build_pipeline(fps)) as device:
        intrinsics = save_intrinsics(device, session / "intrinsics.json")
        print(f"Intrinsics: fx={intrinsics['fx']:.1f} fy={intrinsics['fy']:.1f} "
              f"cx={intrinsics['cx']:.1f} cy={intrinsics['cy']:.1f}")

        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        rgb, depth_m, timestamp = None, None, 0.0
        timestamps, frame_id = [], 0
        print("\nRECORDING -- press q in a preview window to stop\n")

        while True:
            packet = q_rgb.tryGet()
            if packet is not None:
                rgb = packet.getCvFrame()
                timestamp = packet.getTimestamp().total_seconds()

            packet = q_depth.tryGet()
            if packet is not None:
                depth_m = packet.getFrame().astype(np.float32) / 1000.0

            if rgb is None or depth_m is None:
                continue

            cv2.imshow("RGB", cv2.resize(rgb, PREVIEW_SIZE))
            vis = np.clip(depth_m / DEPTH_VIS_RANGE_M, 0, 1)
            vis = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
            cv2.imshow("Depth", cv2.resize(vis, PREVIEW_SIZE))

            if frame_id >= WARMUP_FRAMES:
                cv2.imwrite(str(rgb_dir / f"{frame_id:06d}.png"), rgb)
                np.save(str(depth_dir / f"{frame_id:06d}.npy"), depth_m)
                timestamps.append(f"{frame_id:06d} {timestamp:.9f}")
            frame_id += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    (session / "timestamps.txt").write_text("\n".join(timestamps))
    return max(frame_id - WARMUP_FRAMES, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="recordings", help="root for session folders")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    session = create_session(Path(args.out))
    print(f"Recording to {session}")
    t0 = time.time()
    try:
        saved = record(session, args.fps)
    finally:
        cv2.destroyAllWindows()
    elapsed = time.time() - t0
    print(f"Done. {saved} frames in {elapsed:.1f}s "
          f"({saved / max(elapsed, 1e-9):.1f} fps)")


if __name__ == "__main__":
    main()
