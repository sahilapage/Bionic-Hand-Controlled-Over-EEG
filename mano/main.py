import torch
import cv2

from camera import Camera
from hand_detector import HandDetector
from mano_optimizer import MANOOptimizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    cam = Camera()
    detector = HandDetector()

    optimizer = MANOOptimizer(
        model_path="models/mano",   # directory containing MANO_RIGHT.pkl
        device=device
    )

    print("System initialized. Show your hand to the camera.")
    print("Press 'q' to quit.")

    while True:
        rgb, depth, K = cam.read()
        if rgb is None:
            break

        # Show camera feed
        cv2.imshow("RGB", rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        joints_2d, mask = detector.detect(rgb)
        if joints_2d is None:
            continue

        params = optimizer.step(
            j2d=joints_2d,
            depth=depth,
            mask=mask,
            K=K
        )

        theta = params["theta"]

        print("θ shape:", theta.shape)
        print("θ (first 6):", theta[0][:6])
        print("β:", params["beta"])
        print("r (orient):", params["global_orient"])
        print("r (transl):", params["transl"])
        print("-" * 40)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()