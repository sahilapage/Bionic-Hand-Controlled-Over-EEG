import torch
import argparse
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

PALM_ID = 0
FINGERTIP_IDS = [4, 8, 12, 16, 20]
MIDDLE_IDS = [3, 7, 11, 15, 19]

def compute_tsvs(joints_3d):
    palm = joints_3d[PALM_ID]
    tsvs = []
    for tip, mid in zip(FINGERTIP_IDS, MIDDLE_IDS):
        tsvs.append(joints_3d[tip] - palm)
        tsvs.append(joints_3d[mid] - palm)
    return np.stack(tsvs)

LIGHT_BLUE = (0.65, 0.74, 0.86)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--img_folder', type=str, required=True)
    parser.add_argument('--out_folder', type=str, default='demo_out')
    parser.add_argument('--save_mesh', action='store_true')
    parser.add_argument('--rescale_factor', type=float, default=2.0)
    args = parser.parse_args()

    img_folder = Path(args.img_folder)
    out_folder = Path(args.out_folder)
    out_folder.mkdir(exist_ok=True)

    print("Loading MediaPipe...")
    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    print("Loading HAMER...")
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    renderer = Renderer(model_cfg, faces=model.mano.faces)

    if img_folder.is_file():
        img_paths = [img_folder]
    else:
        img_paths = sorted(img_folder.glob("*.jpg")) + sorted(img_folder.glob("*.png"))

    print(f"Processing {len(img_paths)} images...")

    for img_path in tqdm(img_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        results = detector.process(img_rgb)
        if not results.multi_hand_landmarks:
            continue

        bboxes = []
        is_right = []

        for i, lm in enumerate(results.multi_hand_landmarks):
            xs = [p.x * W for p in lm.landmark]
            ys = [p.y * H for p in lm.landmark]
            bboxes.append([min(xs), min(ys), max(xs), max(ys)])
            handedness = results.multi_handedness[i].classification[0].label
            is_right.append(1 if handedness == "Right" else 0)

        boxes = np.array(bboxes)
        right = 1 - np.array(is_right)

        dataset = ViTDetDataset(
            model_cfg,
            img_bgr,
            boxes,
            right,
            rescale_factor=args.rescale_factor
        )

        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

        all_verts = []
        all_cam_t = []
        all_right = []

        for batch in loader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch["right"] - 1)
            pred_cam = out["pred_cam"].clone()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                model_cfg.EXTRA.FOCAL_LENGTH
                / model_cfg.MODEL.IMAGE_SIZE
                * img_size.max()
            )

            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).cpu().numpy()

            B = batch["img"].shape[0]

            for n in range(B):
                img_fn = img_path.stem
                pid = int(batch["personid"][n])

                mano_params = {
                    "global_orient": out["pred_mano_params"]["global_orient"][n].cpu().numpy(),
                    "hand_pose": out["pred_mano_params"]["hand_pose"][n].cpu().numpy(),
                    "betas": out["pred_mano_params"]["betas"][n].cpu().numpy(),
                    "cam_t": pred_cam_t_full[n],
                    "is_right": bool(batch["right"][n].item())
                }
                np.save(out_folder / f"{img_fn}_{pid}_mano.npy", mano_params)

                joints_3d = out["pred_keypoints_3d"][n].cpu().numpy()
                tsvs = compute_tsvs(joints_3d)
                np.save(out_folder / f"{img_fn}_{pid}_tsv.npy", tsvs)

                verts = out["pred_vertices"][n].cpu().numpy()
                is_r = batch["right"][n].cpu().numpy()
                verts[:, 0] = (2 * is_r - 1) * verts[:, 0]
                cam_t = pred_cam_t_full[n]

                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_r)

                if args.save_mesh:
                    mesh = renderer.vertices_to_trimesh(
                        verts, cam_t, LIGHT_BLUE, is_right=is_r
                    )
                    mesh.export(out_folder / f"{img_fn}_{pid}.obj")

        if len(all_verts) > 0:
            cam_view = renderer.render_rgba_multiple(
                all_verts,
                cam_t=all_cam_t,
                render_res=img_size[n],
                is_right=all_right,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )

            input_img = img_rgb.astype(np.float32) / 255.0
            input_img = np.concatenate(
                [input_img, np.ones_like(input_img[:, :, :1])], axis=2
            )

            overlay = (
                input_img[:, :, :3] * (1 - cam_view[:, :, 3:])
                + cam_view[:, :, :3] * cam_view[:, :, 3:]
            )

            cv2.imwrite(
                str(out_folder / f"{img_path.stem}_overlay.png"),
                (overlay[:, :, ::-1] * 255).astype(np.uint8),
            )

        print(f"✓ {img_path.name}")

    detector.close()
    print(f"\nDone! Results saved in {out_folder}")

if __name__ == "__main__":
    main()