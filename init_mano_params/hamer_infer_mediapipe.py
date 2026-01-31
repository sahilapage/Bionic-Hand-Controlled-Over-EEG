"""HAMER with MediaPipe - using correct loading from demo.py"""

import torch
import argparse
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp

# Import exactly as demo.py does
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--img_folder', type=str, required=True)
    parser.add_argument('--out_folder', type=str, default='demo_out')
    parser.add_argument('--side_view', action='store_true', default=False)
    parser.add_argument('--save_mesh', action='store_true', default=False)
    parser.add_argument('--rescale_factor', type=float, default=2.0)
    args = parser.parse_args()
    
    img_folder = Path(args.img_folder)
    out_folder = Path(args.out_folder)
    os.makedirs(out_folder, exist_ok=True)
    
    # MediaPipe detector
    print("Loading MediaPipe...")
    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    
    # Download and load HAMER - exactly like demo.py
    print("Loading HAMER...")
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    # Setup renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    
    # Get images
    if img_folder.is_file():
        img_paths = [img_folder]
    else:
        img_paths = sorted(img_folder.glob('*.jpg')) + sorted(img_folder.glob('*.png'))
    
    print(f"Processing {len(img_paths)} images...")
    
    for img_path in tqdm(img_paths):
        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            continue
        
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_height, img_width = img_rgb.shape[:2]
        
        # Detect hands with MediaPipe
        results = detector.process(img_rgb)
        if not results.multi_hand_landmarks:
            continue
        
        # Extract bounding boxes
        bboxes = []
        is_right = []
        
        for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
            xs = [lm.x * img_width for lm in landmarks.landmark]
            ys = [lm.y * img_height for lm in landmarks.landmark]
            
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            bbox = [x_min, y_min, x_max, y_max]
            bboxes.append(bbox)
            
            # Get handedness
            handedness = results.multi_handedness[hand_idx]
            is_right_hand = 1 if handedness.classification[0].label == 'Right' else 0
            is_right.append(is_right_hand)
        
        if len(bboxes) == 0:
            continue
        
        boxes = np.array(bboxes)
        right = np.array(is_right)
        
        # Run HAMER reconstruction - using ViTDetDataset like demo.py
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        all_verts = []
        all_cam_t = []
        all_right = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            
            # Process predictions
            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get person ID and filename
                img_fn = img_path.stem
                person_id = int(batch['personid'][n])
                
                # Prepare images for rendering
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:, None, None] / 255) / (DEFAULT_STD[:, None, None] / 255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:, None, None] / 255) + (DEFAULT_MEAN[:, None, None] / 255)
                input_patch = input_patch.permute(1, 2, 0).numpy()
                
                # Render
                regression_img = renderer(
                    out['pred_vertices'][n].detach().cpu().numpy(),
                    out['pred_cam_t'][n].detach().cpu().numpy(),
                    batch['img'][n],
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                )
                
                if args.side_view:
                    side_img = renderer(
                        out['pred_vertices'][n].detach().cpu().numpy(),
                        out['pred_cam_t'][n].detach().cpu().numpy(),
                        white_img,
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                        side_view=True
                    )
                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    final_img = np.concatenate([input_patch, regression_img], axis=1)
                
                # Save individual hand
                cv2.imwrite(str(out_folder / f'{img_fn}_{person_id}.png'), 255 * final_img[:, :, ::-1])
                
                # Collect for full frame rendering
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right_hand = batch['right'][n].cpu().numpy()
                verts[:, 0] = (2 * is_right_hand - 1) * verts[:, 0]
                cam_t = pred_cam_t_full[n]
                
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right_hand)
                
                # Save mesh
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right_hand)
                    tmesh.export(str(out_folder / f'{img_fn}_{person_id}.obj'))
                    #save mano parameters
                    mano_params = {
                    'global_orient': out['pred_mano_params']['global_orient'][n].cpu().numpy(),
                    'hand_pose': out['pred_mano_params']['hand_pose'][n].cpu().numpy(),
                    'betas': out['pred_mano_params']['betas'][n].cpu().numpy(),
                    'cam_t': cam_t,
                    'is_right': bool(is_right_hand)
                }
                np.save(out_folder / f'{img_fn}_{person_id}_mano.npy', mano_params)
        
        # Render full frame with all hands
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(
                all_verts, 
                cam_t=all_cam_t, 
                render_res=img_size[n], 
                is_right=all_right, 
                **misc_args
            )
            
            # Overlay on original image
            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
            input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
            
            cv2.imwrite(str(out_folder / f'{img_path.stem}_all.jpg'), 255 * input_img_overlay[:, :, ::-1])
        
        print(f"✓ {img_path.name} ({len(all_verts)} hands)")
    
    detector.close()
    print(f"\nDone! Results in {out_folder}")

if __name__ == '__main__':
    main()