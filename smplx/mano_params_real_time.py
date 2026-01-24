import cv2
import depthai as dai
import numpy as np
import mediapipe as mp
import torch
from smplx import SMPLX
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

smplx_model = SMPLX(
    model_path="models/smplx",
    gender="neutral",
    use_pca=False,
).to(device)

NUM_BETAS = smplx_model.num_betas  # = 16

def project_3d_to_2d(points_3d, fx, fy, cx, cy):
    """
    Project 3D points to 2D image coordinates
    points_3d: (N, 3) tensor [X, Y, Z]
    Returns: (N, 2) tensor [u, v]
    """
    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2] + 1e-6  # avoid division by zero
    
    u = (X / Z) * fx + cx
    v = (Y / Z) * fy + cy
    
    return torch.stack([u, v], dim=-1)

def render_depth_map(joints_3d, depth_shape, fx, fy, cx, cy, radius=5):
    """
    Simple depth rendering by rasterizing joint positions
    joints_3d: (N, 3) tensor
    depth_shape: (H, W)
    Returns: rendered depth map (H, W)
    """
    H, W = depth_shape
    depth_render = torch.zeros((H, W), device=device)
    
    joints_2d = project_3d_to_2d(joints_3d, fx, fy, cx, cy)
    
    for i in range(joints_3d.shape[0]):
        u = int(joints_2d[i, 0].item())
        v = int(joints_2d[i, 1].item())
        z = joints_3d[i, 2].item()
        
        if 0 <= u < W and 0 <= v < H:
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    if dx*dx + dy*dy <= radius*radius:
                        u_new = u + dx
                        v_new = v + dy
                        if 0 <= u_new < W and 0 <= v_new < H:
                            depth_render[v_new, u_new] = max(depth_render[v_new, u_new], z)
    
    return depth_render

right_hand_pose = torch.zeros(1, 45, device=device, requires_grad=True)   # θ
betas           = torch.zeros(1, NUM_BETAS, device=device, requires_grad=True)  # β
global_orient   = torch.zeros(1, 3, device=device, requires_grad=True)    # R 
transl          = torch.zeros(1, 3, device=device, requires_grad=True)    # t 

prev_right_hand_pose = None
prev_betas = None

LAMBDA_DEPTH = 1.0  
LAMBDA_TEMPORAL = 10.0  
LAMBDA_SHAPE_REG = 5.0  
LAMBDA_POSE_REG = 0.1  
OPT_ITERS = 20  
LEARNING_RATE = 0.001

def create_pipeline():
    p = dai.Pipeline()

    # RGB Camera
    cam = p.createColorCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(640, 480)
    cam.setInterleaved(False)
    cam.setFps(30)

    xout = p.createXLinkOut()
    xout.setStreamName("rgb")
    cam.video.link(xout.input)

    # Mono Cameras
    mono_l = p.createMonoCamera()
    mono_r = p.createMonoCamera()
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # Stereo Depth
    stereo = p.createStereoDepth()
    
    stereo.setLeftRightCheck(True)  
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)
    
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.initialConfig.setConfidenceThreshold(200)
    
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    xout_d = p.createXLinkOut()
    xout_d.setStreamName("depth")
    stereo.depth.link(xout_d.input)

    return p

with dai.Device(create_pipeline()) as dev:

    calib = dev.readCalibration()
    K = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 480)
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]

    q_rgb = dev.getOutputQueue("rgb", maxSize=2, blocking=False)
    q_d   = dev.getOutputQueue("depth", maxSize=2, blocking=False)

    print("\n[INFO] DexMV-style SMPL-X Hand Optimization")
    print("[INFO] Show your RIGHT hand")
    print(f"[INFO] Camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print("[INFO] Checking depth stream...")

    frame_id = 0
    
    # Test depth map first
    print("[INFO] Testing depth map for 30 frames...")
    for test_frame in range(30):
        frame = q_rgb.get().getCvFrame()
        depth_raw = q_d.get().getFrame().astype(np.float32) / 1000.0
        h, w = frame.shape[:2]
        depth = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_NEAREST)
        
        valid_depth_count = np.sum((depth > 0.1) & (depth < 2.0))
        depth_nonzero = depth[depth > 0]
        
        if test_frame % 10 == 0:
            if len(depth_nonzero) > 0:
                print(f"  Frame {test_frame}: Valid depth pixels: {valid_depth_count}/{h*w}, "
                      f"Min: {np.min(depth_nonzero):.3f}m, Max: {np.max(depth):.3f}m")
            else:
                print(f"  Frame {test_frame}: WARNING - No depth data! All values are 0")
                print(f"  Depth raw shape: {depth_raw.shape}, Resized: {depth.shape}")
                print(f"  Depth raw range: {depth_raw.min():.3f} to {depth_raw.max():.3f}")
    
    print("[INFO] Depth test complete. Starting optimization...\n")

    while True:
        frame = q_rgb.get().getCvFrame()
        depth_raw = q_d.get().getFrame().astype(np.float32) / 1000.0

        h, w = frame.shape[:2]
        
        depth = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_NEAREST)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        j2d_list = []  # 2D joint positions
        hand_mask = np.zeros((h, w), dtype=np.uint8)

        if res.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                res.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS
            )

            # Extract 2D landmarks and create better hand mask
            landmark_points = []
            for lm in res.multi_hand_landmarks[0].landmark:
                u = lm.x * w
                v = lm.y * h
                j2d_list.append([u, v])
                landmark_points.append((int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))))
            
            if len(landmark_points) > 0:
                landmark_array = np.array(landmark_points, dtype=np.int32)
                hull = cv2.convexHull(landmark_array)
                cv2.fillConvexPoly(hand_mask, hull, 255)
                
                for pt in landmark_points:
                    cv2.circle(hand_mask, pt, 15, 255, -1)

        
        if len(j2d_list) >= 12:
            j2d_gt = torch.tensor(j2d_list, device=device, dtype=torch.float32).unsqueeze(0)  # (1, 21, 2)
            depth_masked = depth * (hand_mask > 0)
            depth_tensor = torch.tensor(depth_masked, device=device, dtype=torch.float32)
            
            # Estimate initial translation from depth
            if frame_id < 5: 
                depth_values = depth_masked[hand_mask > 0]
                if len(depth_values) > 0:
                    median_depth = np.median(depth_values[depth_values > 0.05])
                    if median_depth > 0:
                        hand_coords = np.argwhere(hand_mask > 0)
                        center_v, center_u = np.mean(hand_coords, axis=0)
                        center_x = (center_u - cx) * median_depth / fx
                        center_y = (center_v - cy) * median_depth / fy
                        
                        with torch.no_grad():
                            transl[0, 0] = center_x
                            transl[0, 1] = center_y
                            transl[0, 2] = median_depth
            
            optimizer = torch.optim.Adam(
                [
                    {'params': right_hand_pose, 'lr': LEARNING_RATE},
                    {'params': betas, 'lr': LEARNING_RATE * 0.1}, 
                    {'params': global_orient, 'lr': LEARNING_RATE},
                    {'params': transl, 'lr': LEARNING_RATE}
                ]
            )

            best_loss = float('inf')
            best_params = None

            for iter_idx in range(OPT_ITERS):
                optimizer.zero_grad()

                # Forward pass through SMPL-X
                out = smplx_model(
                    global_orient=global_orient,
                    transl=transl,
                    body_pose=torch.zeros(1, 63, device=device),
                    left_hand_pose=torch.zeros(1, 45, device=device),
                    right_hand_pose=right_hand_pose,
                    betas=betas,
                )

                j3d_pred = out.joints[:, 40:61]  # (1, 21, 3)
                j3d_pred_squeezed = j3d_pred.squeeze(0)  # (21, 3)

                j2d_pred = project_3d_to_2d(j3d_pred_squeezed, fx, fy, cx, cy)
                j2d_pred = j2d_pred.unsqueeze(0)  # (1, 21, 2)
                
                loss_2d = torch.mean((j2d_pred - j2d_gt) ** 2) / (w * h) 

                loss_depth = 0.0
                num_valid_depth = 0
                j2d_pred_squeezed = j2d_pred.squeeze(0)  # (21, 2)
                
                sample_radius = 5
                for i in range(j2d_pred_squeezed.shape[0]):
                    u_center = int(torch.clamp(j2d_pred_squeezed[i, 0], 0, w-1).item())
                    v_center = int(torch.clamp(j2d_pred_squeezed[i, 1], 0, h-1).item())
                    
                    depth_samples = []
                    for dv in range(-sample_radius, sample_radius + 1):
                        for du in range(-sample_radius, sample_radius + 1):
                            u = u_center + du
                            v = v_center + dv
                            
                            if 0 <= u < w and 0 <= v < h and hand_mask[v, u] > 0:
                                z_obs = depth_tensor[v, u].item()
                                if 0.1 < z_obs < 2.0:  
                                    depth_samples.append(z_obs)
                    
                    if len(depth_samples) > 0:
                        z_observed = np.median(depth_samples)
                        z_predicted = j3d_pred_squeezed[i, 2]
                        loss_depth += (z_predicted - z_observed) ** 2
                        num_valid_depth += 1
                
                if num_valid_depth > 0:
                    loss_depth = loss_depth / num_valid_depth
                else:
                    loss_depth = torch.tensor(0.0, device=device)

                loss_temporal = torch.tensor(0.0, device=device)
                if prev_right_hand_pose is not None:
                    loss_temporal += torch.mean((right_hand_pose - prev_right_hand_pose) ** 2)
                if prev_betas is not None:
                    loss_temporal += torch.mean((betas - prev_betas) ** 2)

                loss_shape_reg = torch.mean(betas ** 2)
                
                loss_pose_reg = torch.mean(right_hand_pose ** 2)

                loss = (loss_2d + 
                       LAMBDA_DEPTH * loss_depth + 
                       LAMBDA_TEMPORAL * loss_temporal +
                       LAMBDA_SHAPE_REG * loss_shape_reg +
                       LAMBDA_POSE_REG * loss_pose_reg)

                loss.backward()
                optimizer.step()
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_params = {
                        'pose': right_hand_pose.detach().clone(),
                        'betas': betas.detach().clone(),
                        'orient': global_orient.detach().clone(),
                        'transl': transl.detach().clone()
                    }

                if iter_idx == OPT_ITERS - 1 and frame_id % PRINT_EVERY == 0:
                    print(f"\n[Frame {frame_id}] Final Loss: {loss.item():.6f}")
                    print(f"  L_2D: {loss_2d.item():.6f}, L_depth: {loss_depth.item():.6f}")
                    print(f"  L_temporal: {loss_temporal.item():.6f}, L_shape_reg: {loss_shape_reg.item():.6f}")
                    print(f"  Valid depth points: {num_valid_depth}/21")
            
            if best_params is not None:
                right_hand_pose.data.copy_(best_params['pose'])
                betas.data.copy_(best_params['betas'])
                global_orient.data.copy_(best_params['orient'])
                transl.data.copy_(best_params['transl'])

            prev_right_hand_pose = right_hand_pose.detach().clone()
            prev_betas = betas.detach().clone()

        if frame_id % PRINT_EVERY == 0:
            print("θ (hand pose):", right_hand_pose.detach().cpu().numpy()[0, :6], "...")
            print("β (shape):", betas.detach().cpu().numpy()[0, :4], "...")
            print("r (global orient):", global_orient.detach().cpu().numpy())
            print("t (translation):", transl.detach().cpu().numpy())

        cv2.putText(
            frame,
            f"2D joints: {len(j2d_list)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        if np.any(hand_mask > 0):
            hand_depth_values = depth[hand_mask > 0]
            valid_depth = hand_depth_values[(hand_depth_values > 0.1) & (hand_depth_values < 2.0)]
            if len(valid_depth) > 0:
                cv2.putText(
                    frame,
                    f"Hand depth: {np.median(valid_depth):.3f}m ({len(valid_depth)} pts)",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "No valid depth in hand region!",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        mask_display = cv2.applyColorMap((hand_mask).astype(np.uint8), cv2.COLORMAP_JET)
        combined = cv2.addWeighted(frame, 0.7, mask_display, 0.3, 0)

        depth_vis = np.clip(depth * 255, 0, 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        
        depth_masked_vis = depth_colored.copy()
        if np.any(hand_mask > 0):
            mask_indices = hand_mask > 0
            depth_masked_vis[mask_indices] = cv2.addWeighted(
                depth_colored[mask_indices], 0.5,
                mask_display[mask_indices], 0.5, 0
            )

        cv2.imshow("SMPL-X Hand Optimization (DexMV)", combined)
        cv2.imshow("Depth Map", depth_masked_vis)
        
        if cv2.waitKey(1) == 27:
            break

        frame_id += 1

cv2.destroyAllWindows()
print("\n[INFO] Optimization complete!")