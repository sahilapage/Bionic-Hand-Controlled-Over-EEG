"""
Visualize MANO parameters to verify correctness
Shows: 3D mesh, skeleton, and overlay on original image
"""

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_results(json_path):
    """Load MANO results from JSON"""
    with open(json_path, 'r') as f:
        return json.load(f)

def visualize_3d_mesh(vertices, joints_3d, save_path=None):
    """Visualize the 3D hand mesh and skeleton"""
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: 3D mesh points
    ax1 = fig.add_subplot(131, projection='3d')
    vertices = np.array(vertices)
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                c='blue', marker='.', s=1, alpha=0.3)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Hand Mesh (778 vertices)')
    
    # Plot 2: 3D skeleton
    ax2 = fig.add_subplot(132, projection='3d')
    joints = np.array(joints_3d)
    
    # Plot joints
    ax2.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                c='red', marker='o', s=50)
    
    # Draw skeleton connections
    # Define finger chains: wrist -> finger tips
    connections = [
        [0, 1, 2, 3, 4],      # Thumb
        [0, 5, 6, 7, 8],      # Index
        [0, 9, 10, 11, 12],   # Middle
        [0, 13, 14, 15, 16],  # Ring
        [0, 17, 18, 19, 20]   # Pinky
    ]
    
    for chain in connections:
        for i in range(len(chain) - 1):
            j1, j2 = chain[i], chain[i+1]
            ax2.plot([joints[j1, 0], joints[j2, 0]],
                    [joints[j1, 1], joints[j2, 1]],
                    [joints[j1, 2], joints[j2, 2]], 'b-', linewidth=2)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Hand Skeleton (21 joints)')
    
    # Plot 3: Top-down view
    ax3 = fig.add_subplot(133)
    ax3.scatter(vertices[:, 0], vertices[:, 1], c='blue', s=1, alpha=0.3)
    ax3.scatter(joints[:, 0], joints[:, 1], c='red', s=50)
    
    for chain in connections:
        for i in range(len(chain) - 1):
            j1, j2 = chain[i], chain[i+1]
            ax3.plot([joints[j1, 0], joints[j2, 0]],
                    [joints[j1, 1], joints[j2, 1]], 'b-', linewidth=2)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Top-down View (XY plane)')
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def project_to_2d(joints_3d, image_shape):
    """Simple orthographic projection for visualization"""
    # This is approximate - real projection needs camera intrinsics
    joints = np.array(joints_3d)
    
    # Scale to image space (rough approximation)
    # Normalize to [0, 1] range
    joints_2d = joints[:, :2].copy()
    
    # Center and scale
    min_vals = joints_2d.min(axis=0)
    max_vals = joints_2d.max(axis=0)
    joints_2d = (joints_2d - min_vals) / (max_vals - min_vals)
    
    # Scale to image size with padding
    padding = 0.1
    joints_2d = joints_2d * (1 - 2*padding) + padding
    joints_2d[:, 0] *= image_shape[1]  # width
    joints_2d[:, 1] *= image_shape[0]  # height
    
    return joints_2d.astype(int)

def overlay_on_image(image_path, joints_3d, bbox, save_path=None):
    """Overlay skeleton on original image using bbox for positioning"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Extract bbox
    x1, y1, x2, y2 = map(int, bbox)
    
    # Project 3D joints to 2D within bbox
    joints = np.array(joints_3d)
    
    # Use XY coordinates, normalize to bbox
    joints_xy = joints[:, :2]
    min_vals = joints_xy.min(axis=0)
    max_vals = joints_xy.max(axis=0)
    
    # Normalize to [0, 1]
    joints_norm = (joints_xy - min_vals) / (max_vals - min_vals + 1e-8)
    
    # Map to bbox coordinates
    joints_2d = np.zeros_like(joints_norm)
    joints_2d[:, 0] = x1 + joints_norm[:, 0] * (x2 - x1)
    joints_2d[:, 1] = y1 + joints_norm[:, 1] * (y2 - y1)
    joints_2d = joints_2d.astype(int)
    
    # Draw bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw skeleton
    connections = [
        [0, 1, 2, 3, 4],      # Thumb
        [0, 5, 6, 7, 8],      # Index
        [0, 9, 10, 11, 12],   # Middle
        [0, 13, 14, 15, 16],  # Ring
        [0, 17, 18, 19, 20]   # Pinky
    ]
    
    # Draw connections
    for chain in connections:
        for i in range(len(chain) - 1):
            j1, j2 = chain[i], chain[i+1]
            pt1 = tuple(joints_2d[j1])
            pt2 = tuple(joints_2d[j2])
            cv2.line(img, pt1, pt2, (255, 0, 0), 2)
    
    # Draw joints
    for i, pt in enumerate(joints_2d):
        color = (0, 0, 255) if i == 0 else (255, 255, 0)  # Red for wrist, yellow for others
        cv2.circle(img, tuple(pt), 4, color, -1)
    
    if save_path:
        cv2.imwrite(save_path, img)
        print(f"Saved overlay to {save_path}")
    else:
        cv2.imshow('Hand Overlay', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return img

def check_hand_metrics(vertices, joints_3d, is_right):
    """Verify hand dimensions are reasonable"""
    vertices = np.array(vertices)
    joints = np.array(joints_3d)
    
    print("\n" + "="*60)
    print("HAND METRICS VERIFICATION")
    print("="*60)
    
    # Hand size (bounding box)
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    hand_size = v_max - v_min
    
    print(f"\nHand Bounding Box Size:")
    print(f"  Width  (X): {hand_size[0]*1000:.1f} mm")
    print(f"  Height (Y): {hand_size[1]*1000:.1f} mm")
    print(f"  Depth  (Z): {hand_size[2]*1000:.1f} mm")
    
    # Typical adult hand: 70-90mm wide, 170-200mm long
    if 60 < hand_size[0]*1000 < 120 and 150 < hand_size[1]*1000 < 250:
        print("  ✓ Hand size looks reasonable")
    else:
        print("  ⚠ Warning: Hand size seems unusual")
    
    # Finger lengths
    print(f"\nFinger Lengths (wrist to tip):")
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    finger_indices = [[0, 4], [0, 8], [0, 12], [0, 16], [0, 20]]
    
    for name, (start, end) in zip(finger_names, finger_indices):
        length = np.linalg.norm(joints[end] - joints[start])
        print(f"  {name:7s}: {length*1000:.1f} mm")
    
    # Check handedness
    print(f"\nHandedness:")
    print(f"  Detected as: {'RIGHT' if is_right else 'LEFT'} hand")
    
    # Palm center
    wrist = joints[0]
    print(f"\nWrist Position (camera coordinates):")
    print(f"  X: {wrist[0]:.3f} m")
    print(f"  Y: {wrist[1]:.3f} m")
    print(f"  Z: {wrist[2]:.3f} m (distance from camera)")
    
    print("\n" + "="*60)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize MANO parameters')
    parser.add_argument('--json', required=True, help='MANO results JSON file')
    parser.add_argument('--img_folder', required=True, help='Original images folder')
    parser.add_argument('--output', default='visualization', help='Output folder')
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.json)
    
    import os
    os.makedirs(args.output, exist_ok=True)
    
    # Process each image
    for img_name, hands in results.items():
        print(f"\n{'='*60}")
        print(f"Processing: {img_name}")
        print(f"{'='*60}")
        print(f"Found {len(hands)} hand(s)")
        
        for i, hand_data in enumerate(hands):
            print(f"\n--- Hand {i+1} ---")
            
            # Extract data
            vertices = hand_data['vertices']
            joints_3d = hand_data['joints_3d']
            is_right = hand_data['is_right']
            bbox = hand_data['bbox']
            
            # Check metrics
            check_hand_metrics(vertices, joints_3d, is_right)
            
            # Create visualizations
            img_base = img_name.rsplit('.', 1)[0]
            
            # 3D visualization
            vis_3d_path = os.path.join(args.output, f"{img_base}_hand{i}_3d.png")
            visualize_3d_mesh(vertices, joints_3d, vis_3d_path)
            
            # Overlay on image
            img_path = os.path.join(args.img_folder, img_name)
            overlay_path = os.path.join(args.output, f"{img_base}_hand{i}_overlay.png")
            overlay_on_image(img_path, joints_3d, bbox, overlay_path)
    
    print(f"\n✓ Visualizations saved to {args.output}/")

if __name__ == '__main__':
    main()