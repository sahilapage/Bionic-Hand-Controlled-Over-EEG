import torch
import mano

# MediaPipe → MANO finger joints 
MP_TO_MANO = [
    2,              # thumb IP
    5, 6, 7,        # index
    9, 10, 11,      # middle
    13, 14, 15,     # ring
    17, 18, 19      # pinky
]


class MANOOptimizer:
    def __init__(self, model_path, device):
        self.device = device

        self.mano = mano.load(
            model_path=model_path,
            is_rhand=True,
            num_pca_comps=45,
            batch_size=1,
            flat_hand_mean=False
        ).to(device)

        self.hand_pose = torch.zeros(1, 45, device=device, requires_grad=True)
        self.betas = torch.zeros(1, 10, device=device, requires_grad=True)
        self.global_orient = torch.zeros(1, 3, device=device, requires_grad=True)
        self.transl = torch.zeros(1, 3, device=device, requires_grad=True)

        self.optimizer = torch.optim.Adam(
            [self.hand_pose, self.betas, self.global_orient, self.transl],
            lr=1e-2
        )

        self.prev_joints = None
        self.frame_count = 0

    def project(self, j3d, K):
        x = j3d[:, 0]
        y = j3d[:, 1]
        z = j3d[:, 2] + 1e-9

        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]

        return torch.stack([u, v], dim=-1)

    def step(self, j2d, depth, mask, K, iters=8):
        j2d = torch.as_tensor(j2d, device=self.device, dtype=torch.float32)
        depth = torch.as_tensor(depth, device=self.device, dtype=torch.float32)
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.float32)
        K = torch.as_tensor(K, device=self.device, dtype=torch.float32)

        for _ in range(iters):
            output = self.mano(
                hand_pose=self.hand_pose,
                betas=self.betas,
                global_orient=self.global_orient,
                transl=self.transl
            )

            joints_3d = output.joints[0]          # (16,3)
            proj_2d = self.project(joints_3d, K)

            # Drop wrist
            proj_2d_fingers = proj_2d[1:]
            j2d_fingers = j2d[MP_TO_MANO]

            # Dynamic alignment
            n = min(proj_2d_fingers.shape[0], j2d_fingers.shape[0])
            proj_2d_fingers = proj_2d_fingers[:n]
            j2d_fingers = j2d_fingers[:n]

            # 2D reprojection loss
            loss_2d = ((proj_2d_fingers - j2d_fingers) ** 2).mean()

            # Weak depth loss
            depth_pred = output.vertices[0, :, 2]
            depth_gt = depth[mask > 0].mean()
            loss_depth = ((depth_pred - depth_gt) ** 2).mean()

            loss = loss_2d + 0.001 * loss_depth

            # Temporal smoothness
            if self.prev_joints is not None:
                loss += 0.1 * ((joints_3d - self.prev_joints) ** 2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Lock shape after a few frames
        self.frame_count += 1
        if self.frame_count > 10:
            self.betas.requires_grad_(False)

        self.prev_joints = joints_3d.detach()

        return {
            "theta": self.hand_pose.detach().cpu().numpy(),           # (1,45)
            "beta": self.betas.detach().cpu().numpy(),                # (1,10)
            "global_orient": self.global_orient.detach().cpu().numpy(),# (1,3)
            "transl": self.transl.detach().cpu().numpy()               # (1,3)
        }