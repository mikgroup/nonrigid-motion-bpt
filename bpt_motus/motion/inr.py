"""
Implicit neural representation (INR) motion field: an MLP mapping normalized
spatial coordinates -- optionally concatenated with each frame's BPT PCA
components -- directly to a voxel-unit displacement field.

Implements the same forward(frame_ids) -> (batch, nx, ny, nz, 3) and
get_trainable_parameters() contract as MotionFieldModel, so it drops into
MotionFieldOptimizer / MotionFieldWarp unchanged.
"""
import numpy as np
import torch
import torch.nn as nn


class ImplicitMotionFieldModel(nn.Module):
    def __init__(self,
                 im_shape: tuple,
                 n_frames: int,
                 mode: str = 'inr',
                 bpt_frames: np.ndarray | torch.Tensor | None = None,
                 max_disp_frac: float = 0.05,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 verbose: bool = False,
                 device: str = "cpu"):
        super().__init__()
        self.im_shape: tuple = tuple(im_shape)
        self.n_frames: int = n_frames
        self.use_bpt: bool = "bpt" in mode  # mode='inr' -> space only, mode='inr_bpt' -> space+BPT
        self.verbose: bool = verbose
        self.device: str = device
        self.max_disp: float = max(self.im_shape) * max_disp_frac

        in_dim = 3
        self.bpt_frames = None
        if self.use_bpt:
            if bpt_frames is None:
                raise ValueError(f"mode='{mode}' requires bpt_frames.")
            self.bpt_frames = torch.tensor(bpt_frames, dtype=torch.float32, device=device).detach()
            in_dim += self.bpt_frames.shape[1]

        dims = [in_dim] + [hidden_dim] * n_layers + [3]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.Softplus()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        nn.init.zeros_(layers[-1].weight)  # start at ~zero displacement, like max_t_init/max_disp_frac elsewhere
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)

        grid = torch.stack(torch.meshgrid(
            *[torch.linspace(-1, 1, s, device=device) for s in self.im_shape], indexing='ij'), dim=-1)
        self.coords = grid.reshape(-1, 3)  # (n_voxels, 3), normalized to [-1, 1] per axis

        self.to(device)

    def initialize(self):
        # no-op: all setup happens in __init__. Kept for interface parity with MotionFieldModel.
        pass

    def get_trainable_parameters(self):
        return dict(self.named_parameters())

    def forward(self, frame_ids=None):
        if frame_ids is None:
            frame_ids = range(self.n_frames)

        if not self.use_bpt:
            # No per-frame conditioning -- a single field shared by every requested frame.
            disp = torch.tanh(self.net(self.coords)) * self.max_disp
            disp = disp.reshape(*self.im_shape, 3)
            return disp.unsqueeze(0).expand(len(frame_ids), *disp.shape)

        frames = []
        for f in frame_ids:
            bpt_vec = self.bpt_frames[f].expand(self.coords.shape[0], -1)
            net_in = torch.cat([self.coords, bpt_vec], dim=-1)
            disp = torch.tanh(self.net(net_in)) * self.max_disp
            frames.append(disp.reshape(*self.im_shape, 3))
        return torch.stack(frames, dim=0)
