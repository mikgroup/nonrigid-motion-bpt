"""
Implicit neural representation (INR) motion field: an MLP mapping normalized
spatial coordinates -- optionally concatenated with each frame's BPT PCA
components -- directly to a voxel-unit displacement field. The coordinate is
also expanded with a few low-frequency Fourier features (see
_encode_coords): a plain MLP's bias toward smooth, low-frequency spatial
functions is desirable for a mostly-rigid field, but the same bias prevented
it from representing the sharper, spatially-localized deviation expected near
the jaw/neck (confirmed: without this, the trained field was dominated by one
smooth, near-linear spatial gradient, real BPT/temporal variation notwithstanding).

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
                 n_freq_bands: int = 4,
                 verbose: bool = False,
                 device: str = "cpu"):
        super().__init__()
        self.im_shape: tuple = tuple(im_shape)
        self.n_frames: int = n_frames
        self.use_bpt: bool = "bpt" in mode  # mode='inr' -> space only, mode='inr_bpt' -> space+BPT
        self.verbose: bool = verbose
        self.device: str = device
        self.max_disp: float = max(self.im_shape) * max_disp_frac
        self.n_freq_bands: int = n_freq_bands

        grid = torch.stack(torch.meshgrid(
            *[torch.linspace(-1, 1, s, device=device) for s in self.im_shape], indexing='ij'), dim=-1)
        self.coords = grid.reshape(-1, 3)  # (n_voxels, 3), normalized to [-1, 1] per axis
        self.coord_features = self._encode_coords(self.coords)  # (n_voxels, 3 + 3*2*n_freq_bands)

        in_dim = self.coord_features.shape[1]
        self.bpt_frames = None
        if self.use_bpt:
            if bpt_frames is None:
                raise ValueError(f"mode='{mode}' requires bpt_frames.")
            bpt_frames = torch.tensor(bpt_frames, dtype=torch.float32, device=device)
            # PCA components carry wildly different scales (component 0's std can be
            # 30x component 15's) and are all much larger than the [-1,1] coordinate
            # input -- left unnormalized, training found it easier to fit almost
            # entirely through the spatial pathway and drove the BPT-connected weights
            # toward irrelevance (confirmed: trained output was bit-identical across
            # frames despite genuinely different BPT input). Per-component z-score
            # brings every dimension to a comparable, coordinate-like scale.
            bpt_std = bpt_frames.std(dim=0, keepdim=True).clamp_min(1e-6)
            self.bpt_frames = (bpt_frames / bpt_std).detach()
            in_dim += self.bpt_frames.shape[1]

        dims = [in_dim] + [hidden_dim] * n_layers + [3]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.Softplus()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        nn.init.zeros_(layers[-1].weight)  # start at ~zero displacement, like max_t_init/max_disp_frac elsewhere
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)

        self.to(device)

    def initialize(self):
        # no-op: all setup happens in __init__. Kept for interface parity with MotionFieldModel.
        pass

    def _encode_coords(self, coords):
        """Raw coordinate plus a few low (not NeRF-many/high) frequency
        sin/cos bands per axis -- enough added expressivity to represent a
        spatially localized deviation, without encouraging the network to
        fit per-voxel noise the way many high-frequency bands would."""
        if self.n_freq_bands == 0:
            return coords
        freqs = (2.0 ** torch.arange(self.n_freq_bands, device=coords.device, dtype=coords.dtype)) * torch.pi
        scaled = coords.unsqueeze(-1) * freqs  # (N, 3, n_freq_bands)
        enc = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).reshape(coords.shape[0], -1)
        return torch.cat([coords, enc], dim=-1)

    def get_trainable_parameters(self):
        return dict(self.named_parameters())

    def forward(self, frame_ids=None):
        if frame_ids is None:
            frame_ids = range(self.n_frames)

        if not self.use_bpt:
            # No per-frame conditioning -- a single field shared by every requested frame.
            disp = torch.tanh(self.net(self.coord_features)) * self.max_disp
            disp = disp.reshape(*self.im_shape, 3)
            return disp.unsqueeze(0).expand(len(frame_ids), *disp.shape)

        frames = []
        for f in frame_ids:
            bpt_vec = self.bpt_frames[f].expand(self.coord_features.shape[0], -1)
            net_in = torch.cat([self.coord_features, bpt_vec], dim=-1)
            disp = torch.tanh(self.net(net_in)) * self.max_disp
            frames.append(disp.reshape(*self.im_shape, 3))
        return torch.stack(frames, dim=0)
