"""
Classes for modeling motion fields with implicit neural representations (INRs).
"""
import numpy as np
import torch
import torch.nn as nn
import interpol


def _scaling_and_squaring(v, n_steps):
    """
    Integrate a stationary voxel-unit velocity field into a displacement field by repeated
    self-composition. Composing a flow with itself can't introduce folding, so the result
    is guaranteed invertible regardless of what produced v.

    Args:
    v (torch.Tensor): Velocity field, in voxel units. (Shape: (nx, ny, nz, 3))
    n_steps (int): Number of composition steps.

    Returns:
    d (torch.Tensor): Displacement field, in the same raw voxel-index convention as
    interpol.api.affine_grid (matches MotionFieldWarp's grid_push/grid_pull). (Shape: (nx, ny, nz, 3))
    """
    identity = interpol.api.affine_grid(torch.eye(4, device=v.device), tuple(v.shape[:3]))
    d = v / (2 ** n_steps)
    for _ in range(n_steps):
        sample_coords = (identity + d).unsqueeze(0)
        d_chw = d.permute(3, 0, 1, 2).unsqueeze(0)
        composed = interpol.grid_pull(d_chw, sample_coords, interpolation=1, bound='zero', extrapolate=False)
        d = d + composed.squeeze(0).permute(1, 2, 3, 0)
    return d


class _INRMotionFieldBase(nn.Module):
    """
    Shared setup for coordinate-network motion field models.

    Handles the Nyquist-safe Fourier-encoded spatial grid, per-frame conditioning, and a
    small MLP builder, shared by all INR motion field parameterizations. Every subclass
    conditions on one of two per-frame signals, selected by whether 'bpt' appears in mode:
    - '*_bpt' modes: each frame's BPT PCA components.
    - all other modes: a normalized frame-index (time) scalar.
    """

    def __init__(self, im_shape, n_frames, mode, bpt_frames=None, max_disp_frac=0.05,
                 hidden_dim=64, n_layers=3, n_freq_bands=4, n_integration_steps=6,
                 verbose=False, device="cpu"):
        super().__init__()
        self.im_shape: tuple = tuple(im_shape) # Spatial dimensions of the image. (Shape: (nx, ny, nz))
        self.n_frames: int = n_frames # Total number of temporal frames.
        self.use_bpt: bool = "bpt" in mode # True: condition on BPT PCA components. False: condition on a time scalar.
        self.verbose: bool = verbose
        self.device: str = device
        self.max_disp: float = max(self.im_shape) * max_disp_frac # Maximum displacement/velocity magnitude, in voxels.
        self.n_integration_steps: int = n_integration_steps # Scaling-and-squaring steps, velocity-based subclasses only.
        self.hidden_dim: int = hidden_dim # MLP hidden layer width.
        self.n_layers: int = n_layers # Number of MLP hidden layers.

        # Band k contributes 2**k full cycles across the [-1,1] coordinate range -- beyond
        # the grid's own Nyquist limit (min(im_shape)/2 cycles), that band aliases into a
        # spurious high-frequency pattern instead of representing real structure. Cap
        # silently to the grid's safe ceiling.
        max_safe_bands = int(np.floor(np.log2(min(self.im_shape) / 2))) + 1
        if n_freq_bands > max_safe_bands:
            print(f"[{type(self).__name__}] n_freq_bands={n_freq_bands} exceeds the "
                  f"Nyquist-safe limit ({max_safe_bands}) for im_shape={self.im_shape} -- "
                  f"capping to {max_safe_bands} to avoid aliasing artifacts.")
            n_freq_bands = max_safe_bands
        self.n_freq_bands: int = n_freq_bands

        grid = torch.stack(torch.meshgrid(
            *[torch.linspace(-1, 1, s, device=device) for s in self.im_shape], indexing='ij'), dim=-1)
        self.register_buffer("coords", grid.reshape(-1, 3)) # Normalized voxel coordinates. (Shape: (n_voxels, 3))
        self.register_buffer("frame_scalars", torch.linspace(-1, 1, n_frames, device=device)
                              if n_frames > 1 else torch.zeros(n_frames, device=device)) # (Shape: (n_frames,))

        if self.use_bpt:
            if bpt_frames is None:
                raise ValueError(f"mode='{mode}' requires bpt_frames.")
            bpt_t = torch.tensor(bpt_frames, dtype=torch.float32, device=device)
            # PCA components carry very different scales from each other and from the
            # [-1,1] coordinate input; per-component z-score puts every input on a
            # comparable scale so training doesn't ignore the smaller ones.
            bpt_std = bpt_t.std(dim=0, keepdim=True).clamp_min(1e-6)
            self.register_buffer("bpt_frames", (bpt_t / bpt_std).detach()) # Z-scored BPT PCA components. (Shape: (n_frames, n_bpt_components))
            self.cond_dim: int = self.bpt_frames.shape[1] # Width of the per-frame conditioning vector.
        else:
            self.bpt_frames = None
            self.cond_dim: int = 1

    def initialize(self):
        """
        No-op: all setup happens in __init__. Kept for interface parity with MotionFieldModel.
        """
        pass

    def get_trainable_parameters(self):
        """
        Get all trainable parameters.

        Returns:
        params (dict): Dictionary mapping parameter names to tensors.
        """
        return dict(self.named_parameters())

    def _encode_coords(self, coords):
        """
        Encode coordinates as raw values plus a few low sin/cos frequency bands per axis,
        giving the MLP access to sharper spatial detail than raw coordinates alone allow.

        Args:
        coords (torch.Tensor): Normalized coordinates. (Shape: (n_points, 3))

        Returns:
        features (torch.Tensor): Encoded coordinates. (Shape: (n_points, 3 + 3*2*n_freq_bands))
        """
        if self.n_freq_bands == 0:
            return coords
        freqs = (2.0 ** torch.arange(self.n_freq_bands, device=coords.device, dtype=coords.dtype)) * torch.pi
        scaled = coords.unsqueeze(-1) * freqs
        enc = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).reshape(coords.shape[0], -1)
        return torch.cat([coords, enc], dim=-1)

    def _cond_for_frame(self, f, n_points):
        """
        Get the per-frame conditioning vector (BPT components or a time scalar), broadcast
        to every spatial point.

        Args:
        f (int): Frame index.
        n_points (int): Number of spatial points to broadcast the conditioning vector across.

        Returns:
        cond (torch.Tensor): Conditioning vector, repeated per point. (Shape: (n_points, cond_dim))
        """
        cond = self.bpt_frames[f] if self.use_bpt else self.frame_scalars[f].unsqueeze(0)
        return cond.expand(n_points, -1)

    def _build_mlp(self, in_dim, out_dim, zero_init_last=True, last_bias=True):
        """
        Build a Softplus MLP with a zero-initialized last layer, so training starts at
        ~zero output.

        Args:
        in_dim (int): Input feature width.
        out_dim (int): Output width.
        zero_init_last (bool): Whether to zero-initialize the last layer's weight (and bias, if present).
        last_bias (bool): Whether the last layer has a bias term. Set False for potentials that
        only ever get differentiated (curl/gradient), where a constant offset has no effect on
        the output.

        Returns:
        net (torch.nn.Sequential): The MLP.
        """
        dims = [in_dim] + [self.hidden_dim] * self.n_layers + [out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.Softplus()]
        layers.append(nn.Linear(dims[-2], dims[-1], bias=last_bias))
        if zero_init_last:
            nn.init.zeros_(layers[-1].weight)
            if last_bias:
                nn.init.zeros_(layers[-1].bias)
        return nn.Sequential(*layers)


class DeformationFieldINR(_INRMotionFieldBase):
    """
    Motion field parameterization that directly regresses a voxel-unit displacement field
    from encoded spatial coordinates and a per-frame conditioning signal.

    The simplest of the three INR motion models: nothing prevents the result from folding
    or shearing. Supports two modes:
    - 'inr': conditioned on space and a per-frame time scalar only.
    - 'inr_bpt': conditioned on space and each frame's BPT PCA components.
    """

    def __init__(self, im_shape, n_frames, mode='inr', **kwargs):
        super().__init__(im_shape, n_frames, mode, **kwargs)
        self.register_buffer("coord_features", self._encode_coords(self.coords)) # Fixed for all frames. (Shape: (n_voxels, coord_feature_dim))
        self.net = self._build_mlp(self.coord_features.shape[1] + self.cond_dim, out_dim=3)
        self.to(kwargs.get("device", "cpu"))

    def forward(self, frame_ids=None):
        """
        Generate motion fields for the specified frames.

        Args:
        frame_ids (list | range | None): Frame indices to generate motion fields for. Defaults to all frames.

        Returns:
        motion_fields (torch.Tensor): Motion fields. (Shape: (batch_size, nx, ny, nz, 3))
        """
        if frame_ids is None:
            frame_ids = range(self.n_frames)
        frames = []
        for f in frame_ids:
            cond = self._cond_for_frame(f, self.coord_features.shape[0])
            net_in = torch.cat([self.coord_features, cond], dim=-1)
            disp = torch.tanh(self.net(net_in)) * self.max_disp
            frames.append(disp.reshape(*self.im_shape, 3))
        return torch.stack(frames, dim=0)


class VelocityFieldINR(_INRMotionFieldBase):
    """
    Motion field parameterization that regresses a per-step velocity field and integrates
    it via scaling-and-squaring into a displacement field.

    Composing a flow with itself can't introduce folding, so the result is guaranteed
    invertible regardless of what the network outputs -- unlike DeformationFieldINR.
    Supports two modes:
    - 'inr_vel': conditioned on space and a per-frame time scalar only.
    - 'inr_vel_bpt': conditioned on space and each frame's BPT PCA components.
    """

    def __init__(self, im_shape, n_frames, mode='inr_vel', **kwargs):
        super().__init__(im_shape, n_frames, mode, **kwargs)
        self.register_buffer("coord_features", self._encode_coords(self.coords)) # Fixed for all frames. (Shape: (n_voxels, coord_feature_dim))
        self.net = self._build_mlp(self.coord_features.shape[1] + self.cond_dim, out_dim=3)
        self.to(kwargs.get("device", "cpu"))

    def forward(self, frame_ids=None):
        """
        Generate motion fields for the specified frames.

        Args:
        frame_ids (list | range | None): Frame indices to generate motion fields for. Defaults to all frames.

        Returns:
        motion_fields (torch.Tensor): Motion fields. (Shape: (batch_size, nx, ny, nz, 3))
        """
        if frame_ids is None:
            frame_ids = range(self.n_frames)
        frames = []
        for f in frame_ids:
            cond = self._cond_for_frame(f, self.coord_features.shape[0])
            net_in = torch.cat([self.coord_features, cond], dim=-1)
            v = (torch.tanh(self.net(net_in)) * self.max_disp).reshape(*self.im_shape, 3)
            frames.append(_scaling_and_squaring(v, self.n_integration_steps))
        return torch.stack(frames, dim=0)


class HelmholtzVelocityINR(_INRMotionFieldBase):
    """
    Motion field parameterization that regresses a vector potential A and a scalar
    potential phi, combines curl(A) + grad(phi) into a velocity field, and integrates it
    via scaling-and-squaring into a displacement field.

    curl(A) is automatically divergence-free (volume-preserving, rotation-like); grad(phi)
    is automatically curl-free (expansion/contraction-like). Useful when the expected
    motion is mostly rotational (e.g. head nodding/turning) with an optional compressive
    component (e.g. breathing). Needs derivatives of the network's own output with respect
    to its coordinate input, so unlike the other two classes it can't precompute a fixed
    coordinate encoding. Supports two modes:
    - 'inr_helmholtz': conditioned on space and a per-frame time scalar only.
    - 'inr_helmholtz_bpt': conditioned on space and each frame's BPT PCA components.
    """

    def __init__(self, im_shape, n_frames, mode='inr_helmholtz', **kwargs):
        super().__init__(im_shape, n_frames, mode, **kwargs)
        coord_features = self._encode_coords(self.coords)
        self.net = self._build_mlp(coord_features.shape[1] + self.cond_dim, out_dim=4, last_bias=False)
        self.to(kwargs.get("device", "cpu"))

    def _velocity_field(self, f):
        """
        Compute the velocity field for one frame as curl(A) + grad(phi), via autograd
        through the network's own coordinate input.

        Args:
        f (int): Frame index.

        Returns:
        v (torch.Tensor): Velocity field, in voxel units. (Shape: (nx, ny, nz, 3))
        """
        coords = self.coords.clone().requires_grad_(True)
        feats = self._encode_coords(coords)
        cond = self._cond_for_frame(f, feats.shape[0])
        out = self.net(torch.cat([feats, cond], dim=-1))
        A, phi = out[:, :3], out[:, 3]

        jac_A = []
        for i in range(3):
            grad_outputs = torch.zeros_like(A)
            grad_outputs[:, i] = 1.0
            (g,) = torch.autograd.grad(A, coords, grad_outputs=grad_outputs,
                                        create_graph=self.training, retain_graph=True)
            jac_A.append(g)
        jac_A = torch.stack(jac_A, dim=1)  # jac_A[:, i, j] = dA_i/dx_j

        (grad_phi,) = torch.autograd.grad(phi, coords, grad_outputs=torch.ones_like(phi),
                                           create_graph=self.training, retain_graph=self.training)

        curl = torch.stack([
            jac_A[:, 2, 1] - jac_A[:, 1, 2],
            jac_A[:, 0, 2] - jac_A[:, 2, 0],
            jac_A[:, 1, 0] - jac_A[:, 0, 1],
        ], dim=-1)
        v = torch.tanh(curl + grad_phi) * self.max_disp
        return v.reshape(*self.im_shape, 3)

    def forward(self, frame_ids=None):
        """
        Generate motion fields for the specified frames.

        Args:
        frame_ids (list | range | None): Frame indices to generate motion fields for. Defaults to all frames.

        Returns:
        motion_fields (torch.Tensor): Motion fields. (Shape: (batch_size, nx, ny, nz, 3))
        """
        if frame_ids is None:
            frame_ids = range(self.n_frames)
        frames = []
        for f in frame_ids:
            with torch.enable_grad():
                v = self._velocity_field(f)
            frames.append(_scaling_and_squaring(v, self.n_integration_steps))
        return torch.stack(frames, dim=0)


_MODE_TO_CLASS = {
    "inr": DeformationFieldINR, "inr_bpt": DeformationFieldINR,
    "inr_vel": VelocityFieldINR, "inr_vel_bpt": VelocityFieldINR,
    "inr_helmholtz": HelmholtzVelocityINR, "inr_helmholtz_bpt": HelmholtzVelocityINR,
}
INR_MODES = set(_MODE_TO_CLASS)


def build_inr_motion_model(mode, **kwargs):
    """
    Construct the INR motion model class matching a given mode.

    Args:
    mode (str): One of INR_MODES.
    **kwargs: Forwarded to the model class's constructor.

    Returns:
    model (_INRMotionFieldBase): The constructed motion model.
    """
    try:
        cls = _MODE_TO_CLASS[mode]
    except KeyError:
        raise ValueError(f"Unknown INR mode '{mode}', expected one of {sorted(INR_MODES)}")
    return cls(mode=mode, **kwargs)
