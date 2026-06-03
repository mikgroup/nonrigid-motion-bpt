"""
Classes for modeling and optimizing non-rigid motion fields with MR-MOTUS and BPT-MOTUS.
"""
import os
import numpy as np
import torch
import logging
from typing import Literal, Optional, Dict, Tuple, List
from scipy.interpolate import BSpline

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MotionFieldModel:
    """
    Motion field parameterization and generation using B-splines.
    
    Handles spatial and temporal B-spline bases, control point initialization,
    and motion field computation. Supports multiple modes:
    - 'mrmotus': MR-MOTUS with learned spatial and temporal B-splines
    - 'bpt_motus': BPT temporal components with learned spatial B-splines
    - 'mrmotus_rigid': Rigid motion components with learned temporal B-splines
    - 'bpt_rigid': BPT temporal components with rigid spatial components
    """
    
    def __init__(self, 
                 im_shape: tuple,
                 n_frames: int,
                 mode: str = 'bpt_rigid',
                 xyz_downsampling: list = [4, 8, 16],
                 t_downsampling: list = [1, 2, 5],
                 n_mfcomponents: int = 6,
                 max_disp_frac: float = 0.05,
                 max_t_init: float = 0.0,
                 bpt_frames: np.ndarray | torch.Tensor | None = None,
                 verbose: bool = False,
                 degree: int = 3,
                 device: str = "cpu"):
        # Store input arguments as class attributes
        self.verbose: bool = verbose
        self.degree: int = degree # B-spline polynomial degree.
        self.device: str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")  # Compute device ('cpu' or 'cuda').
        self.mode: str = mode 
        
        self.im_shape: tuple = im_shape # Spatial dimensions of the image. (Shape: (nx, ny, nz))
        self.nx: int = im_shape[0]
        self.ny: int = im_shape[1]
        self.nz: int = im_shape[2]
        self.n_frames: int = n_frames # Total number of temporal frames.
        
        self.xyz_downsampling: list = xyz_downsampling # Downsampling factors for multi-scale spatial B-splines.
        self.t_downsampling: list = t_downsampling # Downsampling factors for multi-scale temporal B-splines.
        self.n_mfcomponents: int = n_mfcomponents # Number of motion field components.
        self.max_disp_frac: float = max_disp_frac # Maximum allowed displacement as a fraction of the FOV.
        self.max_disp: float = self.nx * max_disp_frac
        self.max_t_init: float = max_t_init # Maximum initialization value for temporal components.
        
        self.bpt_frames: np.ndarray | torch.Tensor | None = bpt_frames # Pre-calculated B+PT navigator signals, per frame. (Shape: (n_frames, num_bpt_signals))
        
        # B-spline bases
        self.xyz_basis_scales: dict | None = None
        self.t_basis_scales: dict | None = None
        
        # Control points
        self.xyz_ctrls: dict | None = None
        self.t_ctrls: dict | None = None
        
        # Dense coefficients (computed from control points)
        self.xyz_coeffs: torch.Tensor | None = None
        self.t_coeffs: torch.Tensor | None = None
        
        # BPT-related
        self.mixing: torch.Tensor | None = None  
        
    def initialize(self):
        """
        Initializes B-spline basis functions, control points, and rigid components based on the selected mode.
        """
        if self.verbose:
            logger.info(f"MotionFieldModel initialized: mode={self.mode}, shape={self.im_shape}, frames={self.n_frames}")
            
        if self.mode == 'mrmotus':
            self._init_mrmotus()
        elif self.mode == 'bpt_motus':
            self._init_bpt_motus()
        elif self.mode == 'mrmotus_rigid':
            self._init_mrmotus_rigid()
        elif self.mode == 'bpt_rigid':
            self._init_bpt_rigid()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_trainable_parameters(self):
        """
        Get the list of trainable parameters based on the mode.
        
        Returns:
        params (dict): Dictionary mapped with parameter names indicating optimization tracking.
        """
        params = {}
        if self.mode == 'mrmotus':
            for k, v in self.xyz_ctrls.items(): params[f'xyz_ctrls_{k}'] = v
            for k, v in self.t_ctrls.items(): params[f't_ctrls_{k}'] = v
        elif self.mode == 'bpt_motus':
            for k, v in self.xyz_ctrls.items(): params[f'xyz_ctrls_{k}'] = v
        elif self.mode == 'mrmotus_rigid':
            for k, v in self.t_ctrls.items(): params[f't_ctrls_{k}'] = v
        elif self.mode == 'bpt_rigid':
            params['mixing'] = self.mixing
        return params
    
    def forward(self, frame_ids=None):
        """
        Generate motion fields for specified frames.
        
        Args:
        frame_ids (list | torch.Tensor | None): Frame indices to generate motion fields for. (Shape: (batch_size,))
        
        Returns:
        motion_fields (torch.Tensor): Motion fields. (Shape: (batch_size, nx, ny, nz, 3))
        """
        # Get spatial coefficients
        if frame_ids is None:
            frame_ids = torch.arange(self.n_frames, device=self.device)
        if self.xyz_coeffs is None:
            self.xyz_coeffs = self._eval_xyz_coeffs(self.xyz_ctrls, self.xyz_basis_scales)
        
        # Get temporal coefficients for batch
        if self.mode == 'mrmotus':
            t_coeffs_batch = self._eval_t_coeffs_batch(self.t_ctrls, self.t_basis_scales, frame_ids)
        elif self.mode == 'bpt_motus':
            t_coeffs_batch = self.t_coeffs[frame_ids]
        elif self.mode == 'mrmotus_rigid':
            t_coeffs_batch = self._eval_t_coeffs_batch(self.t_ctrls, self.t_basis_scales, frame_ids)
        elif self.mode == 'bpt_rigid':
            t_coeffs_batch = (self.t_coeffs @ self.mixing)[frame_ids]
        
        # Combine spatial and temporal to get motion fields
        motion_fields = self._eval_motion_fields_batch(self.xyz_coeffs, t_coeffs_batch)
        
        return motion_fields

    def _init_mrmotus(self):
        """
        Initialize for full MR-MOTUS: learned spatial and temporal B-splines.
        
        Stores:
        xyz_basis_scales (dict): Spatial basis scales.
        t_basis_scales (dict): Temporal basis scales.
        xyz_ctrls (dict): Initialized spatial control points.
        t_ctrls (dict): Initialized temporal control points.
        """
        if self.verbose:
            logger.info("Initializing full MR-MOTUS mode...")
        
        # Build multi-scale spatial bases
        self.xyz_basis_scales = self._build_xyz_basis()
        
        # Build multi-scale temporal bases
        self.t_basis_scales = self._build_t_basis()
        
        # Initialize spatial control points (zeros)
        self.xyz_ctrls = self._init_xyz_ctrls(self.xyz_basis_scales)
        
        # Initialize temporal control points (random)
        self.t_ctrls = self._init_t_ctrls(self.t_basis_scales)
    
    def _init_bpt_motus(self):
        """
        Initialize for BPT-MOTUS: BPT temporal components with learned spatial B-splines.
        
        Stores:
        bpt_frames (torch.Tensor): Stored BPT navigation signals. (Shape: (n_frames, num_bpt_signals))
        n_mfcomponents (int): Updated number of motion field components based on BPT signals.
        xyz_basis_scales (dict): Spatial basis scales.
        xyz_ctrls (dict): Initialized spatial control points.
        t_coeffs (torch.Tensor): Fixed temporal coefficients from BPT. (Shape: (n_frames, num_bpt_signals))
        """
        if self.verbose:
            logger.info("Initializing BPT-MOTUS mode...")
        
        if self.bpt_frames is None:
            raise ValueError("bpt_frames required for bpt_motus mode")
        
        # Store BPT frames
        self.bpt_frames = torch.tensor(np.abs(self.bpt_frames), dtype=torch.float32, device=self.device).detach()
        self.n_mfcomponents = self.bpt_frames.shape[1]
        
        # Build multi-scale spatial bases
        self.xyz_basis_scales = self._build_xyz_basis()
        
        # Initialize spatial control points (zeros)
        self.xyz_ctrls = self._init_xyz_ctrls(self.xyz_basis_scales)
        
        # Temporal coefficients are fixed from BPT
        self.t_coeffs = self.bpt_frames
    
    def _init_mrmotus_rigid(self):
        """
        Initialize for rigid motion + learned time: rigid spatial components with learned temporal B-splines.
        
        Stores:
        n_mfcomponents (int): Forced to 6 components for rigid tracking.
        t_basis_scales (dict): Temporal basis scales.
        t_ctrls (dict): Initialized temporal control points.
        xyz_coeffs (torch.Tensor): Rigid spatial components directly as dense coefficients. (Shape: (3, nx, ny, nz, 6))
        """
        if self.verbose:
            logger.info("Initializing rigid motion + learned temporal mode...")
        
        if self.n_mfcomponents != 6:
            logger.warning(f"n_mfcomponents={self.n_mfcomponents} for rigid motion, expected 6. Using 6.")
            self.n_mfcomponents = 6
        
        # Build temporal bases
        self.t_basis_scales = self._build_t_basis()
        
        # Initialize temporal control points
        self.t_ctrls = self._init_t_ctrls(self.t_basis_scales)
        
        # Initialize rigid spatial components directly as dense coefficients
        self.xyz_coeffs = torch.zeros((3, self.nx, self.ny, self.nz, 6), 
                                      dtype=torch.float32, device=self.device)
        self._init_rigid_xyz_coeffs(self.xyz_coeffs)
    
    def _init_bpt_rigid(self):
        """
        Initialize for BPT + rigid motion: BPT temporal components with rigid spatial components.
        
        Stores:
        bpt_frames (torch.Tensor): Stored BPT navigation signals. (Shape: (n_frames, num_bpt_signals))
        t_coeffs (torch.Tensor): Fixed temporal coefficients from BPT. (Shape: (n_frames, num_bpt_signals))
        n_mfcomponents (int): Forced to 6 components for rigid tracking.
        mixing (torch.Tensor): Matrix learning the mapping from BPT frames to 6 rigid components. (Shape: (num_bpt_signals, 6))
        xyz_coeffs (torch.Tensor): Rigid spatial components directly as dense coefficients. (Shape: (3, nx, ny, nz, 6))
        """
        if self.verbose:
            logger.info("Initializing BPT + rigid motion mode...")
        
        if self.bpt_frames is None:
            raise ValueError("bpt_frames required for bpt_rigid mode")
        
        # Store BPT frames
        self.bpt_frames = torch.tensor(self.bpt_frames, dtype=torch.float32, device=self.device)
        self.t_coeffs = self.bpt_frames  # Fixed temporal coefficients from BPT
        self.n_mfcomponents = 6
        
        # Initialize mixing matrix (learns how to combine BPTs into 6 rigid components)
        self.mixing = torch.zeros((self.bpt_frames.shape[1], self.n_mfcomponents), 
                                  dtype=torch.float32, device=self.device, requires_grad=True)
        
        # Initialize rigid spatial components directly as dense coefficients
        self.xyz_coeffs = torch.zeros((3, self.nx, self.ny, self.nz, self.n_mfcomponents), 
                                      dtype=torch.float32, device=self.device)
        self._init_rigid_xyz_coeffs(self.xyz_coeffs)

    # ========== B-spline Basis Generation ==========
    
    def _generate_1d_basis(self, domain_size, factor):
        """
        Generate a dense 1D B-spline basis matrix.
        
        Args:
        domain_size (int): Number of points in the domain.
        factor (int): Downsampling factor (n_control_points = domain_size // factor).
        
        Returns:
        B (torch.Tensor): Basis matrix. (Shape: (domain_size, n_control_points))
        """
        k = self.degree
        n_cp = max(k + 1, domain_size // factor)
        
        eval_points = np.linspace(0, 1, domain_size, endpoint=True)
        n_internal = n_cp - k + 1
        t_uniform = np.linspace(0, 1, n_internal, endpoint=True)
        knots = np.concatenate((np.zeros(k), t_uniform, np.ones(k)))
        
        try:
            B = BSpline.design_matrix(eval_points, knots, k, extrapolate=False).todense()
        except ValueError as e:
            logger.error(f"B-spline generation failed for size {domain_size} and factor {factor}. Detail: {e}")
            raise
        
        return torch.from_numpy(np.asarray(B)).float().to(self.device)
    
    def _build_xyz_basis(self):
        """
        Precompute 3D spatial B-spline basis functions for multiple scales.
        
        Returns:
        basis_scales (dict): Keys are scale names (e.g., 'scale_8'), values are tuples (B_x, B_y, B_z).
        """
        if self.verbose:
            logger.info(f"Building spatial B-spline bases: nx={self.nx}, ny={self.ny}, nz={self.nz}, factors={self.xyz_downsampling}")
        
        basis_scales = {}
        for factor in self.xyz_downsampling:
            B_x = self._generate_1d_basis(self.nx, factor)
            B_y = self._generate_1d_basis(self.ny, factor)
            B_z = self._generate_1d_basis(self.nz, factor)
            basis_scales[f'scale_{factor}'] = (B_x, B_y, B_z)
        
        return basis_scales
    
    def _build_t_basis(self):
        """
        Precompute 1D temporal B-spline basis functions for multiple scales.
        
        Returns:
        basis_scales (dict): Keys are scale names, values are B_t tensors.
        """
        if self.verbose:
            logger.info(f"Building temporal B-spline bases: n_frames={self.n_frames}, factors={self.t_downsampling}")
        
        basis_scales = {}
        for factor in self.t_downsampling:
            B_t = self._generate_1d_basis(self.n_frames, factor)
            basis_scales[f'scale_{factor}'] = B_t
        
        return basis_scales
    
    # ========== Control Point Initialization ==========
    
    def _init_xyz_ctrls(self, basis_scales):
        """
        Initialize spatial control point tensors (zeros).
        
        Args:
        basis_scales (dict): Spatial basis scales from _build_xyz_basis.
        
        Returns:
        ctrls (dict): Keys are scale names, values are control point tensors. (Shape: (3, n_cpx, n_cpy, n_cpz, n_mfcomponents))
        """
        ctrls = {}
        for scale_name, (B_x, B_y, B_z) in basis_scales.items():
            n_cpx, n_cpy, n_cpz = B_x.shape[1], B_y.shape[1], B_z.shape[1]
            P = torch.zeros((3, n_cpx, n_cpy, n_cpz, self.n_mfcomponents),
                           dtype=torch.float32, device=self.device, requires_grad=True)
            ctrls[scale_name] = P
        
        return ctrls
    
    def _init_t_ctrls(self, basis_scales):
        """
        Initialize temporal control point tensors (uniform random).
        
        Args:
        basis_scales (dict): Temporal basis scales from _build_t_basis.
        
        Returns:
        ctrls (dict): Keys are scale names, values are control point tensors. (Shape: (n_cpt, n_mfcomponents))
        """
        ctrls = {}
        for scale_name, B_t in basis_scales.items():
            n_cpt = B_t.shape[1]
            T = torch.tensor(
                np.random.uniform(-self.max_t_init, self.max_t_init, (n_cpt, self.n_mfcomponents)),
                dtype=torch.float32, device=self.device, requires_grad=True
            )
            ctrls[scale_name] = T
        
        return ctrls
    
    def _init_rigid_xyz_coeffs(self, xyz_coeffs):
        """
        Initialize motion field components as 3 rotations and 3 translations.
        
        Args:
        xyz_coeffs (torch.Tensor): Tensor to fill in-place. (Shape: (3, nx, ny, nz, 6))
        """
        M_target = self.max_disp / max(self.max_t_init, 1.0)
        nx, ny, nz = self.nx, self.ny, self.nz
        
        # Normalized grid coordinates
        x = torch.linspace(-1, 1, nx, device=self.device)
        y = torch.linspace(-1, 1, ny, device=self.device)
        z = torch.linspace(-1, 1, nz, device=self.device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
        # Rotation fields (centered at origin)
        rot_x = torch.stack([torch.zeros_like(X), -Z, Y])
        rot_y = torch.stack([Z, torch.zeros_like(Y), -X])
        rot_z = torch.stack([-Y, X, torch.zeros_like(Z)])
        rotations = [rot_x, rot_y, rot_z]
        
        # Normalize each rotation so max magnitude = M_target
        for i in range(3):
            mag = torch.sqrt((rotations[i] ** 2).sum(0))
            max_mag = mag.max().clamp_min(1e-8)
            rotations[i] = rotations[i] * (M_target / max_mag)
        
        # Translation fields
        trans_x = torch.stack([torch.ones_like(X), torch.zeros_like(Y), torch.zeros_like(Z)]) * M_target
        trans_y = torch.stack([torch.zeros_like(X), torch.ones_like(Y), torch.zeros_like(Z)]) * M_target
        trans_z = torch.stack([torch.zeros_like(X), torch.zeros_like(Y), torch.ones_like(Z)]) * M_target
        
        # Fill in xyz_coeffs
        components = rotations + [trans_x, trans_y, trans_z]
        for i, component in enumerate(components):
            xyz_coeffs[..., i].copy_(component)
    
    # ========== Coefficient Evaluation ==========
    
    def _eval_xyz_coeffs(self, ctrls, basis_scales):
        """
        Evaluate dense 3D spatial motion field coefficients from control points.
        
        Args:
        ctrls (dict): Control point tensors from _init_xyz_ctrls.
        basis_scales (dict): Basis tensors from _build_xyz_basis.
        
        Returns:
        S (torch.Tensor): Dense spatial coefficients combined across all scales. (Shape: (3, nx, ny, nz, n_mfcomponents))
        """
        def evaluate_scale(B_x, B_y, B_z, P):
            # Contract over Z-dimension
            S = torch.einsum('zk, cijkm -> cijzm', B_z, P)
            # Contract over Y-dimension
            S = torch.einsum('yj, cijzm -> ciyzm', B_y, S)
            # Contract over X-dimension
            S = torch.einsum('xi, ciyzm -> cxyzm', B_x, S)
            return S
        
        return sum(
            evaluate_scale(*basis_scales[scale_name], ctrls[scale_name])
            for scale_name in basis_scales.keys()
        )
    
    def _eval_t_coeffs_batch(self, ctrls, basis_scales, frame_ids):
        """
        Evaluate temporal coefficients for a batch of frames.
        
        Args:
        ctrls (dict): Control point tensors.
        basis_scales (dict): Basis tensors.
        frame_ids (list | torch.Tensor): Frame indices for the batch. (Shape: (batch_size,))
        
        Returns:
        t_coeffs (torch.Tensor): Temporal coefficients. (Shape: (batch_size, n_mfcomponents))
        """
        return sum(
            torch.einsum('ft, tm -> fm', 
                        B_t_full[frame_ids], 
                        ctrls[scale_name])
            for scale_name, B_t_full in basis_scales.items()
        )
    
    def _eval_motion_fields_batch(self, xyz_coeffs, t_coeffs_batch):
        """
        Combine spatial and temporal coefficients to produce motion fields.
        
        Args:
        xyz_coeffs (torch.Tensor): Spatial coefficients. (Shape: (3, nx, ny, nz, n_mfcomponents))
        t_coeffs_batch (torch.Tensor): Temporal coefficients. (Shape: (batch_size, n_mfcomponents))
        
        Returns:
        motion_fields_batch (torch.Tensor): Combined motion fields. (Shape: (batch_size, nx, ny, nz, 3))
        """
        motion_fields_batch = torch.einsum('cxyzm, fm -> fxyzc', xyz_coeffs, t_coeffs_batch)
        return motion_fields_batch