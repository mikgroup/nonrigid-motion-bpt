"""
Classes for modeling and optimizing non-rigid motion fields with MR-MOTUS and BPT-MOTUS.
"""
import os
import numpy as np
import torch
import logging
from typing import Literal, Optional, Dict, Tuple, List
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix
from tqdm import tqdm
import gc

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MotionFieldModel:
    """
    Motion field parameterization and generation using B-splines.
    
    Handles spatial and temporal B-spline bases, control point initialization,
    and motion field computation. Supports multiple modes:
    - 'mrmotus': MR-MOTUS with learned spatial and temporal B-splines
    - 'bpt_nonrigid': BPT temporal components with learned spatial B-splines
    - 'mrmotus_rigid': Rigid motion components with learned temporal B-splines
    - 'bpt_rigid': BPT temporal components with rigid spatial components
    """
    
    def __init__(self, 
                 im_shape: Tuple[int, int, int],
                 n_frames: int,
                 mode: Literal['mrmotus', 'bpt_nonrigid', 'mrmotus_rigid', 'bpt_rigid'] = 'bpt_rigid',
                 verbose: bool = False,
                 degree: int = 3,
                 device: str = "cpu"):
        
        self.verbose: bool = verbose
        self.degree: int = degree # b-spline degree (default: 3 for cubic splines).
        self.device: str = device
        self.mode: str = mode # motion parameterization mode
        
        self.nx, self.ny, self.nz = im_shape # image dimensions
        self.n_frames: int = n_frames # number of temporal frames
        
        # B-spline bases
        self.xyz_basis_scales: Optional[Dict] = None
        self.t_basis_scales: Optional[Dict] = None
        
        # Control points
        self.xyz_ctrls: Optional[Dict] = None
        self.t_ctrls: Optional[Dict] = None
        
        # Dense coefficients (computed from control points)
        self.xyz_coeffs: Optional[torch.Tensor] = None
        self.t_coeffs: Optional[torch.Tensor] = None
        
        # BPT-related
        self.bpt_frames: Optional[torch.Tensor] = None
        self.mixing: Optional[torch.Tensor] = None  # for BPT rigid mode
        
        # Configuration parameters (set during initialize)
        self.n_mfcomponents: int = 6  # default for rigid
        self.max_disp: float = 5.0  # maximum displacement
        self.max_t_init: float = 0.5  # max initial temporal coeff value
        
        if self.verbose:
            logger.info(f"MotionFieldModel initialized: mode={mode}, shape={im_shape}, frames={n_frames}")
    
    def initialize(self,
                   xyz_downsampling: List[int] = [4, 8, 16],
                   t_downsampling: List[int] = [1, 2, 5],
                   n_mfcomponents: int = 6,
                   max_disp_frac: float = 0.05,
                   max_t_init: float = 0.0,
                   bpt_frames: Optional[np.ndarray] = None):
        """
        Initialize B-spline bases and control points based on the selected mode.
        
        Args:
            xyz_downsampling (List[int]): Spatial downsampling factors for multi-scale B-splines.
            t_downsampling (List[int]): Temporal downsampling factors for multi-scale B-splines.
            n_mfcomponents (int): Number of motion field components.
            max_disp_frac (float): Maximum displacement as fraction of image size.
            max_t_init (float): Maximum initial temporal coefficient value.
            bpt_frames (Optional[np.ndarray]): BPT/PT frames for BPT modes (n_frames, n_bpt_components).
        
        Stores:
            xyz_basis_scales (dict): Spatial B-spline bases.
            t_basis_scales (dict): Temporal B-spline bases (if applicable).
            xyz_ctrls (dict): Spatial control points.
            t_ctrls (dict): Temporal control points (if applicable).
            bpt_frames (torch.Tensor): BPT frames (if applicable).
            mixing (torch.Tensor): BPT mixing matrix (if applicable).
        """
        self.n_mfcomponents = n_mfcomponents
        self.max_disp = self.nx * max_disp_frac
        self.max_t_init = max_t_init
        
        if self.verbose:
            logger.info(f"Initializing motion field model in mode: {self.mode}")
        
        if self.mode == 'mrmotus':
            self._init_mrmotus(xyz_downsampling, t_downsampling, n_mfcomponents, max_t_init)
        elif self.mode == 'bpt_nonrigid':
            self._init_bpt_nonrigid(xyz_downsampling, bpt_frames)
        elif self.mode == 'mrmotus_rigid':
            self._init_mrmotus_rigid(t_downsampling, n_mfcomponents, max_t_init)
        elif self.mode == 'bpt_rigid':
            self._init_bpt_rigid(bpt_frames)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _init_mrmotus(self, xyz_downsampling, t_downsampling, n_mfcomponents, max_t_init):
        """
        Initialize for full MR-MOTUS: learned spatial and temporal B-splines.
        
        Stores:
            xyz_basis_scales, t_basis_scales, xyz_ctrls, t_ctrls
        """
        if self.verbose:
            logger.info("Initializing full MR-MOTUS mode...")
        
        # Build multi-scale spatial bases
        self.xyz_basis_scales = self._build_xyz_basis(self.nx, self.ny, self.nz, xyz_downsampling)
        
        # Build multi-scale temporal bases
        self.t_basis_scales = self._build_t_basis(self.n_frames, t_downsampling)
        
        # Initialize spatial control points (zeros)
        self.xyz_ctrls = self._init_xyz_ctrls(self.xyz_basis_scales, n_mfcomponents)
        
        # Initialize temporal control points (random)
        self.t_ctrls = self._init_t_ctrls(self.t_basis_scales, n_mfcomponents, max_t_init)
    
    def _init_bpt_nonrigid(self, xyz_downsampling, bpt_frames):
        """
        Initialize for BPT + MR-MOTUS: BPT temporal components with learned spatial B-splines.
        
        Args:
            bpt_frames (np.ndarray): BPT frames (n_frames, n_bpt_components).
        
        Stores:
            bpt_frames, xyz_basis_scales, xyz_ctrls
        """
        if self.verbose:
            logger.info("Initializing BPT + nonrigid mode...")
        
        if bpt_frames is None:
            raise ValueError("bpt_frames required for bpt_nonrigid mode")
        
        # Store BPT frames
        self.bpt_frames = torch.tensor(np.abs(bpt_frames), dtype=torch.float32, device=self.device).detach()
        self.n_mfcomponents = self.bpt_frames.shape[1] + 1  # +1 for bias/baseline
        
        # Build multi-scale spatial bases
        self.xyz_basis_scales = self._build_xyz_basis(self.nx, self.ny, self.nz, xyz_downsampling)
        
        # Initialize spatial control points (zeros)
        self.xyz_ctrls = self._init_xyz_ctrls(self.xyz_basis_scales, self.n_mfcomponents)
        
        # Temporal coefficients are fixed from BPT
        self.t_coeffs = self.bpt_frames
    
    def _init_mrmotus_rigid(self, t_downsampling, n_mfcomponents, max_t_init):
        """
        Initialize for rigid motion + learned time: rigid spatial components with learned temporal B-splines.
        
        Args:
            n_mfcomponents (int): Should be 6 (3 rotations + 3 translations).
        
        Stores:
            xyz_basis_scales, xyz_coeffs, t_basis_scales, t_ctrls
        """
        if self.verbose:
            logger.info("Initializing rigid motion + learned temporal mode...")
        
        if n_mfcomponents != 6:
            logger.warning(f"n_mfcomponents={n_mfcomponents} for rigid motion, expected 6. Using 6.")
            self.n_mfcomponents = 6
        
        # Build temporal bases
        self.t_basis_scales = self._build_t_basis(self.n_frames, t_downsampling)
        
        # Initialize temporal control points
        self.t_ctrls = self._init_t_ctrls(self.t_basis_scales, 6, max_t_init)
        
        # Initialize rigid spatial components directly as dense coefficients
        self.xyz_coeffs = torch.zeros((3, self.nx, self.ny, self.nz, 6), 
                                      dtype=torch.float32, device=self.device)
        self._initialize_rigid_transformation_components(self.xyz_coeffs)
    
    def _init_bpt_rigid(self, bpt_frames):
        """
        Initialize for BPT + rigid motion: BPT temporal components with rigid spatial components.
        
        Args:
            bpt_frames (np.ndarray): BPT frames (n_frames, n_bpt_components).
        
        Stores:
            bpt_frames, mixing, xyz_coeffs
        """
        if self.verbose:
            logger.info("Initializing BPT + rigid motion mode...")
        
        if bpt_frames is None:
            raise ValueError("bpt_frames required for bpt_rigid mode")
        
        # Store BPT frames
        self.bpt_frames = torch.tensor(bpt_frames, dtype=torch.float32, device=self.device)
        self.n_mfcomponents = 6
        
        # Initialize mixing matrix (learns how to combine BPTs into 6 rigid components)
        self.mixing = torch.zeros((self.bpt_frames.shape[1], 6), 
                                  dtype=torch.float32, device=self.device, requires_grad=True)
        
        # Initialize rigid spatial components directly as dense coefficients
        self.xyz_coeffs = torch.zeros((3, self.nx, self.ny, self.nz, 6), 
                                      dtype=torch.float32, device=self.device)
        self._initialize_rigid_transformation_components(self.xyz_coeffs)
    
    def get_trainable_parameters(self) -> List[torch.Tensor]:
        """
        Get the list of trainable parameters based on the mode.
        
        Returns:
            List[torch.Tensor]: List of parameters to optimize.
        """
        params = []
        
        if self.mode == 'mrmotus':
            # Optimize both spatial and temporal control points
            params.extend(self.xyz_ctrls.values())
            params.extend(self.t_ctrls.values())
        elif self.mode == 'bpt_nonrigid':
            # Optimize only spatial control points (temporal is fixed from BPT)
            params.extend(self.xyz_ctrls.values())
        elif self.mode == 'mrmotus_rigid':
            # Optimize only temporal control points (spatial is fixed rigid)
            params.extend(self.t_ctrls.values())
        elif self.mode == 'bpt_rigid':
            # Optimize only mixing matrix (both spatial and temporal are fixed)
            params.append(self.mixing)
        
        return params
    
    def forward(self, frame_ids: List[int]) -> torch.Tensor:
        """
        Generate motion fields for specified frames.
        
        Args:
            frame_ids (List[int]): Frame indices to generate motion fields for.
        
        Returns:
            torch.Tensor: Motion fields (batch_size, nx, ny, nz, 3).
        """
        # Get spatial coefficients
        if self.xyz_coeffs is None:
            # Evaluate from control points
            self.xyz_coeffs = self._eval_xyz_coeffs(self.xyz_ctrls, self.xyz_basis_scales)
        
        # Get temporal coefficients for batch
        if self.mode == 'mrmotus':
            t_coeffs_batch = self._eval_t_coeffs_batch(self.t_ctrls, self.t_basis_scales, frame_ids)
        elif self.mode == 'bpt_nonrigid':
            t_coeffs_batch = self.t_coeffs[frame_ids]
        elif self.mode == 'mrmotus_rigid':
            t_coeffs_batch = self._eval_t_coeffs_batch(self.t_ctrls, self.t_basis_scales, frame_ids)
        elif self.mode == 'bpt_rigid':
            t_coeffs_batch = (self.bpt_frames @ self.mixing)[frame_ids]
        
        # Combine spatial and temporal to get motion fields
        motion_fields = self._eval_motion_fields_batch(self.xyz_coeffs, t_coeffs_batch)
        
        return motion_fields
    
    # ========== B-spline Basis Generation ==========
    
    def _generate_1d_basis(self, domain_size: int, factor: int) -> torch.Tensor:
        """
        Generate a dense 1D B-spline basis matrix.
        
        Args:
            domain_size (int): Number of points in the domain.
            factor (int): Downsampling factor (n_control_points = domain_size // factor).
        
        Returns:
            torch.Tensor: Basis matrix of shape (domain_size, n_control_points).
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
    
    def _build_xyz_basis(self, nx: int, ny: int, nz: int, 
                         downsample_factors: List[int]) -> Dict:
        """
        Precompute 3D spatial B-spline basis functions for multiple scales.
        
        Args:
            nx, ny, nz (int): Spatial dimensions.
            downsample_factors (List[int]): Downsampling factors for each scale.
        
        Returns:
            Dict: Keys are scale names (e.g., 'scale_8'), values are tuples (B_x, B_y, B_z).
        """
        if self.verbose:
            logger.info(f"Building spatial B-spline bases: nx={nx}, ny={ny}, nz={nz}, factors={downsample_factors}")
        
        basis_scales = {}
        for factor in downsample_factors:
            B_x = self._generate_1d_basis(nx, factor)
            B_y = self._generate_1d_basis(ny, factor)
            B_z = self._generate_1d_basis(nz, factor)
            basis_scales[f'scale_{factor}'] = (B_x, B_y, B_z)
        
        return basis_scales
    
    def _build_t_basis(self, n_frames: int, downsample_factors: List[int]) -> Dict:
        """
        Precompute 1D temporal B-spline basis functions for multiple scales.
        
        Args:
            n_frames (int): Number of temporal frames.
            downsample_factors (List[int]): Downsampling factors for each scale.
        
        Returns:
            Dict: Keys are scale names, values are B_t tensors.
        """
        if self.verbose:
            logger.info(f"Building temporal B-spline bases: n_frames={n_frames}, factors={downsample_factors}")
        
        basis_scales = {}
        for factor in downsample_factors:
            B_t = self._generate_1d_basis(n_frames, factor)
            basis_scales[f'scale_{factor}'] = B_t
        
        return basis_scales
    
    # ========== Control Point Initialization ==========
    
    def _init_xyz_ctrls(self, basis_scales: Dict, n_mfcomponents: int) -> Dict:
        """
        Initialize spatial control point tensors (zeros).
        
        Args:
            basis_scales (Dict): Spatial basis scales from _build_xyz_basis.
            n_mfcomponents (int): Number of motion field components.
        
        Returns:
            Dict: Keys are scale names, values are control point tensors 
                  (3, n_cpx, n_cpy, n_cpz, n_mfcomponents).
        """
        ctrls = {}
        for scale_name, (B_x, B_y, B_z) in basis_scales.items():
            n_cpx, n_cpy, n_cpz = B_x.shape[1], B_y.shape[1], B_z.shape[1]
            P = torch.zeros((3, n_cpx, n_cpy, n_cpz, n_mfcomponents),
                           dtype=torch.float32, device=self.device, requires_grad=True)
            ctrls[scale_name] = P
        
        return ctrls
    
    def _init_t_ctrls(self, basis_scales: Dict, n_mfcomponents: int, 
                      max_t_init: float) -> Dict:
        """
        Initialize temporal control point tensors (uniform random).
        
        Args:
            basis_scales (Dict): Temporal basis scales from _build_t_basis.
            n_mfcomponents (int): Number of motion field components.
            max_t_init (float): Maximum initial value for uniform distribution.
        
        Returns:
            Dict: Keys are scale names, values are control point tensors (n_cpt, n_mfcomponents).
        """
        ctrls = {}
        for scale_name, B_t in basis_scales.items():
            n_cpt = B_t.shape[1]
            T = torch.tensor(
                np.random.uniform(-max_t_init, max_t_init, (n_cpt, n_mfcomponents)),
                dtype=torch.float32, device=self.device, requires_grad=True
            )
            ctrls[scale_name] = T
        
        return ctrls
    
    def _initialize_rigid_transformation_components(self, xyz_coeffs: torch.Tensor):
        """
        Initialize motion field components as 3 rotations and 3 translations.
        
        Modifies xyz_coeffs in-place. Components 0-2 are rotations around x, y, z axes;
        components 3-5 are translations along x, y, z axes.
        
        Args:
            xyz_coeffs (torch.Tensor): Tensor to fill (3, nx, ny, nz, 6).
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
    
    def _eval_xyz_coeffs(self, ctrls: Dict, basis_scales: Dict) -> torch.Tensor:
        """
        Evaluate dense 3D spatial motion field coefficients from control points.
        
        Uses sequential einsum factorization to avoid large intermediate tensors.
        
        Args:
            ctrls (Dict): Control point tensors from _init_xyz_ctrls.
            basis_scales (Dict): Basis tensors from _build_xyz_basis.
        
        Returns:
            torch.Tensor: Dense spatial coefficients (3, nx, ny, nz, n_mfcomponents).
        """
        def evaluate_scale(B_x, B_y, B_z, P):
            """Sequential einsum for a single scale."""
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
    
    def _eval_t_coeffs_batch(self, ctrls: Dict, basis_scales: Dict, 
                             frame_ids: List[int]) -> torch.Tensor:
        """
        Evaluate temporal coefficients for a batch of frames.
        
        Args:
            ctrls (Dict): Control point tensors from _init_t_ctrls.
            basis_scales (Dict): Basis tensors from _build_t_basis.
            frame_ids (List[int]): Frame indices for the batch.
        
        Returns:
            torch.Tensor: Temporal coefficients (batch_size, n_mfcomponents).
        """
        return sum(
            torch.einsum('ft, tm -> fm', 
                        B_t_full[frame_ids], 
                        ctrls[scale_name])
            for scale_name, B_t_full in basis_scales.items()
        )
    
    def _eval_motion_fields_batch(self, xyz_coeffs: torch.Tensor, 
                                   t_coeffs_batch: torch.Tensor) -> torch.Tensor:
        """
        Combine spatial and temporal coefficients to produce motion fields.
        
        Args:
            xyz_coeffs (torch.Tensor): Spatial coefficients (3, nx, ny, nz, n_mfcomponents).
            t_coeffs_batch (torch.Tensor): Temporal coefficients (batch_size, n_mfcomponents).
        
        Returns:
            torch.Tensor: Motion fields (batch_size, nx, ny, nz, 3).
        """
        motion_fields_batch = torch.einsum('cxyzm, fm -> fxyzc', xyz_coeffs, t_coeffs_batch)
        return motion_fields_batch


class MotionOptimizer:
    """
    Optimization framework for estimating motion fields from k-space data.
    
    Handles forward model, loss computation, training loop, and early stopping.
    """
    
    def __init__(self,
                 motion_model: MotionFieldModel,
                 S: torch.Tensor,
                 csm: torch.Tensor,
                 xk_frames: List[np.ndarray],
                 coords_frames: List[np.ndarray],
                 dcf_frames: List[np.ndarray],
                 scaling_per_frame: np.ndarray,
                 verbose: bool = False):
        """
        Initialize the motion optimizer.
        
        Args:
            motion_model (MotionFieldModel): Initialized motion field model.
            S (torch.Tensor): Reference image (nx, ny, nz), complex.
            csm (torch.Tensor): Coil sensitivity maps (n_coils, nx, ny, nz), complex.
            xk_frames (List[np.ndarray]): K-space data per frame (n_coils, n_spokes, n_samples).
            coords_frames (List[np.ndarray]): K-space coordinates per frame (n_spokes, n_samples, 3).
            dcf_frames (List[np.ndarray]): Density compensation per frame (n_spokes, n_samples).
            scaling_per_frame (np.ndarray): Scaling factors per frame (n_frames,).
            verbose (bool): Enable verbose logging.
        """
        self.verbose: bool = verbose
        self.motion_model: MotionFieldModel = motion_model
        self.device: str = motion_model.device
        
        # Store reference image and coil maps
        self.S: torch.Tensor = S.to(self.device)
        self.csm: torch.Tensor = csm.to(self.device)
        
        # Convert data to torch tensors
        self.n_frames = len(xk_frames)
        self.xk_frames_t = [torch.from_numpy(x).to(self.device).to(torch.complex64) for x in xk_frames]
        self.coords_frames_t = [torch.from_numpy(c).to(self.device).to(torch.float32) for c in coords_frames]
        self.dcf_frames_t = [torch.from_numpy(d).to(self.device).to(torch.float32) for d in dcf_frames]
        self.scaling_per_frame = scaling_per_frame
        
        # NUFFT operator
        self.im_size = S.shape
        self.oversamp = 1.25
        import torchkbnufft as tkbn
        self.kb_nufft = tkbn.KbNufft(
            im_size=self.im_size,
            grid_size=torch.round(torch.tensor(self.im_size) * self.oversamp).to(torch.int64)
        ).to(self.device)
        
        # Training state
        self.optimizer = None
        self.best_params = None
        self.best_loss = float('inf')
        self.best_epoch = 0
        
        # Loss logs
        self.dc_loss_log = []
        
        if self.verbose:
            logger.info(f"MotionOptimizer initialized: {self.n_frames} frames, device={self.device}")
    
    def fit(self,
            epochs: int = 40,
            batch_size: int = 50,
            step_size: float = 5e-2,
            patience: int = 4,
            relative_change_threshold: float = 1e-3,
            seed: int = 4) -> Dict:
        """
        Run the optimization to estimate motion fields.
        
        Args:
            epochs (int): Maximum number of training epochs.
            batch_size (int): Number of frames per batch.
            step_size (float): Learning rate for Adam optimizer.
            patience (int): Number of epochs without improvement before early stopping.
            relative_change_threshold (float): Minimum relative loss change to continue training.
            seed (int): Random seed for reproducibility.
        
        Returns:
            Dict: Results containing best parameters, motion fields, and loss logs.
        """
        np.random.seed(seed)
        
        # Setup optimizer
        trainable_params = self.motion_model.get_trainable_parameters()
        self.optimizer = torch.optim.Adam(trainable_params, lr=step_size)
        
        # Pre-calculate max radius for edge weighting
        max_radius_all = 0.0
        for coords_t in self.coords_frames_t:
            max_radius_all = max(max_radius_all, torch.linalg.norm(coords_t, dim=-1).max().item())
        
        # Early stopping
        previous_loss = float('inf')
        no_improvement_epochs = 0
        
        if self.verbose:
            logger.info(f"Starting optimization: {epochs} epochs, batch_size={batch_size}, lr={step_size}")
        
        # Training loop
        pbar = tqdm(range(epochs), desc="Training")
        try:
            for epoch in pbar:
                epoch_loss = 0.0
                frame_idxes = np.arange(self.n_frames)
                np.random.shuffle(frame_idxes)
                num_batches = int(np.ceil(self.n_frames / batch_size))
                
                for batch_idx in range(num_batches):
                    self.optimizer.zero_grad(set_to_none=True)
                    batch_loss = torch.tensor(0.0, device=self.device)
                    
                    # Get frame IDs for this batch
                    cur_frame_ids = frame_idxes[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                    real_batch_size = len(cur_frame_ids)
                    
                    # Update spatial coefficients if needed
                    if self.motion_model.xyz_ctrls is not None:
                        self.motion_model.xyz_coeffs = self.motion_model._eval_xyz_coeffs(
                            self.motion_model.xyz_ctrls, 
                            self.motion_model.xyz_basis_scales
                        )
                    
                    # Process each frame in batch (for GPU memory management)
                    for i in range(real_batch_size):
                        frame_id = cur_frame_ids[i]
                        
                        # Get motion field for this frame
                        motion_field_frame = self.motion_model.forward([frame_id])
                        
                        # Warp image
                        S_warped = self._warp_img(self.S, motion_field_frame)
                        
                        # Forward model: image -> k-space
                        k_pred = self._forward_model(
                            S_warped.unsqueeze(0),
                            self.coords_frames_t[frame_id].unsqueeze(0),
                            self.csm
                        ) * self.scaling_per_frame[frame_id]
                        
                        # Data consistency loss with edge weighting
                        frame_loss = self._compute_dc_loss(
                            k_pred,
                            self.xk_frames_t[frame_id].unsqueeze(0),
                            self.dcf_frames_t[frame_id].unsqueeze(0),
                            self.coords_frames_t[frame_id].unsqueeze(0),
                            max_radius_all,
                            real_batch_size
                        )
                        
                        # Backprop
                        frame_loss.backward(retain_graph=False)
                        batch_loss += frame_loss
                        
                        # Memory management
                        del motion_field_frame, S_warped, k_pred, frame_loss
                        if i % 20 == 0:
                            gc.collect()
                            torch.cuda.empty_cache()
                    
                    # Gradient clipping and optimizer step
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5000.0)
                    self.optimizer.step()
                    
                    epoch_loss += batch_loss.item()
                    del batch_loss
                
                # Log epoch loss
                self.dc_loss_log.append(epoch_loss)
                
                # Store best parameters
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.best_epoch = epoch
                    self.best_params = self._save_current_params()
                
                # Logging
                if epoch % 1 == 0:
                    pbar.set_postfix({
                        'DC Loss': f'{epoch_loss:.4e}',
                        'Best': f'{self.best_loss:.4e} (epoch {self.best_epoch})'
                    })
                
                # Early stopping
                if previous_loss != 0:
                    relative_change = abs(previous_loss - epoch_loss) / previous_loss if previous_loss != float('inf') else float('inf')
                else:
                    relative_change = 0
                
                if relative_change < relative_change_threshold:
                    no_improvement_epochs += 1
                else:
                    no_improvement_epochs = 0
                
                if no_improvement_epochs >= patience:
                    if self.verbose:
                        logger.info(f"Early stopping at epoch {epoch} due to minimal relative change in loss.")
                    break
                
                previous_loss = epoch_loss
        
        except KeyboardInterrupt:
            if self.verbose:
                logger.info(f"Training interrupted by user at epoch {epoch}")
        
        torch.cuda.empty_cache()
        
        if self.verbose:
            logger.info(f"Returning best parameters from epoch {self.best_epoch} with loss {self.best_loss:.4e}")
        
        return {
            'best_params': self.best_params,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'dc_loss_log': self.dc_loss_log
        }
    
    def _warp_img(self, S: torch.Tensor, motion_field_batch: torch.Tensor) -> torch.Tensor:
        """
        Warp image using motion field.
        
        Args:
            S (torch.Tensor): Image to warp (nx, ny, nz), complex.
            motion_field_batch (torch.Tensor): Motion field (batch_size, nx, ny, nz, 3).
        
        Returns:
            torch.Tensor: Warped image (batch_size, nx, ny, nz), complex.
        """
        import interpol
        from interpol.api import affine_grid
        
        B = motion_field_batch.shape[0]
        imshape = S.shape
        
        # Create coordinate grid and apply motion
        new_coords = affine_grid(torch.eye(4, device=self.device), imshape)
        new_coords = new_coords - torch.tensor(imshape, device=self.device).view(1, 1, 1, 3) // 2
        new_coords = new_coords[None, ...] + motion_field_batch
        new_coords = new_coords + torch.tensor(imshape, device=self.device).view(1, 1, 1, 1, 3) // 2
        
        # Warp real and imaginary parts separately
        S_real = S.real[None, None, ...].expand(B, -1, -1, -1, -1)
        S_imag = S.imag[None, None, ...].expand(B, -1, -1, -1, -1)
        
        img_warped_re = interpol.grid_push(S_real, new_coords, interpolation=1)
        img_warped_im = interpol.grid_push(S_imag, new_coords, interpolation=1)
        
        S_warped = torch.squeeze(img_warped_re + 1j * img_warped_im)
        return S_warped
    
    def _forward_model(self, S_warped_batch: torch.Tensor, coords_batch: torch.Tensor,
                       csm: torch.Tensor) -> torch.Tensor:
        """
        Forward model: apply coil sensitivities and NUFFT.
        
        Args:
            S_warped_batch (torch.Tensor): Warped images (batch_size, nx, ny, nz).
            coords_batch (torch.Tensor): K-space coordinates (batch_size, n_spokes, n_samples, 3).
            csm (torch.Tensor): Coil sensitivity maps (n_coils, nx, ny, nz).
        
        Returns:
            torch.Tensor: K-space data (batch_size, n_coils, n_spokes, n_samples).
        """
        # Apply coil sensitivity maps
        S_warped_batch = S_warped_batch.unsqueeze(1) * csm.unsqueeze(0)
        
        # Reshape coordinates for NUFFT
        f, spokes, samp, dim = coords_batch.shape
        coords_batch = coords_batch.reshape(f, -1, dim).permute(0, 2, 1)
        coords_batch = coords_batch / S_warped_batch.shape[-1] * 2 * torch.pi
        
        # Forward NUFFT
        xk_batch = self.kb_nufft(S_warped_batch, coords_batch, norm='ortho')
        xk_batch = xk_batch.view(f, S_warped_batch.shape[1], spokes, samp)
        
        return xk_batch
    
    def _compute_dc_loss(self, k_pred: torch.Tensor, k_target: torch.Tensor,
                         dcf: torch.Tensor, coords: torch.Tensor,
                         max_radius: float, batch_size: int) -> torch.Tensor:
        """
        Compute data consistency loss with density compensation and edge weighting.
        
        Args:
            k_pred (torch.Tensor): Predicted k-space (batch_size, n_coils, n_spokes, n_samples).
            k_target (torch.Tensor): Target k-space (batch_size, n_coils, n_spokes, n_samples).
            dcf (torch.Tensor): Density compensation (batch_size, n_spokes, n_samples).
            coords (torch.Tensor): K-space coordinates (batch_size, n_spokes, n_samples, 3).
            max_radius (float): Maximum k-space radius for normalization.
            batch_size (int): Batch size for normalization.
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Edge weighting (sigmoid decay at outer k-space)
        r = torch.linalg.norm(coords.squeeze(0), dim=-1) / max_radius
        edge_weight = 1 / (1 + torch.exp(40 * (r - 0.9)))
        
        # Combined weighting
        dcf = dcf.unsqueeze(1)
        combined_weight = dcf * edge_weight.unsqueeze(0).unsqueeze(1)
        
        # Weighted L2 loss
        loss = torch.sum(combined_weight * torch.abs(k_pred - k_target)**2) / batch_size
        
        return loss
    
    def _save_current_params(self) -> Dict:
        """
        Save current model parameters.
        
        Returns:
            Dict: Dictionary containing current parameter values.
        """
        params = {}
        
        if self.motion_model.xyz_ctrls is not None:
            params['xyz_ctrls'] = {k: v.clone().detach().cpu() for k, v in self.motion_model.xyz_ctrls.items()}
        if self.motion_model.xyz_coeffs is not None:
            params['xyz_coeffs'] = self.motion_model.xyz_coeffs.clone().detach().cpu()
        
        if self.motion_model.t_ctrls is not None:
            params['t_ctrls'] = {k: v.clone().detach().cpu() for k, v in self.motion_model.t_ctrls.items()}
        if self.motion_model.t_coeffs is not None:
            params['t_coeffs'] = self.motion_model.t_coeffs.clone().detach().cpu()
        
        if self.motion_model.mixing is not None:
            params['mixing'] = self.motion_model.mixing.clone().detach().cpu()
        
        return params
