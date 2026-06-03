"""
Classes and functions for optimizing motion fields with BPT-MOTUS and MR-MOTUS.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import gc
import torchkbnufft as tkbn
import interpol
import json

from .bsplines import MotionFieldModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MotionFieldOptimizer:
    def __init__(self, 
                 calib_inpdir: str, 
                 nomotion_inpdir: str, 
                 out_dir: str | None = None,
                 mode: str = 'bpt_rigid',
                 xyz_downsampling: list = [4, 8, 16],
                 t_downsampling: list = [1, 2, 5],
                 n_mfcomponents: int = 6,
                 crop_factor_nomotion: int = 3,
                 crop_factor_motion: int = 8,
                 max_disp_frac: float = 0.05,
                 max_t_init: float = 0.0,
                 epochs: int = 40,
                 batch_size: int = 50,
                 learning_rate: float = 5e-2,
                 patience: int = 4,
                 lambda_l1: float = 0.0,
                 lambda_tv: float = 0.0,
                 lambda_disp: float = 0.0,
                 verbose: bool = False, 
                 device: str | None = None):
        # Core settings
        self.calib_inpdir: str = calib_inpdir # Directory containing calibration/motion data.
        self.nomotion_inpdir: str = nomotion_inpdir # Directory containing the static no-motion reference data.
        self.verbose: bool = verbose 
        self.device: str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")  # Compute device ('cpu' or 'cuda').
        self.mode: str = mode # Optimization mode (e.g., 'bpt_rigid', 'mrmotus').
        self.out_dir: str = os.path.join(calib_inpdir, mode) if out_dir is None else out_dir # Directory to save outputs. Defaults to a subfolder in calib_inpdir.
        
        # Hyperparameters
        self.epochs: int = epochs # Number of training epochs.
        self.batch_size: int = batch_size # Batch size (number of frames per gradient step).
        self.learning_rate: float = learning_rate # Adam optimizer learning rate.
        self.patience: int = patience # Early stopping patience epochs.
        self.lambda_l1: float = lambda_l1 # L1 regularization weight.
        self.lambda_tv: float = lambda_tv # Total Variation regularization weight.
        self.lambda_disp: float = lambda_disp # Displacement penalty weight.
        self.xyz_downsampling: list = xyz_downsampling # Spatial B-spline downsampling factors.
        self.t_downsampling: list = t_downsampling # Temporal B-spline downsampling factors.
        self.n_mfcomponents: int = n_mfcomponents # Number of motion field components.
        self.max_disp_frac: float = max_disp_frac # Maximum allowed displacement as a fraction of the FOV.
        self.max_t_init: float = max_t_init # Maximum initialization value for temporal components.
        self.oversamp: float = 1.25

        # Filenames
        self.S_target_fname: str = os.path.join(nomotion_inpdir, f"crop_{crop_factor_nomotion}", "S_reference.npy") # crop_factor_nomotion (int): Cropping factor for the no-motion data.
        self.csm_fname: str = os.path.join(nomotion_inpdir, f"crop_{crop_factor_nomotion}", "csm_reference.npy")
        self.xk_frames_fname: str = os.path.join(calib_inpdir, f"crop_{crop_factor_motion}", "xk_frames.npy") # crop_factor_motion (int): Cropping factor for the motion data.
        self.coords_frames_fname: str = os.path.join(calib_inpdir, f"crop_{crop_factor_motion}", "coords_frames.npy")
        self.dcf_frames_fname: str = os.path.join(calib_inpdir, f"crop_{crop_factor_motion}", "dcf_frames.npy")
        self.bpts_frames_fname: str = os.path.join(calib_inpdir, "bpts_frames.npy")
        
        # Attributes to be populated sequentially
        self.S_target: torch.Tensor | None = None
        self.csm: torch.Tensor | None = None
        self.xk_frames: np.ndarray | None = None
        self.coords_frames: np.ndarray | None = None
        self.dcf_frames: np.ndarray | None = None
        self.bpt_frames: np.ndarray | None = None
        
        self.im_shape: tuple | None = None
        self.n_frames: int | None = None
        self.max_disp: float | None = None
        
        self.nufft: tkbn.KbNufft | None = None
        self.motion_model: MotionFieldModel | None = None
        self.scaling_per_frame: np.ndarray | None = None
        self.best_params: dict | None = None
        
        # Optimization logs
        self.dc_loss_log: list = []
        self.l1_loss_log: list = []
        self.total_loss_log: list = []

    def optimize(self):
        """
        Run the motion field training loop to fit the motion model to acquired k-space chunks.

        Stores:
        Optimal parameters from best_params (dict): Dictionary mapping parameter names to their optimal tensors.
        Loss logs per epoch and optimization parameters for reproducibility
        """
        # Execute Setup Steps
        if self.xk_frames is None:
            self._load_data()
        if self.motion_model is None:
            self._setup_models()
        if self.scaling_per_frame is None:
            if self.verbose: 
                logger.info("Calculating scaling per frame dynamically...")
            self.scaling_per_frame = self._compute_scaling_per_frame()

        xk_frames_t = torch.from_numpy(self.xk_frames).to(self.device).to(torch.complex64)
        coords_frames_t = torch.from_numpy(self.coords_frames).to(self.device).to(torch.float32)
        dcf_frames_t = torch.from_numpy(self.dcf_frames).to(self.device).to(torch.float32)
        
        max_radius_all = max([torch.linalg.norm(c, dim=-1).max().item() for c in coords_frames_t])
        
        # Get parameter keys and values mapped securely over training scope
        learnable_params_dict = self.motion_model.get_trainable_parameters()
        param_names = list(learnable_params_dict.keys())
        learnable_params = list(learnable_params_dict.values())
        
        optimizer = torch.optim.Adam(learnable_params, lr=self.learning_rate)
        
        best_loss, previous_loss = float('inf'), float('inf')
        no_improvement_epochs, best_epoch = 0, 0
        relative_change_threshold = 1e-3
        
        # Clear logs in case of re-running
        self.dc_loss_log.clear()
        self.l1_loss_log.clear()
        self.total_loss_log.clear()
        
        pbar = tqdm(range(self.epochs), disable=not self.verbose)
        try:
            for epoch in pbar:
                epoch_dc, epoch_l1, epoch_total = 0.0, 0.0, 0.0
                
                frame_idxes = np.arange(self.n_frames)
                np.random.shuffle(frame_idxes)
                num_batches = int(np.ceil(self.n_frames / self.batch_size))
                
                for batch_idx in range(num_batches):
                    optimizer.zero_grad(set_to_none=True)
                    batch_dc_loss, batch_l1_loss, batch_loss = 0.0, 0.0, 0.0
                    
                    cur_frames = frame_idxes[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
                    real_batch_size = len(cur_frames)
                    
                    accumulated_grads = {name: torch.zeros_like(p) for name, p in zip(param_names, learnable_params)}
                    self.motion_model.xyz_coeffs = None # Ensure re-eval on step
                    
                    for i, frame_id in enumerate(cur_frames):
                        xk_frame = xk_frames_t[:,frame_id].unsqueeze(0)
                        coords_frame = coords_frames_t[frame_id].unsqueeze(0)
                        dcf_frame = dcf_frames_t[frame_id].unsqueeze(0).unsqueeze(1)
                        
                        mf_frame_batch = self.motion_model.forward([frame_id])
                        
                        # Use class attributes instead of passing S_target and csm
                        S_warped = self._warp_img(mf_frame_batch)
                        k_pred = self._forward_model(S_warped, coords_frame) * self.scaling_per_frame[frame_id]
                        
                        r_frame = torch.linalg.norm(coords_frame.squeeze(0), dim=-1) / max_radius_all
                        edge_weight = 1 / (1 + torch.exp(40 * (r_frame - 0.9)))
                        combined_weight = (dcf_frame.squeeze(1) * edge_weight.unsqueeze(0)).unsqueeze(1)
                        
                        frame_dc = torch.sum(combined_weight * torch.abs(k_pred - xk_frame)**2) / real_batch_size
                        frame_l1 = self.lambda_l1 * torch.sum(torch.abs(mf_frame_batch)) / mf_frame_batch.numel() / real_batch_size
                        
                        frame_loss = frame_dc + frame_l1
                        
                        if self.lambda_tv > 0: frame_loss += self.lambda_tv * self._tv_loss(mf_frame_batch)
                        if self.lambda_disp > 0: frame_loss += self.lambda_disp * self._displacement_loss(mf_frame_batch)
                        
                        current_grads = torch.autograd.grad(frame_loss, learnable_params, retain_graph=(i < real_batch_size - 1), allow_unused=True)
                        for name, grad in zip(param_names, current_grads):
                            if grad is not None:
                                accumulated_grads[name] += grad
                        
                        batch_dc_loss += frame_dc.item()
                        batch_l1_loss += frame_l1.item()
                        batch_loss += frame_loss.item()
                        
                        del mf_frame_batch, S_warped, k_pred, edge_weight, combined_weight, frame_dc, frame_loss, current_grads
                        if i % 2 == 0:
                            gc.collect()
                            torch.cuda.empty_cache()
                    
                    for name, param in zip(param_names, learnable_params):
                        if param.grad is None:
                            param.grad = accumulated_grads[name]
                        else:
                            param.grad += accumulated_grads[name]
                    
                    torch.nn.utils.clip_grad_norm_(learnable_params, max_norm=5000.0)
                    optimizer.step()
                    
                    epoch_dc += batch_dc_loss
                    epoch_l1 += batch_l1_loss
                    epoch_total += batch_loss
                
                self.dc_loss_log.append(epoch_dc)
                self.l1_loss_log.append(epoch_l1)
                self.total_loss_log.append(epoch_total)
                
                if epoch_total < best_loss:
                    best_loss = epoch_total
                    best_epoch = epoch
                    self.best_params = {name: p.clone().detach().cpu() for name, p in zip(param_names, learnable_params)}
                
                if self.verbose:
                    logger.info(f"Epoch {epoch:03d} | DC: {epoch_dc:.4e} | L1: {epoch_l1:.4e} | Best: {best_loss:.4e} (Epoch {best_epoch})")
                
                if previous_loss != float('inf'):
                    rel_change = abs(previous_loss - epoch_total) / previous_loss
                    if rel_change < relative_change_threshold:
                        no_improvement_epochs += 1
                    else:
                        no_improvement_epochs = 0
                
                if no_improvement_epochs >= self.patience:
                    if self.verbose:
                        logger.info(f"Stopping early at epoch {epoch} due to minimal relative change.")
                    break
                previous_loss = epoch_total
                
        except KeyboardInterrupt:
            logger.warning(f"Training interrupted at Epoch {epoch}. Get best parameters from epoch {best_epoch} with loss {best_loss:.4e} at self.best_params.")
        
        torch.cuda.empty_cache()
        if self.verbose:
            logger.info(f"Saving best parameters from epoch {best_epoch} and logs to {self.out_dir}...")
            
        self._save_results()

    # ================== Initialization Helper Methods ==================
    def _load_data(self):
        """
        Loads reference image, sensitivity maps, and dynamic k-space frames from disk.

        Stores:
        S_target (torch.Tensor): Target no-motion image. (Shape: (nx, ny, nz))
        csm (torch.Tensor): Coil sensitivity maps. (Shape: (ncoils, nx, ny, nz))
        xk_frames (np.ndarray): Framed dynamic k-space data. (Shape: (ncoils, nframes, spokes, samples))
        coords_frames (np.ndarray): Framed trajectory coordinates. (Shape: (nframes, spokes, samples, 3))
        dcf_frames (np.ndarray): Framed density compensation functions. (Shape: (nframes, spokes, samples))
        im_shape (tuple): Spatial image dimensions derived from S_target.
        n_frames (int): Number of temporal frames derived from xk_frames.
        max_disp (float): Absolute maximum displacement threshold based on im_shape.
        bpt_frames (np.ndarray | None): BPT navigation frames, if applicable.
        """
        self.S_target = torch.from_numpy(np.load(self.S_target_fname)).to(self.device).to(torch.complex64)
        self.csm = torch.from_numpy(np.load(self.csm_fname)).to(self.device).to(torch.complex64)
        
        self.xk_frames = np.load(self.xk_frames_fname)
        self.coords_frames = np.load(self.coords_frames_fname)
        self.dcf_frames = np.load(self.dcf_frames_fname)

        self.im_shape = self.S_target.shape
        self.n_frames = self.xk_frames.shape[1]
        self.max_disp = self.im_shape[0] * self.max_disp_frac

        if "bpt" in self.mode:
            try:
                self.bpt_frames = np.load(self.bpts_frames_fname)
            except FileNotFoundError:
                logger.warning(f"Mode is {self.mode} but bpts_frames.npy not found in {self.calib_inpdir}")

    def _setup_models(self):
        """
        Initializes the NUFFT operator and B-spline motion field model.

        Stores:
        nufft (tkbn.KbNufft): Non-uniform FFT operator.
        motion_model (MotionFieldModel): Initialized parameterization of the motion fields.
        """
        grid_size = torch.round(torch.tensor(self.im_shape) * self.oversamp).to(torch.int64)
        self.nufft = tkbn.KbNufft(im_size=self.im_shape, grid_size=grid_size).to(self.device)

        self.motion_model = MotionFieldModel(
            im_shape=self.im_shape, 
            n_frames=self.n_frames, 
            mode=self.mode,
            xyz_downsampling=self.xyz_downsampling,
            t_downsampling=self.t_downsampling,
            n_mfcomponents=self.n_mfcomponents,
            max_disp_frac=self.max_disp_frac,
            max_t_init=self.max_t_init,
            bpt_frames=self.bpt_frames,
            verbose=self.verbose,
            device=self.device
        )
        
    def _save_results(self):
        """
        Save the optimal trained parameters and loss histories to the output directory
        using the data stored in the class attributes.
        """
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            for name, p in self.best_params.items():
                torch.save(p, os.path.join(self.out_dir, f"{name}.pt"))
                
            np.save(os.path.join(self.out_dir, "dc_loss.npy"), np.array(self.dc_loss_log))
            np.save(os.path.join(self.out_dir, "l1_loss.npy"), np.array(self.l1_loss_log))
            np.save(os.path.join(self.out_dir, "total_loss.npy"), np.array(self.total_loss_log))
            
            opt_params = {
                "mode": self.mode,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "patience": self.patience,
                "lambda_l1": self.lambda_l1,
                "lambda_tv": self.lambda_tv,
                "lambda_disp": self.lambda_disp,
                "xyz_downsampling": getattr(self.motion_model, "xyz_downsampling", None),
                "t_downsampling": getattr(self.motion_model, "t_downsampling", None),
                "n_mfcomponents": getattr(self.motion_model, "n_mfcomponents", None),
                "max_disp_frac": self.max_disp_frac,
                "max_t_init": getattr(self.motion_model, "max_t_init", 0.0),
                "n_frames": self.n_frames,
                "im_shape": list(self.im_shape)
            }
            with open(os.path.join(self.out_dir, "optimization_params.json"), "w") as f:
                json.dump(opt_params, f, indent=4)
            
    # ================== Loss Helper Methods ==================
    def _tv_loss(self, motion_fields_batch):
        """
        Calculates TV loss. 
        
        Args:
        motion_fields_batch (torch.Tensor): Extracted fields. (Shape: (batch_size, nx, ny, nz, 3))
        
        Returns:
        normalized_grad_sq_sum (torch.Tensor): TV scalar summation.
        """
        epsilon = 1e-8
        mf_batch = motion_fields_batch.permute(0, 4, 1, 2, 3)
        mf_mag_sq = torch.sum(mf_batch**2, dim=1, keepdim=True) + epsilon
        
        dx2 = (mf_batch[:, :, 1:, :, :] - mf_batch[:, :, :-1, :, :])**2
        dy2 = (mf_batch[:, :, :, 1:, :] - mf_batch[:, :, :, :-1, :])**2
        dz2 = (mf_batch[:, :, :, :, 1:] - mf_batch[:, :, :, :, :-1])**2

        min_x = min(dx2.shape[2], dy2.shape[2], dz2.shape[2]) 
        min_y = min(dx2.shape[3], dy2.shape[3], dz2.shape[3]) 
        min_z = min(dx2.shape[4], dy2.shape[4], dz2.shape[4]) 
        
        dx2 = dx2[:, :, :min_x, :min_y, :min_z]
        dy2 = dy2[:, :, :min_x, :min_y, :min_z]
        dz2 = dz2[:, :, :min_x, :min_y, :min_z]
        
        mf_mag_sq_trimmed = mf_mag_sq[:, :, :min_x, :min_y, :min_z]
        normalized_grad_sq = (dx2 + dy2 + dz2) / mf_mag_sq_trimmed
        
        return normalized_grad_sq.sum() / normalized_grad_sq.numel()

    def _displacement_loss(self, motion_fields_batch):
        """
        Penalize squared displacement magnitude exceeding the threshold. 
        
        Args:
        motion_fields_batch (torch.Tensor): Extracted fields. (Shape: (batch_size, nx, ny, nz, 3))
        
        Returns:
        displacement loss (float): Loss due to displacements exceeding the maximum allowed threshold.
        """
        displacement_squared = torch.sum(motion_fields_batch**2, dim=-1)
        excess_displacement_squared = torch.relu(displacement_squared - self.max_disp**2)
        return torch.sum(excess_displacement_squared) / motion_fields_batch.numel()


    # ================== Physics Helper Methods ==================
    def _warp_img(self, motion_field_batch):
        """
        Deform the static reference grid targeting B-splines using self.S_target.
        
        Args:
        motion_field_batch (torch.Tensor): Evaluated coordinates displacement batch. (Shape: (batch_size, nx, ny, nz, 3))
        
        Returns: 
        S_warped (torch.Tensor): warped batch results. (Shape: (batch_size, nx, ny, nz))
        """
        B = motion_field_batch.shape[0]
        imshape = self.S_target.shape
        new_coords = interpol.api.affine_grid(torch.eye(4, device=self.device), imshape)
        new_coords = new_coords - torch.tensor(imshape, device=self.device).view(1, 1, 1, 3) // 2
        new_coords = new_coords[None, ...] + motion_field_batch
        new_coords = new_coords + torch.tensor(imshape, device=self.device).view(1, 1, 1, 1, 3) // 2

        S_real = self.S_target.real[None, None, ...].expand(B, -1, -1, -1, -1)
        S_imag = self.S_target.imag[None, None, ...].expand(B, -1, -1, -1, -1)

        img_warped_re = interpol.grid_push(S_real, new_coords, interpolation=1)
        img_warped_im = interpol.grid_push(S_imag, new_coords, interpolation=1)
        S_warped = torch.squeeze(img_warped_re + 1j * img_warped_im, dim=1)
        return S_warped

    def _forward_model(self, S_warped_batch, coords_batch):
        """
        Gets simulated k-space from the warped baseline No-Motion image.
        
        Args:
        S_warped_batch (torch.Tensor): Source warped grid mappings. (Shape: (batch_size, nx, ny, nz))
        coords_batch (torch.Tensor): Trajectory arrays. (Shape: (batch_size, spokes_per_frame, samples_per_spoke, 3))
        
        Returns:
        xk_lowres_batch (torch.Tensor): Resolving back projections. (Shape: (batch_size, ncoils, spokes_per_frame, samples_per_spoke))
        """
        S_warped_batch = S_warped_batch.unsqueeze(1) * self.csm.unsqueeze(0)
        f, spokes, samp, dim = coords_batch.shape
        coords_batch = coords_batch.reshape(f, -1, dim).permute(0, 2, 1)
        coords_batch = coords_batch / S_warped_batch.shape[-1] * 2 * torch.pi

        xk_lowres_batch = self.nufft(S_warped_batch, coords_batch, norm='ortho')
        xk_lowres_batch = xk_lowres_batch.view(S_warped_batch.shape[0], S_warped_batch.shape[1], spokes, samp)
        return xk_lowres_batch
        
    def _compute_scaling_per_frame(self):
        """
        Dynamically compute scale factors using the baseline No-Motion image 
        to match simulated k_space intensity and acquired k_space intensity (potentially needed due to BPT signal leakage).
        
        Returns:
        scalings (np.ndarray): Scaling factors for each frame. (Shape: (nframes,))
        """
        scalings = []
        S_tensor = self.S_target.unsqueeze(0)
        
        for i in tqdm(range(self.n_frames), desc="Computing scaling", disable=not self.verbose):
            coords = torch.from_numpy(self.coords_frames[i]).to(self.device).to(torch.float32).unsqueeze(0)
            xk_comp = torch.from_numpy(self.xk_frames[:,i]).to(self.device).to(torch.complex64).unsqueeze(0)
            
            with torch.no_grad():
                # self._forward_model uses self.csm intrinsically now
                xk_sim = self._forward_model(S_tensor, coords)
                numerator = (xk_comp * xk_sim.conj()).sum()
                denominator = torch.norm(xk_sim)**2
                scaling_factor = numerator / denominator
                scalings.append(scaling_factor.item())
        scalings = np.array(scalings, dtype=np.complex64)
        return scalings