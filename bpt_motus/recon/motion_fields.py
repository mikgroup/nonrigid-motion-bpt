import torch
import numpy as np
import interpol
from interpol.api import affine_grid
import os
import json
import logging
import sys

from ..motion.bsplines import MotionFieldModel

logger = logging.getLogger(__name__)

class MotionFieldWarp:
    def __init__(self, 
                 model_dir: str | None = None, 
                 ref_dir: str | None = None, 
                 out_dir: str | None = None, 
                 bpts_dir: str | None = None,
                 ref_file: str | None = None, 
                 ref_type: str = "s_target",
                 params_file: str = "optimization_params.json",
                 mode: str = "grid_push", 
                 verbose: bool = True, 
                 force_reload: bool = False, 
                 device: str = None):
        
        # Core settings
        self.model_dir: str | None = model_dir # Directory containing the trained motion model parameters and configuration.
        self.ref_dir: str | None = ref_dir # Directory containing the reference image file (e.g., calib, inf, or no_motion directories).
        self.out_dir: str | None = out_dir if out_dir is not None else model_dir # Directory where output warped frames will be saved.
        self.bpts_dir: str | None = bpts_dir # Directory containing B+PT signal files, if applicable for motion field construction.
        self.ref_type: str = ref_type # Reference type selection ('s_target', 'pics', or 'custom').
        self.mode: str = mode # Warping interpolation mode ('grid_push' or 'grid_pull').
        self.verbose: bool = verbose # Whether to enable verbose log outputs.
        self.force_reload: bool = force_reload # Force recalculation even if the output file already exists.
        self.device: str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")  # Compute device ('cpu' or 'cuda').

        # Dynamic reference filename resolution based on previous class parameters
        if ref_file is None:
            if self.ref_type == "s_target":
                self.ref_file: str = "S_target.npy"
            elif self.ref_type == "pics":
                self.ref_file: str = "pics_frames.npy"
            elif self.ref_type == "custom":
                raise ValueError("ref_file must be explicitly specified when ref_type is set to 'custom'.")
            else:
                raise ValueError(f"Unknown ref_type: '{self.ref_type}'. Must be 's_target', 'pics', or 'custom'.")
        else:
            self.ref_file: str = ref_file

        # Filenames
        self.config_fname: str | None = os.path.join(model_dir, params_file) if model_dir else None # optimization parameters used in motion model training
        self.mf_fname: str | None = os.path.join(model_dir, "motion_fields.npy") if model_dir else None # Optional precomputed motion field file path
        self.ref_fname: str | None = os.path.join(self.ref_dir, self.ref_file) if self.ref_dir else None # Reference image file path
        self.bpts_fname: str | None = os.path.join(bpts_dir, "bpts_frames.npy") if bpts_dir else None # Optional B+PT signals file path, for motion field construction
        self.save_fname: str | None = os.path.join(self.out_dir, f"warped_{ref_type}_frames.npy") if self.out_dir else None # Output file for warped frames

        # Attributes to be populated sequentially
        self.reference_frame: np.ndarray | None = None
        self.opt_params: dict = {}
        self.full_motion_field: np.ndarray | None = None
        self.warped_frames: np.ndarray | None = None
        self.bpt_frames: np.ndarray | None = None

    def run(self):
        """
        Warp the reference image with the fully sampled motion fields.

        Stores:
        warped_frames (torch.Tensor): Complex array of motion-corrected/warped image volumes. (Shape: (Nframes, Nx, Ny, Nz))
        full_motion_field (torch.Tensor): Dense spatial displacement coordinates. (Shape: (Nframes, Nx, Ny, Nz, 3))
        """
        if not self.force_reload and os.path.exists(self.save_fname) and os.path.exists(self.mf_fname):
            if self.verbose:
                logger.info(f"Loading existing warped frames from {self.save_fname} and motion fields from {self.mf_fname}.")
            self.warped_frames = torch.from_numpy(np.load(self.save_fname)).to(dtype=torch.complex64, device=self.device)
            self.full_motion_field = torch.from_numpy(np.load(self.mf_fname)).to(dtype=torch.float32, device=self.device)
            return

        self._load_data() # Loads opt_params, reference_frame, and full_motion_field (either from disk or by constructing from params)

        if self.verbose:
            logger.info("Running Motion Field Warping (run_reconstruction logic)...")

        self._run_reconstruction() # Warps reference_frame according to full_motion_field and stores in self.warped_frames

        if self.save_fname:
            os.makedirs(self.out_dir, exist_ok=True)
            np.save(self.save_fname, self.warped_frames.cpu().numpy())
            np.save(self.mf_fname, self.full_motion_field.cpu().numpy())
            if self.verbose:
                logger.info(f"Saved warped frames to {self.save_fname} and motion fields to {self.mf_fname}.")

    def _load_data(self):
        """
        Load configuration parameters, reference frames, and motion fields from disk.

        Stores:
        opt_params (dict): Optimization configuration dictionary.
        reference_frame (torch.Tensor): The resolved complex 3D reference image volume. (Shape: (Nx, Ny, Nz))
        full_motion_field (torch.Tensor): Fully sampled dense displacement array. (Shape: (Nframes, Nx, Ny, Nz, 3))
        """
        # 1. Load optimization parameters
        if self.config_fname and os.path.exists(self.config_fname):
            with open(self.config_fname, 'r') as f:
                self.opt_params = json.load(f)
                if 'mode' in self.opt_params:
                    self.mode = self.opt_params['mode']
                if self.verbose:
                    logger.info(f"Loaded optimization config from {self.config_fname}")

        # 2. Load reference frame
        if self.ref_fname and os.path.exists(self.ref_fname):
            if self.verbose:
                logger.info(f"Loading reference from {self.ref_fname} (type: {self.ref_type})")
            loaded_ref = np.load(self.ref_fname)
            
            if self.ref_type == "pics":
                self.reference_frame = loaded_ref[0]
            else:
                self.reference_frame = loaded_ref
            self.reference_frame = torch.from_numpy(self.reference_frame).to(dtype=torch.complex64, device=self.device)
        else:
            logger.warning(f"Reference file not found at {self.ref_fname}")

        # 3. Load or construct full_motion_field
        if self.mf_fname and os.path.exists(self.mf_fname):
            if self.verbose:
                logger.info(f"Loading fully sampled motion fields from {self.mf_fname}")
            self.full_motion_field = torch.from_numpy(np.load(self.mf_fname)).to(dtype=torch.float32, device=self.device)
        else:
            if self.verbose:
                logger.info("Constructing motion fields from best_params...")
            self._construct_motion_fields()

    def _construct_motion_fields(self):
        """
        Construct the fully sampled motion fields using the saved best parameters from optimization.

        Stores:
        bpt_frames (np.ndarray | None): Loaded physiological navigation references. (Shape: (Nframes, nrank))
        full_motion_field (torch.Tensor): Generated dense displacement field. (Shape: (Nframes, Nx, Ny, Nz, 3))
        """
        im_shape = self.reference_frame.shape
        n_frames = self.opt_params.get("n_frames")
    
        if n_frames is None:
            logger.warning("n_frames not found, cannot construct motion model automatically.")
            return

        if "bpt" in self.opt_params.get("mode", ""):
            if self.bpts_fname and os.path.exists(self.bpts_fname):
                self.bpt_frames = np.load(self.bpts_fname)

        try:
            motion_model = MotionFieldModel(
                im_shape=list(im_shape),
                n_frames=n_frames,
                mode=self.opt_params.get("mode"),
                xyz_downsampling=self.opt_params.get("xyz_downsampling"),
                t_downsampling=self.opt_params.get("t_downsampling"),
                n_mfcomponents=self.opt_params.get("n_mfcomponents"),
                max_disp_frac=self.opt_params.get("max_disp_frac"),
                max_t_init=self.opt_params.get("max_t_init"),
                bpt_frames=self.bpt_frames,
                verbose=self.verbose,
                device=self.device
            )

            for name, param in motion_model.get_trainable_parameters().items():
                pt_path = os.path.join(self.model_dir, f"{name}.pt")
                if os.path.exists(pt_path):
                    param.data.copy_(torch.load(pt_path, map_location=self.device))
                else:
                    logger.warning(f"Parameter file {pt_path} not found.")

            with torch.no_grad():
                self.full_motion_field = motion_model.forward(np.arange(n_frames).tolist()).to(device=self.device)
            
            if self.verbose:
                logger.info(f"Constructed full_motion_field of shape {self.full_motion_field.shape}")
                
        except Exception as e:
            logger.error(f"Failed to construct full motion field from params: {e}")
        

    def _run_reconstruction(self):
        """
        Warps `self.reference_frame` according to `self.full_motion_field`.

        Stores:
        warped_frames (torch.Tensor): Complex array of motion-corrected/warped image volumes. (Shape: (Nframes, Nx, Ny, Nz))
        """
        nframes = self.full_motion_field.shape[0]
        nx, ny, nz = self.reference_frame.shape

        self.warped_frames = torch.zeros((nframes, nx, ny, nz), dtype=torch.complex64, device=self.device)
        
        for cur_frame_idx in range(nframes):
            new_coords = affine_grid(np.eye(4), self.reference_frame.shape).to(dtype=torch.float32, device=self.device)
            new_coords = new_coords - torch.tensor(self.reference_frame.shape, device=self.device).view(1, 1, 1, 3) // 2
            
            new_coords = new_coords + self.full_motion_field[cur_frame_idx]
            new_coords = new_coords + torch.tensor(self.reference_frame.shape, device=self.device).view(1, 1, 1, 3) // 2

            S_real = self.reference_frame.real[None, None, ...]
            S_imag = self.reference_frame.imag[None, None, ...]
            c_new = new_coords[None, ...]
            
            if self.mode == "grid_push":
                img_warped_re = interpol.grid_push(S_real, c_new, interpolation=1)[0, 0, ...]
                img_warped_im = interpol.grid_push(S_imag, c_new, interpolation=1)[0, 0, ...]
            elif self.mode == "grid_pull":
                img_warped_re = interpol.grid_pull(S_real, c_new, interpolation=1)[0, 0, ...]
                img_warped_im = interpol.grid_pull(S_imag, c_new, interpolation=1)[0, 0, ...]
            else:
                raise ValueError("Mode must be 'grid_push' or 'grid_pull'.")

            self.warped_frames[cur_frame_idx, ...] = img_warped_re + 1j * img_warped_im