"""
PICS reconstruction module for dynamic MRI data.
"""
import numpy as np
import os
import logging
from bart import bart

logger = logging.getLogger(__name__)

class PICSRecon:
    def __init__(self, 
                 motion_frames_dir: str, 
                 no_motion_dir: str, 
                 save_dir: str | None = None, 
                 save_file: str = "pics_frames.npy", 
                 lamda: float = 1e-2, 
                 max_iter: int = 50, 
                 use_gpu: bool = True, 
                 verbose: bool = True, 
                 force_reload: bool = False):
        
        # Core settings
        self.motion_frames_dir: str = motion_frames_dir # Motion frames directory (can be calib or inference dir)
        self.no_motion_dir: str = no_motion_dir # No motion directory
        self.save_dir: str = save_dir if save_dir is not None else motion_frames_dir # Directory to save output
        self.lamda: float = lamda # Regularization parameter for PICS
        self.max_iter: int = max_iter # Maximum iterations for BART
        self.use_gpu: bool = use_gpu # Whether to use GPU in BART
        self.verbose: bool = verbose 
        self.force_reload: bool = force_reload 

        # Filenames
        self.xk_frames_fname: str = os.path.join(self.motion_frames_dir, "xk_frames.npy")
        self.csm_fname: str = os.path.join(self.no_motion_dir, "csm_reference.npy")
        self.coords_frames_fname: str = os.path.join(self.motion_frames_dir, "coords_frames.npy")
        self.save_fname: str = os.path.join(self.save_dir, save_file)

        # Attributes to be populated sequentially
        self.xk_frames: np.ndarray | None = None
        self.csm: np.ndarray | None = None
        self.coords_frames: np.ndarray | None = None
        self.pics: np.ndarray | None = None

    def run(self, frame_by_frame: bool = True):
        """
        Execute the PICS reconstruction workflow either sequentially per frame or for all data.

        Args:
        frame_by_frame (bool): If True, loops over and reconstructs individual temporal frames.

        Stores:
        pics (np.ndarray): The full array of reconstructed image frames. (Shape: (Nframes, Nx, Ny, Nz) or (Nx, Ny, Nz))
        """
        if not self.force_reload and os.path.exists(self.save_fname):
            if self.verbose:
                logger.info(f"Loading existing PICS recon from {self.save_fname}")
            self.pics = np.load(self.save_fname)
            return

        self._load_data()

        if self.xk_frames is None or self.csm is None:
            raise ValueError("k-space frames and CSMs were not loaded.")

        if self.verbose:
            logger.info("Running PICS reconstruction...")

        if frame_by_frame and self.xk_frames.ndim > (3 if self.coords_frames is None else 4):
            n_frames = self.xk_frames.shape[1]
            recons = []
            for i in range(n_frames):
                if self.verbose:
                    print(f"Reconstructing frame {i+1}/{n_frames}")
                x_f = self.xk_frames[:,i]
                c_f = self.coords_frames[i] if (self.coords_frames is not None and self.coords_frames.ndim > 3) else self.coords_frames
                recon = self._bart_pics_recon(x_f, c_f)
                recons.append(recon)
            self.pics = np.stack(recons)
        else:
            self.pics = self._bart_pics_recon(self.xk_frames, self.coords_frames)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            np.save(self.save_fname, self.pics)
            if self.verbose:
                logger.info(f"Saved PICS recons to {self.save_dir}")

    def _load_data(self):
        """
        Load k-space data, coil sensitivity maps, and trajectory coordinates from disk.

        Stores:
        xk_frames (np.ndarray | None): Frames of dynamic k-space data. (Shape: (Ncoils, Nframes, Nspokes, Nsamples))
        csm (np.ndarray | None): Coil sensitivity maps reference array. (Shape: (Ncoils, Nx, Ny, Nz))
        coords_frames (np.ndarray | None): Frames of coordinate trajectories. (Shape: (Nframes, Nspokes, Nsamples, 3))
        """
        if self.verbose:
            logger.info("Loading frames and CSM data...")
        self.xk_frames = np.load(self.xk_frames_fname) if os.path.exists(self.xk_frames_fname) else None
        self.csm = np.load(self.csm_fname) if os.path.exists(self.csm_fname) else None
        self.coords_frames = np.load(self.coords_frames_fname) if os.path.exists(self.coords_frames_fname) else None

    def _bart_pics_recon(self, xk, coords):
        """
        Call the BART PICS command to compute a regularized compressed sensing reconstruction.

        Args:
        xk (np.ndarray): Input k-space data to be reconstructed. (Shape: (Ncoils, Nspokes, Nsamples) or (Ncoils, Nframes, Nspokes, Nsamples))
        coords (np.ndarray | None): Coordinates per k-space frame. (Shape: (Nframes, Nspokes, Nsamples, 3))

        Returns:
        recon (np.ndarray): Reconstructed complex spatial image array volume. (Shape: (Nx, Ny, Nz) or (Nframes, Nx, Ny, Nz))
        """
        base_cmd = f"pics -i {self.max_iter} -R T:7:0:{self.lamda} -t"
        
        if self.use_gpu:
            base_cmd += " -g"
            
        if coords is not None:
            recon = bart(1, base_cmd, coords, xk, self.csm)
        else:
            recon = bart(1, base_cmd, xk, self.csm)
            
        return recon