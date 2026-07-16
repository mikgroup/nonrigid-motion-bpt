"""
Functions for processing k-space data, in cases with and without motion.
"""
import os
import numpy as np
import pickle as pkl
import torch
from torchkbnufft import KbNufftAdjoint
from bart import bart
import sigpy as sp
import sigpy.mri as mr
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class NoMotionReference:
    """Extracts a reference image and coil sensitivity maps from continuous radial data, in cases without motion."""
    def __init__(self, inp_dir: str, verbose: bool = False, 
                 center_out: bool = False, crop_factor: int = 3):
        self.verbose: bool = verbose
        self.inp_dir: str = inp_dir
        self.save_dir: str = os.path.join(inp_dir, f"crop_{crop_factor}")
        
        self.xk_cleaned_fname: str = os.path.join(self.inp_dir, "xk_cleaned_comp.npy")
        self.coords_raw_fname: str = os.path.join(self.inp_dir, "coords.npy")
        self.dcf_raw_fname: str = os.path.join(self.inp_dir, "dcf.npy")

        self.S_fname: str = os.path.join(self.save_dir, "S_reference.npy")
        self.csm_fname: str = os.path.join(self.save_dir, "csm_reference.npy")
        
        self.oversamp: float = 1.25
        self.crop_factor: int = crop_factor
        self.center_out: bool = center_out

        self.xk: np.ndarray | torch.Tensor | None = None
        self.coords: np.ndarray | torch.Tensor | None = None
        self.dcf: np.ndarray | torch.Tensor | None = None
        self.xk_cart: np.ndarray | torch.Tensor | None = None
        self.ref_im: np.ndarray | None = None
        self.adj_nufft: KbNufftAdjoint | None = None
        self.im_size: tuple | None = None
        self.S: np.ndarray | None = None
        self.csm: np.ndarray | None = None

    def run(self, force_reload: bool = False):
        """
        Run the extraction and reconstruction of the reference image and coil sensitivity maps.

        Args:
        force_reload (bool): If True, re-extract and overwrite even if files exist.

        Stores:
        S (np.ndarray): High-resolution reference image. (Shape: (Nx, Ny, Nz))
        csm (np.ndarray): Coil sensitivity maps. (Shape: (Nc, Nx, Ny, Nz))
        """
        if (os.path.exists(self.S_fname) and os.path.exists(self.csm_fname)) and not force_reload:
            logger.info("Reference image and CSMs found. Opening...")
            self.S = np.load(self.S_fname)
            self.csm = np.load(self.csm_fname)
        else:
            logger.info(f"Reference image and CSMs not found. Extracting with crop factor {self.crop_factor}...")
            self._load_data()
            self._crop_spokes()
            self._prep_nufft()
            self._get_ref_xk_cart()
            self._get_ref_csm()
            self._get_S()
            
            os.makedirs(self.save_dir, exist_ok=True)
            np.save(self.S_fname, self.S)
            np.save(self.csm_fname, self.csm)

    def _load_data(self):
        """
        Load continuous radial data arrays from disk.

        Stores:
        xk (np.ndarray): Cleaned k-space data. (Shape: (Nc, Nsp, Nr))
        coords (np.ndarray): Coordinate data. (Shape: (Nsp, Nr, 3))
        dcf (np.ndarray): Density compensation function. (Shape: (Nsp, Nr))
        """
        if self.verbose:
            logger.info("Getting xk, coords, dcf from radial data.")
        self.xk = np.load(self.xk_cleaned_fname)
        self.coords = np.load(self.coords_raw_fname)
        self.dcf = np.load(self.dcf_raw_fname)

    def _crop_spokes(self):
        """
        Crop the ends or centers of radial spokes using the specified crop factor.

        Stores:
        xk (np.ndarray): Cropped k-space data. (Shape: (Nc, Nsp, Nr))
        coords (np.ndarray): Cropped coordinates. (Shape: (Nsp, Nr, 3))
        dcf (np.ndarray): Cropped DCF. (Shape: (Nsp, Nr))
        """
        if self.verbose:
            logger.info(f"Cropping spokes by {self.crop_factor}.")
        if self.crop_factor == 1:
            logger.info("Crop factor is 1, so no cropping applied...")    
            return
            
        if self.center_out:
            ro_off = int((self.coords.shape[1] - self.coords.shape[1] / self.crop_factor))
            logger.info("Center-out cropping selected. Cropping spoke ends, not beginnings.")
            self.xk = self.xk[:, :, :-ro_off]
            self.coords = self.coords[:, :-ro_off]
            self.dcf = self.dcf[:, :-ro_off]
        else:
            ro_off = int((self.coords.shape[1] - self.coords.shape[1] / self.crop_factor) / 2)
            self.xk = self.xk[:, :, ro_off:-ro_off]
            self.coords = self.coords[:, ro_off:-ro_off]
            self.dcf = self.dcf[:, ro_off:-ro_off]

    def _prep_nufft(self):
        """
        Prepare the adjoint NUFFT operator and format input tensors.

        Stores:
        im_size (tuple): Estimated spatial shape of the image.
        adj_nufft (KbNufftAdjoint): Adjoint NUFFT operator.
        coords (torch.Tensor): Flattened and scaled coordinates. (Shape: (1, 3, Nsp*Nr))
        dcf (torch.Tensor): Flattened and scaled DCF. (Shape: (1, Nsp*Nr,))
        xk (torch.Tensor): Flattened k-space. (Shape: (1, Nc, Nsp*Nr))
        """
        if self.verbose:
            logger.info("Preparing adjoint NUFFT operator and inputs.")
        self.im_size = sp.fourier.estimate_shape(self.coords)
        grid_size = self._get_grid_size()
        self.adj_nufft = KbNufftAdjoint(im_size=self.im_size, grid_size=grid_size)

        self.coords = self.coords / self.im_size[0] * 2 * np.pi
        self.dcf = self.dcf / self.dcf.sum() * np.prod(self.im_size)

        self.xk = torch.tensor(self.xk).view(self.xk.shape[0], -1).unsqueeze(0)
        self.coords = torch.tensor(self.coords).view(-1,3).permute(1,0).unsqueeze(0)
        self.dcf = torch.tensor(self.dcf).view(-1).unsqueeze(0)

    def _get_grid_size(self):
        """
        Calculate the oversampled grid size for the NUFFT algorithm.

        Returns:
        grid_size (torch.Tensor): Rounded integer elements of the grid size.
        """
        grid_size = torch.round(torch.tensor(self.im_size)*self.oversamp).to(torch.int64).cpu()
        return grid_size
        
    def _get_ref_xk_cart(self):
        """
        Reconstruct a Cartesian k-space reference via adjoint NUFFT.

        Stores:
        ref_im (np.ndarray): Image volume constructed from adjoint NUFFT. (Shape: (Nx, Ny, Nz))
        xk_cart (np.ndarray): Cartesian k-space derived from reference image. (Shape: (Nc, Nx, Ny, Nz))
        """
        if self.verbose:
            logger.info("Getting cartesian k-space of reference.")
        self.ref_im = self.adj_nufft(self.xk * self.dcf, self.coords, norm="ortho").numpy()
        self.xk_cart = sp.fft(self.ref_im, axes=(-3,-2,-1), norm="ortho")[0]

    def _get_ref_csm(self):
        """
        Estimate coil sensitivity maps using BART and the Cartesian k-space.

        Stores:
        csm (np.ndarray): Coil sensitivity maps. (TODO: get shape)
        """
        if self.verbose:
            logger.info("Getting coil sensitivity maps of reference.")
        self.csm = bart(1, "ecalib -d2 -m1 -c0 -S", self.xk_cart.transpose(1,2,3,0)).transpose(3,0,1,2)

    def _get_S(self):
        """
        Reconstruct the final SENSE reference image using SigPy.

        Stores:
        S (np.ndarray): Reconstructed reference image. (TODO: get shape)
        """
        if self.verbose:
            logger.info("Getting reference image.")
        recon = mr.app.SenseRecon(
            self.xk_cart, mps=self.csm, lamda=0.001, max_iter=30, show_pbar=self.verbose
        )
        self.S = recon.run()

class MotionFrames:
    """Splits continuous radial data (and B+PT signals) into overlapping temporal frames, for use in motion estimation and motion-resolved reconstruction."""
    def __init__(self, inp_dir: str, verbose: bool = False, 
                 center_out: bool = False, spokes_per_frame: int = 500, 
                 stride: int = 100, crop_factor: int = 8):
        self.verbose: bool = verbose
        self.inp_dir: str = inp_dir
        self.save_dir: str = os.path.join(inp_dir, f"crop_{crop_factor}")
        
        self.xk_cleaned_fname: str = os.path.join(self.inp_dir, "xk_cleaned_comp.npy")
        self.coords_raw_fname: str = os.path.join(self.inp_dir, "coords.npy")
        self.dcf_raw_fname: str = os.path.join(self.inp_dir, "dcf.npy")
        self.bpts_proc_fname: str = os.path.join(self.inp_dir, "bpts_proc.npy")

        self.xk_fname: str = os.path.join(self.save_dir, "xk_frames.npy")
        self.coords_fname: str = os.path.join(self.save_dir, "coords_frames.npy")
        self.dcf_fname: str = os.path.join(self.save_dir, "dcf_frames.npy")
        self.bpts_fname: str = os.path.join(self.inp_dir, "bpts_frames.npy")
        self.frames_center_spokes_fname: str = os.path.join(self.save_dir, "frames_center_spokes.npy")
        self.params_fname: str = os.path.join(self.save_dir, "motion_frames_processing_params.pkl")

        self.spokes_per_frame: int = spokes_per_frame
        self.stride: int = stride
        self.crop_factor: int = crop_factor
        self.center_out: bool = center_out

        self.xk: np.ndarray | None = None
        self.coords: np.ndarray | None = None
        self.dcf: np.ndarray | None = None
        self.bpts: np.ndarray | None = None
        self.xk_frames: np.ndarray | None = None
        self.coords_frames: np.ndarray | None = None
        self.dcf_frames: np.ndarray | None = None
        self.bpts_frames: np.ndarray | None = None
        self.frames_center_spokes: np.ndarray | None = None

    def run(self, force_reload: bool = False):
        """
        Split the radial data acquisition into overlapping temporal frames.

        Args:
        force_reload (bool): If True, re-extract and overwrite even if files exist.

        Stores:
        xk_frames (np.ndarray): Framed k-space data. (Shape: (Nc, Nframes, spokes_per_frame, Nr))
        coords_frames (np.ndarray): Framed coordinate data. (Shape: (Nframes, spokes_per_frame, Nr, 3))
        dcf_frames (np.ndarray): Framed density compensation functions. (Shape: (Nframes, spokes_per_frame, Nr))
        bpts_frames (np.ndarray): Framed navigation points. (Shape: (Nframes, nrank)) 
        frames_center_spokes (np.ndarray): Center spoke indices for each frame. (Shape: (Nframes,))
        """
        if (os.path.exists(self.xk_fname) and os.path.exists(self.coords_fname) 
            and os.path.exists(self.dcf_fname)) and not force_reload:
            logger.info("Radial acquisition split into frames found. Opening...")
            self.xk_frames = np.load(self.xk_fname)
            self.coords_frames = np.load(self.coords_fname)
            self.dcf_frames = np.load(self.dcf_fname)
            try:
                self.bpts_frames = np.load(self.bpts_fname)
            except:
                logger.warning("BPT/PTs frames not found.")
        else:
            logger.info(f"Radial acquisition split into frames not found. Extracting with crop factor {self.crop_factor}...")
            self._load_data()
            self._crop_spokes()
            self._split_frames()
            
            os.makedirs(self.save_dir, exist_ok=True)
            np.save(self.xk_fname, self.xk_frames)
            np.save(self.coords_fname, self.coords_frames)
            np.save(self.dcf_fname, self.dcf_frames)
            if self.bpts_frames is not None:
                np.save(self.bpts_fname, self.bpts_frames)
            np.save(self.frames_center_spokes_fname, self.frames_center_spokes)

    def save_processing_params(self):
        """
        Save sliding window frame and cropping parameters to a pickle file.
        """
        params = {
            "spokes_per_frame": self.spokes_per_frame,
            "stride": self.stride,
            "crop_factor": self.crop_factor
        }
        with open(self.params_fname, "wb") as f:
            pkl.dump(params, f)
        if self.verbose:
            logger.info(f"Saved processing parameters to {self.params_fname}.")

    def _load_data(self):
        """
        Load continuous radial data arrays and BPT navigation points from disk.

        Stores:
        xk (np.ndarray): Cleaned k-space data. (Shape: (Nc, Nsp, Nr))
        coords (np.ndarray): Coordinate data. (Shape: (Nsp, Nr, 3))
        dcf (np.ndarray): Density compensation function. (Shape: (Nsp, Nr))
        bpts (np.ndarray): Processed BPT/PT navigation signals. (Shape: (Nsp, nrank))
        """
        if self.verbose:
            logger.info("Getting xk, coords, dcf, and bpts from radial data.")
        self.xk = np.load(self.xk_cleaned_fname)
        self.coords = np.load(self.coords_raw_fname)
        self.dcf = np.load(self.dcf_raw_fname)
        try:
            self.bpts = np.load(self.bpts_proc_fname)
        except:
            self.bpts = None
            logger.warning("Processed BPT/PTs not found.")

    def _crop_spokes(self):
        """
        Crop the ends of radial spokes using the specified crop factor.

        Stores:
        xk (np.ndarray): Cropped k-space data. (Shape: (Nc, Nsp, Nr))
        coords (np.ndarray): Cropped coordinates. (Shape: (Nsp, Nr, 3))
        dcf (np.ndarray): Cropped DCF. (Shape: (Nsp, Nr))
        """
        if self.verbose:
            logger.info(f"Cropping spokes by {self.crop_factor}.")
        if self.crop_factor == 1:
            logger.info("Crop factor is 1, so no cropping applied...")    
            return
        if self.center_out:
            ro_off = int((self.coords.shape[1] - self.coords.shape[1] / self.crop_factor))
            logger.info("Center-out cropping selected. Cropping spoke ends, not beginnings.")
            self.xk = self.xk[:, :, :-ro_off]
            self.coords = self.coords[:, :-ro_off]
            self.dcf = self.dcf[:, :-ro_off]
        else:
            ro_off = int((self.coords.shape[1] - self.coords.shape[1] / self.crop_factor) / 2)
            self.xk = self.xk[:, :, ro_off:-ro_off]
            self.coords = self.coords[:, ro_off:-ro_off]
            self.dcf = self.dcf[:, ro_off:-ro_off]

    def _split_frames(self):
        """
        Split cropped radial arrays into sliding window temporal frames.

        Stores:
        xk_frames (np.ndarray): Framed k-space array. (Shape: (Nc, Nframes, spokes_per_frame, Nr))
        coords_frames (np.ndarray): Framed coordinates. (Shape: (Nframes, spokes_per_frame, Nr, 3))
        dcf_frames (np.ndarray): Framed DCF. (Shape: (Nframes, spokes_per_frame, Nr))
        bpts_frames (np.ndarray): Navigation points sampled at frame centers. (Shape: (Nframes, nrank))
        frames_center_spokes (np.ndarray): Center spoke indices. (Shape: (Nframes,))
        """
        if self.verbose:
            logger.info("Splitting radial data into frames.")
        Nc, Nsp, Nr = self.xk.shape
        starts = np.arange(0, Nsp - self.spokes_per_frame + 1, self.stride)
        self.frames_center_spokes = starts + self.spokes_per_frame // 2
        self.xk_frames = np.stack(
            [self.xk[:, s:s + self.spokes_per_frame] for s in starts],
            axis=1)
        self.coords_frames = np.stack(
            [self.coords[s:s + self.spokes_per_frame] for s in starts],
            axis=0)
        self.dcf_frames = np.stack(
            [self.dcf[s:s + self.spokes_per_frame] for s in starts],
            axis=0)
        if self.bpts is not None:
            self.bpts_frames = self.bpts[self.frames_center_spokes]
