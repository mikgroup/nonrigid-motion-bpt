"""
Two classes for processing cleaned k-space data — one for the BPT-MOTUS reference image, and one for generating frames with motion, to be resolved by BPT-MOTUS.
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

def load_radial(inp_dir, verbose=False):
    """
    Get radial data.
    Returns: 
        xk (np.ndarray): cleaned k-space (Nc, Nsp, Nr)
        coords (np.ndarray): time-ordered coords (Nsp, Nr, 3)
        dcf (np.ndarray): time-ordered dcf (Nsp, Nr)
        If BPTs are processed: bpts (np.ndarray): processed BPT/PTs (Nsp, nrank); else None
    """
    if verbose:
        logger.info("Getting xk, coords, dcf, and bpts from radial data.")
    xk = np.load(os.path.join(inp_dir, "xk_cleaned_comp.npy"))
    coords = np.load(os.path.join(inp_dir, "coords.npy"))
    dcf = np.load(os.path.join(inp_dir, "dcf.npy"))
    try:
        bpts = np.load(os.path.join(inp_dir, "bpts_proc.npy"))
    except:
        bpts = None
        logger.warning("Processed BPT/PTs not found.")
    return xk, coords, dcf, bpts
    
def crop_spokes(xk, coords, dcf, crop_factor, verbose=False, center_out=False):
    """
    Crop radial spokes according to crop_factor. 
    Stores:
        xk (np.ndarray): cropped cleaned k-space (Nc, Nsp, Nr)
        coords (np.ndarray): cropped time-ordered coords (Nsp, Nr, 3)
        dcf (np.ndarray): cropped time-ordered dcf (Nsp, Nr)
    """
    if verbose:
        logger.info(f"Cropping spokes by {crop_factor}.")
    Nc, Nsp, Nr = xk.shape
    if center_out:
        ro_off = int((coords.shape[1] - coords.shape[1] / crop_factor))
        logger.info("Center-out cropping selected. Cropping spoke ends, not beginnings.")
        xk_crop = xk[:, :, :-ro_off]
        coords_crop = coords[:, :-ro_off]
        dcf_crop = dcf[:, :-ro_off]
    else:
        ro_off = int((coords.shape[1] - coords.shape[1] / crop_factor) / 2)
        xk_crop = xk[:, :, ro_off:-ro_off]
        coords_crop = coords[:, ro_off:-ro_off]
        dcf_crop = dcf[:, ro_off:-ro_off]
    if crop_factor == 1:
        logger.info("Crop factor is 1, so no cropping applied...")    
        xk_crop = xk[:,:,:]
        coords_crop = coords[:,:]
        dcf_crop = dcf[:,:]
    return xk_crop, coords_crop, dcf_crop
    
class NoMotionReference:
    """
    Build a no-motion reference image S and coil sensitivity
    maps from the radial, processed xk / coords / dcf files.
    """
    def __init__(self, inp_dir: str, verbose: bool = False, 
                 center_out: bool = False, crop_factor: int = 3):
        self.verbose: bool = verbose
        self.inp_dir: str = inp_dir
        self.save_dir: str = os.path.join(inp_dir, f"crop_{crop_factor}")
        self.S_fname: str = os.path.join(self.save_dir, "S_reference.npy")
        self.csm_fname: str = os.path.join(self.save_dir, "csm_reference.npy")
        self.S: np.ndarray
        self.csm: np.ndarray
        
        # Internal intermediates
        self.xk: np.ndarray | torch.Tensor
        self.xk_cart: np.ndarray | torch.Tensor
        self.coords: np.ndarray | torch.Tensor
        self.dcf: np.ndarray | torch.Tensor
        self.adj_nufft = None
        self.im_size = None

        # Processing parameters
        self.oversamp: float = 1.25
        self.crop_factor: int = crop_factor
        self.center_out = center_out

    def run(self, force_reload: bool = False):
        """
        Get reference image, from radial k-space without motion.
        Stores and saves:
            S (np.ndarray): high-resolution reference (Nx, Ny, Nz)
            csm (np.ndarray): coil sensitivity maps (Nc, Nx, Ny, Nz)
        """
        if (os.path.exists(self.S_fname) and os.path.exists(self.csm_fname)) and not force_reload:
            logger.info("Reference image and CSMs found. Opening...")
            self.S = np.load(self.S_fname)
            self.csm = np.load(self.csm_fname)
        else:
            logger.info(f"Reference image and CSMs not found. Extracting with crop factor {self.crop_factor}...")
            self.xk, self.coords, self.dcf, _ = load_radial(self.inp_dir, self.verbose)
            self.xk, self.coords, self.dcf = crop_spokes(self.xk, self.coords, self.dcf, self.crop_factor, self.verbose, self.center_out)
            self._prep_nufft()
            self._get_ref_xk_cart()
            self._get_ref_csm()
            self._get_S()
            
            # save
            os.makedirs(self.save_dir, exist_ok=True)
            np.save(self.S_fname, self.S)
            np.save(self.csm_fname, self.csm)
    
    # def _prep_nufft(self):
    #     """
    #     Get adjoint nufft operator and input tensors.
    #     Stores:
    #         adj_nufft: KbNufftAdjoint instance (on default device)
    #         xk (torch.tensor): flattened k-space (1, Nc, Nsp * Nr)
    #         coords (torch.tensor): flattened and permuted coords (3, Nsp * Nr)
    #         dcf (torch.tensor): flattened dcf (1, Nsp * Nr)
    #         im_size: image size (estimated from coords)
    #     """
    #     if self.verbose:
    #         logger.info("Preparing adjoint NUFFT operator and inputs.")
    #     self.im_size = sp.fourier.estimate_shape(self.coords)
    #     grid_size = self._get_grid_size()
    #     self.adj_nufft = KbNufftAdjoint(im_size=self.im_size, grid_size=grid_size)

    #     # Normalize coords and dcf for orthonormal adjoint NUFFT
    #     # scale coords so extent of reference image goes from -pi to pi
    #     self.coords = self.coords / self.im_size[0] * 2 * np.pi
    #     # scale DCF so sum equals image volume
    #     self.dcf = self.dcf / self.dcf.sum() * np.prod(self.im_size)

    #     # get torch tensors
    #     self.xk = torch.tensor(self.xk).view(self.xk.shape[0], -1).unsqueeze(0)
    #     self.coords = torch.tensor(self.coords).view(-1,3).permute(1,0).unsqueeze(0)
    #     self.dcf = torch.tensor(self.dcf).view(-1).unsqueeze(0)

    def _prep_nufft(self):
        # TODO: remove this (just for unaliasing messed up volunteer data 4/2/26)
        """
        Get adjoint nufft operator and input tensors for 2x FOV reconstruction.
        """
        if self.verbose:
            logger.info("Preparing adjoint NUFFT operator for 2x FOV.")

        # 1. Estimate the base resolution from the trajectory
        base_shape = sp.fourier.estimate_shape(self.coords)
        
        # 2. Set im_size to be 2x larger than the base resolution
        # This creates the larger canvas for the expanded FOV
        self.im_size = tuple(int(s * 2) for s in base_shape)
        
        # 3. Get the oversampled grid size based on the NEW im_size
        grid_size = self._get_grid_size()
        self.adj_nufft = KbNufftAdjoint(im_size=self.im_size, grid_size=grid_size)

        # 4. Normalize coords based on the BASE resolution, not the new im_size
        # This ensures that 1 k-space unit = 1 pixel at the original resolution.
        # We do NOT multiply by 0.5 here if we want the data to span the 2x im_size.
        self.coords = (self.coords / base_shape[0]) * 2 * np.pi

        # 5. Scale DCF so sum equals the NEW (larger) image volume
        self.dcf = self.dcf / self.dcf.sum() * np.prod(self.im_size)

        # 6. Convert to torch tensors
        # xk: (1, Nc, Nsp * Nr)
        self.xk = torch.tensor(self.xk).view(self.xk.shape[0], -1).unsqueeze(0)
        # coords: (1, 3, Nsp * Nr)
        self.coords = torch.tensor(self.coords).view(-1, 3).permute(1, 0).unsqueeze(0)
        # dcf: (1, Nsp * Nr)
        self.dcf = torch.tensor(self.dcf).view(-1).unsqueeze(0)

    def _get_grid_size(self):
        """
        Given image size (in pixels), get dimensions of oversampled grid (in pixels).
        Returns:
            grid_size (torch.tensor): grid size
        """
        grid_size = torch.round(torch.tensor(self.im_size)*self.oversamp).to(torch.int64).cpu()
        return grid_size
        
    def _get_ref_xk_cart(self):
        """
        Get cartesian k-space of reference, to estimate CSMs and get SENSE recon.
        Stores:
            xk_cart (torch.tensor): cartesian k-space (Nc, Nx, Ny, Nz)
        """
        if self.verbose:
            logger.info("Getting cartesian k-space of reference.")
        self.ref_im = self.adj_nufft(self.xk * self.dcf, self.coords, norm="ortho").numpy()
        self.xk_cart = sp.fft(self.ref_im, axes=(-3,-2,-1), norm="ortho")[0]

    def _get_ref_csm(self):
        """
        Get coil sensitivity maps of reference, from cartesian k-space with BART.
        Stores:
            csm (np.ndarray): coil sensitivity maps (TODO: shape)
        """
        if self.verbose:
            logger.info("Getting coil sensitivity maps of reference.")
        self.csm = bart(1, "ecalib -d2 -m1 -c0 -S", self.xk_cart.transpose(1,2,3,0)).transpose(3,0,1,2)

    def _get_S(self):
        """
        Get reference image with SENSE reconstruction.
        Stores:
            S (np.ndarray): reference image (TODO: shape)
        """
        if self.verbose:
            logger.info("Getting reference image.")
        recon = mr.app.SenseRecon(
            self.xk_cart, mps=self.csm, lamda=0.001, max_iter=30, show_pbar=self.verbose
        )
        self.S = recon.run()

class MotionFrames:
    """
    Split radial acquisition with motion into frames.
    """
    def __init__(self, inp_dir: str, verbose: bool = False, 
                 spokes_per_frame: int = 500, stride: int = 100, crop_factor: int = 8):
        self.verbose: bool = verbose
        self.inp_dir: str = inp_dir
        self.save_dir: str = os.path.join(inp_dir, f"crop_{crop_factor}")
        self.xk_fname: str = os.path.join(self.save_dir, "xk_frames.npy")
        self.coords_fname: str = os.path.join(self.save_dir, "coords_frames.npy")
        self.dcf_fname: str = os.path.join(self.save_dir, "dcf_frames.npy")
        self.bpts_fname: str = os.path.join(self.inp_dir, "bpts_frames.npy")
        self.frames_center_spokes_fname: str = os.path.join(self.save_dir, "frames_center_spokes.npy")
        self.xk_frames: np.ndarray
        self.coords_frames: np.ndarray
        self.dcf_frames: np.ndarray
        self.bpts_frames: np.ndarray
        self.frames_center_spokes: np.ndarray

        # Internal intermediates
        self.xk: np.ndarray
        self.coords: np.ndarray
        self.dcf: np.ndarray
        self.bpts: np.ndarray

        # Processing parameters
        self.spokes_per_frame = spokes_per_frame
        self.stride = stride
        self.crop_factor = crop_factor

    def run(self, force_reload: bool = False):
        """
        Split radial acquisition with motion into frames.
        Stores and saves:
            xk_frames (np.ndarray): radial cleaned k-space, split into frames (Nc, Nframes, spokes_per_frame, Nr)
            coords_frames (np.ndarray): coords, split into frames (Nframes, spokes_per_frame, Nr, 3)
            dcf_frames (np.ndarray): dcf, split into frames (Nframes, spokes_per_frame, Nr)
        """
        if (os.path.exists(self.xk_fname) and os.path.exists(self.coords_fname) 
            and os.path.exists(self.dcf_fname)) and not force_reload:
            logger.info("Radial acquisition split into frames found. Opening...")
            self.xk_frames = np.load(self.xk_fname)
            self.coords_frames = np.load(self.coords_fname)
            self.dcf_frames = np.load(self.dcf_fname)
            try:
                self.bpts_fname = np.load(self.bpts_fname)
            except:
                logger.warning("BPT/PTs frames not found.")
        else:
            logger.info(f"Radial acquisition split into frames not found. Extracting with crop factor {self.crop_factor}...")
            self.xk, self.coords, self.dcf, self.bpts = load_radial(self.inp_dir, self.verbose)
            self.xk, self.coords, self.dcf = crop_spokes(self.xk, self.coords, self.dcf, self.crop_factor, self.verbose)
            self._split_frames()
            
            # save
            os.makedirs(self.save_dir, exist_ok=True)
            np.save(self.xk_fname, self.xk_frames)
            np.save(self.coords_fname, self.coords_frames)
            np.save(self.dcf_fname, self.dcf_frames)
            if self.bpts_frames is not None:
                np.save(self.bpts_fname, self.bpts_frames)
            np.save(self.frames_center_spokes_fname, self.frames_center_spokes)
        
    def save_processing_params(self):
        """
        Save processing parameters to a pickle file.
        """
        params = {
            "spokes_per_frame": self.spokes_per_frame,
            "stride": self.stride,
            "crop_factor": self.crop_factor
        }
        params_fname = os.path.join(self.save_dir, "motion_frames_processing_params.pkl")
        with open(params_fname, "wb") as f:
            pkl.dump(params, f)
        if self.verbose:
            logger.info(f"Saved processing parameters to {params_fname}.")

    def _split_frames(self):
        """
        Split radial data into frames, given the spokes per frame and stride.
        Stores:
            xk_frames (np.ndarray): split k-space (Nc, Nframes, spokes_per_frame, Nr)
            coords_frames (np.ndarray): split coords (Nframes, spokes_per_frame, Nr, 3)
            dcf_frames (np.ndarray): split dcf (Nframes, spokes_per_frame, Nr)
        """
        if self.verbose:
            logger.info("Splitting radial data into frames.")
        Nc, Nsp, Nr = self.xk.shape
        # All valid frame-start spokes
        starts = np.arange(0, Nsp - self.spokes_per_frame + 1, self.stride)
        # Midpoint spoke for each frame
        self.frames_center_spokes = starts + self.spokes_per_frame // 2
        # Build frames
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