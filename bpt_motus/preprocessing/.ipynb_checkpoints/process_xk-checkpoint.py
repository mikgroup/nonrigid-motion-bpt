"""
Two classes for processing cleaned k-space data — one for the MR-MOTUS reference image, and one for generating frames with motion, to be resolved by MR-MOTUS.
"""
import os
import numpy as np
import pickle
import torch
from torchkbnufft import KbNufftAdjoint
from bart import bart
import sigpy as sp
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_radial(preproc_dir, raw_data_dir, verbose=False):
    """
    Get radial data.
    Returns: 
        xk (np.ndarray): cleaned k-space (Nc, Nsp, Nr)
        coords (np.ndarray): time-ordered coords (Nsp, Nr, 3)
        dcf (np.ndarray): time-ordered dcf (Nsp, Nr)
    """
    if verbose:
        logger.info("Getting xk, dcf, and coords from radial data.")
    xk = np.load(os.path.join(preproc_dir, "xk_cleaned_comp.npy"))
    raw_data_dict = os.path.join(raw_data_dir, "data_dict.pkl")
    with open(raw_data_dict, "rb") as f:
        data_dict = pickle.load(f)
    coords = data_dict['coords_ordered']
    dcf = data_dict['dcf_ordered']
    return xk, coords, dcf

def crop_spokes(xk, coords, dcf, crop_factor, verbose=False):
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
    ro_off = int((coords.shape[1] - coords.shape[1] / crop_factor) / 2)
    xk_crop = xk[:, :, ro_off:-ro_off]
    coords_crop = coords[:, ro_off:-ro_off]
    dcf_crop = dcf[:, ro_off:-ro_off]
    return xk_crop, coords_crop, dcf_crop
    
class NoMotionReference:
    """
    Build a no-motion reference image S and coil sensitivity
    maps from the radial, processed xk / coords / dcf files.
    """
    def __init__(self, inp_dir: str, verbose: bool = False, 
                 save_dir: str = "no_motion_reference", preproc_dir: str = "preprocessed_data", 
                 raw_dir: str = "raw_data", S_file: str = "S_reference.npy", 
                 csm_file: str = "csm_reference.npy", no_motion_spokes = None, crop_factor: int = 3):
        self.verbose: bool = verbose
        self.save_dir: str = os.path.join(inp_dir, save_dir, f'crop_{crop_factor}')
        self.raw_data_dir = os.path.join(inp_dir, raw_dir)
        self.preproc_dir = os.path.join(inp_dir, preproc_dir)
        self.S_fname: str = os.path.join(self.save_dir, S_file)
        self.csm_fname: str = os.path.join(self.save_dir, csm_file)
        self.S: np.ndarray
        self.csm: np.ndarray
        
        # Internal intermediates
        self.xk = None
        self.xk_cart = None
        self.coords = None
        self.dcf = None
        self.adj_nufft = None
        self.im_size = None

        # Processing parameters
        self.oversamp = 1.25
        self.no_motion_spokes = no_motion_spokes
        self.crop_factor = crop_factor

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
            self.xk, self.coords, self.dcf = load_radial(self.preproc_dir, self.raw_data_dir, self.verbose)
            self._keep_no_motion_spokes()
            self.xk, self.coords, self.dcf = crop_spokes(self.xk, self.coords, self.dcf, self.crop_factor, self.verbose)
            self._prep_nufft()
            self._get_ref_xk_cart()
            self._get_ref_csm()
            self._get_S()
            
            # save
            os.makedirs(self.save_dir, exist_ok=True)
            np.save(self.S_fname, self.S)
            np.save(self.csm_fname, self.csm)

    def _keep_no_motion_spokes(self):
        """
        Keep the no motion spokes only, from the beginning.
        Stores:
            xk (np.ndarray): no-motion k-space (Nc, Nsp, Nr)
            coords (np.ndarray): no-motion coords (Nsp, Nr, 3)
            dcf (np.ndarray): no-motion dcf (Nsp, Nr)
        """
        # If no_motion_spokes is None, use all spokes
        if self.no_motion_spokes is None:
            self.no_motion_spokes = self.xk.shape[1] # TODO: check shape
            if self.verbose:
                logger.info(f"Using all spokes: {self.no_motion_spokes}")
        use_spokes = min(self.no_motion_spokes, self.xk.shape[1])
        self.xk = self.xk[:, :use_spokes]
        self.coords = self.coords[:use_spokes]
        self.dcf = self.dcf[:use_spokes]
        
    def _prep_nufft(self):
        """
        Get adjoint nufft operator and input tensors.
        Stores:
            adj_nufft: KbNufftAdjoint instance (on default device)
            xk (torch.tensor): cropped cleaned k-space (1, Nc, Nsp * Nr)
            coords (torch.tensor): cropped time-ordered coords (3, Nsp * Nr)
            dcf (torch.tensor): cropped time-ordered dcf (1, Nsp * Nr)
            im_size: image size (estimated from coords)
        """
        if self.verbose:
            logger.info("Preparing adjoint NUFFT operator and inputs.")
        self.im_size = sp.fourier.estimate_shape(self.coords)
        grid_size = self._get_grid_size()
        self.adj_nufft = KbNufftAdjoint(im_size=self.im_size, grid_size=grid_size)

        # normalize coords and dcf for orthonormal adjoint NUFFT
        self.coords = self.coords / self.im_size[0] * 2 * np.pi # scaled by 2*pi / reference image extent
        self.dcf = self.dcf / self.dcf.sum() * np.prod(self.im_size) # make DCF sum the image volume

        # get torch tensors
        self.xk = torch.tensor(self.xk).view(self.xk.shape[0], -1).unsqueeze(0)
        self.coords = torch.tensor(self.coords).view(-1,3).permute(1,0).unsqueeze(0)
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
        ref_im = self.adj_nufft(self.xk * self.dcf, self.coords, norm="ortho").numpy()
        self.xk_cart = sp.fft(ref_im, axes=(-3,-2,-1), norm="ortho")[0]

    def _get_ref_csm(self):
        """
        Get coil sensitivity maps of reference, from cartesian k-space with BART.
        Stores:
            csm (np.ndarray): coil sensitivity maps (TODO: shape)
        """
        if self.verbose:
            logger.info("Getting coil sensitivity maps of reference.")
        self.csm = bart(1, 'ecalib -d2 -m1 -c0 -S', self.xk_cart.transpose(1,2,3,0)).transpose(3,0,1,2)

    def _get_S(self):
        """
        Get reference image with SENSE reconstruction.
        Stores:
            S (np.ndarray): reference image (TODO: shape)
        """
        if self.verbose:
            logger.info("Getting reference image.")
        recon = sp.mri.app.SenseRecon(
            self.xk_cart, mps=self.csm, lamda=0.001, max_iter=30, show_pbar=self.verbose
        )
        self.S = recon.run()

class MotionFrames:
    """
    Split radial acquisition with motion into frames.
    """
    def __init__(self, inp_dir: str, verbose: bool = False, 
                 save_dir: str = "motion_frames", preproc_dir: str = "preprocessed_data", 
                 raw_dir: str = "raw_data", xk_frames_file: str = "xk_frames.npy", 
                 coords_frames_file: str = "coords_frames.npy", dcf_frames_file: str = "dcf_frames.npy",
                 motion_spokes = None, spokes_per_frame: int = 500, crop_factor: int = 8):
        self.verbose: bool = verbose
        self.save_dir: str = os.path.join(inp_dir, save_dir, f'crop_{crop_factor}')
        self.raw_data_dir = os.path.join(inp_dir, raw_dir)
        self.preproc_dir = os.path.join(inp_dir, preproc_dir)
        self.xk_fname: str = os.path.join(self.save_dir, xk_frames_file)
        self.coords_fname: str = os.path.join(self.save_dir, coords_frames_file)
        self.dcf_fname: str = os.path.join(self.save_dir, dcf_frames_file)
        self.xk_frames: np.ndarray
        self.coords_frames: np.ndarray
        self.dcf_frames: np.ndarray

        # Processing parameters
        self.motion_spokes = motion_spokes
        self.spokes_per_frame = spokes_per_frame
        self.crop_factor = crop_factor

    def run(self, force_reload: bool = False):
        """
        Split radial acquisition with motion into frames.
        Stores and saves:
            xk_frames (np.ndarray): radial cleaned k-space, split into frames (TODO: shape)
            coords_frames (np.ndarray): coords, split into frames (TODO: shape)
            dcf_frames (np.ndarray): dcf, split into frames (TODO: shape)
        """
        if (os.path.exists(self.xk_fname) and os.path.exists(self.coords_fname) 
            and os.path.exists(self.dcf_fname)) and not force_reload:
            logger.info("Radial acquisition split into frames found. Opening...")
            self.xk_frames = np.load(self.xk_fname)
            self.coords_frames = np.load(self.coords_fname)
            self.dcf_frames = np.load(self.dcf_fname)
        else:
            logger.info(f"Radial acquisition split into frames not found. Extracting with crop factor {self.crop_factor}...")
            self.xk, self.coords, self.dcf = load_radial(self.preproc_dir, self.raw_data_dir, self.verbose)
            self._keep_motion_spokes()
            self.xk, self.coords, self.dcf = crop_spokes(self.xk, self.coords, self.dcf, self.crop_factor, self.verbose)
            self._split_frames()
            
            # save
            os.makedirs(self.save_dir, exist_ok=True)
            np.save(self.xk_fname, self.xk_frames)
            np.save(self.coords_fname, self.coords_frames)
            np.save(self.dcf_fname, self.dcf_frames)
    
    def _keep_motion_spokes(self):
        """
        Keep the motion spokes only, from the end.
        Stores:
            xk (np.ndarray): motion k-space (Nc, Nsp, Nr)
            coords (np.ndarray): motion coords (Nsp, Nr, 3)
            dcf (np.ndarray): motion dcf (Nsp, Nr)
        """
        # If motion_spokes is None, use all spokes
        if self.motion_spokes is None:
            self.motion_spokes = self.xk.shape[1] # TODO: check shape
            if self.verbose:
                logger.info(f"Using all spokes: {self.motion_spokes}")
        use_spokes = min(self.motion_spokes, self.xk.shape[1])
        self.xk = self.xk[:, -use_spokes:]
        self.coords = self.coords[-use_spokes:]
        self.dcf = self.dcf[-use_spokes:]

    def _split_frames(self):
        """
        Split radial data into frames, given the spokes per frame and stride.
        Stores:
            xk (np.ndarray): split k-space (Nc, Nframes, spokes_per_frame, Nr)
            coords (np.ndarray): split coords (Nframes, spokes_per_frame, Nr, 3)
            dcf (np.ndarray): split dcf (Nframes, spokes_per_frame, Nr)
        """
        Nc, Nsp, Nr = self.xk.shape
        xk_f, coords_f, dcf_f = [], [], []
        start = 0
        while start + self.spokes_per_frame <= Nsp:
            stop = start + self.spokes_per_frame
            xk_f.append(self.xk[:, start:stop])
            coords_f.append(self.coords[:, start:stop])
            dcf_f.append(self.dcf[:, start:stop])
            start += self.stride
        self.xk_frames = np.stack(xk_f)
        
            
# TODO: write split_frames
# TODO: test it all