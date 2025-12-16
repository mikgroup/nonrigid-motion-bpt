"""
Classes for processing BPT/PT signals, to be used as temporal components in calibration OR inference.
"""
import os
import numpy as np
import sigpy as sp
import logging
from typing import Literal 
from scipy.signal import medfilt, butter, filtfilt 
from tqdm import tqdm 
import torch 
from sklearn.decomposition import PCA 
import pickle as pkl 

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
    
class ProcessBPT:
    """
    Get processed BPT/PT signals from the raw BPT/PT signals, for calibration OR inference.
    """
    def __init__(self, inp_dir: str, verbose: bool = False, device: str = "cpu", 
                 nrank :int = 16, phase: Literal["calib", "inf"] = "calib", 
                 bpts_pca_fname:str = None):
        self.verbose: bool = verbose
        self.device = device
        self.inp_dir: str = inp_dir
        self.bpts_proc_fname: str = os.path.join(self.inp_dir, "bpts_proc.npy")
        if bpts_pca_fname is None:
            self.bpts_pca_fname: str = os.path.join(os.path.dirname(self.inp_dir), "calib", "bpts_pca.pkl")
        else:
            self.bpts_pca_fname:str = bpts_pca_fname
        self.phase: str = phase
        self.bpts_proc: np.ndarray = None
        self.bpts_pca = None
        
        # Internal intermediates
        self.bpts_raw: np.ndarray
        self.bpts_flat: np.ndarray
        self.bpts_med: np.ndarray
        self.bpts_filt: np.ndarray
        self.bpts_ssa: np.ndarray
        self.bpts_norm: np.ndarray
        
        # Processing parameters
        self.tr: float = 4e-3 # in seconds
        self.median_window: int = 11
        self.lpf_cutoff_hz: float = 0.4
        self.lpf_order: int = 5
        self.ssa_window: int = 300
        self.ssa_components_removed: int = 100
        self.nbpts: int = 1
        self.nrank: int = nrank
        self.coupler: bool = False

    def run(self, force_reload: bool = False):
        """
        Get processed BPT/PTs.
        Stores and saves:
            bpts_proc (np.ndarray): processed BPT/PTs
            If calibration: bpt_pca: PCA model from calibration BPT/PTs
        """
        if (os.path.exists(self.bpts_proc_fname) and os.path.exists(self.bpts_pca_fname)) and not force_reload:
            logger.info("Processed BPT/PTs found. Opening them...")
            self.bpts_proc = np.load(self.bpts_proc_fname)
        else:
            logger.info("Processed BPT/PTs not found. Extracting them...")
            self._get_tr()
            self._load_raw_bpts()
            if self.coupler:
                self.bpts_raw = self.bpts_raw[...,:-2] # remove final 2 coupler channels
            self._flatten_bpts()
            self._med_filt_bpts()
            self._lpf_bpts()
            self._ssa_bpts()
            self._norm_bpts()
            self._comp_bpts()
            
            # save
            os.makedirs(self.inp_dir, exist_ok=True)
            np.save(self.bpts_proc_fname, self.bpts_proc)
            if self.phase == "calib":
                with open(self.bpts_pca_fname, "wb") as f:
                    pkl.dump(self.bpts_pca, f)

    def _get_tr(self):
        """
        Get TR for the scan the BPT/PTs came from, if possible.
        Stores:
            tr (float): TR in seconds
        """
        try: 
            with open(os.path.join(self.inp_dir, "metadata_dict.pkl"), "rb") as f:
                metadata = pkl.load(f)
            self.tr = metadata['tr']
            if self.verbose:
                logger.info(f"TR found from metadata: {self.tr*1e3:.2f} ms")
        except Exception as e:
            if self.verbose:
                logger.warning(f"Could not get TR from metadata: {e}. Using default TR: {self.tr*1e3:.2f} ms")

    def _load_raw_bpts(self):
        """
        Get saved raw BPT/PTs.
        Stores:
            bpts_raw (np.ndarray): raw BPT/PTs (num_bpts, Nsp, Nc)
        """
        if self.verbose:
            logger.info("Loading raw BPT/PTs...")
        self.bpts_raw = np.load(os.path.join(self.inp_dir, "bpts.npy"))

    def _flatten_bpts(self):
        """
        Flatten BPT/PTs, combining all the MIMO signals.
        Stores:
            bpts_flat (np.ndarray): flattened BPT/PTs (Nsp, num_bpts*Nc)
        """
        self.nbpts, n_spokes, n_coils = self.bpts_raw.shape
        self.bpts_flat = self.bpts_raw.transpose(1,0,2).reshape(n_spokes, self.nbpts*n_coils)

    def _med_filt_bpts(self):
        """
        Median filter BPT/PTs, to remove spikes due to object entering BPT/PT frequencies.
        Stores:
            bpts_med (np.ndarray): median filtered BPT/PTs (Nsp, num_bpts*Nc)
        """
        if self.verbose:
            logger.info("Applying median filter to BPT/PTs...")
        self.bpts_med = medfilt(self.bpts_flat, kernel_size=(self.median_window, 1))

    def _lpf_bpts(self):
        """
        Low-pass filter BPT/PTs, to only keep the fastest expected motion
        Stores:
            bpts_lpf (np.ndarray): LPF BPT/PTs (Nsp, num_bpts*Nc)
        """
        if self.verbose:
            logger.info("Applying low-pass filter to BPT/PTs...")
        nyq = 0.5 / self.tr
        Wn = min(self.lpf_cutoff_hz / nyq, 0.99)  # make sure <= 1
        b, a = butter(self.lpf_order, Wn, btype='low')
        self.bpts_lpf = filtfilt(b, a, self.bpts_med, axis=0)

    def _ssa_bpts(self):
        """
        Removes top SSA components of BPT/PTs, to remove regular oscillation artifacts
        Stores:
            bpts_ssa (np.ndarray): post-SSA BPT/PTs (Nsp, num_bpts*Nc)
        """
        if self.verbose:
            logger.info("Applying SSA to BPT/PTs...")
        bpts_lpf = self.bpts_lpf.copy()
        N, F = self.bpts_lpf.shape
        out = np.zeros_like(bpts_lpf)
    
        for i in tqdm(range(F), desc="SSA"):
            x = torch.tensor(bpts_lpf[:, i], device=self.device, dtype=torch.float32)
            # Zero-copy Hankel matrix (trajectory) using as_strided
            D = x.as_strided((N - self.ssa_window + 1, self.ssa_window),
                             (x.stride(0), x.stride(0)))
            # Center and perform SVD
            means = D.mean(1, keepdim=True)
            u, s, vh = torch.linalg.svd(D - means, full_matrices=False)
            # Filter out leading components and reconstruct trajectory
            S = (u[:, self.ssa_components_removed:] * s[self.ssa_components_removed:].unsqueeze(0)) @ vh[self.ssa_components_removed:] + means
            # Vectorized overlap-add reconstruction
            res = torch.zeros(N + self.ssa_window, device=self.device)
            counts = torch.zeros(N + self.ssa_window, device=self.device)
            idx = torch.arange(self.ssa_window, device=self.device)
            base = torch.arange(S.shape[0], device=self.device).unsqueeze(1)
            positions = base + idx  # shape: (rows, self.ssa_window)
            # Scatter-add values and counts
            res.index_add_(0, positions.flatten(), S.flatten())
            counts.index_add_(0, positions.flatten(), torch.ones_like(S).flatten())
            # Extract valid region, normalize, and store result
            valid_counts = counts[:N]
            y = res[:N]
            y[valid_counts > 0] /= valid_counts[valid_counts > 0]
            out[:, i] = y.cpu().numpy()
    
        self.bpts_ssa = out
    
    def _norm_bpts(self):
        """
        Remove shared multiplicative component (eg. drift) from BPT/PTs, but preserve relative magnitudes.
        Stores:
            bpts_norm (np.ndarray): normalized BPT/PTs (num_bpts, Nsp, Nc)
        """
        if self.verbose:   
            logger.info("Normalizing BPT/PTs...")
        N, M = self.bpts_ssa.shape
        if M % self.nbpts != 0:
            logger.error(f"Unflattening BPT/PTs failed. M={M} must be divisible by num_bpts={self.nbpts}.")
        channels = M // self.nbpts
        # Reshape and transpose -> (num_bpts, N, channels)
        x = self.bpts_ssa.reshape(N, self.nbpts, channels).transpose(1, 0, 2)
        # Compute norms over the last axis -> (num_bpts, N)
        norms = np.linalg.norm(x, axis=2)
        # Mean norms per BPT -> (num_bpts,)
        mean_norms = norms.mean(axis=1)
        # Normalize and rescale
        x_scaled = x / norms[..., None] * mean_norms[:, None, None]
        self.bpts_norm = x_scaled.transpose(1, 0, 2).reshape(N, M)

    def _comp_bpts(self):
        """
        Compress BPT/PTs using PCA.
        Stores:
            bpt_comp (np.ndarray): compressed BPT/PTs (Nsp, nrank)
            if calib: bpt_pca: PCA model from calibration BPT/PTs
        """
        if self.verbose:
            logger.info("Applying PCA to BPT/PTs...")
        if self.phase == "calib":
            pca = PCA(n_components=self.nrank)
            self.bpts_proc = pca.fit_transform(self.bpts_norm)
            self.bpts_pca = pca
        else: # use PCA from calibration
            try:
                with open(self.bpts_pca_fname, "rb") as f:
                    self.bpts_pca = pkl.load(f)
                self.bpts_proc = self.bpts_pca.transform(self.bpts_norm)
            except Exception as e:
                logger.error(f"Could not load calibration phase PCA model: {e}.")
