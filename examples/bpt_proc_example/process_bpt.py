"""
Classes for processing BPT/PT signals, to be used as temporal components in calibration OR inference.
"""
import os
import numpy as np
import sigpy as sp
import matplotlib.pyplot as plt
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
                 nrank :int = 16):
        self.verbose: bool = verbose
        self.device = device
        self.inp_dir: str = inp_dir
        self.bpts_proc_fname: str = os.path.join(self.inp_dir, "bpts_proc.npy")
        self.bpts_proc: np.ndarray = None
        
        # Internal intermediates
        self.bpts_raw: np.ndarray
        self.bpts_flat: np.ndarray
        self.bpts_med: np.ndarray
        self.bpts_filt: np.ndarray
        self.bpts_norm: np.ndarray
        
        # Processing parameters
        self.tr: float = 4e-3 # in seconds
        self.median_window: int = 11
        self.lpf_cutoff_hz: float = 5.0
        self.lpf_order: int = 5
        self.nbpts: int = 1
        self.nrank: int = nrank
        self.coupler: bool = False

    def run(self, force_reload: bool = False):
        """
        Get processed BPT/PTs.
        Stores and saves:
            bpts_proc (np.ndarray): processed BPT/PTs
        """
        if os.path.exists(self.bpts_proc_fname) and not force_reload:
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
            self._norm_bpts()
            self._comp_bpts()
            
            # save
            os.makedirs(self.inp_dir, exist_ok=True)
            np.save(self.bpts_proc_fname, self.bpts_proc)

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
    
    def _norm_bpts(self):
        """
        Remove shared multiplicative component (eg. drift) from BPT/PTs, but preserve relative magnitudes.
        Stores:
            bpts_norm (np.ndarray): normalized BPT/PTs (num_bpts, Nsp, Nc)
        """
        if self.verbose:   
            logger.info("Normalizing BPT/PTs...")
        N, M = self.bpts_lpf.shape
        if M % self.nbpts != 0:
            logger.error(f"Unflattening BPT/PTs failed. M={M} must be divisible by num_bpts={self.nbpts}.")
        channels = M // self.nbpts
        # Reshape and transpose -> (num_bpts, N, channels)
        x = self.bpts_lpf.reshape(N, self.nbpts, channels).transpose(1, 0, 2)
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
            bpts_proc (np.ndarray): compressed BPT/PTs (Nsp, nrank)
        """
        if self.verbose:
            logger.info("Applying PCA to BPT/PTs...")
        pca = PCA(n_components=self.nrank)
        self.bpts_proc = pca.fit_transform(self.bpts_norm)

    def _unflatten_bpts(self, bpts):
            """
            Unflatten BPT/PTs at any stage, reversing the flattening operation.
            Returns:
                bpts_raw (np.ndarray): unflattened BPT/PTs (num_bpts, Nsp, Nc)
            """
            n_spokes, flat_dim = bpts.shape
            n_coils = flat_dim // self.nbpts
            return bpts.reshape(n_spokes, self.nbpts, n_coils).transpose(1, 0, 2)
    
# Visualize B+PT signals

def plot_bpts(bpts, tr=1, shift=None, figsize=(10,10), titles=None):
    """
    Plot BPT/PTs across all coils. Automatically shifts coil signals for visibility.
    """
    # Handle 2D input (npe, ncoils) vs 3D (nbpts, npe, ncoils)
    if bpts.ndim == 2:
        bpts = bpts[np.newaxis, ...]
        
    nbpts, npe, ncoils = bpts.shape
    
    # Auto-generate titles if not provided
    if titles is None:
        titles = [f"B+PT {i+1}" for i in range(nbpts)]
    
    # Demean signals for plotting
    bpt_dm = bpts - np.mean(bpts, axis=1, keepdims=True)
    
    # Automate shift: 0.5 * max magnitude across all data
    if shift is None:
        shift = 0.5 * np.max(np.abs(bpt_dm))
        
    # Calculate grid size for subplots
    ncols = 2 if nbpts > 1 else 1
    nrows = int(np.ceil(nbpts / ncols))
    
    plt.figure(figsize=figsize)
    t = np.arange(npe) * tr
    
    # Plot
    for i in range(nbpts):
        plt.subplot(nrows, ncols, i+1)
        # Normalize by shift and offset by 1 per coil
        plt.plot(t, (bpt_dm[i] / shift) + np.arange(ncoils));
        plt.title(titles[i])
        # X-label only on the bottom row
        if i >= (nrows - 1) * ncols:
            plt.xlabel("Time (s)")