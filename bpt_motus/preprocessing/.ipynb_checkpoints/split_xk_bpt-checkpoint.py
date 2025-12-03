"""
Functions for splitting acquired k-space into cleaned k-space (MRI) data and BPT data 
"""
import os
import numpy as np
import pickle
import scipy.signal
import warnings
from tqdm import tqdm
import sigpy as sp
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SplitXkBPT:
    """
    Split acquired time-ordered k-space 
    into cleaned k-space data and BPT data.
    """
    def __init__(self, inp_dir: str, verbose: bool = False, save_dir: str = 'preprocessed_data', raw_dir: str = 'raw_data', xk_file: str = "xk_cleaned_comp.npy", bpts_file: str = "bpts.npy"):
        self.verbose: bool = verbose 
        self.save_dir: str = os.path.join(inp_dir, save_dir)
        self.raw_data_dir = os.path.join(inp_dir, raw_dir)
        self.xk_cleaned_fname: str = os.path.join(self.save_dir, xk_file)
        self.bpts_fname: str = os.path.join(self.save_dir, bpts_file)
        self.xk_cleaned: np.ndarray
        self.bpts: np.ndarray

        # Processing variables, filled in sequentially
        self.xk_ordered = None
        self.xk_f = None
        self.coarse_peaks = None
        self.best_coil = None
        self.best_peak = None
        self.offsets = None
        self.xk_aligned = None
        self.bpts = None
        self.xk_aligned_cleaned = None
        self.xk_cleaned = None

        # Processing parameters
        self.num_bpts: int = 4 # number of BPT/PT signals
        self.edge_frac: float = 0.4 # fraction of edge of readout BPT/PT signals are in
        self.zpad: int = 10 # zero-padding interpolation
        self.offset_win: int = self.zpad * 1 # window around zero-padded peak to search
        self.polyinterp: int = 25 # polynomial interpolation
        self.order: int = 5 # polynomial order
        self.bpt_win: int = 2 # window around BPT/PT peak to get signal 
        self.lpf_tolerance: int = 40 # number of samples beyond max peak to include in stopband
        self.comp_channels: int = 6 # number of compressed coils

    def run(self, force_reload: bool = False):
        """
        Split time-ordered k-space into cleaned k-space and raw BPT signals.
        Stores and saves: 
            xk_cleaned (np.ndarray): BPT-free k-space (Nc, Nsp, Nr)
            bpts (np.ndarray): BPT/PT signals - shape (num_bpts, Nsp, Nc)
        """
        if (os.path.exists(self.xk_cleaned_fname) and os.path.exists(self.bpts_fname)) and not force_reload:
            logger.info("Cleaned k-space and raw BPT/PT signals found. Opening...")
            self.xk_cleaned = np.load(self.xk_cleaned_fname)
            self.bpts = np.load(self.bpts_fname)
        else:
            logger.info("Cleaned k-space and raw BPT/PT signals not found. Extracting...")
            self._get_raw_xk()
            self._get_xk_f()
            self._find_coarse_peaks()
            self._find_strongest_tone()
            self.xk_f = None # free memory
            self._compute_offsets()
            self._align_kspace()
            self.xk_ordered = None # free memory
            self._extract_bpts()
            self._clean_kspace()
            self.xk_aligned = None # free memory
            self._unalign_kspace()
            self._compress_kspace()

            # save
            os.makedirs(self.save_dir, exist_ok=True)
            np.save(self.xk_cleaned_fname, self.xk_cleaned)
            np.save(self.bpts_fname, self.bpts)

    def _get_raw_xk(self):
        """
        Get time-ordered k-space from ScanArchive.
        Stores: xk_ordered (np.ndarray): raw k-space ()
        """
        if self.verbose:
            logger.info("Getting raw time-ordered k-space.")
        raw_data_dict = os.path.join(self.raw_data_dir, "data_dict.pkl")
        with open(raw_data_dict, "rb") as f:
            data_dict = pickle.load(f)
        self.xk_ordered = data_dict['xk_time']

    def _get_xk_f(self):
        """ 
        Get hybrid raw k-space.
        Stores: xk_f (np.ndarray): hybrid k-space ()
        """
        if self.verbose:
            logger.info("Getting hybrid raw k-space.")
        self.xk_f = sp.ifft(self.xk_ordered, axes=(-1,))
        
    def _find_coarse_peaks(self):
        """
        Find the coarse frequency locations of the BPT/PTs.

        Stores (np.ndarray): coarse_peaks (num_bpts,)
        """
        if self.verbose:
            logging.info("Getting coarse peaks.")
        # Get RSS of readouts
        xk_rss = sp.rss(self.xk_f, axes=(0,1))
        nr = xk_rss.shape[0]
        # Remove middle of readouts' RSS
        edge_id = int(xk_rss.shape[0] * self.edge_frac)
        xk_rss_edge = np.concatenate([xk_rss[:edge_id], xk_rss[-edge_id:]])
        edge_indices = np.concatenate([np.arange(edge_id), np.arange(nr - edge_id, nr)])
        # Indices of strongest peaks, ordered by peak strength
        coarse_peaks = edge_indices[(lambda p: p[np.argsort(xk_rss_edge[p])[-self.num_bpts:]])(scipy.signal.find_peaks(xk_rss_edge)[0])]
        self.coarse_peaks = np.sort(coarse_peaks)

    def _find_strongest_tone(self):
        """
        Find the coil and frequency index of the strongest BPT/PT, which will be used as a reference.
    
        Stores:
            best_coil (int): Coil index of strongest BPT/PT
            best_peak (int): Frequency index of strongest BPT/PT
        """
        if self.verbose:
            logger.info("Getting strongest tone.")
        coarse_bpts = self.xk_f[:,:,self.coarse_peaks] # nc x nsp x num_bpts
        self.best_coil, best_peak_id = np.unravel_index(np.argmax(np.abs(coarse_bpts).max(axis=1)), coarse_bpts.shape[::2])
        self.best_peak = self.coarse_peaks[best_peak_id]

    def _compute_offsets(self):
        """
        Get the offsets between the actual BPT/PT peaks, which move per readout, and the coarse peak estimates. Use zero-padding and polynomial interpolation to finely estimate the actual peaks.
        
        Stores: offsets (np.ndarray): Estimates of the offset between actual peaks and coarse estimates over all spokes (Nsp,)
        """
        if self.verbose:
            logger.info("Getting offsets.")
        xk_best_channel = self.xk_ordered[self.best_coil]
        nsp, nr = xk_best_channel.shape
        self.offsets = np.zeros(nsp)
        for spoke_id in tqdm(range(nsp)):
            spoke = xk_best_channel[spoke_id]
            # Interpolate with zero-padding
            pad_spoke = np.pad(spoke, ((nr*self.zpad-nr)//2, (nr*self.zpad-nr)//2))
            spoke_f = sp.ifft(pad_spoke)
    
            # Get window around best peak
            start, end = max(0, self.best_peak * self.zpad - self.offset_win), min(nr * self.zpad, self.best_peak * self.zpad + self.offset_win + 1)
            spoke_window = np.abs(spoke_f[start:end])
            window_idx = np.arange(start, end, dtype=np.float32)
            window_idx_centered = window_idx - window_idx.mean() # center coordinates
            
            # Polynomial fit
            try:
                with warnings.catch_warnings(): # Don't show warnings
                    warnings.simplefilter('ignore', np.RankWarning)
                    coeffs = np.polyfit(window_idx_centered, spoke_window, deg=self.order) 
                poly = np.poly1d(coeffs) # Fit polynomial
                npoints = self.offset_win * 2 * self.polyinterp
                fine_window_idx = np.linspace(window_idx_centered[0], window_idx_centered[-1], npoints)
                fine_vals = poly(fine_window_idx) # Sample polynomial with additional interpolation
                # Find peak in fine grid
                fine_peak_idx = fine_window_idx[int(np.argmax(fine_vals))] + window_idx.mean()
                offset = fine_peak_idx / float(self.zpad) - self.best_peak
                self.offsets[spoke_id] = offset
            except: # If polynomial fitting fails, use zero drift
                logger.warning(f"Spoke localization failed for spoke {spoke_id}.")
                self.offsets[spoke_id] = 0.0

    def _align_kspace(self):
        """
        Aligns k-space so that BPT/PT frequencies are steady
        Stores: xk_aligned (np.ndarray): aligned k-space, with offsets removed (Nc, Nsp, Nr)
        """
        if self.verbose:
            logger.info("Aligning k-space.")
        nc, nsp, nr = self.xk_ordered.shape
        
        # Apply offset correction
        centered_idx = (np.arange(nr, dtype=np.float32) - (nr - 1) / 2.0) / nr
        phase_ramps = np.exp(1j * 2 * np.pi * self.offsets[:, None] * centered_idx[None, :])
        self.xk_aligned = self.xk_ordered * phase_ramps[None, :, :]

    def _extract_bpts(self):
        """
        Extract BPT/PT signals from aligned k-space.
        Stores: bpts (np.ndarray): BPT/PT signals - shape (num_bpts, Nsp, Nc)
        """
        if self.verbose:
            logger.info("Extracting BPT/PTs.")
        xk_f_aligned = sp.ifft(self.xk_aligned, axes=(-1,))
        # Stack the windows around each BPT/PT peak
        bpt_windows = np.stack([xk_f_aligned[:, :, peak_id - self.bpt_win:peak_id + self.bpt_win + 1] for peak_id in self.coarse_peaks], axis=0)
        # Take RSS over the windows around each peak
        self.bpts = sp.rss(bpt_windows, axes=(-1,)).transpose(0, 2, 1)

    def _clean_kspace(self):
        """
        Remove BPT's contribution to k-space via a zero-phase LPF.
    
        Stores: xk_aligned_cleaned (np.ndarray): BPT-free k-space, still aligned (Nc, Nsp, Nr)
        """
        if self.verbose:
            logger.info("Cleaning k-space, still aligned.")
        nc, nsp, nr = self.xk_aligned.shape
        # Assumes peaks are to the left of k-space
        cutoff_index = abs(np.max(self.coarse_peaks) - nr // 2 + self.lpf_tolerance)
        cutoff_freq = min(0.999, cutoff_index / (nr // 2))
        # Design filter
        b, a = scipy.signal.butter(10, cutoff_freq, btype='low')
        self.xk_aligned_cleaned = scipy.signal.filtfilt(b, a, self.xk_aligned, axis=-1)
        
    def _unalign_kspace(self):
        """
        Undoes alignment of cleaned k-space now that BPT/PT is removed.
        Stores: xk_cleaned (np.ndarray): BPT-free k-space (Nc, Nsp, Nr)
        """
        if self.verbose:
            logger.info("Unaligning cleaned k-space.")
        nc, nsp, nr = self.xk_aligned_cleaned.shape
        centered_idx = (np.arange(nr, dtype=np.float32) - (nr - 1) / 2.0) / nr
        # Note negative sign in exponent
        phase_ramps = np.exp(-1j * 2 * np.pi * self.offsets[:, None] * centered_idx[None, :]) 
        self.xk_cleaned = self.xk_aligned_cleaned * phase_ramps[None, :, :]

    def _compress_kspace(self):
        """
        Compresses k-space into fewer channels with PCA.
        Stores: xk_cleaned (np.ndarray): replaces BPT-free k-space with coil-compressed, BPT-free k-space (Nc_comp, Nsp, Nr)
        """
        if self.verbose:
            logger.info("Coil compressing k-space with PCA.")
        orig_coils = self.xk_cleaned.shape[0]
        # Random subsampling (to reduce SVD memory)
        mask = np.random.choice([True,False], size=self.xk_cleaned.shape[1:], p=[0.05, 1-0.05])
        # Subsampled coil x points matrix
        xk_masked = self.xk_cleaned[:, mask]
        # SVD (u: Nc x Nc)
        u,_,_ = np.linalg.svd(xk_masked, full_matrices=False)
        self.xk_cleaned = np.tensordot(u[:, :self.comp_channels], self.xk_cleaned, axes=(0, 0))