"""
Classes for processing BPT/PT signals, to be used as temporal components in calibration OR inference.
"""
import os
import numpy as np
import sigpy as sp
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
    
class ProcBPT:
    """
    Get processed BPT/PT signals from the raw BPT/PT signals, for calibration OR inference.
    """
    def __init__(self, inp_dir: str, verbose: bool = False, 
                 phase: Literal["calib", "inf"] = "calib"):
        self.verbose: bool = verbose
        self.inp_dir: str = inp_dir
        self.save_dir: str = os.path.join(inp_dir, save_dir, )
        self.bpts_proc_fname: str = os.path.join(self.save_dir, "bpts_proc.npy")
        self.bpts_pca_fname: str = os.path.join(self.save_dir, "bpts_pca.pkl")
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
        self.median_window: int = 11
        self.lpf_cutoff_hz: float = 0.4
        self.lpf_order: int = 5
        self.tr: float
        self.ssa_window: int = 300
        self.ssa_components_removed: int = 100
        self.nrank: int = nrank
        self.coupler: bool = False

    def run(self, force_reload: bool = False):
        """
        Get processed BPT/PTs.
        Stores and saves:
            bpts_proc (np.ndarray): processed BPT/PTs
            If calibration: bpt_pca: PCA model from calibration BPT/PTs
        """
        if (os.path.exists(self.bpt_proc_fname) and os.path.exists(self.bpt_pca_fname)) and not force_reload:
            logger.info("Processed BPT/PTs found. Opening them...")
            self.bpt_proc = np.load(self.bpt_proc_fname)
        else:
            logger.info("Processed BPT/PTs not found. Extracting them...")
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
            os.makedirs(self.save_dir, exist_ok=True)
            np.save(self.bpts_proc_fname, self.bpts_proc)
            if self.phase == "calib":
                np.save(self.bpts_pca_fname, self.bpts_pca)

    def _load_raw_bpts(self):
        """
        Get saved raw BPT/PTs.
        Stores:
            bpts_raw (np.ndarray): raw BPT/PTs (num_bpts, Nsp, Nc)
        """
        self.bpts_raw = np.load(os.path.join(self.inp_dir, "bpts.npy"))

    def _flatten_bpts(self):
        """
        Flatten BPT/PTs, combining all the MIMO signals.
        Stores:
            bpts_flat (np.ndarray): flattened BPT/PTs (Nsp, num_bpts*Nc)
        """
        n_bpts, n_spokes, n_coils = self.bpts_raw.shape
        self.bpts_flat = self.bpts_raw.transpose(1,0,2).reshape(n_spokes, n_bpts*n_coils)

    def _med_filt_bpts(self):
        """
        Median filter BPT/PTs, to remove spikes due to object entering BPT/PT frequencies.
        Stores:
            bpts_med (np.ndarray): median filtered BPT/PTs (Nsp, num_bpts*Nc)
        """
        self.bpts_med = medfilt(self.bpts_flat, kernel_size=(self.median_window, 1))

    def _lpf_bpts(self):
        """
        Low-pass filter BPT/PTs, to only keep the fastest expected motion
        Stores:
            bpts_lpf (np.ndarray): LPF BPT/PTs (Nsp, num_bpts*Nc)
        """
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
        N, F = self.bpts_lpf.shape
        out = np.zeros_like(self.bpts_lpf)
    
        for i in tqdm(range(F), desc="SSA"):
            x = torch.tensor(self.bpts_lpf[:, i], device=my_device, dtype=torch.float32)
            # Zero-copy Hankel matrix (trajectory) using as_strided
            D = x.as_strided((N - window_len + 1, window_len),
                             (x.stride(0), x.stride(0)))
            # Center and perform SVD
            means = D.mean(1, keepdim=True)
            U, s, Vh = torch.linalg.svd(D - means, full_matrices=False)
            # Filter out leading components and reconstruct trajectory
            S = (U[:, components_removed:] * s[components_removed:].unsqueeze(0)) @ Vh[components_removed:] + means
            # Vectorized overlap-add reconstruction
            res = torch.zeros(N + window_len, device=my_device)
            counts = torch.zeros(N + window_len, device=my_device)
            idx = torch.arange(window_len, device=my_device)
            base = torch.arange(S.shape[0], device=my_device).unsqueeze(1)
            positions = base + idx  # shape: (rows, window_len)
            # Scatter-add values and counts
            res.index_add_(0, positions.flatten(), S.flatten())
            counts.index_add_(0, positions.flatten(), torch.ones_like(S).flatten())
            # Extract valid region, normalize, and store result
            y = res[:N] / counts[:N]
            out[:, i] = y.cpu().numpy()
    
        return out