"""
Functions for splitting raw data from the head motion study acquisitions into the three BPT-MOTUS phases: no-motion reference, motion calibration, and motion inference.
"""
import os
import numpy as np
from typing import Tuple, Literal
import logging
import pickle as pkl

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SplitRadialAcq:
    """
    Split raw data from the head motion study acquisitions into the 
    three BPT-MOTUS phases: 
    1. No-motion reference (always from the hires acquisition)
    2. Motion calibration (either from the lowres or hires radial acquisition)
    3. Motion inference (either on the lowres or remaining hires radial acquisition).
    """
    def __init__(self, inp_dir: str, verbose: bool = True, 
                 hires_dir: str = "hires_ute", lowres_dir: str = "lowres_ute", 
                 calib_source: Literal["hires", "lowres"] = "lowres", 
                 no_motion_range: Tuple[int, int] = (None, None), 
                 calib_range: Tuple[int, int] = (None, None), inf_range: Tuple[int, int] = (None, None)):
        self.verbose: bool = verbose
        self.inp_dir: str = inp_dir
        self.hires_dir: str = os.path.join(inp_dir, hires_dir)
        self.lowres_dir: str = os.path.join(inp_dir, lowres_dir)
        self.calib_source: str = calib_source

        self.no_motion_range = no_motion_range
        self.calib_range = calib_range 
        self.inf_range = inf_range 

        # Output directories
        self.no_motion_dir = os.path.join(self.inp_dir, "bpt_motus", "no_motion")
        self.calib_inf_dir = self._make_calib_inf_dir_name()
        self.calib_dir = os.path.join(self.calib_inf_dir, "calib")
        self.inf_dir = os.path.join(self.calib_inf_dir, "inf")
        
        if not os.path.isdir(self.hires_dir):
            logger.error(f"Hires directory not found: {self.hires_dir}")
        if self.calib_source == "lowres" and not os.path.isdir(self.lowres_dir):
            logger.error(f"Lowres directory not found: {self.lowres_dir}")

    def run(self, force_reload: bool = False):
        """
        Reorganize the radial acquisition(s) into three phases.
        """
        if not force_reload and os.path.exists(self.no_motion_dir):
            if self.verbose:
                logger.info("Found split datasets. No need to split again.") 
        else:
            # No motion data
            if self.no_motion_range is not None:
                if self.verbose:
                    logger.info("Generating no motion dataset.")
                nm_s, nm_e = self.no_motion_range
                xk_hr, coords_hr, dcf_hr = self._load_raw_radial(self.hires_dir)
                xk_nm, coords_nm, dcf_nm = self._subset(xk_hr, coords_hr, dcf_hr, nm_s, nm_e)
                md_hr = self._load_metadata(self.hires_dir)
                self._save_raw_radial(self.no_motion_dir, xk_nm, coords_nm, dcf_nm)
                self._save_metadata(self.no_motion_dir, md_hr)
                del xk_nm, coords_nm, dcf_nm
            # Calibration data
            if self.verbose:
                logger.info("Generating calibration dataset.")
            c_s, c_e = self.calib_range
            if self.calib_source == "lowres":
                xk_lr, coords_lr, dcf_lr = self._load_raw_radial(self.lowres_dir)
                xk_c, coords_c, dcf_c = self._subset(xk_lr, coords_lr, dcf_lr, c_s, c_e)
                md_lr = self._load_metadata(self.lowres_dir)
                self._save_metadata(self.calib_dir, md_lr)
                del xk_lr, coords_lr, dcf_lr
            else:
                xk_c, coords_c, dcf_c = self._subset(xk_hr, coords_hr, dcf_hr, c_s, c_e)
                self._save_metadata(self.calib_dir, md_hr)
            self._save_raw_radial(self.calib_dir, xk_c, coords_c, dcf_c)
            # Inference data
            if self.verbose:
                logger.info("Generating inference dataset.")
            i_s, i_e = self.inf_range
            xk_inf, coords_inf, dcf_inf = self._subset(xk_hr, coords_hr, dcf_hr, i_s, i_e)
            self._save_raw_radial(self.inf_dir, xk_inf, coords_inf, dcf_inf)
            self._save_metadata(self.inf_dir, md_hr)
    
    def _make_calib_inf_dir_name(self):
        """
        Make top folder to hold calibration / inference data and results.
        Returns: calib_inf_dir (str): Name of folder created.
        """
        A, B = self.calib_range
        C, D = self.inf_range
        if self.calib_source == "lowres":
            out_folder = f"calib_lr_{A}_{B}_inf_hr_{C}_{D}"
        else:  # "hires"
            out_folder = f"calib_hr_{A}_{B}_inf_hr_{C}_{D}"

        return os.path.join(self.inp_dir, "bpt_motus", "calib_inf", out_folder)
        
    def _load_raw_radial(self, raw_data_dir: str):
        """Load raw radial data that was extracted from ScanArchives."""
        xk = np.load(os.path.join(raw_data_dir, "xk.npy"))
        coords = np.load(os.path.join(raw_data_dir, "coords.npy"))
        dcf = np.load(os.path.join(raw_data_dir, "dcf.npy"))
        return xk, coords, dcf

    def _subset(self, xk, coords, dcf, s, e):
        """Take subset (from s to e) of spokes from raw radial data."""
        return xk[:,s:e], coords[s:e], dcf[s:e]
    
    def _load_metadata(self, raw_data_dir: str):
        """Load metadata dictionary from original raw data."""
        with open(os.path.join(raw_data_dir, "metadata_dict.pkl"), "rb") as f:
            metadata = pkl.load(f)
        return metadata

    def _save_raw_radial(self, save_dir: str, xk, coords, dcf):
        """Save raw radial data (now subsetted)."""
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "xk.npy"), xk)
        np.save(os.path.join(save_dir, "coords.npy"), coords)
        np.save(os.path.join(save_dir, "dcf.npy"), dcf)

    def _save_metadata(self, save_dir: str, metadata: dict):
        """Save metadata dictionary from original raw data."""
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "metadata_dict.pkl"), "wb") as f:
            pkl.dump(metadata, f)