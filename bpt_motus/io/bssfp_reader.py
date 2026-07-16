"""
Functions for reading bSSFP MRI data extracted from ScanArchives.
"""
import os
import pickle
import logging
import subprocess
import h5py
import numpy as np
import copy
from dataclasses import dataclass
from GERecon import Archive
from tqdm.notebook import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class bSSFPArchive:
    """
    Wrapper for extracting, loading, and caching raw MRI data 
    from a folder containing a bSSFP ScanArchive.
    """
    def __init__(self, inp_dir: str):
        self.inp_dir: str = inp_dir
        self.archive_fname: str = ""
        self.metadata_dict: dict = {}
        self.xk_time: np.ndarray = None
        self.xk_recon: np.ndarray = None

    # -------------------------
    # Public API
    # -------------------------

    def get_metadata(self, force_reload: bool = False):
        """
        Load cached metadata if available, otherwise extract from archive.

        Args:
            force_reload (bool): If True, re-extract metadata even if cached.

        Stores and saves:
            metadata_dict (dict): Metadata.
        """
        metadata_fname = os.path.join(self.inp_dir, "metadata_dict.pkl")

        if not force_reload and os.path.exists(metadata_fname):
            logger.info(f"Loading cached metadata from {metadata_fname}")
            with open(metadata_fname, "rb") as f:
                self.metadata_dict = pickle.load(f)
            return

        logger.info("Cached metadata not found / used — extracting.")
        self.archive_fname = self._find_archive_fname()
        archive = Archive(self.archive_fname)
        metadata = archive.Metadata()
        header = archive.Header()

        self.metadata_dict = dict(
            bw = header["rdb_hdr_image"]["vbw"],
            tr = header["rdb_hdr_image"]["tr"] * 1e-6, # in seconds
            fov = header["rdb_hdr_image"]["dfov"] * 1e-1, # in cm, RO direction
            xres = metadata["acquiredXRes"],
            yres = metadata["acquiredYRes"],
            ncontrol = metadata["controlCount"],
            nslices_per_pass = archive.SlicesPerPass(),
            ncoils = metadata["numChannels"],
            npasses = metadata["passes"],
        )

        with open(metadata_fname, "wb") as f:
            pickle.dump(self.metadata_dict, f)

    def get_ksp(self, force_reload: bool = False):
        """
        Load cached k-space data if available, otherwise extract from archive.

        Args:
            force_reload (bool): If True, re-extract data even if cached.

        Stores:
            data_dict (dict): Data dictionary with keys:
                - 'xk_time'  : time-ordered k-space, (Ncoils, Npe * Nslice, Nro)
                - 'xk_recon' : trajectory-ordered k-space, (Ncoils, Nro, Npe, Nslice)
        """
        xk_time_fname  = os.path.join(self.inp_dir, "xk.npy")
        xk_recon_fname = os.path.join(self.inp_dir, "xk_recon.npy")

        if not force_reload and \
           os.path.exists(xk_time_fname) and \
           os.path.exists(xk_recon_fname):

            logger.info(f"Loading cached k-space from {self.inp_dir}")
            self.xk_time  = np.load(xk_time_fname)
            self.xk_recon = np.load(xk_recon_fname)
            return

        logger.info("Cached data not found / used — extracting k-space.")
        self._extract_data()
        np.save(xk_time_fname,  self.xk_time)
        np.save(xk_recon_fname, self.xk_recon)

    # -------------------------
    # Internals
    # -------------------------

    def _extract_data(self):
        """Extract k-space and coordinates from ScanArchive."""
        if not self.metadata_dict:
            self.get_metadata()

        self.xk_time, self.xk_recon = self._extract_xk()

    # def _extract_xk(self):
    #     """Extract time-ordered k-space (Ncoils, Npe * Nslice, Nro) and trajectory-ordered k-space (Ncoils, Nro, Npe, Nslice)."""
    #     self.archive_fname = self._find_archive_fname()
    #     archive = Archive(self.archive_fname)

    #     # Initialize both k-spaces
    #     xk_recon_all_passes = []
    #     pass_num = 0
    #     current_ksp = np.zeros(
    #         [self.metadata_dict['xres'], self.metadata_dict['yres'], self.metadata_dict['ncoils'], self.metadata_dict['nslices_per_pass'][pass_num]], 
    #         dtype=np.complex64
    #     )
    #     xk_time = []

    #     # Loop over packets
    #     for i in tqdm(range(self.metadata_dict['ncontrol']), desc="Extracting k-space"):
    #         control = archive.NextControl()
        
    #         # raw control packet; don't fill k-space
    #         if control['opcode'] == 16: 
    #             next_frame = np.squeeze(archive.NextFrame()) # keep control and frames in sync
        
    #         # programmable control packet; fill kspace
    #         elif control['opcode'] == 1 and 0 < control['viewNum'] <= self.metadata_dict['yres'] and control['sliceNum'] <= self.metadata_dict['nslices_per_pass'][pass_num]:
    #             next_frame = np.squeeze(archive.NextFrame())
    #             if len(next_frame.shape) == 1: # if 1 coil, add coil dimension
    #                 next_frame = next_frame[:, None]    
    #             current_ksp[:, control['viewNum'] - 1, :, control['sliceNum']] = next_frame
    #             xk_time.append(next_frame)
        
    #         # scan control packet; pass finished
    #         elif control['opcode'] == 0:
    #             pass_num += 1
    #             xk_recon_all_passes.append(current_ksp)
    #             if pass_num < self.metadata_dict['npasses']: # next pass
    #                 num_slices = self.metadata_dict['nslices_per_pass'][pass_num]
    #                 current_ksp = np.zeros([self.metadata_dict['xres'], self.metadata_dict['yres'], self.metadata_dict['ncoils'], num_slices], dtype=np.complex64)

    #     xk_time = np.array(xk_time).transpose(2,0,1) # (Ncoils, Npe * Nslice, Nro)
    #     xk_recon = np.array(xk_recon_all_passes).squeeze().transpose(0, 3, 1, 2) # (Npasses, Ncoils, Nro, Npe)
        
    #     # If only one pass, squeeze to maintain original 4D shape (Ncoils, Nro, Npe, Nslice)
    #     if xk_recon.shape[0] == 1:
    #         xk_recon = xk_recon.squeeze(axis=0)
            
    #     return xk_time, xk_recon

    def _extract_xk(self):
        """Extract time-ordered k-space (Ncoils, Npe * Nslice, Nro) and trajectory-ordered k-space (Ncoils, Nro, Npe, Nslice)."""
        self.archive_fname = self._find_archive_fname()
        archive = Archive(self.archive_fname)

        # Cache metadata for cleaner code
        xres = self.metadata_dict['xres']
        yres = self.metadata_dict['yres']
        ncoils = self.metadata_dict['ncoils']
        npasses = self.metadata_dict['npasses']
        nslices_per_pass = self.metadata_dict['nslices_per_pass']

        # Initialize tracking variables
        xk_recon_all_passes = []
        pass_num = 0
        
        # Grid order matrix shape: [Xres, Yres, Ncoils, Nslices]
        current_ksp = np.zeros([xres, yres, ncoils, nslices_per_pass[pass_num]], dtype=np.complex64)
        xk_time = []

        # Loop over packets
        for i in tqdm(range(self.metadata_dict['ncontrol']), desc="Extracting k-space"):
            control = archive.NextControl()
            
            # Raw control packet; don't fill k-space
            if control['opcode'] == 16: 
                _ = archive.NextFrame() # Read and discard to keep frames in sync
                
            # Programmable control packet; fill k-space
            elif control['opcode'] == 1 and 0 < control['viewNum'] <= yres and control['sliceNum'] <= nslices_per_pass[pass_num]:
                # Force the frame to always retain its 2D structure: [Nro, Ncoils]
                next_frame = np.atleast_2d(archive.NextFrame())
                if next_frame.shape[1] != ncoils and next_frame.shape[0] == ncoils:
                    next_frame = next_frame.T # Safeguard for orientation edge-cases
                
                # Map into our grid-ordered k-space matrix
                # next_frame[:, :] matches [xres, ncoils] exactly, even if ncoils == 1
                current_ksp[:, control['viewNum'] - 1, :, control['sliceNum']] = next_frame
                
                # Store for the time-ordered array (append transposed to match target [Ncoils, Nro])
                xk_time.append(next_frame.T)
                
            # Scan control packet; pass finished
            elif control['opcode'] == 0:
                xk_recon_all_passes.append(current_ksp)
                pass_num += 1
                if pass_num < npasses: # Initialize next pass
                    current_ksp = np.zeros([xres, yres, ncoils, nslices_per_pass[pass_num]], dtype=np.complex64)

        # Ensure the remaining data is appended if the last pass lacked a clean opcode 0
        if len(xk_recon_all_passes) < pass_num + 1 and pass_num < npasses:
            xk_recon_all_passes.append(current_ksp)

        # finalize Time-Ordered k-space: (Ncoils, Npe * Nslice, Nro)
        xk_time = np.stack(xk_time, axis=1) 
        # finalize Trajectory/Grid-Ordered k-space: (Npasses, Ncoils, Nro, Npe, Nslices)
        xk_recon = np.stack(xk_recon_all_passes, axis=0)
        xk_recon = xk_recon.transpose(0, 3, 1, 2, 4)
        if xk_recon.shape[0] == 1: # drop Npasses dimension cleanly if it's a single-pass scan
            xk_recon = xk_recon[0]
        if xk_recon.shape[-1] == 1: # drop Nslices dimension cleanly if it's a single-slice scan
            xk_recon = xk_recon[..., 0]
            
        return xk_time, xk_recon

    def _find_archive_fname(self):
        """Return the largest Scan*.h5 file in the input directory."""
        if self.archive_fname:
            return self.archive_fname

        archive_fnames = [f for f in os.listdir(self.inp_dir)
                          if f.startswith("Scan") and f.endswith(".h5")]

        if not archive_fnames:
            raise FileNotFoundError(f"No Scan*.h5 found in {self.inp_dir}")

        sizes = [os.path.getsize(os.path.join(self.inp_dir, f)) for f in archive_fnames]
        return os.path.join(self.inp_dir, archive_fnames[int(np.argmax(sizes))])