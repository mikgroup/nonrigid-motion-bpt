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

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class bSSFPArchive:
    """
    Wrapper for extracting, loading, and caching raw MRI data 
    from a folder containing a bSSFP ScanArchive.
    """
    def __init__(self, inp_dir: str, save_dir: str = "raw_data", data_file: str = "data_dict.pkl", metadata_file: str = "metadata_dict.pkl"):
        self.inp_dir: str = os.path.join(inp_dir, save_dir)
        self.archive_fname: str = ""
        self.data_fname: str = os.path.join(self.inp_dir, data_file)
        self.metadata_fname: str = os.path.join(self.inp_dir, metadata_file)
        self.data_dict: dict = {}
        self.metadata_dict: dict = {}

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
        if not force_reload and self.metadata_dict:
            return # if metadata_dict is already stored
        if os.path.exists(self.metadata_fname):
            logger.info(f"Loading cached metadata from {self.metadata_fname}")
            with open(self.metadata_fname, "rb") as f:
                self.metadata_dict = pickle.load(f)
            return
        logger.info("Cached metadata not found / used — extracting.")
        self.archive_fname = self._find_archive_fname()
        archive = Archive(self.archive_fname)
        metadata = archive.Metadata()
        header = archive.Header()
        self.metadata_dict = dict(
            bw = header['rdb_hdr_image']['vbw'],
            tr = header['rdb_hdr_image']['tr'] * 1e-6, # in seconds
            fov = header['rdb_hdr_image']['dfov'] * 1e-1, # in cm, RO direction
            xres = metadata['acquiredXRes'],
            yres = metadata['acquiredYRes'],
            ncontrol = metadata['controlCount'],
            nslices = int(header['rdb_hdr_rec']['rdb_hdr_nslices']),
            ncoils = metadata['numChannels'],
            npasses = metadata['passes'],
        )
        with open(self.metadata_fname, "wb") as f:
            pickle.dump(self.metadata_dict, f)
        return
        
    def get_ksp(self, force_reload: bool = False):
        """
        Load cached k-space data if available, otherwise extract from archive.

        Args:
            force_reload (bool): If True, re-extract data even if cached.

        Stores:
            data_dict (dict): Data dictionary with keys:
                - 'xk_time'  : time-ordered k-space, (Ncoils, Npe * Nslice, Nro)
                - 'xk_recon' : trajectory-ordered k-space, (Ncoils, Nro, Npe, Nslice)
                - 'coords' : coords, (Npe * Nslice, Nro, Ndim)
        """
        if not force_reload and os.path.exists(self.data_fname):
            logger.info(f"Loading cached k-space from {self.data_fname}")
            with open(self.data_fname, "rb") as f:
                self.data_dict = pickle.load(f)
            return

        logger.info("Cached data not found / used — extracting k-space.")
        self.data_dict = self._extract_data_dict()
        with open(self.data_fname, "wb") as f:
            pickle.dump(self.data_dict, f)

    # -------------------------
    # Internals
    # -------------------------
    def _extract_data_dict(self):
        """Extract all data from ScanArchive."""
        # --- Load k-space ---
        xk_time, xk_recon = self._extract_xk()
        # --- Load coords ---
        kacq = self._extract_kacq()
        view, pass_no, ky, kz = kacq.T
        coords = self._create_coords(ky, kz, N=xk_time.shape[-1])

        return dict(
            xk_time=xk_time,
            xk_recon=xk_recon,
            coords=coords,
        )

    def _extract_xk(self):
        """Extract time-ordered k-space (Ncoils, Npe * Nslice, Nro) and trajectory-ordered k-space (Ncoils, Nro, Npe, Nslice)."""
        self.archive_fname = self._find_archive_fname()
        if not self.metadata_dict:
            self.get_metadata()
        assert self.metadata_dict['npasses'] == 1 # Assume 1 pass
        archive = Archive(self.archive_fname)

        # Initialize both k-spaces
        xk_recon = np.zeros([self.metadata_dict['xres'], self.metadata_dict['yres'], self.metadata_dict['ncoils'], self.metadata_dict['nslices']], dtype=np.complex64)
        xk_time = []

        # Loop over packets
        for i in range(self.metadata_dict['ncontrol']):
            control = archive.NextControl()
        
            # raw control packet; don't fill k-space
            if control['opcode'] == 16: 
                next_frame = np.squeeze(archive.NextFrame()) # keep control and frames in sync
        
            # programmable control packet; fill kspace
            elif control['opcode'] == 1 and 0 < control['viewNum'] <= self.metadata_dict['yres'] and control['sliceNum'] < self.metadata_dict['nslices']:
                next_frame = np.squeeze(archive.NextFrame())
                xk_recon[:, control['viewNum'] - 1, :, control['sliceNum']] = next_frame
                xk_time.append(next_frame)
        
            # scan control packet; pass finished
            elif control['opcode'] == 0:
                pass # do nothing, as we're assuming there's 1 pass
        xk_time = np.array(xk_time).transpose(2,0,1) # (Ncoils, Npe * Nslice, Nro)
        xk_recon = xk_recon.transpose(2,0,1,3) # (Ncoils, Nro, Npe, Nslice)
        return xk_time, xk_recon

    def _extract_kacq(self):
        """Generate or read kacq (k-space ordering) from ScanArchive."""
        self.archive_fname = self._find_archive_fname()
        f = h5py.File(self.archive_fname, 'r')
        keys = ['usr', 'g', 'psddata']
        
        # If there is no kacq object, generate the trajectory from the Archive directly
        if self.check_h5_keys(f, keys) is None:
            logger.info("No kacq file at all, estimating trajectory from ScanArchive control packets (in ksp_traj.txt)")
            return self._estimate_kacq()
        else:
            # Get kacq file name
            kacq_fname = self._get_kacq_fname()
            # If kacq .txt file exists, read it
            if os.path.exists(kacq_fname):
                logger.info(f"Kacq .txt file exists. Reading {kacq_fname}")
                kacq = np.loadtxt(kacq_fname, skiprows=16)
            # Otherwise, try to read from ScanArchive
            else:
                logger.info(f"Kacq data is in ScanArchive. Generating {kacq_fname}")
                kacq = self._get_kacq_from_archive(f, kacq_fname)
        return kacq

    def _check_h5_keys(self, h5_obj, keys):
        """
        Check if a series of nested keys exists in an HDF5 object.
        
        Args:
        - h5_obj: h5py object (File, Group, or Dataset)
        - keys: list of strings representing the nested keys
        
        Returns:
        - The final HDF5 object if all keys exist, otherwise None
        """
        for key in keys:
            if key in h5_obj:
                h5_obj = h5_obj[key]
            else:
                return None
        return h5_obj
    
    def _estimate_kacq(self):
        """Estimate kacq from ScanArchive control packets."""
        txt_file = os.path.join(self.inp_dir, "ksp_traj.txt")
        if not os.path.exists(txt_file):
            logging.info("No ksp_traj.txt saved, generating now.")
            self.archive_fname = self._find_archive_fname() 
            if not self.metadata_dict:
                self.get_metadata()
            assert self.metadata_dict['npasses'] == 1 # Assume 1 pass
            archive = Archive(self.archive_fname)
            
            # Loop over packets
            pass_num=0
            c=0
            for i in range(self.metadata_dict['ncontrol']):
                control = archive.NextControl()
        
                # raw control packet
                if control['opcode'] == 16: 
                    next_frame = np.squeeze(archive.NextFrame()) # keep control and frames in sync
            
                # programmable control packet; fill trajectory
                elif control['opcode'] == 1 and 0 < control['viewNum'] <= self.metadata_dict['yres'] and control['sliceNum'] < self.metadata_dict['nslices']:
                    next_frame = np.squeeze(archive.NextFrame())
                    if np.any(next_frame):
                        with open(txt_file, 'a') as file:
                            line = f"{c},{pass_num},{control['viewNum']-1},{control['sliceNum']}\n"
                            file.write(line)
                    c += 1
            
                # scan control packet; pass finished
                elif control['opcode'] == 0:
                    pass # do nothing, as we're assuming there's 1 pass
        # Load written file
        kacq = np.loadtxt(txt_file, delimiter=",")
        return kacq

    def _get_kacq_fname(self):
        """Get kacq filename from ScanArchive."""
        self.archive_fname = self._find_archive_fname()
        f = h5py.File(self.archive_fname, 'r')
        kacq_fname = list(f['usr']['g']['psddata'].keys())[0]
        return os.path.join(self.inp_dir, kacq_fname)

    def _get_kacq_from_archive(self, f, kacq_fname):
        """Get kacq by reading h5 directly. Cache to kacq .txt file."""
        kacq_bytes = f['usr']['g']['psddata'][os.path.basename(kacq_fname)][0]
        # Get data
        kacq_str = kacq_bytes.decode('UTF-8')
        kacq_data = kacq_str.split('\n')[16:]
        # Make array
        kacq = np.empty((len(kacq_data),4))
        for i in range(len(kacq_data)-1):
            kacq[i,:] = np.array(kacq_data[i].split('\t')).astype(int)
        # Write header and data to .txt file
        with open(kacq_fname, 'w') as f:
            for i in range(len(kacq_str)):
                f.write(kacq_str[i])
        return kacq

    def _create_coords(self, ky, kz, N=256):
        """Create coords array (Npe * Nslice, Nro, Ndim) based on ky, kz (from kacq)."""
        ky_max = np.amax(ky)
        kz_max = np.amax(kz)
        nro = np.arange(N) - N//2

        ky, kx = np.meshgrid(ky - ky_max//2, nro)
        kz, _ = np.meshgrid(kz - kz_max//2, nro)
        coords = np.stack((kx, ky, kz), axis=(-1)).transpose(1,0,2)
        return coords
        
    def _find_archive_fname(self):
        """Return the largest ScanArchive file ('Scan*.h5') in the input directory."""
        if self.archive_fname:
            return self.archive_fname # if it's already found
        archive_fnames = [f for f in os.listdir(self.inp_dir) if f.startswith("Scan") and f.endswith(".h5")]
    
        if not archive_fnames:
            raise FileNotFoundError(f"No Scan*.h5 files found in {self.inp_dir}")
    
        # Find the largest file
        sizes = [os.path.getsize(os.path.join(self.inp_dir, fname)) for fname in archive_fnames]
        max_ind = int(np.argmax(sizes))
        return os.path.join(self.inp_dir, archive_fnames[max_ind])