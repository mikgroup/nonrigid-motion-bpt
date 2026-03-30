"""
Functions for reading BRAVO MRI data extracted from ScanArchives.
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

class BravoArchive:
    """
    Wrapper for extracting, loading, and caching raw MRI data 
    from a folder containing a BRAVO ScanArchive.
    """
    def __init__(self, inp_dir: str):
        self.inp_dir: str = inp_dir
        self.archive_fname: str = ""
        self.metadata_dict: dict = {}
        self.xk_time: np.ndarray = None
        self.xk_recon: np.ndarray = None
        self.coords: np.ndarray = None

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
            nslices = int(header["rdb_hdr_rec"]["rdb_hdr_nslices"]),
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
                - 'coords' : coords, (Npe * Nslice, Nro, Ndim)
        """
        xk_time_fname  = os.path.join(self.inp_dir, "xk_time.npy")
        xk_recon_fname = os.path.join(self.inp_dir, "xk_recon.npy")
        coords_fname   = os.path.join(self.inp_dir, "coords.npy")

        if not force_reload and \
           os.path.exists(xk_time_fname) and \
           os.path.exists(xk_recon_fname) and \
           os.path.exists(coords_fname):

            logger.info(f"Loading cached k-space from {self.inp_dir}")
            self.xk_time  = np.load(xk_time_fname)
            self.xk_recon = np.load(xk_recon_fname)
            self.coords   = np.load(coords_fname)
            return

        logger.info("Cached data not found / used — extracting k-space.")
        self._extract_data()
        np.save(xk_time_fname,  self.xk_time)
        np.save(xk_recon_fname, self.xk_recon)
        np.save(coords_fname,   self.coords)

    # -------------------------
    # Internals
    # -------------------------

    def _extract_data(self):
        """Extract k-space and coordinates from ScanArchive."""
        if not self.metadata_dict:
            self.get_metadata()

        self.xk_time, self.xk_recon = self._extract_xk()
        kacq = self._extract_kacq() # used to generate coords

        _, _, ky, kz = kacq.T
        self.coords = self._create_coords(ky, kz, N=self.xk_time.shape[-1])

    def _extract_xk(self):
        """Extract time-ordered k-space (Ncoils, Npe * Nslice, Nro) and trajectory-ordered k-space (Ncoils, Nro, Npe, Nslice)."""
        self.archive_fname = self._find_archive_fname()
        archive = Archive(self.archive_fname)

        assert self.metadata_dict["npasses"] == 1 # Assume 1 pass

        xk_recon = np.zeros(
            [self.metadata_dict["xres"], self.metadata_dict["yres"], self.metadata_dict["ncoils"], self.metadata_dict["nslices"]],
            dtype=np.complex64
        )
        xk_time = []

        for _ in range(self.metadata_dict["ncontrol"]):
            control = archive.NextControl()

            # raw control packet; don't fill k-space
            if control["opcode"] == 16:
                _ = np.squeeze(archive.NextFrame())

            # programmable control packet; fill kspace
            elif control["opcode"] == 1 and \
                 0 < control["viewNum"] <= self.metadata_dict["yres"] and \
                 control["sliceNum"] < self.metadata_dict["nslices"]:

                frame = np.squeeze(archive.NextFrame())
                xk_recon[:, control["viewNum"]-1, :, control["sliceNum"]] = frame
                xk_time.append(frame)

            # scan control packet; pass finished
            elif control["opcode"] == 0:
                pass

        xk_time  = np.array(xk_time).transpose(2,0,1) # (Ncoils, Npe * Nslice, Nro)
        xk_recon = xk_recon.transpose(2,0,1,3) # (Ncoils, Nro, Npe, Nslice)
        return xk_time, xk_recon

    def _extract_kacq(self):
        """
        Extract k-space acquisition ordering (kacq).

        Priority:
        1) Embedded in ScanArchive
        2) Cached ksp_traj.txt
        3) Estimated from control packets
        """
        self.archive_fname = self._find_archive_fname()
        kacq_txt = os.path.join(self.inp_dir, "ksp_traj.txt")

        # --- Case 1: embedded kacq inside archive ---
        with h5py.File(self.archive_fname, "r") as f:
            psddata = None
            if "usr" in f and "g" in f["usr"] and "psddata" in f["usr"]["g"]:
                psddata = f["usr"]["g"]["psddata"]

            if psddata is not None:
                embedded_fname = list(psddata.keys())[0]
                full_path = os.path.join(self.inp_dir, embedded_fname)

                if os.path.exists(full_path):
                    logger.info(f"Loading cached kacq from {full_path}")
                    return np.loadtxt(full_path, skiprows=16)
                logger.info(f"Extracting kacq from ScanArchive to {full_path}")
                return self._get_kacq_from_archive(f, embedded_fname)

        # --- Case 2: cached kacq.txt exists ---
        if os.path.exists(kacq_txt):
            logger.info(f"Loading cached kacq from {kacq_txt}")
            return np.loadtxt(kacq_txt, delimiter=",")

        # --- Case 3: fallback estimation ---
        logger.info("No embedded or cached kacq found — estimating from control packets.")
        return self._estimate_kacq()

    def _get_kacq_from_archive(self, f, embedded_fname):
        """
        Extract kacq text from ScanArchive and save to .txt file.
        """
        txt_path = os.path.join(self.inp_dir, embedded_fname)

        kacq_bytes = f["usr"]["g"]["psddata"][embedded_fname][0]
        kacq_str = kacq_bytes.decode("utf-8")

        # Write full file (header + data)
        with open(txt_path, "w") as out:
            out.write(kacq_str)

        # Load numeric data (skip header)
        return np.loadtxt(txt_path, skiprows=16)
    
    def _estimate_kacq(self):
        """
        Estimate kacq from ScanArchive control packets.
        Used only as fallback when scanner does not embed trajectory.
        """
        txt_file = os.path.join(self.inp_dir, "ksp_traj.txt")

        if not self.metadata_dict:
            self.get_metadata()

        archive = Archive(self._find_archive_fname())
        assert self.metadata_dict["npasses"] == 1 # Assume 1 pass

        # Loop over packets
        pass_num = 0
        c = 0
        with open(txt_file, "w") as f:
            for _ in range(self.metadata_dict["ncontrol"]):
                control = archive.NextControl()

                # raw control packet; don't fill k-space
                if control["opcode"] == 16:
                    _ = np.squeeze(archive.NextFrame())

                # programmable control packet; fill kspace
                elif control["opcode"] == 1 and \
                    0 < control["viewNum"] <= self.metadata_dict["yres"] and \
                    control["sliceNum"] < self.metadata_dict["nslices"]:

                    frame = np.squeeze(archive.NextFrame())
                    if np.any(frame):
                        f.write(f"{c},{pass_num},{control['viewNum']-1},{control['sliceNum']}\n")
                    c += 1

                # scan control packet; pass finished
                elif control["opcode"] == 0:
                    pass

        return np.loadtxt(txt_file, delimiter=",")

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
        """Return the largest Scan*.h5 file in the input directory."""
        if self.archive_fname:
            return self.archive_fname

        archive_fnames = [f for f in os.listdir(self.inp_dir)
                          if f.startswith("Scan") and f.endswith(".h5")]

        if not archive_fnames:
            raise FileNotFoundError(f"No Scan*.h5 found in {self.inp_dir}")

        sizes = [os.path.getsize(os.path.join(self.inp_dir, f)) for f in archive_fnames]
        return os.path.join(self.inp_dir, archive_fnames[int(np.argmax(sizes))])