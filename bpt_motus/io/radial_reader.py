"""
Functions for reading radial MRI data (e.g., UTE) extracted from ScanArchives.
"""
import os
import pickle
import logging
import subprocess
import h5py
import numpy as np
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MRIRaw:
    """Simple container for raw MRI data and metadata."""
    kdata: np.ndarray
    kcoords: np.ndarray
    dcf: np.ndarray
    timevec: np.ndarray
    num_coils: int
    num_encodings: int
    num_frames: int


class RadialArchive:
    """
    Wrapper for extracting, loading, and caching raw MRI data
    from a folder containing a radial ScanArchive.
    """

    def __init__(self, inpdir: str, save_name: str = "data_dict.pkl"):
        self.inpdir = inpdir
        self.archive_name = None
        self.save_name = os.path.join(inpdir, save_name)
        self.h5_file = os.path.join(inpdir, "MRI_Raw.h5")
        self.header_file = os.path.join(inpdir, "data_header.txt")
        self.data_dict = None

    # -------------------------
    # Public API
    # -------------------------
    def get_ksp(self, force_reload: bool = False):
        """
        Load cached k-space data if available, otherwise extract from archive.

        Args:
            force_reload (bool): If True, re-extract data even if cached.

        Returns:
            dict[str, np.ndarray]: Dictionary with keys:
                - 'kdata'  : k-space, (Nc, Nproj, Nr)
                - 'coords' : coords, (Nproj, Nr, 3)
                - 'dcf'    : density compensation function, (Nproj, Nr)
                - 'time_ordering' : time ordering of spokes, (Nproj,)
                - 'data_header'   : dictionary of metadata
        """
        if not force_reload and os.path.exists(self.save_name):
            logger.info(f"Loading cached k-space from {self.save_name}")
            with open(self.save_name, "rb") as f:
                self.data_dict = pickle.load(f)
            return self.data_dict

        logger.info("Cached data not found — extracting k-space.")
        self.data_dict = self._extract_ksp()
        with open(self.save_name, "wb") as f:
            pickle.dump(self.data_dict, f)
        return self.data_dict

    # -------------------------
    # Internals
    # -------------------------
    def _extract_ksp(self):
        """Extract data from cached MRI_Raw.h5 with pcvipr."""
        if not os.path.exists(self.h5_file):
            self._run_pcvipr()

        MRI_Raw = self._load_MRI_Raw()
        data_header = self._read_pcvipr_txt()

        kdata = MRI_Raw.kdata
        coords = MRI_Raw.kcoords
        dcf = np.reshape(MRI_Raw.dcf, (kdata.shape[1], kdata.shape[2]))
        timevec_proj = MRI_Raw.timevec.reshape((-1, kdata.shape[2]))[:, 0]
        time_ordering = np.argsort(timevec_proj, kind="stable")

        return dict(
            kdata=kdata,
            coords=coords,
            dcf=dcf,
            time_ordering=time_ordering,
            data_header=data_header,
        )

    def _run_pcvipr(self):
        """Run pcvipr binary to extract raw data from ScanArchive into MRI_Raw.h5."""
        # Set environmental vars
        os.environ["VDS_GRADIENT_PATH"] = "/mikQNAP/sanand/UTE/support_files/"
        # Define command
        if not self.archive_name:
            self.archive_name = self._find_archive_file() # choose largest ScanArchive in directory
        cmd = [
            "/mikQNAP/sanand/UTE/pcvipr_recon_binary",
            "-export_kdata",
            "-hdf5",
            "-f", self.archive_name,
            "-dont_use_ge_channel_weights"
        ]
        log_fname = "pcvipr_log.txt"
        logger.info(f"Running command: {' '.join(cmd)}")
        
        original_cwd = os.getcwd()
        os.chdir(self.inpdir)
        try:
            with open(log_fname, 'w') as log_file:
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False # Do not raise an exception automatically
                )
            if result.returncode != 0:
                logger.error(f"pcvipr command failed with return code {result.returncode}.")
                raise RuntimeError(
                    f"pcvipr command failed (return code {result.returncode}). Check {os.path.join(self.inpdir, log_fname)} for details."
                )
        
        except FileNotFoundError:
            logger.error(f"pcvipr binary not found at: {cmd[0]}")
        except Exception as e:
            logger.error(f"Error during pcvipr execution: {e}")
        finally:
            os.chdir(original_cwd)
        
    def _load_MRI_Raw(self) -> MRIRaw:
        """Load MRI_Raw.h5 file produced by pcvipr into MRIRaw object."""
        with h5py.File(self.h5_file, "r") as hf:
            num_coils = hf.attrs.get("Num_Coils", 1)
            num_encodings = hf.attrs.get("Num_Encodings", 1)
            num_frames = hf.attrs.get("Num_Frames", 1)

            kdata = hf["Kdata"][()]
            kcoords = hf["Kcoords"][()]
            dcf = hf["DCF"][()]
            timevec = hf["Time"][()]

        return MRIRaw(
            kdata=kdata,
            kcoords=kcoords,
            dcf=dcf,
            timevec=timevec,
            num_coils=num_coils,
            num_encodings=num_encodings,
            num_frames=num_frames,
        )

    def _read_pcvipr_txt(self):
        """Read the data_header.txt file into a metadata dictionary."""
        header = {}
        if not os.path.exists(self.header_file):
            logger.warning(f"No header file found at {self.header_file}")
            return header

        with open(self.header_file, "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":", 1)
                    header[key.strip().lower()] = val.strip()
        return header

    def _find_archive_file(self):
        """Return the largest ScanArchive file ('Scan*.h5') in the input directory."""
        archive_fnames = [f for f in os.listdir(self.inpdir) if f.startswith("Scan") and f.endswith(".h5")]
    
        if not archive_fnames:
            raise FileNotFoundError(f"No Scan*.h5 files found in {self.inpdir}")
    
        # Find the largest file
        sizes = [os.path.getsize(os.path.join(self.inpdir, fname)) for fname in archive_fnames]
        max_ind = int(np.argmax(sizes))
        return os.path.join(self.inpdir, archive_fnames[max_ind])