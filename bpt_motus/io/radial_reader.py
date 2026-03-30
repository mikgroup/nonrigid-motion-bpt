"""
Functions for reading UTE data extracted from ScanArchives.
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class MRIRaw:
    """Simple container for raw MRI data and metadata."""
    xk: np.ndarray
    coords: np.ndarray
    dcf: np.ndarray
    time: np.ndarray
    num_coils: int
    num_encodings: int
    num_frames: int
    trajectory_type = None
    ecg = None
    resp = None  
    prep = None

class RadialArchive:
    """
    Extract, load, and cache raw MRI data
    from a folder containing a radial ScanArchive and 2 Gating Track files.
    """
    def __init__(self, inp_dir: str):
        self.inp_dir: str = inp_dir
        self.archive_fname: str = ""
        self.metadata_fname: str = os.path.join(self.inp_dir, "metadata_dict.pkl")
        self.metadata_dict: dict = {}
        self.xk_time: np.ndarray = None
        self.coords_time: np.ndarray = None
        self.dcf_time: np.ndarray = None
        self.time_ordering: np.ndarray = None
        

    def get_metadata(self, force_reload: bool = False):
        """
        Load cached metadata if available, otherwise extract from archive and pcvipr output (if available).

        Args:
            force_reload (bool): If True, re-extract metadata even if cached.
        Stores:
            metadata_dict (dict): Dictionary with metadata, only cached if pcvipr metadata is also available.
        """
        if not force_reload and os.path.exists(self.metadata_fname):
            logger.info(f"Loading cached metadata from {self.metadata_fname}")
            with open(self.metadata_fname, "rb") as f:
                self.metadata_dict = pickle.load(f)
            return
        logger.info("Cached metadata not found / used — extracting.")
        # Get archive metadata
        self.archive_fname = self._find_archive_fname()
        archive = Archive(self.archive_fname)
        metadata = archive.Metadata()
        header = archive.Header()
        self.metadata_dict = dict(
            bw = header["rdb_hdr_image"]["vbw"],
            tr = header["rdb_hdr_image"]["tr"] * 1e-6, # in seconds
            fov = header["rdb_hdr_image"]["dfov"] * 1e-1, # in cm, RO direction
            nproj = header["rdb_hdr_rec"]["rdb_hdr_user8"],
            ncoils = metadata["numChannels"],
        )
        
        # Get post-pcvipr processing metadata
        header_fname = os.path.join(self.inp_dir, "pcvipr_header.txt")
        if not os.path.exists(header_fname):
            logger.warning("Pcvipr header not yet generated. Getting incomplete metadata without saving.")
            return
        pcvipr_header = self._read_pcvipr_header()
        self.metadata_dict.update(
            nr = pcvipr_header["nr"],
            imsize = (pcvipr_header["matrixx"], pcvipr_header["matrixy"], pcvipr_header["matrixz"])
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
            - xk_time (np.ndarray): time-ordered k-space, (Nc, Nsp, Nr)
            - coords_time (np.ndarray): time-ordered coords, (Nsp, Nr, 3)
            - dcf_time (np.ndarray): time-ordered density compensation function, (Nsp, Nr)
            - time_ordering (np.ndarray): time ordering of spokes, (Nsp,)
        """
        if not force_reload and os.path.exists(os.path.join(self.inp_dir, "xk.npy")):
            logger.info(f"Loading cached raw radial data from {self.inp_dir}...")
            self.xk_time = np.load(os.path.join(self.inp_dir, "xk.npy"))
            self.coords_time = np.load(os.path.join(self.inp_dir, "coords.npy"))
            self.dcf_time = np.load(os.path.join(self.inp_dir, "dcf.npy"))
            self.time_ordering = np.load(os.path.join(self.inp_dir, "time_ordering.npy"))
            return
        logger.info("Cached data not found... Extracting k-space.")
        self._extract_data_dict(force_reload=force_reload)
        np.save(os.path.join(self.inp_dir, "xk.npy"), self.xk_time)
        np.save(os.path.join(self.inp_dir, "coords.npy"), self.coords_time)
        np.save(os.path.join(self.inp_dir, "dcf.npy"), self.dcf_time)
        np.save(os.path.join(self.inp_dir, "time_ordering.npy"), self.time_ordering)

    def _extract_data_dict(self, force_reload: bool = False):
        """Extract data from cached MRI_Raw.h5, or generate it with Kevin's pcvipr recon."""
        mri_raw_fname = os.path.join(self.inp_dir, "MRI_Raw.h5")
        if not os.path.exists(mri_raw_fname) or force_reload:
            # Make MRI_Raw.h5
            self._run_pcvipr()

        # Read metadata
        if not self.metadata_dict:
            self.get_metadata()
        nr = self.metadata_dict["nr"]
        # Get k-space, coords, dcf, time ordering
        xk, coords, dcf, time = self._load_MRI_Raw()
        xk = copy.deepcopy(xk).reshape((xk.shape[0], -1, nr)) # (Nc, Nsp, Nr)
        coords = coords.reshape((-1, nr, 3)) # (Nsp, Nr, 3)
        dcf = dcf.reshape(xk.shape[1:]) # (Nsp, Nr)
        self.time_ordering = np.argsort(time, kind="stable")
        # Order k-space, coords, dcf in time
        self.xk_time = xk[:,self.time_ordering]
        self.coords_time = coords[self.time_ordering]
        self.dcf_time = dcf[self.time_ordering]

    def _run_pcvipr(self):
        """Run pcvipr binary from docker container to extract raw data from ScanArchive into MRI_Raw.h5. See GE3T wiki for details."""
        # Define command
        self.archive_fname = self._find_archive_fname()
        # We run as root inside to read gradients, then chown to your local UID:GID
        inner_cmd = (
            f"pcvipr_recon_binary -export_kdata -hdf5 -f {os.path.basename(self.archive_fname)} "
            f"-dont_use_ge_channel_weights -gradwarp > pcvipr_log_myrecon.txt 2>&1; "
            f"chown -R {os.getuid()}:{os.getgid()} ."
        ) # command within docker container
        docker_cmd = (
            f"docker run --rm --user root "
            f"-v {os.path.abspath(self.inp_dir)}:/data -w /data "
            f"ubuntu_orc3 /bin/bash -c '{inner_cmd}'"
        ) # mounts inp_dir to /data in container
        logger.info(f"Running Docker command: {inner_cmd}")
        
        try:
            # Note: Redirection is handled INSIDE the docker command now
            subprocess.run(docker_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker pcvipr execution failed: {e}")
            raise RuntimeError(f"pcvipr failed. Check {self.inp_dir}/pcvipr_log.txt")
        
    def _load_MRI_Raw(self) -> MRIRaw:
        """Load data from MRI_Raw.h5 after pcvipr extraction."""
        mri_raw_fname = os.path.join(self.inp_dir, "MRI_Raw.h5")
        try:
            with h5py.File(mri_raw_fname, "r") as hf:
                num_encodings = int(np.squeeze(hf["Kdata"].attrs["Num_Encodings"]))
                num_coils = int(np.squeeze(hf["Kdata"].attrs["Num_Coils"]))
                num_frames = int(np.squeeze(hf["Kdata"].attrs["Num_Frames"]))
        
                assert num_encodings == 1 and num_frames == 1
                encode = 0
        
                # --- Load coords ---
                coords_list = []
                for i in ["Z", "Y", "X"]:
                    kcoord = np.array(hf["Kdata"][f"K{i}_E{encode}"]).flatten()
                    if i == "Z":
                        if np.max(kcoord) - np.min(kcoord) < 1e-3:
                            continue
                    coords_list.append(kcoord)
        
                coords = np.stack(coords_list, axis=-1)   # (N,2) or (N,3)
        
                # --- Load DCF ---
                dcf = np.array(hf["Kdata"][f"KW_E{encode}"]).flatten() # (N,)
        
                # --- Load k-space ---
                xk = []
                for c in range(num_coils):
                    logging.info(f"Loading kspace, coil {c + 1} / {num_coils}.")
                    k = hf["Kdata"][f"KData_E{encode}_C{c}"]
                    try:
                        xk.append(np.array(k["real"] + 1j * k["imag"]).flatten())
                    except:
                        xk.append(k)
                xk = np.stack(xk, axis=0)  # (coils, N)
        
                # --- Load time + correct for mismatch ---
                try:
                    time_readout = np.array(hf["Gating"]["time"]).flatten()
                except Exception:
                    time_readout = np.array(hf["Gating"][f"TIME_E{encode}"]).flatten()
                return xk, coords, dcf, time_readout
        except:
            logger.error(f"MRI_Raw.h5 not found at: {mri_raw_fname}")
            return

    def _read_pcvipr_header(self):
        """Read the pcvipr_header.txt and pcvipr_log.txt files into a metadata dictionary."""
        header = {}
        header_fname = os.path.join(self.inp_dir, "pcvipr_header.txt")
        if not os.path.exists(header_fname):
            logger.warning(f"No header file found at {header_fname}")
            return header

        with open(header_fname, "r") as f:
            lines = f.read().split("\n")
            for line in lines:
                if not line=="" and not line[:5]=="pfile":
                    try:
                        key, val = line.split(" ")
                        header[key] = int(float(val))
                    except:
                        logger.warning("Cannot read line {line}")

        log_fname = os.path.join(self.inp_dir, "pcvipr_log.txt")
        if not os.path.exists(log_fname):
            logger.warning(f"No log file found at {log_fname}")
            return header
        with open(log_fname, "r") as f:
            lines = f.read().split("\n")
            for line in lines:
                if line.startswith("Xres:"):
                    header["nr"] = int(line.split()[-1])
        return header

    def _find_archive_fname(self):
        """Return the largest ScanArchive file ("Scan*.h5") in the input directory."""
        if self.archive_fname:
            return self.archive_fname # if it's already found
        archive_fnames = [f for f in os.listdir(self.inp_dir) if f.startswith("Scan") and f.endswith(".h5")]
    
        if not archive_fnames:
            raise FileNotFoundError(f"No Scan*.h5 files found in {self.inp_dir}")
    
        # Find the largest file
        sizes = [os.path.getsize(os.path.join(self.inp_dir, fname)) for fname in archive_fnames]
        max_ind = int(np.argmax(sizes))
        return os.path.join(self.inp_dir, archive_fnames[max_ind])