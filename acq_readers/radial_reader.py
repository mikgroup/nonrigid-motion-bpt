"""
Functions for reading radial MRI data (e.g., UTE) extracted from ScanArchives.
"""

import os
import numpy as np
import h5py
import logging
import copy
import pickle
import subprocess # Use subprocess instead of os.system
from typing import Dict, Any, Tuple

# Set up logging for better control over output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# High-level wrapper (Caching)
# -----------------------------------------------------------------------------

def get_ksp(inpdir: str, save_name: str = "data_dict.pkl") -> Dict[str, Any]:
    """
    Wrapper for extracting or loading cached k-space data.

    Checks if a pickled data_dict exists; if so, loads it.
    Otherwise, calls extract_ksp(), saves the result, and returns it.
    """
    save_path = os.path.join(inpdir, save_name)

    if os.path.exists(save_path):
        logger.info(f"Loading cached k-space data from {save_path}")
        try:
            with open(save_path, "rb") as f:
                return pickle.load(f)
        except pickle.UnpicklingError as e:
            logger.warning(f"Failed to unpickle cache ({e}). Re-extracting data.")
            # Fall through to extraction if cache is corrupt

    data = extract_ksp(inpdir)
    try:
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved extracted data to {save_path}")
    except Exception as e:
        # Catch exceptions during saving (e.g., permission issues)
        logger.error(f"Failed to save extracted data to {save_path}: {e}")
        
    return data


# -----------------------------------------------------------------------------
# Core data loading
# -----------------------------------------------------------------------------

def extract_ksp(inpdir: str) -> Dict[str, Any]:
    """
    Extract k-space, coordinates, DCF, and time ordering from an MRI_Raw.h5 file.
    Runs pcvipr if the .h5 file doesn’t exist yet.
    """
    file_path = os.path.join(inpdir, 'MRI_Raw.h5')

    # Automatically generate MRI_Raw.h5 if missing
    if not os.path.exists(file_path):
        scanarchives = [f for f in os.listdir(inpdir) if f.endswith(".ScanArchive")]
        
        if not scanarchives:
            raise FileNotFoundError(
                f"No MRI_Raw.h5 or .ScanArchive file found in {inpdir}"
            )
        
        # Only process the first ScanArchive found
        fname = scanarchives[0]
        logger.info(f"MRI_Raw.h5 not found; running pcvipr on {fname}")
        run_pcvipr(inpdir, fname)
        
        # Check if the file was created successfully
        if not os.path.exists(file_path):
            raise RuntimeError(f"pcvipr failed to create MRI_Raw.h5 in {inpdir}. Check pcvipr_log.txt.")

    logger.info(f"Reading {file_path}...")
    raw_kdata = load_MRI_raw(file_path)

    header_path = os.path.join(inpdir, 'data_header.txt')
    if not os.path.exists(header_path):
        raise FileNotFoundError(f"{header_path} missing in {inpdir}. This file is required for metadata.")

    logger.info("Extracting parameters from .txt files...")
    data_header = ReadPcviprTxt(header_path)
    
    # Ensure keys exist before accessing
    try:
        Nr = data_header['xres']
        Nrecon = (data_header['rcxres'], data_header['rcyres'], data_header['rczres'])
        Nproj = data_header['nproj']
    except KeyError as e:
        raise ValueError(f"Required parameter {e} not found in data_header.txt")

    logger.info("Extracting data from MRI_Raw object...")
    kdata_raw, coords, dcf, timevec = Raw2Array(raw_kdata)

    logger.info("Reshaping and sorting data...")
    # NOTE: Removed copy.deepcopy as reshape is typically a view or handled by assignment
    # kdata_raw.shape is (Ncoils, Nprojections * Nreadouts)
    Ncoils = kdata_raw.shape[0]
    
    kdata = kdata_raw.reshape((Ncoils, -1, Nr)) # Shape: (Ncoils, Nprojections, Nreadouts)
    coords = coords.reshape((-1, Nr, 3))        # Shape: (Nprojections, Nreadouts, 3)
    dcf = dcf.reshape(kdata.shape[1:])          # Shape: (Nprojections, Nreadouts)
    
    # Time vector handling
    # timevec.shape is (Nprojections * Nreadouts, )
    # Reshape to (Nprojections, Nreadouts) and take the first column (time per projection)
    timevec_proj = timevec.reshape((-1, Nr))[:, 0]

    # Calculate time ordering (redundant line removed)
    time_ordering = np.argsort(timevec_proj)

    data = dict(
        kdata=kdata,
        coords=coords,
        dcf=dcf,
        time_ordering=time_ordering,
        Nr=Nr,
        Nrecon=Nrecon,
        Nproj=Nproj,
    )
    return data


# -----------------------------------------------------------------------------
# Run pcvipr binary (if raw data hasn't been extracted yet)
# -----------------------------------------------------------------------------

def run_pcvipr(inpdir: str, fname: str):
    """
    Run the pcvipr binary to extract raw MRI data from a ScanArchive.
    Creates an MRI_Raw.h5 file in the same directory.
    Uses subprocess.run for safer execution.
    """
    # NOTE: Hardcoded paths are common in research, but limit portability.
    # Consider making these configurable via environment variables or arguments.
    os.environ["VDS_GRADIENT_PATH"] = "/mikQNAP/sanand/UTE/support_files/"

    # Define the command and arguments as a list
    cmd = [
        "/mikQNAP/sanand/UTE/pcvipr_recon_binary",
        "-export_kdata",
        "-hdf5",
        "-f", fname,
        "-dont_use_ge_channel_weights"
    ]
    
    log_fname = "pcvipr_log.txt"

    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Change directory before running the command
    original_cwd = os.getcwd()
    os.chdir(inpdir)
    
    try:
        with open(log_fname, 'w') as log_file:
            # Execute the command, writing stdout and stderr to the log file
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False # Do not raise an exception automatically
            )
        
        if result.returncode != 0:
            logger.error(f"pcvipr command failed with return code {result.returncode}.")
            logger.error(f"See {os.path.join(inpdir, log_fname)} for details.")
            # Do not raise here; let extract_ksp check for the resulting H5 file.

    except FileNotFoundError:
        logger.error(f"pcvipr binary not found at: {cmd[0]}")
    except Exception as e:
        logger.error(f"Error during pcvipr execution: {e}")
    finally:
        os.chdir(original_cwd)


# -----------------------------------------------------------------------------
# MRI_Raw container and helpers
# -----------------------------------------------------------------------------

class MRI_Raw:
    """Lightweight structure for storing MRI raw data and attributes."""
    def __init__(self):
        self.Num_Encodings = 0
        self.Num_Coils = 0
        self.Num_Frames = 0
        self.trajectory_type = None
        self.dft_needed = None
        self.coords = []
        self.dcf = []
        self.kdata = []
        self.time = []
        self.ecg = []
        self.resp = []
        self.prep = []
        self.target_image_size = [256, 256, 64]


def load_MRI_raw(h5_filename: str, max_coils: int = None, max_encodes: int = None,
                 compress_coils: bool = False, scale_kspace: bool = False) -> MRI_Raw:
    """Load the MRI_Raw.h5 file exported by pcvipr."""
    with h5py.File(h5_filename, 'r') as hf:
        mri_raw = MRI_Raw()

        try:
            # Use .item() or [()] for scalar extraction from h5py dataset
            mri_raw.Num_Encodings = int(hf['Kdata'].attrs['Num_Encodings'].item())
            mri_raw.Num_Coils = int(hf['Kdata'].attrs['Num_Coils'].item())
            mri_raw.Num_Frames = int(hf['Kdata'].attrs['Num_Frames'].item())
        except (KeyError, AttributeError, ValueError):
            logger.warning("Missing header attributes in Kdata; continuing without explicit attributes.")

        # Handle optional truncation for coils/encodes
        if max_coils is not None:
            mri_raw.Num_Coils = min(max_coils, mri_raw.Num_Coils)
        if max_encodes is not None:
            mri_raw.Num_Encodings = min(max_encodes, mri_raw.Num_Encodings)

        # Loop through all available encodings/frames
        total_encodings_to_read = mri_raw.Num_Encodings * mri_raw.Num_Frames
        
        for encode in range(total_encodings_to_read):
            coords_list = []
            
            # Load Kx, Ky, Kz
            for i, axis in enumerate(['X', 'Y', 'Z']):
                kcoord = np.array(hf['Kdata'][f'K{axis}_E{encode}']).flatten()
                
                # Check for 2D data where Kz is essentially zero
                if axis == 'Z' and (np.max(kcoord) - np.min(kcoord)) < 1e-3:
                    continue
                
                coords_list.append(kcoord)
                
            # Stack coords (typically Kx, Ky, Kz)
            coords = np.stack(coords_list, axis=-1)

            # Load DCF (Density Compensation Factor)
            dcf = np.array(hf['Kdata'][f'KW_E{encode}'])
            
            # Load K-space data for all coils
            ksp_list = []
            for c in range(mri_raw.Num_Coils):
                ksp_real = np.array(hf['Kdata'][f'KData_E{encode}_C{c}']['real']).flatten()
                ksp_imag = np.array(hf['Kdata'][f'KData_E{encode}_C{c}']['imag']).flatten()
                ksp_list.append(ksp_real + 1j * ksp_imag)
            
            ksp = np.stack(ksp_list, axis=0) # Shape: (Num_Coils, Nreadouts)

            # Gating data safety checks (robust use of .get with default zero array)
            time = np.array(hf['Gating'].get('time', np.zeros_like(dcf)))
            resp = np.array(hf['Gating'].get('resp', np.zeros_like(dcf)))
            ecg = np.array(hf['Gating'].get('ecg', np.zeros_like(dcf)))
            prep = np.array(hf['Gating'].get('prep', np.zeros_like(dcf)))

            mri_raw.coords.append(coords)
            mri_raw.dcf.append(dcf)
            mri_raw.kdata.append(ksp)
            mri_raw.time.append(time)
            mri_raw.resp.append(resp)
            mri_raw.ecg.append(ecg)
            mri_raw.prep.append(prep)

        if scale_kspace:
            # Normalize k-space data by the global maximum magnitude
            kmax = max(np.abs(k).max() for k in mri_raw.kdata)
            if kmax > 0:
                for i in range(len(mri_raw.kdata)):
                    mri_raw.kdata[i] /= kmax
            else:
                logger.warning("K-space data appears to be all zero; skipping scaling.")

        return mri_raw


def Raw2Array(MRI_raw: MRI_Raw) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert MRI_Raw lists into NumPy arrays."""
    if MRI_raw.Num_Encodings != 1:
        raise ValueError(
            f"Expected Num_Encodings == 1, but found {MRI_raw.Num_Encodings}. "
            "Raw2Array currently only supports single-encoding data."
        )
        
    # The structure holds lists of data if multiple encodings/frames were loaded.
    # Since we checked for Num_Encodings == 1, we can safely take the first element (index 0).
    kdata = MRI_raw.kdata[0]
    coords = MRI_raw.coords[0]
    dcf = MRI_raw.dcf[0]
    time = MRI_raw.time[0]
    return kdata, coords, dcf, time


def ReadPcviprTxt(filename: str) -> Dict[str, int]:
    """
    Read key parameters from the data_header.txt file.
    Improved parsing robustness against varied whitespace.
    """
    vals = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('pfile'):
                continue
            
            parts = line.split()
            
            if len(parts) >= 2:
                key = parts[0]
                value_str = parts[1]
                try:
                    # Parse value as float first, then convert to integer
                    # This handles values like "128.000"
                    value = int(float(value_str))
                    vals[key] = value
                except ValueError:
                    logger.warning(f"Cannot parse value '{value_str}' as number for key '{key}'. Skipping line: {line}")
                except Exception as e:
                    logger.warning(f"Unexpected error processing line: {line}. Error: {e}")
            else:
                logger.warning(f"Line does not contain a key-value pair: {line}")
                
    return vals