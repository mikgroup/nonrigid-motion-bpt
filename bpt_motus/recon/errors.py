import numpy as np
import logging
import os
import json

logger = logging.getLogger(__name__)

class ErrorEvaluator:
    def __init__(self, 
                 pics_dir: str | None = None, 
                 mr_motus_dir: str | None = None, 
                 bpt_motus_dir: str | None = None, 
                 out_dir: str | None = None, 
                 start_idx: int = 0,
                 pics_filename: str = "pics_frames.npy",
                 warped_filename: str = "warped_pics_frames.npy",
                 params_filename: str = "transform_parameters.npy",
                 save_filename: str = "error_report.json",
                 verbose: bool = True,
                 force_reload: bool = False):
        
        # Core settings
        self.pics_dir: str | None = pics_dir # Directory containing ground truth PICS baseline.
        self.mr_motus_dir: str | None = mr_motus_dir # Directory containing MR-MOTUS results.
        self.bpt_motus_dir: str | None = bpt_motus_dir # Directory containing BPT-MOTUS results.
        self.out_dir: str | None = out_dir if out_dir is not None else pics_dir # Directory to save error report.
        self.start_idx: int = start_idx # Frame index to start evaluation from (skipping early transients).
        self.verbose: bool = verbose # Whether to print logs.
        self.force_reload: bool = force_reload # Whether to force recalculation if output exists.

        # Filenames - Dynamic path resolution
        self.pics_img_fname: str | None = os.path.join(pics_dir, pics_filename) if pics_dir else None
        self.pics_params_fname: str | None = os.path.join(pics_dir, params_filename) if pics_dir else None

        self.mr_img_fname: str | None = os.path.join(mr_motus_dir, warped_filename) if mr_motus_dir else None
        self.mr_params_fname: str | None = os.path.join(mr_motus_dir, params_filename) if mr_motus_dir else None

        self.bpt_img_fname: str | None = os.path.join(bpt_motus_dir, warped_filename) if bpt_motus_dir else None
        self.bpt_params_fname: str | None = os.path.join(bpt_motus_dir, params_filename) if bpt_motus_dir else None

        self.save_fname: str | None = os.path.join(self.out_dir, save_filename) if self.out_dir else None

        # Tracking Attributes
        self.pics_frames: np.ndarray | None = None
        self.pics_params: np.ndarray | None = None

        self.mr_frames: np.ndarray | None = None
        self.mr_params: np.ndarray | None = None

        self.bpt_frames: np.ndarray | None = None
        self.bpt_params: np.ndarray | None = None

        self.error_report: dict = {}

    def run(self):
        """
        Orchestrate the loading of data and the evaluation of all available pipelines against the PICS baseline.

        Stores:
        error_report (dict): Nested dictionary containing all calculated RMSE metrics.

        Saves:
        error_report serialized to `self.save_fname` as a JSON file.
        """
        if not self.force_reload and self.save_fname and os.path.exists(self.save_fname):
            if self.verbose:
                logger.info(f"Loading existing error report from {self.save_fname}")
            with open(self.save_fname, 'r') as f:
                self.error_report = json.load(f)
            return

        if self.verbose:
            logger.info("Initializing Error Evaluator pipeline...")
            
        self._load_data()
        self.error_report = {}

        # Evaluate MR-MOTUS
        if self.pics_frames is not None and self.mr_frames is not None:
            self.compute_custom_voxel_errors(self.pics_frames, self.mr_frames, "MR-MOTUS_vs_PICS_Images")
        if self.pics_params is not None and self.mr_params is not None:
            self.compute_custom_param_errors(self.pics_params, self.mr_params, "MR-MOTUS_vs_PICS_Params")

        # Evaluate BPT-MOTUS
        if self.pics_frames is not None and self.bpt_frames is not None:
            self.compute_custom_voxel_errors(self.pics_frames, self.bpt_frames, "BPT-MOTUS_vs_PICS_Images")
        if self.pics_params is not None and self.bpt_params is not None:
            self.compute_custom_param_errors(self.pics_params, self.bpt_params, "BPT-MOTUS_vs_PICS_Params")

        self._save_results()

    def compute_custom_voxel_errors(self, target_frames: np.ndarray, pred_frames: np.ndarray, report_key: str):
        """
        Compute voxel-wise RMSE between two arbitrary image arrays and store the results in the report.

        Args:
        target_frames (np.ndarray): The ground truth image array. (Shape: (Nframes, Nx, Ny) or (Nframes, Nx, Ny, Nz))
        pred_frames (np.ndarray): The predicted/warped image array. (Shape: (Nframes, Nx, Ny) or (Nframes, Nx, Ny, Nz))
        report_key (str): The dictionary key under which the results will be saved.

        Stores:
        error_report (dict): Updates the dictionary with global and frame-wise RMSE lists.
        """
        if self.verbose:
            logger.info(f"Computing voxel-wise RMSE for: {report_key}")
            
        tar = np.abs(target_frames[self.start_idx:])
        prd = np.abs(pred_frames[self.start_idx:])
        
        diff = tar - prd
        
        f_rmses = np.linalg.norm(diff.reshape(len(tar), -1), axis=1) / \
                  (np.linalg.norm(tar.reshape(len(tar), -1), axis=1) + 1e-12)
                  
        global_rmse = np.linalg.norm(diff.ravel()) / (np.linalg.norm(tar.ravel()) + 1e-12)
        
        self.error_report[report_key] = {
            "global_rmse": float(global_rmse),
            "frame_rmses": f_rmses.tolist()
        }

    def compute_custom_param_errors(self, target_params: np.ndarray, pred_params: np.ndarray, report_key: str):
        """
        Compute parameter-wise RMSE between two registration coordinate arrays, handling both 2D and 3D formats.

        Args:
        target_params (np.ndarray): The ground truth parameters. (Shape: (Nframes, 3) or (Nframes, 6))
        pred_params (np.ndarray): The predicted parameters. (Shape: (Nframes, 3) or (Nframes, 6))
        report_key (str): The dictionary key under which the results will be saved.

        Stores:
        error_report (dict): Updates the dictionary with total, rotational, and translational RMSE values.
        """
        if self.verbose:
            logger.info(f"Computing registration parameter RMSE for: {report_key}")

        tar = target_params[self.start_idx:]
        prd = pred_params[self.start_idx:]

        # Safety check: Match lengths if pipelines had different temporal resolutions
        min_len = min(len(tar), len(prd))
        tar = tar[:min_len]
        prd = prd[:min_len]

        total_param_rmse = np.linalg.norm(tar - prd) / (np.linalg.norm(tar) + 1e-12)

        res = {
            "total_param_rmse": float(total_param_rmse)
        }

        # Check if 3D (6 parameters: 3 rotation, 3 translation)
        if tar.shape[1] >= 6:
            rot_rmse = np.linalg.norm(tar[:, :3] - prd[:, :3]) / (np.linalg.norm(tar[:, :3]) + 1e-12)
            trans_rmse = np.linalg.norm(tar[:, 3:] - prd[:, 3:]) / (np.linalg.norm(tar[:, 3:]) + 1e-12)
            res["rot_rmse"] = float(rot_rmse)
            res["trans_rmse"] = float(trans_rmse)

        self.error_report[report_key] = res

    def _load_data(self):
        """
        Gracefully load all available target and prediction data from disk without crashing if partial data is missing.

        Stores:
        pics_frames, pics_params, mr_frames, mr_params, bpt_frames, bpt_params (np.ndarray | None): 
        Loaded arrays for evaluation.
        """
        # Load PICS Ground Truth
        if self.pics_img_fname and os.path.exists(self.pics_img_fname):
            self.pics_frames = np.load(self.pics_img_fname)
        if self.pics_params_fname and os.path.exists(self.pics_params_fname):
            self.pics_params = np.load(self.pics_params_fname)

        # Load MR-MOTUS Data
        if self.mr_img_fname and os.path.exists(self.mr_img_fname):
            self.mr_frames = np.load(self.mr_img_fname)
        if self.mr_params_fname and os.path.exists(self.mr_params_fname):
            self.mr_params = np.load(self.mr_params_fname)

        # Load BPT-MOTUS Data
        if self.bpt_img_fname and os.path.exists(self.bpt_img_fname):
            self.bpt_frames = np.load(self.bpt_img_fname)
        if self.bpt_params_fname and os.path.exists(self.bpt_params_fname):
            self.bpt_params = np.load(self.bpt_params_fname)

    def _save_results(self):
        """
        Write the consolidated error_report dictionary to disk as a JSON file.

        Saves:
        error_report to `self.save_fname`
        """
        if self.save_fname and self.error_report:
            os.makedirs(os.path.dirname(self.save_fname), exist_ok=True)
            with open(self.save_fname, 'w') as f:
                json.dump(self.error_report, f, indent=4)
            if self.verbose:
                logger.info(f"Saved complete metric report to {self.save_fname}")
        elif self.verbose and not self.error_report:
            logger.warning("No metrics were computed. Error report is empty.")