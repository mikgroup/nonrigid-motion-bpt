import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ErrorVisualizer:
    def __init__(self, 
                 report_dir: str,
                 pics_dir: str | None = None,
                 mr_motus_dir: str | None = None,
                 bpt_motus_dir: str | None = None,
                 report_filename: str = "error_report.json",
                 params_filename: str = "rigid_parameters.npy",
                 figsize_rmse: tuple = (10, 6),
                 figsize_params: tuple = (12, 10),
                 verbose: bool = True):
        
        # Core settings
        self.report_dir: str = report_dir # Directory containing error_report.json
        self.pics_dir: str | None = pics_dir # Directory containing GT PICS parameters
        self.mr_motus_dir: str | None = mr_motus_dir # Directory containing MR-MOTUS parameters
        self.bpt_motus_dir: str | None = bpt_motus_dir # Directory containing BPT-MOTUS parameters
        self.figsize_rmse: tuple = figsize_rmse
        self.figsize_params: tuple = figsize_params
        self.verbose: bool = verbose
        
        # Filenames
        self.report_fname: str = os.path.join(self.report_dir, report_filename)
        self.pics_params_fname: str | None = os.path.join(pics_dir, params_filename) if pics_dir else None
        self.mr_params_fname: str | None = os.path.join(mr_motus_dir, params_filename) if mr_motus_dir else None
        self.bpt_params_fname: str | None = os.path.join(bpt_motus_dir, params_filename) if bpt_motus_dir else None
        
        # Data Tracking Attributes
        self.error_report: dict = {}
        self.pics_params: np.ndarray | None = None
        self.mr_params: np.ndarray | None = None
        self.bpt_params: np.ndarray | None = None
        
        # Output Figure States
        self.fig_rmse: plt.Figure | None = None
        self.ax_rmse: plt.Axes | None = None
        self.fig_params: plt.Figure | None = None
        self.axes_params: np.ndarray | None = None

    def plot_voxel_rmses(self, title: str = "Voxel-wise Tracking RMSE"):
        """
        Plot voxel-wise frame-by-frame RMSE over time across all methods found in the error report.

        Stores:
        fig_rmse (plt.Figure): The generated RMSE matplotlib figure.
        ax_rmse (plt.Axes): The generated RMSE matplotlib axis.
        """
        self._load_data()
        
        if not self.error_report:
            logger.warning("Error report is empty. Cannot plot voxel RMSEs.")
            return

        self.fig_rmse, self.ax_rmse = plt.subplots(figsize=self.figsize_rmse)
        
        # Iterate through the JSON and plot any entry containing 'frame_rmses'
        for label, data in self.error_report.items():
            if 'frame_rmses' in data:
                f_rmses = np.array(data['frame_rmses'])
                t_axis = np.arange(len(f_rmses))
                global_val = data.get('global_rmse', np.mean(f_rmses))
                
                # Clean up label for legend (e.g., 'MR-MOTUS_vs_PICS_Images' -> 'MR-MOTUS')
                clean_label = label.split("_vs_")[0] if "_vs_" in label else label
                
                self.ax_rmse.plot(t_axis, f_rmses, label=f"{clean_label} (Global: {global_val:.4f})", alpha=0.9, linewidth=1.5)

        self.ax_rmse.set_title(title, fontsize=15)
        self.ax_rmse.set_xlabel("Frame Index")
        self.ax_rmse.set_ylabel("Relative RMSE")
        self.ax_rmse.grid(True, alpha=0.3)
        self.ax_rmse.legend(loc='upper right')
        
        self.fig_rmse.tight_layout()

    def plot_registration_parameters(self, title: str = "Registration Parameters (GT vs Modeled)"):
        """
        Plots 6-DoF registration parameters (translations/rotations) derived from Elastix, 
        comparing ground truth against all loaded model outputs.

        Stores:
        fig_params (plt.Figure): The generated 6-panel matplotlib figure.
        axes_params (np.ndarray): The 3x2 array of matplotlib axes.
        """
        self._load_data()
        
        if self.pics_params is None:
            logger.warning("PICS ground truth parameters missing. Cannot plot registration tracking.")
            return

        num_frames = self.pics_params.shape[0]
        t = np.arange(num_frames)

        self.fig_params, self.axes_params = plt.subplots(3, 2, figsize=self.figsize_params, sharex=True)
        param_names = ["Rot X", "Rot Y", "Rot Z", "Trans X", "Trans Y", "Trans Z"]
        
        # Create a dictionary of available predictions to loop through
        predictions = {}
        if self.mr_params is not None: predictions['MR-MOTUS'] = self.mr_params
        if self.bpt_params is not None: predictions['BPT-MOTUS'] = self.bpt_params

        for dim in range(6):
            ax = self.axes_params[dim // 2, dim % 2]
            
            # Plot GT reference once
            ax.plot(t, self.pics_params[:, dim], 'k--', linewidth=2.5, alpha=0.7, label='PICS (GT)')
            
            # Plot each model tracking line
            for label, mod_params in predictions.items():
                # Safety crop if sequences have different temporal lengths
                plot_len = min(num_frames, mod_params.shape[0])
                ax.plot(t[:plot_len], mod_params[:plot_len, dim], label=label, alpha=0.85)

            ax.set_title(param_names[dim])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        for ax in self.axes_params[-1, :]:
            ax.set_xlabel("Frame Index")
            
        self.fig_params.suptitle(title, fontsize=16)
        self.fig_params.tight_layout()

    def _load_data(self):
        """
        Load the JSON error report and any available transform parameter .npy arrays.

        Stores:
        error_report (dict): Nested dictionary of evaluation metrics.
        pics_params, mr_params, bpt_params (np.ndarray | None): 6-DoF tracking arrays.
        """
        # Load Report
        if not self.error_report and os.path.exists(self.report_fname):
            if self.verbose: logger.info(f"Loading error report from {self.report_fname}")
            with open(self.report_fname, 'r') as f:
                self.error_report = json.load(f)

        # Load parameter arrays
        if self.pics_params is None and self.pics_params_fname and os.path.exists(self.pics_params_fname):
            self.pics_params = np.load(self.pics_params_fname)
            
        if self.mr_params is None and self.mr_params_fname and os.path.exists(self.mr_params_fname):
            self.mr_params = np.load(self.mr_params_fname)
            
        if self.bpt_params is None and self.bpt_params_fname and os.path.exists(self.bpt_params_fname):
            self.bpt_params = np.load(self.bpt_params_fname)