import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

class BPTVisualizer:
    def __init__(self, 
                 inp_dir: str | None = None,
                 bpts: np.ndarray | None = None,
                 bpts_type: str | None = None,
                 tr: float = 1.0, 
                 rel_shift: float = 0.5,
                 time_range: list | tuple | None = None,
                 figsize: tuple = (10, 10),
                 plot_bpt: bool = True,
                 plot_resp: bool = False,
                 plot_cardio: bool = False,
                 display_fig: bool = True,
                 verbose: bool = True):
        
        # Core settings
        self.inp_dir: str | None = inp_dir # Directory containing BPT arrays and physiological data files.
        self.tr: float = tr # Repetition time or sampling interval for the x-axis.
        self.rel_shift: float = rel_shift # Vertical shift multiplier to separate BPT signals.
        self.time_range: list | tuple | None = time_range # Optional [min, max] time window in seconds to display.
        self.figsize: tuple = figsize # Dimensions of the generated figure.
        
        # Plotting feature flags
        self.plot_bpt: bool = plot_bpt
        self.plot_resp: bool = plot_resp
        self.plot_cardio: bool = plot_cardio
        self.display_fig: bool = display_fig
        self.verbose: bool = verbose
        
        # Filenames and Types
        self.bpts_type: str | None = bpts_type if bpts_type else 'bpts_proc' # 'bpts_proc' for PC labels, anything else for Coil labels.
        self.bpts_fname: str | None = os.path.join(inp_dir, f"{self.bpts_type}.npy") if self.inp_dir and self.bpts_type else None
        print(self.bpts_fname)
        
        # Data Tracking Attributes
        self.bpts: np.ndarray | None = bpts # The raw signals array. (Shape: (Time, Channels) or (Sets, Time, Channels))
        self.resp_data: tuple | None = None # Tuple of (time_axis, normalized_signal) for respiration.
        self.cardio_data: tuple | None = None # Tuple of (time_axis, normalized_signal) for PPG/cardio.
        
        # Internal Processed State
        self.bpts_3d: np.ndarray | None = None # Enforced 3D framing of BPTs. (Shape: (Sets, Time, Channels))
        self.max_time_sec: float = 0.0 # Total temporal duration of the acquisition.
        
        # Output States
        self.fig: plt.Figure | None = None
        self.axes: np.ndarray | None = None

    def plot_bpts(self):
        """
        Main entry point to plot B+PT signals, automatically integrating physiological overlays.

        Stores:
        fig (plt.Figure): The generated matplotlib figure.
        axes (np.ndarray): The array of subplot axes.
        """

        if self.fig is not None:
            plt.close(self.fig)

        self._load_data()
        n_sets = self.bpts_3d.shape[0]

        # Calculate grid layout
        ncols = 2 if n_sets > 1 else 1
        nrows = int(np.ceil(n_sets / ncols))
        
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=self.figsize, squeeze=False)
        self.axes = self.axes.flatten()

        for i in range(n_sets):
            ax = self.axes[i]
            
            current_offset = 0.5
            all_ticks = []
            all_labels = []

            # 1. Plot Cardio
            if self.plot_cardio and self.cardio_data is not None and self.cardio_data[1] is not None:
                ticks, labels = self._add_physio_signals(ax, self.cardio_data, current_offset, color='darkviolet', label="PPG (Cardio)")
                all_ticks.extend(ticks)
                all_labels.extend(labels)
                current_offset += 1.5

            # 2. Plot Respiration
            if self.plot_resp and self.resp_data is not None and self.resp_data[1] is not None:
                ticks, labels = self._add_physio_signals(ax, self.resp_data, current_offset, color='crimson', label="Resp Bellows")
                all_ticks.extend(ticks)
                all_labels.extend(labels)
                current_offset += 1.5

            # 3. Plot BPTs
            if self.plot_bpt:
                label_prefix = "PC" if self.bpts_type == 'bpts_proc' else "Coil"
                title = f"B+PT {i+1} {label_prefix}s, Cutoff=5Hz" if n_sets > 1 else f"B+PT 1 {label_prefix}s, Cutoff=5Hz"
                ax.set_title(title)
                
                ticks, labels, current_offset = self._plot_stacked_signals(
                    ax=ax,
                    bpt_idx=i,
                    bottom_clearance=current_offset
                )
                all_ticks.extend(ticks)
                all_labels.extend(labels)

            # 4. Format the Axis Viewport
            ax.set_yticks(all_ticks)
            ax.set_yticklabels(all_labels)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("")
            
            if self.time_range is not None:
                ax.set_xlim(self.time_range[0], self.time_range[1])
            else:
                ax.set_xlim(0, self.max_time_sec)
                
            lower_limit = 0.5 - (1.0 / self.rel_shift) - 0.5 if self.rel_shift != 0 else 0.0
            ax.set_ylim(lower_limit, current_offset + 0.5) # Added +0.5 to give the top a little breathing room too
            ax.grid(True, alpha=0.15)

        # Hide unused subplots
        for j in range(n_sets, len(self.axes)):
            self.axes[j].axis('off')

        plt.tight_layout()

        if self.display_fig: # plot in notebooks if display_fig is True; otherwise, the user can call plt.show() or use self.fig externally
            plt.show()

    def _load_data(self):
        """
        Load BPT arrays from disk if missing, calculate the global timeline, and load physio traces.
        """
        # 1. Ensure BPTs are loaded
        if self.bpts is None:
            if self.bpts_fname and os.path.exists(self.bpts_fname):
                if self.verbose:
                    logger.info(f"Loading BPT signals from {self.bpts_fname}")
                self.bpts = np.load(self.bpts_fname)
            else:
                raise ValueError("BPT signals must be provided directly or loaded via a valid bpts_file.")

        # Force 3D framing for consistent iteration: (Sets, Time, Channels)
        self.bpts_3d = self.bpts if self.bpts.ndim == 3 else self.bpts[np.newaxis, ...]
        n_time = self.bpts_3d.shape[1]
        self.max_time_sec = (n_time - 1) * self.tr

        # 2. Load Physiological Traces
        if self.plot_resp and self.resp_data is None and self.inp_dir:
            self.resp_data = self._load_physio_trace("RESPData_uwute*", 0.04)
            
        if self.plot_cardio and self.cardio_data is None and self.inp_dir:
            self.cardio_data = self._load_physio_trace("PPGData_uwute*", 0.01)

    def _load_physio_trace(self, file_pattern: str, dt: float) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """
        Load, crop, and normalize a physiological data trace from disk, handling pre-scan delays.
        
        Args:
            file_pattern (str): Glob pattern to match the file (e.g., 'RESPData*').
            dt (float): Sampling interval of the physiological equipment in seconds.
        Returns:
            t_axis (np.ndarray | None): Cropped time axis array.
            sig_norm (np.ndarray | None): Normalized physiological signal array.
        """
        matches = glob.glob(os.path.join(self.inp_dir, file_pattern))
        if not matches:
            if self.verbose:
                logger.warning(f"No physiological file matching '{file_pattern}' found in {self.inp_dir}.")
            return None, None
            
        raw_data = np.loadtxt(matches[0])
        fs = 1.0 / dt
        
        total_physio_duration = len(raw_data) * dt
        
        # Handle scanner pre-scan delay (31 seconds) if physio started early
        discard_sec = 31.0 if total_physio_duration > self.max_time_sec else 0.0
            
        start_idx = int(discard_sec * fs)
        if start_idx >= len(raw_data):
            return None, None
            
        cropped = raw_data[start_idx:]
        
        t_axis = np.arange(len(cropped)) * dt
        time_mask = t_axis <= self.max_time_sec
        t_axis = t_axis[time_mask]
        final_signal = cropped[:len(t_axis)]
        
        # De-mean and normalize
        sig_dm = final_signal - np.mean(final_signal)
        max_val = np.max(np.abs(sig_dm))
        sig_norm = sig_dm / max_val if max_val > 0 else sig_dm
        
        return t_axis, sig_norm

    def _plot_stacked_signals(self, 
                              ax: plt.Axes, 
                              bpt_idx: int,
                              bottom_clearance: float) -> Tuple[List[float], List[str], float]:
        """
        Draw a 2D array of time-series signals with vertical shifts onto a matplotlib axis.

        Args:
            ax (plt.Axes): Matplotlib axes to plot on.
            bpt_idx (int): The index of the BPT set to plot from self.bpts_3d.
            bottom_clearance (float): Base vertical offset to start plotting from.
        Returns:
            y_ticks (list): Calculated y-axis tick positions.
            y_tick_labels (list): Generated string labels for each channel.
            highest_offset (float): The highest vertical limit used by the stack.
        """
        signals = self.bpts_3d[bpt_idx]
        n_time, n_channels = signals.shape
        t = np.arange(n_time) * self.tr

        sig_dm = signals - np.mean(signals, axis=0, keepdims=True)

        # Calculate global shift based on the first 10% of data (ignoring early transients)
        start_idx = int(0.1 * n_time)
        if start_idx >= n_time:
            start_idx = 0
        max_amp = np.max(np.abs(sig_dm[start_idx:, :])) if sig_dm.size > 0 else 1.0
        shift = self.rel_shift * (max_amp + 1e-9)

        y_ticks = []
        y_tick_labels = []
        
        label_prefix = "PC" if self.bpts_type == 'bpts_proc' else "Coil"

        for k in range(n_channels):
            y_offset = k + bottom_clearance
            line_data = (sig_dm[:, k] / shift) + y_offset
            
            # Removed color='k' so Matplotlib automatically cycles colors
            ax.plot(t, line_data, linewidth=1.0)
            
            y_ticks.append(y_offset)
            y_tick_labels.append(f"{label_prefix} {k + 1}")

        highest_offset = n_channels + bottom_clearance
        return y_ticks, y_tick_labels, highest_offset

    def _add_physio_signals(self, 
                            ax: plt.Axes, 
                            physio_data: tuple, 
                            start_offset: float, 
                            color: str, 
                            label: str) -> Tuple[List[float], List[str]]:
        """
        Draw a single physiological trace onto a matplotlib axis.

        Args:
            ax (plt.Axes): The matplotlib axes to plot on.
            physio_data (tuple): Tuple of (time_array, normalized_signal_array).
            start_offset (float): The vertical offset to plot the signal at.
            color (str): Matplotlib color string.
            label (str): The y-axis label for this trace.
        Returns:
            y_ticks (list): Calculated y-axis tick position.
            y_tick_labels (list): Label text.
        """
        t_axis, sig_norm = physio_data
        ax.plot(t_axis, sig_norm + start_offset, color=color, linewidth=1.2 if "Cardio" in label else 1.8)
        
        return [start_offset], [label]