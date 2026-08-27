import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import logging
import warnings
from typing import Tuple, List, Optional, Literal

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
                 save_fig: bool = False,
                 save_fname: str | None = None,
                 dpi: int = 150,
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
        self.save_fig: bool = save_fig # If True, save the figure into inp_dir once plotted.
        self.save_fname: str | None = save_fname # Filename to save under; defaults to "{bpts_type}_plot.png".
        self.dpi: int = dpi
        self.verbose: bool = verbose
        
        # Filenames and Types
        self.bpts_type: str | None = bpts_type if bpts_type else 'bpts_proc' # 'bpts_proc' for PC labels, anything else for Coil labels.
        self.bpts_fname: str | None = os.path.join(inp_dir, f"{self.bpts_type}.npy") if self.inp_dir and self.bpts_type else None
        
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
                title = f"B+PT {i+1} {label_prefix}s" if n_sets > 1 else f"B+PT 1 {label_prefix}s"
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
                
            # Removed the manual y-limits here. 
            # ax.margins handles the padding perfectly, letting Matplotlib fit 
            # exactly to the top and bottom of your signal swings automatically.
            ax.margins(y=0.02) 
            ax.grid(True, alpha=0.15)

        # Hide unused subplots
        for j in range(n_sets, len(self.axes)):
            self.axes[j].axis('off')

        plt.tight_layout()

        if self.save_fig and self.inp_dir:
            fname = self.save_fname or f"{self.bpts_type}_plot.png"
            save_path = os.path.join(self.inp_dir, fname)
            self.fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            if self.verbose:
                logger.info(f"Saved BPT plot to {save_path}")
        elif self.save_fig and self.verbose:
            logger.warning("save_fig=True but no inp_dir set; figure was not saved.")

        if self.display_fig:
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
            self.resp_data = self._load_physio_trace("RESPData*", 0.04)
            
        if self.plot_cardio and self.cardio_data is None and self.inp_dir:
            self.cardio_data = self._load_physio_trace("PPGData*", 0.01)

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

class BPTSweepVisualizer:
    """
    Analyzes a frequency-sweep BPT acquisition, where the transmit tone holds at each frequency
    for `step_time` seconds, from `f_start` to `f_stop` in `f_step` increments (`n_steps` is
    derived from those three, not an input). Detects step boundaries, maps samples to frequency,
    and plots signal / percent-modulation vs frequency instead of time.

    Step detection: channels are combined into one jump-strength signal, thresholded into
    candidate jump clusters, matched to expected step numbers, then fit with a robust linear
    regression (sample_idx vs step number) to get start_idx and samples-per-step directly.
    """

    def __init__(self,
                 visualizer: BPTVisualizer,
                 bpt_idx: int = 0,
                 step_time: float = 2.0,
                 f_start: float = 1.2,
                 f_stop: float = 3.5,
                 f_step: float = 0.02,
                 modulation_method: Literal["peak_to_peak", "std"] = "peak_to_peak",
                 modulation_center_frac: float = 1.0,
                 step_time_tol: float = 0.4,
                 start_time: float | None = None,
                 start_time_tol: float = 1.0,
                 jump_merge_gap: float = 0.5,
                 max_start_time: float | None = None,
                 flatten_steps: bool = False,
                 verbose: bool = True):

        # Core settings
        self.visualizer: BPTVisualizer = visualizer # Source of BPT data, inp_dir, tr, and save/display behavior.
        self.bpt_idx: int = bpt_idx # Which BPT set (of visualizer.bpts_3d) to analyze.

        # Sweep parameters
        self.step_time: float = step_time # Nominal duration of each frequency step, in seconds (initial guess; refined by fit).
        self.f_start: float = f_start # Frequency of the first step.
        self.f_stop: float = f_stop # Frequency of the last step.
        self.f_step: float = f_step # Frequency increment per step.
        self.n_steps: int = int(round((f_stop - f_start) / f_step)) + 1 # Number of frequency steps, derived from the range above.
        self.modulation_method: str = modulation_method # "peak_to_peak" or "std" swing relative to the step's mean.
        self.modulation_center_frac: float = modulation_center_frac # Fraction of each step's window (centered) used for modulation stats.
        self.step_time_tol: float = step_time_tol # Fractional-of-step tolerance for matching/rejecting a detected jump.
        self.start_time: float | None = start_time # Expected time (s) of the first step transition; seeds the fit.
        self.start_time_tol: float = start_time_tol # Max deviation (s) from start_time to accept a detected jump as the seed.
        self.jump_merge_gap: float = jump_merge_gap # Max gap (s) between threshold-crossings to merge into one jump cluster.
        self.max_start_time: float | None = max_start_time # Hard upper bound (s) on the resolved start time.
        self.flatten_steps: bool = flatten_steps # If True, demean each step by its own per-channel median instead of the recording-wide mean.
        self.verbose: bool = verbose

        # Analysis outputs
        self.signal: np.ndarray | None = None # BPT set under analysis. (Shape: (Nsp, Nc))
        self.jump_signal: np.ndarray | None = None # Combined multi-channel jump-strength signal used for detection. (Shape: (Nsp-1,))
        self.detected_jumps: list[tuple[int, float]] | None = None # All candidate (sample_idx, strength) clusters found in jump_signal.
        self.matched_jumps: list[tuple[int, int]] | None = None # (step_k, sample_idx) pairs actually used in the linear fit.
        self.start_idx: int | None = None # Fitted sample index of the first step transition (maps to freq f_start).
        self.step_len_samples: float | None = None # Fitted samples-per-step (replaces step_time/tr for indexing).
        self.freq_axis: np.ndarray | None = None # Frequency assigned to each sample. (Shape: (Nsp,))
        self.step_bounds: np.ndarray | None = None # Sample indices bounding each step. (Shape: (n_steps+1,))
        self.modulation_stats: list[dict] | None = None # One dict per step: freq, per-channel and summary percent modulation.

        # Output figure states
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None
        self.ax_heat: plt.Axes | None = None # Heatmap axis, set by plot_combined().

    def analyze(self) -> list[dict]:
        """
        Detect sweep step boundaries, map samples to frequency, and compute percent-modulation
        statistics per step and channel.

        Stores:
        freq_axis (np.ndarray): Frequency assigned to each sample.
        modulation_stats (list[dict]): Per-step percent modulation, per channel and summary.
        """
        self.visualizer._load_data()
        self.signal = self.visualizer.bpts_3d[self.bpt_idx]  # (Nsp, Nc)
        tr = self.visualizer.tr
        n_sp, n_ch = self.signal.shape

        # Combined jump-strength signal: sum of |diff| across channels, each scaled by its own
        # typical step size, so a real (multi-channel) transition dominates over channel noise.
        diffs = np.diff(self.signal, axis=0)  # (Nsp-1, Nc)
        ch_scale = np.median(np.abs(diffs), axis=0)
        ch_scale = np.where(ch_scale == 0, 1.0, ch_scale)
        self.jump_signal = np.sum(np.abs(diffs) / ch_scale, axis=1)  # (Nsp-1,)

        clusters = self._detect_jump_clusters(self.jump_signal, tr)
        if not clusters:
            raise ValueError("No sweep steps detected; check step_time and tr, or pass a start_time hint.")
        self.detected_jumps = clusters

        nominal_step_len = self.step_time / tr  # initial guess only; refined by the fit below

        # Seed start_idx from a start_time hint. step0->step1 (one step_time after start_time) is
        # usually the first detectable jump, since there's nothing before step0 to jump from.
        if self.start_time is not None:
            expected_jump_time = self.start_time + self.step_time
            lo_idx = max(0, int((expected_jump_time - self.start_time_tol) / tr))
            hi_idx = int((expected_jump_time + self.start_time_tol) / tr)
            in_window = [c for c in clusters if lo_idx <= c[0] <= hi_idx]
            if in_window:
                first_observed_idx = max(in_window, key=lambda c: c[1])[0]
                seed_start_idx = first_observed_idx - nominal_step_len
            else:
                logger.warning(f"No detected jump cluster within {self.start_time_tol}s of "
                                f"start_time + step_time = {expected_jump_time:.3f}s; seeding "
                                "from start_time exactly (the fit below may still recover it).")
                seed_start_idx = max(0, int(round(self.start_time / tr)))
        else:
            seed_start_idx = clusters[0][0] - nominal_step_len

        # Match each detected cluster to its nearest expected step slot k under the seed anchor.
        matches = []  # (k, sample_idx)
        for idx, _strength in clusters:
            k_est = (idx - seed_start_idx) / nominal_step_len
            k = round(k_est)
            if 0 <= k <= self.n_steps and abs(k_est - k) <= self.step_time_tol:
                matches.append((k, idx))

        if len(matches) >= 2:
            ks = np.array([m[0] for m in matches], dtype=float)
            idxs = np.array([m[1] for m in matches], dtype=float)

            # Linear fit of sample_idx vs step number k, with one outlier-rejection pass.
            step_len_samples, start_idx_f = np.polyfit(ks, idxs, 1)
            resid = idxs - (start_idx_f + step_len_samples * ks)
            mad = np.median(np.abs(resid - np.median(resid)))
            thresh = max(6 * mad, 0.1 * nominal_step_len)
            inlier = np.abs(resid) <= thresh
            if 2 <= inlier.sum() < len(ks):
                step_len_samples, start_idx_f = np.polyfit(ks[inlier], idxs[inlier], 1)
                ks, idxs = ks[inlier], idxs[inlier]

            self.matched_jumps = list(zip(ks.astype(int).tolist(), idxs.astype(int).tolist()))
            self.start_idx = int(round(start_idx_f))
            step_len_samples = float(step_len_samples)
        else:
            logger.warning("Fewer than 2 matched step transitions; falling back to the seeded "
                            "start_idx and nominal step_time without drift correction.")
            self.matched_jumps = matches
            self.start_idx = seed_start_idx
            step_len_samples = nominal_step_len

        start_time = self.start_idx * tr
        step_time = step_len_samples * tr  # for logging only

        if self.max_start_time is not None and start_time > self.max_start_time:
            raise ValueError(f"Resolved start_time={start_time:.3f}s exceeds max_start_time={self.max_start_time}s.")

        sample_idx = np.arange(n_sp)
        self.freq_axis = self.f_start + (sample_idx - self.start_idx) / step_len_samples * self.f_step

        edges = self.start_idx + np.arange(self.n_steps + 1) * step_len_samples
        self.step_bounds = np.clip(np.round(edges).astype(int), 0, n_sp)
        self.step_len_samples = step_len_samples

        if self.verbose:
            logger.info(f"start_idx: {self.start_idx} (start_time: {start_time:.3f}s), "
                        f"step_len_samples: {step_len_samples:.3f} (step_time: {step_time:.4f}s), "
                        f"matched {len(self.matched_jumps)}/{len(clusters)} detected jumps")

        self.modulation_stats = self._compute_modulation_stats()
        return self.modulation_stats

    def _detect_jump_clusters(self, jump_signal: np.ndarray, tr: float) -> list[tuple[int, float]]:
        """
        Detect step transitions: flag samples where the combined jump-strength signal exceeds a
        noise-derived threshold, group flagged samples within jump_merge_gap of each other into
        one cluster each, and collapse each to its midpoint sample and peak strength.

        Returns:
        clusters (list[tuple[int, float]]): (candidate_start_idx, strength), sorted by time.
        """
        # Noise baseline must precede the first real transition (start_time + step_time), or a
        # nearby jump would contaminate the threshold estimate.
        baseline_len = max(1, int(4 / tr))
        if self.start_time is not None:
            expected_jump_time = self.start_time + self.step_time
            baseline_len = min(baseline_len, max(1, int((expected_jump_time - self.start_time_tol) / tr)))
        baseline = jump_signal[:baseline_len]
        threshold = np.mean(baseline) + 5 * np.std(baseline)
        raw = np.where(jump_signal > threshold)[0]
        if len(raw) == 0:
            return []

        merge_gap = max(1, int(self.jump_merge_gap / tr))
        breaks = np.where(np.diff(raw) > merge_gap)[0]
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [len(raw) - 1]))

        clusters = []
        for cs, ce in zip(starts, ends):
            cluster = raw[cs:ce + 1]
            midpoint = int(round((cluster[0] + cluster[-1]) / 2))
            candidate_start = midpoint + 1  # first sample of the new level
            clusters.append((candidate_start, float(np.max(jump_signal[cluster]))))

        clusters.sort(key=lambda c: c[0])
        if self.max_start_time is not None:
            max_idx = int(self.max_start_time / tr)
            clusters = [c for c in clusters if c[0] <= max_idx]
        return clusters

    def _compute_modulation_stats(self) -> list[dict]:
        """
        Compute, per step and channel, percent modulation = 100 * (AC swing) / (DC level) over
        the center modulation_center_frac fraction of the step's window.

        Returns:
        rows (list[dict]): step, freq_ghz, coil_{i}_pct_mod per channel, max_pct_mod, best_coil.
        """
        rows = []
        for step in range(self.n_steps):
            lo, hi = self.step_bounds[step], self.step_bounds[step + 1]
            if hi - lo < 2:
                continue
            trim = int(round((hi - lo) * (1 - self.modulation_center_frac) / 2))
            clo, chi = lo + trim, hi - trim
            if chi - clo < 2:
                clo, chi = lo, hi
            window = self.signal[clo:chi]  # (Npts_in_step, Nc)
            dc = np.mean(window, axis=0)
            if self.modulation_method == "std":
                ac = np.std(window, axis=0)
            else:
                ac = np.max(window, axis=0) - np.min(window, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                pct_mod = 100.0 * ac / np.where(dc == 0, np.nan, dc)

            row = {"step": step, "freq_ghz": self.f_start + step * self.f_step}
            for c in range(len(pct_mod)):
                row[f"coil_{c + 1}_pct_mod"] = pct_mod[c]
            row["max_pct_mod"] = float(np.nanmax(pct_mod))
            row["best_coil"] = int(np.nanargmax(pct_mod))
            rows.append(row)
        return rows

    def to_dataframe(self):
        """Convert modulation_stats into a pandas DataFrame (requires pandas)."""
        if self.modulation_stats is None:
            self.analyze()
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for to_dataframe().") from e
        return pd.DataFrame(self.modulation_stats)

    def _flatten_dc(self, arr: np.ndarray) -> np.ndarray:
        """
        Remove each channel's DC level from arr (same shape as self.signal): the recording-wide
        mean by default, or (with flatten_steps set) each step's own per-channel median. Samples
        outside any step window fall back to the recording-wide mean.

        Returns:
        out (np.ndarray): DC-removed version of arr. (Shape: same as arr)
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            out = arr - np.nanmean(arr, axis=0)
            if self.flatten_steps:
                for step in range(self.n_steps):
                    lo, hi = self.step_bounds[step], self.step_bounds[step + 1]
                    if hi - lo < 1:
                        continue
                    out[lo:hi] = arr[lo:hi] - np.nanmedian(arr[lo:hi], axis=0)
        return out

    def plot_sweep(self,
                   figsize: tuple = (10, 10),
                   dpi: int = 300,
                   title: str = "BPT Sweep",
                   plot_f_start: float | None = None,
                   plot_f_stop: float | None = None,
                   shift: float | None = None,
                   save_fig: bool = False,
                   save_fname: str | None = None):
        """
        Plot all channels stacked and offset against frequency, with dashed lines at each
        nominal step frequency. Pass plot_f_start/plot_f_stop to zoom into a sub-range without
        re-running analyze().

        Channels are demeaned via _flatten_dc(); default shift is `visualizer.rel_shift *
        max_amplitude` of that demeaned signal, or pass `shift` to override.

        Stores:
        fig (plt.Figure), ax (plt.Axes): The generated figure/axis.
        """
        if self.freq_axis is None:
            self.analyze()

        n_sp, n_ch = self.signal.shape
        sig_dm = self._flatten_dc(self.signal)

        if shift is None:
            start_idx = int(0.1 * n_sp)
            max_amp = np.max(np.abs(sig_dm[start_idx:])) if n_sp > start_idx else np.max(np.abs(sig_dm))
            shift = self.visualizer.rel_shift * (max_amp + 1e-9)
        denom = shift if shift != 0 else 1

        offset_data = sig_dm / denom + np.arange(n_ch)

        plot_f_start = self.f_start if plot_f_start is None else plot_f_start
        plot_f_stop = self.f_stop if plot_f_stop is None else plot_f_stop
        mask = (self.freq_axis >= plot_f_start) & (self.freq_axis <= plot_f_stop)

        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.ax.plot(self.freq_axis[mask], offset_data[mask], lw=0.5)

        freqs = self.f_start + np.arange(self.n_steps) * self.f_step
        for f in freqs:
            if plot_f_start <= f <= plot_f_stop:
                self.ax.axvline(f, color="black", linestyle="--", alpha=0.25, lw=0.8)

        # Mark the anchor (fitted start_idx, at f_start).
        if plot_f_start <= self.f_start <= plot_f_stop:
            self.ax.axvline(self.f_start, color="crimson", linestyle="-", lw=1.5, alpha=0.85,
                             label="anchor (start_idx)")
            self.ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        self.ax.set_title(f"{title} | {plot_f_start:.2f}-{plot_f_stop:.2f} GHz")
        self.ax.set_xlabel("Frequency (GHz)")
        self.ax.set_ylabel("Signal")
        self.ax.set_xlim(plot_f_start, plot_f_stop)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.fig.tight_layout()

        self._maybe_save(self.fig, save_fig, save_fname or "bpt_sweep.png", dpi)
        if self.visualizer.display_fig:
            plt.show()

    def plot_jump_signal(self,
                         figsize: tuple = (12, 4),
                         dpi: int = 150,
                         title: str = "BPT Sweep - Jump Detection",
                         save_fig: bool = False,
                         save_fname: str | None = None):
        """
        Debug plot of the combined jump-strength signal (see analyze()): matched jumps in green,
        rejected/unmatched candidates in gray, fitted step grid as dashed lines.

        Stores:
        fig (plt.Figure), ax (plt.Axes): The generated figure/axis.
        """
        if self.jump_signal is None:
            self.analyze()

        tr = self.visualizer.tr
        t = np.arange(len(self.jump_signal)) * tr
        matched_idx = {idx for _, idx in self.matched_jumps} if self.matched_jumps else set()

        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.ax.plot(t, self.jump_signal, lw=0.6, color="steelblue")

        for idx, _strength in self.detected_jumps or []:
            matched = idx in matched_idx
            self.ax.axvline(idx * tr, color="green" if matched else "gray",
                             lw=1.0, alpha=0.8 if matched else 0.4)

        if self.start_idx is not None and self.step_len_samples is not None:
            edges = self.start_idx + np.arange(self.n_steps + 1) * self.step_len_samples
            for e in edges:
                self.ax.axvline(e * tr, color="black", linestyle="--", lw=0.5, alpha=0.2)

        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color="green", lw=1.5, label="matched jump (used in fit)"),
            Line2D([0], [0], color="gray", lw=1.5, alpha=0.5, label="rejected / unmatched candidate"),
            Line2D([0], [0], color="black", linestyle="--", lw=1.0, alpha=0.5, label="fitted step grid"),
        ]
        self.ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Combined jump strength")
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.fig.tight_layout()

        self._maybe_save(self.fig, save_fig, save_fname or "bpt_sweep_jump_signal.png", dpi)
        if self.visualizer.display_fig:
            plt.show()

    def plot_modulation(self,
                        kind: Literal["heatmap", "line"] = "heatmap",
                        figsize: tuple | None = None,
                        dpi: int = 150,
                        title: str | None = None,
                        cmap: str = "viridis",
                        aspect: Literal["equal", "auto"] = "equal",
                        save_fig: bool = False,
                        save_fname: str | None = None):
        """
        Plot percent modulation across the swept frequency range.

        kind="heatmap" (default): channel x frequency-step heatmap with a colorbar.
        kind="line": best-channel percent modulation vs frequency, as a single trace.

        Stores:
        fig (plt.Figure), ax (plt.Axes): The generated figure/axis.
        """
        if self.modulation_stats is None:
            self.analyze()

        freqs = [row["freq_ghz"] for row in self.modulation_stats]
        n_ch = self.signal.shape[1]

        if kind == "heatmap":
            mod_matrix = np.array([
                [row.get(f"coil_{c + 1}_pct_mod", np.nan) for row in self.modulation_stats]
                for c in range(n_ch)
            ])  # (n_ch, n_steps)

            self.fig, self.ax = plt.subplots(figsize=figsize or (max(6, 0.12 * len(freqs)), max(3, 0.4 * n_ch)), dpi=dpi)
            im = self.ax.imshow(mod_matrix, aspect=aspect, cmap=cmap)
            cbar = self.fig.colorbar(im, ax=self.ax)
            cbar.set_label("Percent Modulation (%)")

            n_steps = mod_matrix.shape[1]
            tick_idx = np.linspace(0, n_steps - 1, min(10, n_steps)).astype(int)
            self.ax.set_xticks(tick_idx)
            self.ax.set_xticklabels([f"{freqs[i]:.2f}" for i in tick_idx])
            self.ax.set_yticks(np.arange(n_ch))
            self.ax.set_yticklabels([f"Coil {c + 1}" for c in range(n_ch)])
            self.ax.set_xlabel("Frequency (GHz)")
            self.ax.set_ylabel("Channel")
            self.ax.set_title(title or "Percent Modulation per Step per Coil")
        elif kind == "line":
            max_pct_mod = [row["max_pct_mod"] for row in self.modulation_stats]

            self.fig, self.ax = plt.subplots(figsize=figsize or (10, 5), dpi=dpi)
            self.ax.plot(freqs, max_pct_mod, marker="o", ms=3, lw=1.2)
            self.ax.set_xlabel("Frequency (GHz)")
            self.ax.set_ylabel("Percent Modulation (%)")
            self.ax.set_title(title or "Percent Modulation vs Frequency")
            self.ax.spines["top"].set_visible(False)
            self.ax.spines["right"].set_visible(False)
        else:
            raise ValueError(f"Unknown kind: {kind!r}. Expected 'heatmap' or 'line'.")

        self.fig.tight_layout()

        self._maybe_save(self.fig, save_fig, save_fname or f"bpt_sweep_modulation_{kind}.png", dpi)
        if self.visualizer.display_fig:
            plt.show()

    def plot_combined(self,
                      figsize: tuple = (14, 10),
                      dpi: int = 150,
                      title: str = "BPT Sweep",
                      plot_f_start: float | None = None,
                      plot_f_stop: float | None = None,
                      shift: float | None = None,
                      ylim: tuple[float, float] | None = None,
                      cmap: str = "viridis",
                      height_ratios: tuple = (2, 1),
                      top_panel: Literal["absolute", "percent_modulation"] = "absolute",
                      trace_cmap: str = "coolwarm",
                      save_fig: bool = False,
                      save_fname: str | None = None):
        """
        Combined figure: BPT sweep (top) and modulation heatmap (bottom) sharing a frequency
        x-axis, at a fixed figsize -- for building comparable images across many series.

        top_panel="absolute" (default): each channel demeaned (_flatten_dc()), divided by one
            shared shift.
        top_panel="percent_modulation": each channel divided by its own mean first, then the same
            demean+shift. Traces colored by baseline (trace_cmap), with a colorbar and each
            channel's baseline value printed alongside it. Zero-baseline channels are skipped
            (grayed out) with a warning. Saves as "bpt_sweep_percent_modulation_combined.png".

        Channel 1 is at the top and channel n_ch at the bottom in both panels; pass ylim to zoom
        the line plot's y-axis.

        Stores:
        fig (plt.Figure), ax (plt.Axes, top), ax_heat (plt.Axes, bottom).
        """
        if self.modulation_stats is None:
            self.analyze()

        n_sp, n_ch = self.signal.shape

        plot_f_start = self.f_start if plot_f_start is None else plot_f_start
        plot_f_stop = self.f_stop if plot_f_stop is None else plot_f_stop
        mask = (self.freq_axis >= plot_f_start) & (self.freq_axis <= plot_f_stop)

        if top_panel == "percent_modulation":
            baseline = np.mean(self.signal, axis=0)  # each channel's own mean over the full recording
            zero_baseline = np.isclose(baseline, 0)
            if np.any(zero_baseline):
                logger.warning(f"Channel(s) {np.where(zero_baseline)[0].tolist()} have a ~zero "
                                "baseline; percent modulation is undefined for them and they'll be skipped.")
            safe_baseline = np.where(zero_baseline, np.nan, baseline)

            # Divide by each channel's own mean, then the same demean+shift processing as "absolute".
            normalized = self.signal / safe_baseline[None, :]
            plot_data = self._flatten_dc(normalized)
            plot_data *= 100.0  # display as percent; cancels out of the shift ratio below either way

            if shift is None:
                start_idx = int(0.1 * n_sp)
                trace = plot_data[start_idx:] if n_sp > start_idx else plot_data
                max_amp = np.nanmax(np.abs(trace)) if np.any(~np.isnan(trace)) else 1.0
                shift = self.visualizer.rel_shift * (max_amp + 1e-9)
        elif top_panel == "absolute":
            plot_data = self._flatten_dc(self.signal)
            if shift is None:
                start_idx = int(0.1 * n_sp)
                max_amp = np.max(np.abs(plot_data[start_idx:])) if n_sp > start_idx else np.max(np.abs(plot_data))
                shift = self.visualizer.rel_shift * (max_amp + 1e-9)
        else:
            raise ValueError(f"Unknown top_panel: {top_panel!r}. Expected 'absolute' or 'percent_modulation'.")

        denom = shift if shift != 0 else 1
        # Channel 1 at the top, channel n_ch at the bottom -- matches imshow's row order below.
        channel_offset = n_ch - 1 - np.arange(n_ch)
        offset_data = plot_data / denom + channel_offset[None, :]

        # Dedicated colorbar column keeps both rows the same width for a pixel-aligned x-axis.
        self.fig, axes = plt.subplots(
            2, 2, figsize=figsize, dpi=dpi, layout="constrained",
            gridspec_kw=dict(height_ratios=height_ratios, width_ratios=(1, 0.03), wspace=0.02)
        )
        self.ax, cax_spacer = axes[0]
        self.ax_heat, cax = axes[1]
        self.ax.sharex(self.ax_heat)

        # Top: sweep line plot (mirrors plot_sweep)
        if top_panel == "percent_modulation":
            trace_cmap_obj = plt.get_cmap(trace_cmap)
            norm = plt.Normalize(vmin=np.nanmin(baseline), vmax=np.nanmax(baseline))
            for k in range(n_ch):
                color = "lightgray" if zero_baseline[k] else trace_cmap_obj(norm(baseline[k]))
                self.ax.plot(self.freq_axis[mask], offset_data[mask, k], lw=0.5, color=color)
                self.ax.annotate(f"{baseline[k]:.3g}", xy=(1.005, channel_offset[k]), xycoords=("axes fraction", "data"),
                                  va="center", ha="left", fontsize=6, color="black")
            sm = plt.cm.ScalarMappable(norm=norm, cmap=trace_cmap_obj)
            sm.set_array([])
            cbar_trace = self.fig.colorbar(sm, cax=cax_spacer)
            cbar_trace.set_label("Channel Strength (mean signal)", fontsize=8)
            cbar_trace.ax.tick_params(labelsize=6)
        else:
            self.ax.plot(self.freq_axis[mask], offset_data[mask], lw=0.5)
            cax_spacer.axis("off")

        freqs = self.f_start + np.arange(self.n_steps) * self.f_step
        for f in freqs:
            if plot_f_start <= f <= plot_f_stop:
                self.ax.axvline(f, color="black", linestyle="--", alpha=0.25, lw=0.8)
        if plot_f_start <= self.f_start <= plot_f_stop:
            self.ax.axvline(self.f_start, color="crimson", linestyle="-", lw=1.5, alpha=0.85,
                             label="anchor (start_idx)")
            self.ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        self.ax.set_title(f"{title} | {plot_f_start:.2f}-{plot_f_stop:.2f} GHz")
        self.ax.set_ylabel("Percent Modulation (%)" if top_panel == "percent_modulation" else "Signal")
        self.ax.set_yticks(np.arange(n_ch))
        self.ax.set_yticklabels([str(n_ch - o) for o in range(n_ch)])
        if ylim is not None:
            self.ax.set_ylim(*ylim)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

        # Bottom: modulation heatmap, in real frequency units so it aligns with the sweep panel.
        mod_matrix = np.array([
            [row.get(f"coil_{c + 1}_pct_mod", np.nan) for row in self.modulation_stats]
            for c in range(n_ch)
        ])  # (n_ch, n_steps)
        # Each column spans [freq, freq + f_step) -- not centered on freq -- to match freq_axis,
        # where a step's plateau runs from its own nominal frequency up to the next step's.
        step_freqs = np.array([row["freq_ghz"] for row in self.modulation_stats])
        extent = [step_freqs[0], step_freqs[-1] + self.f_step, n_ch - 0.5, -0.5]
        im = self.ax_heat.imshow(mod_matrix, aspect="auto", cmap=cmap, extent=extent)
        cbar = self.fig.colorbar(im, cax=cax)
        cbar.set_label("Percent Modulation (%)")
        self.ax_heat.set_yticks(np.arange(n_ch))
        self.ax_heat.set_yticklabels([f"Coil {c + 1}" for c in range(n_ch)], fontsize=7)
        self.ax_heat.set_xlabel("Frequency (GHz)")
        self.ax_heat.set_ylabel("Channel")
        self.ax_heat.set_xlim(plot_f_start, plot_f_stop)

        default_fname = "bpt_sweep_percent_modulation_combined.png" if top_panel == "percent_modulation" else "bpt_sweep_combined.png"
        self._maybe_save(self.fig, save_fig, save_fname or default_fname, dpi)
        if self.visualizer.display_fig:
            plt.show()

    def _maybe_save(self, fig: plt.Figure, save_fig: bool, default_fname: str, dpi: int):
        """Save a figure into visualizer.inp_dir, if requested and available."""
        if not save_fig:
            return
        if not self.visualizer.inp_dir:
            if self.verbose:
                logger.warning("save_fig=True but visualizer.inp_dir is not set; figure was not saved.")
            return
        save_path = os.path.join(self.visualizer.inp_dir, default_fname)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if self.verbose:
            logger.info(f"Saved figure to {save_path}")