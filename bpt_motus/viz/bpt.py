import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def plot_bpts(bpts, tr=1, rel_shift=0.5, figsize=(10, 10), titles=None, ax=None):
    """
    Plot BPT/PTs across all coils. Automatically shifts coil signals for visibility.
    """
    if bpts.ndim == 2:
        bpts = bpts[np.newaxis, ...]
        
    nbpts, npe, ncoils = bpts.shape
    
    if titles is None:
        titles = [f"B+PT {i+1} PCs" for i in range(nbpts)]
        
    bpt_dm = bpts - np.mean(bpts, axis=1, keepdims=True)
    
    start_idx = int(0.1 * npe)
    shift = rel_shift * np.max(np.abs(bpt_dm[:, start_idx:]))
        
    ncols = 2 if nbpts > 1 else 1
    nrows = int(np.ceil(nbpts / ncols))
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        axes_created = True
    else:
        axes_created = False
        
    t = np.arange(npe) * tr
    
    for i in range(nbpts):
        if axes_created:
            ax_sub = plt.subplot(nrows, ncols, i+1)
        else:
            ax_sub = ax
            
        ax_sub.plot(t, (bpt_dm[i] / shift) + np.arange(ncoils))
        ax_sub.set_title(titles[i])
        
        if axes_created and i >= (nrows - 1) * ncols:
            ax_sub.set_xlabel("Time (s)")

    if axes_created:
        plt.tight_layout()

def load_and_crop_physio(inpdir, file_pattern, dt, max_time_sec):
    """
    Helper to load, dynamically crop, and independently scale a physio file.
    """
    matches = glob.glob(os.path.join(inpdir, file_pattern))
    if not matches:
        return None, None
        
    raw_data = np.loadtxt(matches[0])
    fs = 1.0 / dt
    
    total_physio_duration = len(raw_data) * dt
    discard_sec = 31.0 if total_physio_duration > max_time_sec else 0.0
        
    start_idx = int(discard_sec * fs)
    if start_idx >= len(raw_data):
        return None, None
        
    cropped = raw_data[start_idx:]
    t_axis = np.arange(len(cropped)) * dt
    time_mask = t_axis <= max_time_sec
    t_axis = t_axis[time_mask]
    final_signal = cropped[:len(t_axis)]
    
    sig_dm = final_signal - np.mean(final_signal)
    sig_norm = sig_dm / np.max(np.abs(sig_dm))
    
    return t_axis, sig_norm

def plot_mri_and_physio_combined(bpts, tr, inpdir, plot_bpt=True, plot_resp=True, plot_cardio=False, 
                                 time_range=None, rel_shift=0.5, figsize=(10, 10)):
    """
    Leverages plot_bpts to render the layout, shifts the coils up to place physio
    signals at the bottom, and explicitly customizes the Y-axis labels.
    """
    if plot_bpt:
        plot_bpts(bpts, tr=tr, rel_shift=rel_shift, figsize=figsize)
    else:
        nbpts = bpts.shape[0] if bpts.ndim == 3 else 1
        ncols = 2 if nbpts > 1 else 1
        nrows = int(np.ceil(nbpts / ncols))
        plt.figure(figsize=figsize)
        for i in range(nbpts):
            plt.subplot(nrows, ncols, i + 1)
            plt.title(f"B+PT {i+1} PCs" if bpts.ndim == 3 else "B+PT 1 PCs")
    
    ncoils = bpts.shape[-1]
    npe = bpts.shape[-2]
    max_time_sec = (npe - 1) * tr
    
    resp_data = load_and_crop_physio(inpdir, "RESPData_uwute*", 0.04, max_time_sec) if plot_resp else (None, None)
    cardio_data = load_and_crop_physio(inpdir, "PPGData_uwute*", 0.01, max_time_sec) if plot_cardio else (None, None)
    
    fig = plt.gcf()
    axes = fig.get_axes()
    
    for ax in axes:
        subplot_title = ax.get_title()
        
        bottom_clearance = 0.0
        if plot_resp: bottom_clearance += 1.5
        if plot_cardio: bottom_clearance += 1.5
        
        if plot_bpt:
            for line in list(ax.get_lines()):
                x_val, y_val = line.get_data()
                line.set_ydata(y_val + bottom_clearance)
            
        y_ticks = []
        y_tick_labels = []
        current_vertical_offset = 0.5
        
        t_ppg, ppg_norm = cardio_data
        if ppg_norm is not None:
            ax.plot(t_ppg, ppg_norm + current_vertical_offset, color='darkviolet', linewidth=1.2)
            y_ticks.append(current_vertical_offset)
            y_tick_labels.append("PPG (Cardio)")
            current_vertical_offset += 1.5
            
        t_resp, resp_norm = resp_data
        if resp_norm is not None:
            ax.plot(t_resp, resp_norm + current_vertical_offset, color='crimson', linewidth=1.8)
            y_ticks.append(current_vertical_offset)
            y_tick_labels.append("Resp Bellows")
            current_vertical_offset += 1.5

        if plot_bpt:
            for k in range(ncoils):
                coil_center_y = k + bottom_clearance
                y_ticks.append(coil_center_y)
                prefix = subplot_title.split('PCs', 1)[0].strip() if subplot_title else f"PC"
                y_tick_labels.append(f"{prefix}, {k}")
            y_max_limit = ncoils + bottom_clearance
        else:
            y_max_limit = current_vertical_offset

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        ax.set_ylabel("") 
        ax.set_xlabel("Time (s)")
        
        if time_range is not None:
            ax.set_xlim(time_range[0], time_range[1])
        else:
            ax.set_xlim(0, max_time_sec)
            
        ax.set_ylim(-0.5, y_max_limit)
        ax.grid(True, alpha=0.15)
        
    plt.tight_layout()
    return fig
