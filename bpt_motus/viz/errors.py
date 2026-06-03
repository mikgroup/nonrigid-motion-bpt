import json
import matplotlib.pyplot as plt
import numpy as np

def plot_registration_parameters(json_paths, labels, title="Registration Parameters (GT vs Modeled)", figsize=(12, 10)):
    """
    Plots registration parameters (translations/rotations) derived from Elastix, 
    comparing them across multiple JSON tracker outputs.
    
    Args:
        json_paths (list of str): Paths to registration_errors.json files outputted by ErrorEvaluator.
        labels (list of str): Labels corresponding to each JSON file (e.g. 'MRMOTUS', 'BPT_MOTUS').
    """
    if len(json_paths) != len(labels):
        raise ValueError("Paths and labels lists must match in length.")
        
    all_gt_params = []
    all_model_params = []
    
    for path in json_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            # Both contain frame-by-frame lists
            all_gt_params.append(np.array(data['gt_params']))
            all_model_params.append(np.array(data['model_params']))

    # Assume GT params from the same standard PICS sequence are identical across datasets, 
    # but we just plot the first one as ground truth for cleanliness.
    basline_gt = all_gt_params[0]
    num_frames = basline_gt.shape[0]
    t = np.arange(num_frames)

    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    param_names = ["Rot X", "Rot Y", "Rot Z", "Trans X", "Trans Y", "Trans Z"]
    
    for dim in range(6):
        ax = axes[dim // 2, dim % 2]
        # Plot GT reference once
        ax.plot(t, basline_gt[:, dim], 'k--', linewidth=2.5, alpha=0.7, label='PICS (GT)')
        
        # Plot each model mapping 
        for i, mod_params in enumerate(all_model_params):
            ax.plot(t, mod_params[:, dim], label=labels[i], alpha=0.85)

        ax.set_title(param_names[dim])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    for ax in axes[-1, :]:
        ax.set_xlabel("Frame Index")
        
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig

def plot_voxel_rmses(json_paths, labels, title="Voxel-wise Tracking RMSE", figsize=(10, 6)):
    """
    Plot voxel-wise frame-by-frame RMSE over time across various methods.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for path, label in zip(json_paths, labels):
        with open(path, 'r') as f:
            data = json.load(f)
            f_rmses = data['frame_rmses']
            t_axis = np.arange(len(f_rmses))
            
            global_val = data.get('global_rmse', np.mean(f_rmses))
            ax.plot(t_axis, f_rmses, label=f"{label} (Global: {global_val:.4f})", alpha=0.9, linewidth=1.5)

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Relative RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig
