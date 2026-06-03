import numpy as np
import matplotlib.pyplot as plt
import sigpy.plot as pl

class ImageVisualizer:
    """
    Utility wrapper for plotting multi-dimensional MRI volumes.
    Leverages sigpy.plot natively, alongside custom multi-panel configurations.
    """
    def __init__(self, **sigpy_kwargs):
        self.sigpy_kwargs = sigpy_kwargs
        
    def plot_interactive(self, image_array, title="Volume"):
        """
        Invokes Sigpy's interactive ImagePlot viewer.
        """
        return pl.ImagePlot(image_array, title=title, **self.sigpy_kwargs)

    def plot_side_by_side(self, img_list, titles=None, z_slice=None, figsize=(15, 6), cmap='gray'):
        """
        Plots multiple 3D volumes side-by-side cleanly via Matplotlib.
        img_list: list of 3D arrays (x, y, z)
        """
        num_imgs = len(img_list)
        fig, axes = plt.subplots(1, num_imgs, figsize=figsize)
        if num_imgs == 1:
            axes = [axes]
            
        for i, img in enumerate(img_list):
            ax = axes[i]
            if z_slice is None:
                # center slice
                slice_idx = img.shape[2] // 2
            else:
                slice_idx = z_slice
                
            disp = np.abs(img[:, :, slice_idx])
            im = ax.imshow(disp, cmap=cmap)
            if titles and i < len(titles):
                ax.set_title(titles[i])
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        plt.tight_layout()
        return fig
