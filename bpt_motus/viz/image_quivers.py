import numpy as np
import sigpy.plot as pl

class ImageQuiverPlot(pl.ImagePlot):
    """
    Subclasses SigPy's native ImagePlot to automatically adapt layouts:
      - If only `img` is passed: Standard, full-window SigPy image viewer.
      - If both `img` and `motion` are passed: Automatically creates a 
        side-by-side layout (Image on left, Motion Field Quiver on right).
    """
    def __init__(self, img: np.ndarray, motion: np.ndarray = None, 
                 step: int = 4, scale: float = 1.0, **kwargs):
        
        self.motion = motion
        self.step = step
        self.scale = scale
        self.quiver_artist = None
        self.ax_quiver = None
        
        # 1. Initialize native SigPy plot (creates self.fig and self.ax as a 111 subplot)
        super().__init__(img, **kwargs)
        
        # 2. Automatically split into a side-by-side layout if motion data is provided
        if self.motion is not None:
            self.ax.remove()                            # Clear the default full-window subplot
            self.ax = self.fig.add_subplot(121)         # Left panel: Image
            self.ax_quiver = self.fig.add_subplot(122)  # Right panel: Vector Quiver
            
            # Reset and re-render the image on the new left-hand subplot
            self.axim = None 
            self.update_image()

    def update_image(self):
        """
        Overrides SigPy's core rendering thread to update both the image and the vectors.
        """
        # Let SigPy slice and render the image onto self.ax
        super().update_image()
        
        # Clean up the previous frame's quiver arrows
        if self.quiver_artist is not None:
            self.quiver_artist.remove()
            self.quiver_artist = None
            
        # Automatic exit if we are only viewing images
        if self.motion is None:
            return
            
        # Refresh the quiver axes canvas
        self.ax_quiver.clear()
        self.ax_quiver.set_title("Motion Field (Quiver)")
        self.ax_quiver.set_aspect('equal')
        self.ax_quiver.set_facecolor('black')  # Dark background makes white vectors pop beautifully
        
        # Grab the exact multi-dimensional coordinates SigPy just used to slice the image
        idx = list(self.slices)
        idx[self.x] = slice(None)
        idx[self.y] = slice(None)
        
        try:
            # Slice the motion field to perfectly match the current view matrix
            motion_slice = self.motion[tuple(idx)]
            ny, nx = motion_slice.shape[:2]
            X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
            
            # Subsample grid positions to prevent crowding
            X_sub = X[::self.step, ::self.step]
            Y_sub = Y[::self.step, ::self.step]
            
            # Dynamic plane projection: grab vector components matching screen X and Y
            u_idx = min(self.x, motion_slice.shape[-1] - 1)
            v_idx = min(self.y, motion_slice.shape[-1] - 1)
            
            U_sub = motion_slice[::self.step, ::self.step, u_idx] * self.scale
            V_sub = motion_slice[::self.step, ::self.step, v_idx] * self.scale
            
            # Draw the new vectors onto the right-hand panel
            self.quiver_artist = self.ax_quiver.quiver(X_sub, Y_sub, U_sub, V_sub, 
                                                       color='white', angles='xy', 
                                                       scale_units='xy', scale=1)
            # Match boundary limits to the anatomical image grid
            self.ax_quiver.set_xlim(0, nx)
            self.ax_quiver.set_ylim(0, ny)
            self.ax_quiver.axis('off')
        except Exception:
            pass

        self.fig.canvas.draw_idle()