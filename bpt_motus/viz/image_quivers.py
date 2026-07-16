import numpy as np
import sigpy.plot as pl
from scipy.ndimage import label, binary_fill_holes, generate_binary_structure, binary_closing

class ImageQuiverPlot(pl.ImagePlot):
    """
    Subclasses SigPy's native ImagePlot to precisely match the 
    QuiverAndImagePlot layout and performance from the custom plot.py script.
    """
    def __init__(self, img, motion=None, step=2, scale=5e-2, apply_mask=False, mask_thresh=0.05, **kwargs):
        self.motion = motion
        self.step = step
        self.scale = scale
        self.apply_mask = apply_mask
        self.mask_thresh = mask_thresh
        self.head_mask = None

        self.quiver_artist = None
        self.ax_im = None
        self.ax_quiver = None
        
        # Initialize native SigPy plot
        super().__init__(img, **kwargs)

        # Apply foreground mask if requested
        if self.apply_mask:
            if img.ndim <= 3: # Handle 2D or 3D images
                self.head_mask = self._compute_foreground_mask(img)
            else:
                self.head_mask = self._compute_foreground_mask(img[0])  # Use first frame for mask
        
        # Split into a side-by-side layout
        if self.motion is not None:
            gs = self.fig.add_gridspec(1, 2, wspace=0, hspace=0)
            self.ax.set_subplotspec(gs[0, 0])
            self.ax_im = self.ax  
            
            # Create transparent quiver axis on the right
            self.ax_quiver = self.fig.add_subplot(gs[0, 1])
            self.fig.set_facecolor('None')
            self.fig.patch.set_alpha(0.0)
            self.ax_quiver.set_facecolor('None')
            self.ax_quiver.patch.set_alpha(0.0)
            
            self.update_axes()
            self.update_image()

    def update_axes(self):
        """
        Overrides SigPy's text generation to match plot.py precisely.
        """
        if getattr(self, 'ax_im', None) is None or getattr(self, 'motion', None) is None:
            return super().update_axes()
            
        if not getattr(self, 'hide_axes', False):
            caption = "["
            for i in range(self.ndim):
                if i == getattr(self, 'd', None):
                    caption += "["
                else:
                    caption += " "

                if self.flips[i] == -1 and (i == self.x or i == self.y or i == getattr(self, 'z', None) or i == getattr(self, 'c', None)):
                    caption += "-"

                if i == self.x:
                    caption += "x"
                elif i == self.y:
                    caption += "y"
                elif i == getattr(self, 'z', None):
                    caption += "z"
                elif i == getattr(self, 'c', None):
                    caption += "c"
                elif i == getattr(self, 'd', None) and getattr(self, 'entering_slice', False):
                    caption += str(self.entered_slice) + "_"
                else:
                    caption += str(self.slices[i])

                if i == getattr(self, 'd', None):
                    caption += "]"
                else:
                    caption += " "
            caption += "]"

            # Apply labels
            self.ax_im.set_title(caption)
            self.ax_im.xaxis.set_visible(True)
            self.ax_im.yaxis.set_visible(True)
            self.ax_im.title.set_visible(True)
            
            self.ax_quiver.set_title("Quiver Plot")
            self.ax_quiver.xaxis.set_visible(True)
            self.ax_quiver.yaxis.set_visible(False)
            self.ax_quiver.title.set_visible(True)
            
            if getattr(self, 'title', None):
                self.fig.suptitle(self.title)
        else:
            for ax_obj in [self.ax_im, self.ax_quiver]:
                ax_obj.set_title("")
                ax_obj.xaxis.set_visible(False)
                ax_obj.yaxis.set_visible(False)
                ax_obj.title.set_visible(False)
            self.fig.suptitle("")

    def update_image(self):
        """
        Handles data slicing, transposing, and correctly mapping vector channels.
        """
        super().update_image()
        
        if getattr(self, 'motion', None) is None or getattr(self, 'ax_quiver', None) is None:
            return
            
        # 1. SPATIAL SLICING WITH FLIPS
        idx = []
        for i in range(self.ndim):
            if i in [self.x, self.y]:
                # Apply flips directly to the slicing so the spatial grid mirrors correctly
                idx.append(slice(None, None, self.flips[i]))
            else:
                idx.append(self.slices[i])
                
        if len(self.motion.shape) > self.ndim:
            idx.append(slice(None)) # Keep the [X_mot, Y_mot, Z_mot] channel
            
        try:
            motion_slice = self.motion[tuple(idx)]
            
            # 2. TRANSPOSE TO DISPLAY AXES (Y, X)
            # If self.x < self.y, the array is ordered (X, Y). Matplotlib plots (Rows, Cols), 
            # so we must transpose it to (Y, X) to match the SigPy image display.
            if self.x < self.y:
                motion_slice = np.transpose(motion_slice, (1, 0, 2))
                
            ny, nx = motion_slice.shape[:2]
            X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
            
            # 3. DYNAMIC VECTOR CHANNELS
            # Calculate the offset (e.g., if shape is (frames, x, y, z), offset is 1)
            spatial_offset = max(0, self.ndim - 3) 
            u_comp = max(0, self.x - spatial_offset)
            v_comp = max(0, self.y - spatial_offset)
            
            # Extract components and apply flips so the arrows change direction when inverted
            U = motion_slice[..., u_comp] * self.flips[self.x]
            V = motion_slice[..., v_comp] * self.flips[self.y]
            
            X_sub = X[::self.step, ::self.step]
            Y_sub = Y[::self.step, ::self.step]
            U_sub = U[::self.step, ::self.step]
            V_sub = V[::self.step, ::self.step]
            
            # --- MASKING LOGIC ---
            magnitude = np.sqrt(U_sub**2 + V_sub**2)
            valid_mask = magnitude > 1e-6 
            
            if self.apply_mask and self.head_mask is not None:
                mask_idx = []
                for i in range(self.ndim):
                    if i in [self.x, self.y]:
                        mask_idx.append(slice(None, None, self.flips[i]))
                    else:
                        mask_idx.append(self.slices[i])
                
                # Align mask dimensions with image dimensions
                dim_diff = len(mask_idx) - self.head_mask.ndim
                if dim_diff > 0:
                    mask_idx = mask_idx[dim_diff:]
                    mask_x = self.x - dim_diff
                    mask_y = self.y - dim_diff
                else:
                    mask_x = self.x
                    mask_y = self.y
                    
                struct_mask_slice = self.head_mask[tuple(mask_idx)]
                
                # Transpose the mask to perfectly align with the (Y, X) display axes
                if mask_x < mask_y:
                    struct_mask_slice = struct_mask_slice.T
                    
                struct_mask_sub = struct_mask_slice[::self.step, ::self.step]
                valid_mask = valid_mask & struct_mask_sub
            # --- END MASKING LOGIC ---
            
            X_filtered = X_sub[valid_mask]
            Y_filtered = Y_sub[valid_mask]
            U_filtered = U_sub[valid_mask]
            V_filtered = V_sub[valid_mask]
            
            # DRAW QUIVERS
            if self.quiver_artist is not None:
                self.quiver_artist.remove()
                self.quiver_artist = None
                
            self.quiver_artist = self.ax_quiver.quiver(
                X_filtered, Y_filtered, U_filtered, V_filtered,
                scale=self.scale, 
                color='tab:orange',
                minlength=1
            )
            
            self.ax_quiver.set_xlim(0, nx)
            self.ax_quiver.set_ylim(0, ny)
            self.ax_quiver.set_aspect('equal', adjustable='box')
            self.ax_quiver.autoscale(False)
            
            self.fig.canvas.draw()
            
        except Exception as e:
            print(f"Quiver update failed: {e}")

    # def update_image(self):
    #     """
    #     Handles data slicing and perfectly mimics your high-performance vector updates.
    #     """
    #     super().update_image()
        
    #     if getattr(self, 'motion', None) is None or getattr(self, 'ax_quiver', None) is None:
    #         return
            
    #     idx = list(self.slices)
    #     idx[self.x] = slice(None)
    #     idx[self.y] = slice(None)
        
    #     if len(self.motion.shape) > len(idx):
    #         idx.append(slice(None))
            
    #     try:
    #         motion_slice = self.motion[tuple(idx)]
    #         ny, nx = motion_slice.shape[:2]
    #         X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
            
    #         # Map spatial axes, applying flips just like your original code
    #         u_comp = min(self.x - 1, 2) if self.x > 0 else 0
    #         v_comp = min(self.y - 1, 2) if self.y > 0 else 1
            
    #         U = motion_slice[..., u_comp] * self.flips[self.x]
    #         V = motion_slice[..., v_comp] * self.flips[self.y]
            
    #         # Apply your subsampling
    #         X_sub = X[::self.step, ::self.step]
    #         Y_sub = Y[::self.step, ::self.step]
    #         U_sub = U[::self.step, ::self.step]
    #         V_sub = V[::self.step, ::self.step]
            
    #         # Masking
    #         magnitude = np.sqrt(U_sub**2 + V_sub**2)
    #         valid_mask = magnitude > 1e-6 
    #         if self.apply_mask and self.head_mask is not None:
    #             mask_idx = list(self.slices)
    #             dim_diff = len(mask_idx) - self.head_mask.ndim
    #             if dim_diff > 0:
    #                 mask_idx = mask_idx[dim_diff:]
    #                 mask_x = self.x - dim_diff
    #                 mask_y = self.y - dim_diff
    #             else:
    #                 mask_x = self.x
    #                 mask_y = self.y
    #             mask_idx[mask_x] = slice(None)
    #             mask_idx[mask_y] = slice(None)
    #             struct_mask_slice = self.head_mask[tuple(mask_idx)]
    #             struct_mask_sub = struct_mask_slice[::self.step, ::self.step]
    #             # Combine masks
    #             valid_mask = valid_mask & struct_mask_sub
            
    #         X_filtered = X_sub[valid_mask]
    #         Y_filtered = Y_sub[valid_mask]
    #         U_filtered = U_sub[valid_mask]
    #         V_filtered = V_sub[valid_mask]
            
    #         # Remove old artist gracefully
    #         if self.quiver_artist is not None:
    #             self.quiver_artist.remove()
    #             self.quiver_artist = None
                
    #         # Draw new artist with your exact arguments
    #         self.quiver_artist = self.ax_quiver.quiver(
    #             X_filtered, Y_filtered, U_filtered, V_filtered,
    #             scale=self.scale, 
    #             color='tab:orange',
    #             minlength=1
    #         )
            
    #         self.ax_quiver.set_xlim(0, nx)
    #         self.ax_quiver.set_ylim(0, ny)
    #         self.ax_quiver.set_aspect('equal', adjustable='box')
    #         self.ax_quiver.autoscale(False)
            
    #         self.fig.canvas.draw()
            
    #     except Exception as e:
    #         print(f"Quiver update failed: {e}")

    def _compute_foreground_mask(self, img):
        """
        Calculates a continuous boolean mask representing the head/foreground.
        """
        img_mag = np.abs(img)
        # Basic threshold
        threshold = self.mask_thresh * img_mag.max()
        binary = img_mag > threshold
        
        # Largest connected component
        labeled, num = label(binary)
        if num == 0:
            return binary # Fallback if image is entirely blank
            
        region_sizes = np.bincount(labeled.flat)
        region_sizes[0] = 0 # Ignore background
        largest_label = np.argmax(region_sizes)
        foreground = (labeled == largest_label)
        
        # Fill holes and close gaps
        foreground = binary_fill_holes(foreground)
        structure = generate_binary_structure(img.ndim, 2)
        foreground = binary_closing(foreground, structure=structure, iterations=2)
        foreground = binary_fill_holes(foreground)
        
        return foreground