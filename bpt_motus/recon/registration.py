import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Optional
import os
import logging

logger = logging.getLogger(__name__)

class ElastixRegistration:
    def __init__(self, 
                 inp_dir: str, 
                 out_dir: str | None = None, 
                 fixed_frame_idx: int = 0,
                 frames_type: str = "pics",
                 moving_file: str | None = None,
                 mode: str = 'rigid', 
                 max_iters: list = [20, 20, 20, 20],
                 thres: float = 30, 
                 verbose: bool = True, 
                 force_reload: bool = False,
                 save_filename: str = "rigid_parameters.npy"):
        
        # Core settings
        self.inp_dir: str = inp_dir # Directory containing frames to register.
        self.out_dir: str = out_dir if out_dir is not None else inp_dir # Directory to save registration results.
        self.fixed_frame_idx: int = fixed_frame_idx # Index of the frame to use as the fixed reference.
        self.frames_type: str = frames_type # Type of input frames ('pics', 'warped_pics', or 'custom').
        self.mode: str = mode # Registration mode (e.g., 'rigid').
        self.max_iters: list = max_iters # Maximum iterations per resolution level.
        self.thres: float = thres # Threshold for spatial mask creation (pixels to crop from the bottom).
        self.verbose: bool = verbose
        self.force_reload: bool = force_reload
        
        # Dynamic moving filename resolution based on class parameters
        if moving_file is None:
            if self.frames_type == "pics":
                self.moving_file: str = "pics_frames.npy"
            elif self.frames_type == "warped_pics":
                self.moving_file: str = "warped_pics_frames.npy"
            elif self.frames_type == "custom":
                raise ValueError("moving_file must be explicitly specified when frames_type is set to 'custom'.")
            else:
                raise ValueError(f"Unknown frames_type: '{self.frames_type}'. Must be 'pics', 'warped_pics', or 'custom'.")
        else:
            self.moving_file: str = moving_file

        # Filenames
        self.moving_img_fname: str = os.path.join(self.inp_dir, self.moving_file)
        self.save_fname: str = os.path.join(self.out_dir, save_filename)
        
        # Intermediate data
        self.moving_img: np.ndarray | None = None
        self.moving_frames: np.ndarray | None = None
        self.fixed_img: np.ndarray | None = None
        self.fixed_mask: np.ndarray | None = None
        self.sitk_params: sitk.ParameterMap | None = None
        
        # Outputs
        self.result_imgs: np.ndarray | None = None
        self.motion_params: np.ndarray | None = None

    def run(self):
        """
        Do rigid registration on the loaded frames.

        Stores:
        motion_params (np.ndarray): Rigid motion transform parameters. (Shape: (Nframes, 3) or (Nframes, 6))
        result_imgs (np.ndarray): The registered/transformed array frames. (Shape: (Nframes, Nx, Ny) or (Nframes, Nx, Ny, Nz))
        """
        if not self.force_reload and os.path.exists(self.save_fname):
            if self.verbose:
                logger.info(f"Loading existing registration results from {self.save_fname}")
            self.motion_params = np.load(self.save_fname)
            return

        self._load_data()
        self._setup_registration()
        self._run_registration()

        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            np.save(self.save_fname, self.motion_params)
            if self.verbose:
                logger.info(f"Saved parsed rigid motion parameters to {self.save_fname}")

    def _load_data(self):
        """
        Load the moving image frames from disk and prepare the fixed reference frame.

        Stores:
        moving_img (np.ndarray): The raw loaded image array from disk.
        moving_frames (np.ndarray): The absolute magnitude of the moving image with guaranteed temporal framing.
        fixed_img (np.ndarray): The selected fixed reference frame. (Shape: (Nx, Ny) or (Nx, Ny, Nz))
        """
        if self.verbose:
            logger.info(f"Loading moving images from {self.moving_img_fname}")
            
        if not os.path.exists(self.moving_img_fname):
            raise ValueError(f"Moving image file not found at {self.moving_img_fname}")
            
        self.moving_img = np.load(self.moving_img_fname)
        moving_img_abs = np.abs(self.moving_img)
        
        # Ensure array is 3D (2D+t) or 4D (3D+t) to safely extract frames
        self.moving_frames = moving_img_abs if moving_img_abs.ndim > self.moving_img[0].ndim else moving_img_abs[np.newaxis, ...]
        self.fixed_img = self.moving_frames[self.fixed_frame_idx]
        
    def _setup_registration(self):
        """
        Initialize the fixed image mask and Elastix parameter maps.

        Stores:
        fixed_mask (np.ndarray): Binary mask calculated from the fixed frame. (Shape: (Nx, Ny) or (Nx, Ny, Nz))
        sitk_params (sitk.ParameterMap): The constructed ITK registration parameter mapping.
        """
        self.fixed_mask
        self.sitk_params
        
    def _run_registration(self):
        """
        Iterate over all frames, execute SimpleITK Elastix registration, and extract rigid parameters.

        Stores:
        result_imgs (np.ndarray): Array of the registered warped frames.
        motion_params (np.ndarray): Parsed array of rigid parameters for each frame. (Shape: (Nframes, 3) for 2D, (Nframes, 6) for 3D)
        """
        n_frames = self.moving_frames.shape[0]
        self.result_imgs = np.zeros_like(self.moving_frames)
        
        # Determine number of parameters based on dimensionality (3 for 2D, 6 for 3D)
        ndim = self.fixed_img.ndim
        num_params = 3 if ndim == 2 else 6
        self.motion_params = np.zeros((n_frames, num_params), dtype=np.float32)

        sitk_fixed = sitk.GetImageFromArray(self.fixed_img.astype(np.float32))
        sitk_mask = sitk.GetImageFromArray(self.fixed_mask.astype(np.uint8))
        
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(sitk_fixed)
        elastixImageFilter.SetFixedMask(sitk_mask)
        elastixImageFilter.SetParameterMap(self.sitk_params)

        if self.verbose:
            logger.info(f"Running Elastix Registration on {n_frames} frames ({ndim}D data)...")
            
        for i in range(n_frames):
            if self.verbose and i % 10 == 0:
                logger.info(f"Registering frame {i+1}/{n_frames}...")
                
            sitk_moving = sitk.GetImageFromArray(self.moving_frames[i].astype(np.float32))
            elastixImageFilter.SetMovingImage(sitk_moving)
            
            try:
                if not self.verbose:
                    elastixImageFilter.LogToConsoleOff()
                elastixImageFilter.Execute()
                
                self.result_imgs[i] = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())
                
                # Extract the rigid parameters and store them in the array
                transform_params = elastixImageFilter.GetTransformParameterMap()[0]['TransformParameters']
                self.motion_params[i, :] = [float(p) for p in transform_params]
                
            except Exception as e:
                logger.error(f"Elastix failed on frame {i}: {e}")
                self.result_imgs[i] = self.moving_frames[i]
                self.motion_params[i, :] = 0.0 # Default to no motion on failure
                
        if self.moving_img.ndim <= 3 and self.moving_img.shape[0] != n_frames:
            # Drop the temporal dimension if the original image was a single frame
            self.result_imgs = self.result_imgs[0]

    def _get_param_map(self):
        """
        Construct the SimpleITK ParameterMap dictating the registration optimization scheme.

        Stores:
        sitk_params (sitk.ParameterMap): The customized Elastix configuration.
        """
        if self.verbose:
            logger.info(f"Setting SimpleITK Elastix parameters for {self.mode} registration.")
        self.sitk_params = sitk.GetDefaultParameterMap(self.mode)
        
        self.sitk_params['MaximumNumberOfIterations'] = [str(x) for x in self.max_iters]
        self.sitk_params['Metric'] = ['AdvancedMattesMutualInformation']
        self.sitk_params['AutomaticScalesEstimation'] = ['true']
        self.sitk_params['AutomaticTransformInitialization'] = ['false']
        self.sitk_params['AutomaticParameterEstimation'] = ['true']
        self.sitk_params['AutomaticTransformInitializationMethod'] = ['CenterOfGravity']
        self.sitk_params['NumberOfResolutions'] = [str(len(self.max_iters))]
        
        num_res = len(self.max_iters)
        schedule = []
        for r in reversed(range(num_res)):
            schedule.extend([2**r, 2**r, 2**r])
            
        self.sitk_params['FixedImagePyramidSchedule'] = [str(x) for x in schedule]
        self.sitk_params['MovingImagePyramidSchedule'] = [str(x) for x in schedule]
        self.sitk_params['ErodeMask'] = ['false']
        self.sitk_params['Optimizer'] = ['QuasiNewtonLBFGS']
        self.sitk_params['StopIfWolfeNotSatisfied'] = ['false']
        self.sitk_params['UseJacobianPreconditioning'] = ['false']
        self.sitk_params['UseDirectionCosines'] = ['false']
    
    def _create_mask(self):
        """
        Compute a basic rectangular binary mask for the fixed image to crop out the neck.

        Stores:
        fixed_mask (np.ndarray): Unsigned 8-bit integer mask indicating regions of signal.
        """
        if self.verbose:
            logger.info(f"Computing spatial fixed mask (cropping {int(self.thres)} pixels)...")
            
        # Focus the registration by masking out the neck area
        self.fixed_mask = np.zeros_like(self.fixed_img)
        self.fixed_mask[:-int(self.thres), ...] = 1