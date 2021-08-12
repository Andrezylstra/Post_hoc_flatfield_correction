import os
import argparse
import numpy as np
from skimage.io import imread, MultiImage, imsave
from scipy.ndimage import uniform_filter
from skimage.filters import gaussian
from datetime import datetime

def estimate_illumination(
    image_paths,
    as_timelapse=True,
    ex_frame=1,
    channel=0,
    baseline_value=500,
    dont_blur=False,
    blur_method="uniform",
    gaussian_sigma=10,
    uniform_size=32,
    edge_mode="reflect"
    ):
    """
    Estimates a 2D illumination image from the mean pixel intensities of
    a list of provided image filepaths.

    Parameters
    ----------
    image_paths : list_like
        a list of image filepaths to use in estimation. These should be either
        single images (with or without multiple imaging channels) or time
        lapse image stacks (with or without multiple imaging channels). Images
        should not have more than 4 dimensions.
    
    as_timelapse : bool, optional
        determines whether input images are treated as timelapse images.
        If True, the 1st dimension in image data is treated as the frame 
        index in a time series. Any fourth dimension will be interpreted as
        different imaging channel indices.
        If False, only 2D images are accepted.
        Default is True.

    ex_frame : int, optional
        the frame from time lapse images to use in illumination estimation.
        Ignored if as_timelapse is False.
        Default is 1, i.e. the first frame in the time lapse image stack.
    
    channel : int, optional
        if input images are multichannel (e.g. contain fluorescence data from
        multiple channels) this is the channel index which will be used for 
        estimating illumination. Ignored if input images are 2D or 3D time
        lapse stacks (i.e. no multichannel information). Default is 0.

    baseline_value : int, optional
        the camera pixel baseline intensity, default is 500.
    
    dont_blur : bool, optional
        determines whether the mean image should be blurred/smoothed.
        Default is False, meaning image will be blurred.

    blur_method : {'uniform', 'gaussian'}, optional
        the function used to smooth the mean pixel intensities, ignored if
        dont_blur is True. By default 'uniform'.

    gaussian_sigma : scalar, optional
        value passed to the skimage.filters.gaussian 'sigma' parameter.
        Determines the width of the gaussian filter, larger values result in
        smoother images. Ignored if dont_blur is True or blur_method is
        'uniform'. Default is 10.
    
    uniform_size : int, optional
        value passed to the scipy.ndimage.uniform_filter 'size' parameter.
        Determines the width of the uniform filter, larger values result in 
        smoother images. Ignored if dont_blur is True or blur_method is 
        'gaussian'. Default is 32.

    edge_mode : {'reflect', 'nearest', 'mirror'}, optional
        mode for handling blurring at edges of images, ignored if dont_blur is
        True. Default is 'reflect'.

        'reflect': The input is extended by reflecting about the edge of the 
        last pixel."
        'nearest': The input is extended by replicating the last pixel."
        'mirror': The input is extended by reflecting about the center of the
         last pixel."
    
    Returns
    -------
    blurred_image : ndarray
        the estimated illumination image as a 2D array.  XY shape is the same 
        as input images.
    """
    # ex_frame = int(ex_frame)
    # channel = int(channel)

    # Load images.
    raw_images = MultiImage(image_paths, conserve_memory=True)
    dims = len(raw_images[0].shape)
    # Check images have valid dimensions given inputs.
    if as_timelapse:
        if dims < 3 or dims > 4:
            err_msg = f"Input images have {dims} dimensions. Time lapse " \
            "images should have a minimum of 3 and a maximum of 4."
            raise ValueError(err_msg)
    else:
        if dims != 2:
            err_msg = f"Input images have {dims} dimensions. Single images" \
            "should have 2."
            raise ValueError(err_msg)

    # Remove the baseline value from each pixel. Default is 500.
    # If these are single images without multiple channels.
    if not as_timelapse:
        baseline_image = np.full(raw_images[0].shape, baseline_value)
        images =[np.subtract(i, baseline_image) for i in raw_images]

    # If these are time lapse images without multiple channels.
    elif as_timelapse and dims == 3:
        baseline_image = np.full(
            raw_images[0][ex_frame,:,:].shape, 
            baseline_value
            )
        images = [
            np.subtract(i[ex_frame,:,:], baseline_image) for i in raw_images
        ]
    
    # Otherwise these are time lapse images with multiple channels.
    else:
        baseline_image = np.full(
            raw_images[0][ex_frame,:,:,channel].shape, 
            baseline_value
        )
        images = [
            np.subtract(
                i[ex_frame,:,:,channel], baseline_image
                ) for i in raw_images
        ]

    # Stack images and find average pixel intensities.
    image_stack = np.dstack(images)
    mean_image = np.mean(image_stack, axis=2)

    # Smooth/blur the estimated background image
    if not dont_blur:
        if edge_mode not in ("reflect", "mirror", "nearest"):
            err_msg = f"Unknown edge_mode: {edge_mode}."
            raise ValueError(err_msg)

        if blur_method == 'gaussian':
            blurred_image = gaussian(
                mean_image, 
                gaussian_sigma, 
                mode=edge_mode
                )
        elif blur_method == 'uniform':
            blurred_image = uniform_filter(
                mean_image, 
                uniform_size, 
                mode=edge_mode
                )
        else:
            err_msg = f"Unknown blur_method value: {blur_method}."
            raise ValueError(err_msg)
    else:
        blurred_image = mean_image
    
    return blurred_image

# Handle command line arguments
# argparser = argparse.ArgumentParser()
# argparser.add_argument("input_folder")
# argparser.add_argument("output_folder")
# argparser.add_argument(
#     "--as_single",
#     help="treat input images as single time point images. Only single channel" \
#         "images are accepted in this mode. If not set (default), the 1st " \
#         "dimension in image data is treated as the frame index in a time " \
#         "series. Any fourth dimension will be interpreted as different " \
#         "imaging channel indices.",
#     action="store_true"
# )
# argparser.add_argument(
#     "--ex_frame",
#     type=int,
#     default=1,
#     help="controls which frame is taken from each time-lapse tiff to use for " \
#         "estimating the flatfield illumination. Default is 0, i.e. the first " \
#         "frame in the stack."
# )
# argparser.add_argument(
#     "--channel",
#     type=int,
#     default=0,
#     help="controls which channel will be used to estimate the illumination" \
#         "image. Default is 0, i.e. the first channel."
# )
# argparser.add_argument(
#     "--baseline_value",
#     type=int,
#     default=500,
#     help="baseline pixel value, default is 500"
#     )
# argparser.add_argument(
#     "--dont_blur",
#     help="controls whether the estimated illumination image is " \
#         "blurred/smoothed. Blurring will be performed unless this argument " \
#         "is set.",
#     action='store_true'
# )
# argparser.add_argument(
#     "--blur_method",
#     default="uniform",
#     choices=["uniform", "gaussian"],
#     help="method used to blur the average flatfield image, ignored if " \
#         "dont_blur is True. Default is 'uniform'."
# )
# argparser.add_argument(
#     "--uniform_size",
#     type=int,
#     default=32,
#     help="controls the width of the uniform filter, default is 32. " \
#         "Larger values will result in a more blurred flatfield image. " \
#         "Ignored if dont_blur is True or --blur_method is 'gaussian'"
# )
# argparser.add_argument(
#     "--gaussian_sigma",
#     type=float,
#     default=10,
#     help="controls the width of the gaussian filter, default is 10. " \
#     "Larger values will result in a more blurred flatfield image. " \
#     "Ignored if dont_blur is True or --blur_method is 'uniform'"
# )
# argparser.add_argument(
#     "--edge_mode",
#     default="reflect",
#     choices=["reflect", "nearest", "mirror"],
#     help="mode for handling blurring at edges of images, default is 'reflect'. " \
#     "'reflect': The input is extended by reflecting about the edge of the last " \
#     "pixel. 'nearest': The input is extended by replicating the last pixel. " \
#     "'mirror': The input is extended by reflecting about the center of the " \
#     "last pixel." \
# )
# argparser.add_argument(
#     "--dont_correct",
#     help="if set, input images are not corrected with the estimated " \
#         "illumination image.",
#     action='store_true'
# )
# argparser.add_argument(
#     "--output_format",
#     default="16bit",
#     choices=["16bit", "32bit"],
#     help="determines the bit depth for output images, default is 16bit."
# )

# args = argparser.parse_args()

# in_dir = args.input_folder
# out_dir = args.output_folder
# ex_frame = args.ex_frame
# channel = args.channel
# baseline_value = args.baseline_value
# dont_blur = args.dont_blur
# blur_method = args.blur_method
# uniform_size = args.uniform_size
# gaussian_sigma = args.gaussian_sigma
# edge_mode = args.edge_mode
# dont_correct = args.dont_correct
# output_format = args.output_format
# as_timelapse = not args.as_single

# Debugging inputs
in_dir = "/home/andre/Heinemann_lab/Code/Flatfield_correction/Test_images/Dorien_images"
out_dir = "Single_channel_test"
as_single = True
ex_frame = 0
channel = 0
baseline_value = 500
dont_blur = False
blur_method = "uniform"
uniform_size = 32
gaussian_sigma = 10
edge_mode = "reflect"
dont_correct = True
output_format = "16bit"
as_timelapse = not as_single

# Run checks
# Check input_folder
if not os.path.exists(in_dir):
    err_msg = f"Provided input folder: {in_dir} does not exist."
    raise FileNotFoundError(err_msg)
if not os.path.isdir(in_dir):
    err_msg = f"Provided input folder: {in_dir} is not a folder."
    raise NotADirectoryError(err_msg)

# Create list of tif filepaths for loading. Ignore anything without .tif or
# .tiff file extension
tif_names = os.listdir(in_dir)
tif_names = [
    i for i in tif_names if os.path.splitext(i)[1] in ('.tif', '.tiff')
    ]
tif_paths = [os.path.join(in_dir, i) for i in tif_names]
# Continue as long as there are images present.
if len(tif_paths) == 0:
    err_msg = f"There are no .tif or .tiff files in {in_dir}."
    raise FileNotFoundError(err_msg)

blurred_image = estimate_illumination(
    tif_paths,
    as_timelapse,
    ex_frame,
    channel,
    baseline_value,
    dont_blur,
    blur_method,
    gaussian_sigma,
    uniform_size,
    edge_mode    
)

if not dont_correct:
    # Calculate a per pixel correction factor
    px_max = np.max(blurred_image)
    corr_factor = np.divide(px_max, blurred_image)

    # Apply correction to input images
    corr_images = [np.multiply(corr_factor, i) for i in images]

# Image files normally store pixel values as integers, not floating point
# numbers. Therefore, convert back to either 16 bit or 32 bit integers.
if output_format == "16bit":
    out_type = np.uint16
elif output_format == "32bit":
    out_type = np.uint32
else:
    err_msg = f"Unknown output_format: {output_format}"
    raise ValueError(err_msg)

blurred_image = np.rint(blurred_image).astype(out_type)
if not dont_correct:
    corr_images = [np.rint(i).astype(out_type) for i in corr_images]

# Save output images
flatfield_dir = os.path.join(out_dir, "Flatfield_image")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(flatfield_dir):
    os.mkdir(flatfield_dir)
if not dont_correct:
    corrected_dir = os.path.join(out_dir, "Corrected_images")
    if not os.path.exists(corrected_dir):
        os.mkdir(corrected_dir)

flatfield_path = os.path.join(flatfield_dir, "Flatfield_estimate.tif")
imsave(flatfield_path, blurred_image)

if not dont_correct:
    out_paths = [os.path.join(corrected_dir, i) for i in tif_names]

    for i in range(len(corr_images)):
        imsave(out_paths[i], corr_images[i])

# Write a log of processing
datetime_str = datetime.now().strftime("%d-%m-%Y_%H:%M")
log_path = os.path.join(out_dir, f"{datetime_str}_log.txt")
with open(log_path, "w+") as f:
    f.write(f"Flatfield correction completed at {datetime_str}.\n\n")
    f.write("Provided arguments:\n")
    f.write(f"input_folder: {in_dir}\n")
    f.write(f"output_folder: {out_dir}\n")
    # f.write(f"as_single: {args.as_single}\n")
    f.write(f"as_single: {as_single}\n")
    f.write(f"ex_frame: {ex_frame}\n")
    f.write(f"channel: {channel}\n")
    f.write(f"baseline_value: {baseline_value}\n")
    f.write(f"dont_blur: {dont_blur}\n")
    f.write(f"blur_method: {blur_method}\n")
    f.write(f"uniform_size: {uniform_size}\n")
    f.write(f"gaussian_sigma: {gaussian_sigma}\n")
    f.write(f"edge_mode: {edge_mode}\n")
    f.write(f"dont_correct: {dont_correct}\n")
    f.write(f"output_format: {output_format}\n\n")

    f.write("Input files processed:\n")
    for i in tif_paths:
        f.write(f"{os.path.abspath(i)}\n")
    f.write("\n")

    f.write("Estimated flatfield image:\n")
    f.write(f"{os.path.abspath(flatfield_path)}\n\n")

    if dont_correct:
        f.write("Image correction not performed.")
    else:
        f.write("Corrected image output files:\n")
        for i in out_paths:
            f.write(f"{os.path.abspath(i)}\n")

def estimate_illumination(
    image_paths,
    as_timelapse=True,
    ex_frame=1,
    channel=0,
    baseline_value=500,
    dont_blur=False,
    blur_method="uniform",
    gaussian_sigma=10,
    uniform_size=32,
    edge_mode="reflect"
    ):
    """
    Estimates a 2D illumination image from the mean pixel intensities of
    a list of provided image filepaths.

    Parameters
    ----------
    image_paths : list_like
        a list of image filepaths to use in estimation. These should be either
        single images (with or without multiple imaging channels) or time
        lapse image stacks (with or without multiple imaging channels). Images
        should not have more than 4 dimensions.
    
    as_timelapse : bool, optional
        determines whether input images are treated as timelapse images.
        If True, the 1st dimension in image data is treated as the frame 
        index in a time series. Any fourth dimension will be interpreted as
        different imaging channel indices.
        If False, only 2D images are accepted.
        Default is True.

    ex_frame : int, optional
        the frame from time lapse images to use in illumination estimation.
        Ignored if as_timelapse is False.
        Default is 1, i.e. the first frame in the time lapse image stack.
    
    channel : int, optional
        if input images are multichannel (e.g. contain fluorescence data from
        multiple channels) this is the channel index which will be used for 
        estimating illumination. Ignored if input images are 2D or 3D time
        lapse stacks (i.e. no multichannel information). Default is 0.

    baseline_value : int, optional
        the camera pixel baseline intensity, default is 500.
    
    dont_blur : bool, optional
        determines whether the mean image should be blurred/smoothed.
        Default is False, meaning image will be blurred.

    blur_method : {'uniform', 'gaussian'}, optional
        the function used to smooth the mean pixel intensities, ignored if
        dont_blur is True. By default 'uniform'.

    gaussian_sigma : scalar, optional
        value passed to the skimage.filters.gaussian 'sigma' parameter.
        Determines the width of the gaussian filter, larger values result in
        smoother images. Ignored if dont_blur is True or blur_method is
        'uniform'. Default is 10.
    
    uniform_size : int, optional
        value passed to the scipy.ndimage.uniform_filter 'size' parameter.
        Determines the width of the uniform filter, larger values result in 
        smoother images. Ignored if dont_blur is True or blur_method is 
        'gaussian'. Default is 32.

    edge_mode : {'reflect', 'nearest', 'mirror'}, optional
        mode for handling blurring at edges of images, ignored if dont_blur is
        True. Default is 'reflect'.

        'reflect': The input is extended by reflecting about the edge of the 
        last pixel."
        'nearest': The input is extended by replicating the last pixel."
        'mirror': The input is extended by reflecting about the center of the
         last pixel."
    
    Returns
    -------
    blurred_image : ndarray
        the estimated illumination image as a 2D array.  XY shape is the same 
        as input images.
    """
    # ex_frame = int(ex_frame)
    # channel = int(channel)

    # Load images.
    raw_images = MultiImage(image_paths, conserve_memory=True)
    dims = len(raw_images[0].shape)
    # Check images have valid dimensions given inputs.
    if as_timelapse:
        if dims < 3 or dims > 4:
            err_msg = f"Input images have {dims} dimensions. Time lapse " \
            "images should have a minimum of 3 and a maximum of 4."
            raise ValueError(err_msg)
    else:
        if dims != 2:
            err_msg = f"Input images have {dims} dimensions. Single images" \
            "should have 2."
            raise ValueError(err_msg)

    # Remove the baseline value from each pixel. Default is 500.
    # If these are single images without multiple channels.
    if not as_timelapse:
        baseline_image = np.full(raw_images[0].shape, baseline_value)
        images =[np.subtract(i, baseline_image) for i in raw_images]

    # If these are time lapse images without multiple channels.
    elif as_timelapse and dims == 3:
        baseline_image = np.full(
            raw_images[0][ex_frame,:,:].shape, 
            baseline_value
            )
        images = [
            np.subtract(i[ex_frame,:,:], baseline_image) for i in raw_images
        ]
    
    # Otherwise these are time lapse images with multiple channels.
    else:
        baseline_image = np.full(
            raw_images[0][ex_frame,:,:,channel].shape, 
            baseline_value
        )
        images = [
            np.subtract(
                i[ex_frame,:,:,channel], baseline_image
                ) for i in raw_images
        ]

    # Stack images and find average pixel intensities.
    image_stack = np.dstack(images)
    mean_image = np.mean(image_stack, axis=2)

    # Smooth/blur the estimated background image
    if not dont_blur:
        if edge_mode not in ("reflect", "mirror", "nearest"):
            err_msg = f"Unknown edge_mode: {edge_mode}."
            raise ValueError(err_msg)

        if blur_method == 'gaussian':
            blurred_image = gaussian(
                mean_image, 
                gaussian_sigma, 
                mode=edge_mode
                )
        elif blur_method == 'uniform':
            blurred_image = uniform_filter(
                mean_image, 
                uniform_size, 
                mode=edge_mode
                )
        else:
            err_msg = f"Unknown blur_method value: {blur_method}."
            raise ValueError(err_msg)
    else:
        blurred_image = mean_image
    
    return blurred_image