import os
import argparse
import numpy as np
from skimage.io import imread, MultiImage, imsave
from scipy.ndimage import uniform_filter
from skimage.filters import gaussian
from datetime import datetime

# Handle command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("input_folder")
argparser.add_argument("output_folder")
argparser.add_argument(
    "--as_single",
    help="treat input images as single time point images. Only single channel" \
        "images are accepted in this mode. If not set (default), the 1st " \
        "dimension in image data is treated as the frame index in a time " \
        "series. Any fourth dimension will be interpreted as different " \
        "imaging channel indices.",
    action="store_true"
)
argparser.add_argument(
    "--frame",
    type=int,
    default=1,
    help="controls which frame is taken from each time-lapse tiff to use for " \
        "estimating the flatfield illumination. Default is 0, i.e. the first " \
        "frame in the stack."
)
argparser.add_argument(
    "--channel",
    type=int,
    default=0,
    help="controls which channel will be used to estimate the illumination" \
        "image. Default is 0, i.e. the first channel."
)
argparser.add_argument(
    "--baseline_value",
    type=int,
    default=500,
    help="baseline pixel value, default is 500"
    )
argparser.add_argument(
    "--dont_blur",
    help="controls whether the estimated illumination image is " \
        "blurred/smoothed. Blurring will be performed unless this argument " \
        "is set.",
    action='store_true'
)
argparser.add_argument(
    "--blur_method",
    default="uniform",
    choices=["uniform", "gaussian"],
    help="method used to blur the average flatfield image, ignored if " \
        "dont_blur is True. Default is 'uniform'."
)
argparser.add_argument(
    "--uniform_size",
    type=int,
    default=32,
    help="controls the width of the uniform filter, default is 32. " \
        "Larger values will result in a more blurred flatfield image. " \
        "Ignored if dont_blur is True or --blur_method is 'gaussian'"
)
argparser.add_argument(
    "--gaussian_sigma",
    type=float,
    default=10,
    help="controls the width of the gaussian filter, default is 10. " \
    "Larger values will result in a more blurred flatfield image. " \
    "Ignored if dont_blur is True or --blur_method is 'uniform'"
)
argparser.add_argument(
    "--edge_mode",
    default="reflect",
    choices=["reflect", "nearest", "mirror"],
    help="mode for handling blurring at edges of images, default is 'reflect'. " \
    "'reflect': The input is extended by reflecting about the edge of the last " \
    "pixel. 'nearest': The input is extended by replicating the last pixel. " \
    "'mirror': The input is extended by reflecting about the center of the " \
    "last pixel." \
)
argparser.add_argument(
    "--dont_correct",
    help="if set, input images are not corrected with the estimated " \
        "illumination image.",
    action='store_true'
)
argparser.add_argument(
    "--output_format",
    default="16bit",
    choices=["16bit", "32bit"],
    help="determines the bit depth for output images, default is 16bit."
)

args = argparser.parse_args()

in_dir = args.input_folder
out_dir = args.output_folder
as_single = args.as_single
frame = args.frame
ch = int(args.channel)
baseline_value = args.baseline_value
dont_blur = args.dont_blur
blur_method = args.blur_method
uniform_size = args.uniform_size
gaussian_sigma = args.gaussian_sigma
edge_mode = args.edge_mode
dont_correct = args.dont_correct
output_format = args.output_format
as_timelapse = not as_single

# # Debugging inputs
# in_dir = "testing_inputs/Dorien_images"
# out_dir = "Single_channel_test"
# as_single = True
# ex_frame = 0
# channel = 0
# baseline_value = 500
# dont_blur = False
# blur_method = "uniform"
# uniform_size = 32
# gaussian_sigma = 10
# edge_mode = "reflect"
# dont_correct = True
# output_format = "16bit"
# as_timelapse = not as_single

# Validate some inputs.
# Check input_folder.
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
if len(tif_paths) < 1:
    err_msg = f"There are no .tif or .tiff files in {in_dir}."
    raise FileNotFoundError(err_msg)

# Load images.
raw_images = MultiImage(tif_paths, conserve_memory=True)
dims = len(raw_images[0].shape)
# Check images have valid dimensions given inputs.
if as_timelapse:
    if dims < 3 or dims > 4:
        err_msg = f"Input images have {dims} dimensions. Time lapse " \
        "images should have a minimum of 3 and a maximum of 4. " \
        "If your input images are single images use --as_single."
        raise ValueError(err_msg)
else:
    if dims != 2:
        err_msg = f"Input images have {dims} dimensions. Single images " \
        "should have 2."
        raise ValueError(err_msg)

# Get a group of example images to estimate the illumination. First remove the
# baseline value from each pixel. Default is 500.
# If these are single images without multiple channels.
if not as_timelapse:
    baseline_image = np.full(raw_images[0].shape, baseline_value)
    ex_images = [np.subtract(i, baseline_image) for i in raw_images]

# If these are time lapse images without multiple channels.
elif as_timelapse and dims == 3:
    baseline_image = np.full(
        raw_images[0][frame,:,:].shape, 
        baseline_value
        )
    ex_images = [
        np.subtract(i[frame,:,:], baseline_image) for i in raw_images
    ]

# Otherwise these are time lapse images with multiple channels.
else:
    # Verify channel input is within range
    if ch < 0 or ch > raw_images[0].shape[-1]-1:
        err_msg = f"Input channel value {ch} is out of range. Possible " \
            "values for the input images are between 0 and " \
            f"{raw_images[0].shape[-1]-1} inclusive."
        raise ValueError(err_msg)

    baseline_image = np.full(
        raw_images[0][frame,:,:,ch].shape, 
        baseline_value
    )
    ex_images = [
        np.subtract(i[frame,:,:,ch], baseline_image) for i in raw_images
    ]

# Stack images and find average pixel intensities.
image_stack = np.dstack(ex_images)
mean_image = np.mean(image_stack, axis=2)

# Smooth/blur the estimated illumination image
if not dont_blur:
    if edge_mode not in ("reflect", "mirror", "nearest"):
        err_msg = f"Unknown edge_mode: {edge_mode}. Possible values are " \
            "'reflect', 'mirror', and 'nearest'."
        raise ValueError(err_msg)

    if blur_method == 'gaussian':
        est_illum = gaussian(
            mean_image, 
            gaussian_sigma, 
            mode=edge_mode
            )
    elif blur_method == 'uniform':
        est_illum = uniform_filter(
            mean_image, 
            uniform_size, 
            mode=edge_mode
            )
    else:
        err_msg = f"Unknown blur_method value: {blur_method}. Posssible " \
            "values are 'gaussian' and 'uniform'."
        raise ValueError(err_msg)
else:
    est_illum = mean_image

# Preparing for saving images.
# Image files normally store pixel values as integers, not floating point
# numbers. Therefore, convert back to either 16 bit or 32 bit integers.
if output_format == "16bit":
    out_type = np.uint16
elif output_format == "32bit":
    out_type = np.uint32
else:
    err_msg = f"Unknown output_format: {output_format}. Possible values are " \
        "'16bit' and '32bit'."
    raise ValueError(err_msg)

# Save the estimated illumination image
flatfield_dir = os.path.join(out_dir, "Flatfield_image")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(flatfield_dir):
    os.mkdir(flatfield_dir)

est_illum_out = np.rint(est_illum).astype(out_type)
flatfield_path = os.path.join(flatfield_dir, "Flatfield_estimate.tif")
imsave(flatfield_path, est_illum_out, check_contrast=False)

# Now apply the estimated illumination image to correct the input images. For
# memory conservation reasons, use generator expressions to be lazily evaluated
# when saving below.
if not dont_correct:
    # Set up directory for saving
    corrected_dir = os.path.join(out_dir, "Corrected_images")
    if not os.path.exists(corrected_dir):
        os.mkdir(corrected_dir)

    # Calculate a per pixel correction factor
    px_max = np.max(est_illum)
    corr_factor = np.divide(px_max, est_illum)

    # If single images, re-use the ex_images list used when estimating the
    # illumination.
    if not as_timelapse:
        corr_images = (np.multiply(corr_factor, i) for i in ex_images)
        im_out = (np.rint(i).astype(out_type) for i in corr_images)

    # Otherwise need to baseline and flatfield correct every frame per image
    # For timelapses with a single channel
    elif as_timelapse and dims == 3:
        images = (np.subtract(i, baseline_image) for i in raw_images)
        corr_images = (np.multiply(corr_factor, i) for i in images)
        im_out = (np.rint(i).astype(out_type) for i in corr_images)

    else:
        images = (np.subtract(i[:,:,:,ch], baseline_image) for i in raw_images)
        corr_images = (np.multiply(corr_factor, i) for i in images)
        im_out = (np.rint(i).astype(out_type) for i in corr_images)

    # Process and save images one at a time
    out_paths = [os.path.join(corrected_dir, i) for i in tif_names]
    for i, im in enumerate(im_out):
        imsave(out_paths[i], im, check_contrast=False)

# Write a record of processing (just for info, not intended to help debugging)
datetime_str = datetime.now().strftime("%d-%m-%Y_%H:%M")
log_path = os.path.join(out_dir, f"{datetime_str}_log.txt")
with open(log_path, "w+") as f:
    f.write(f"Flatfield correction completed at {datetime_str}.\n\n")
    f.write("Provided arguments:\n")
    f.write(f"input_folder: {in_dir}\n")
    f.write(f"output_folder: {out_dir}\n")
    f.write(f"as_single: {as_single}\n")
    f.write(f"frame: {frame}\n")
    f.write(f"channel: {ch}\n")
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