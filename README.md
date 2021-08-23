# Flatfield_correction

## Introduction

A Python script for post hoc estimation and correction of uneven illumination in microscopy images.

## Description

The script first estimates the distribution of illumination by taking the mean pixel intensity across a sample set of images. It then optionally blurs this estimated image and uses it to calculate a per-pixel correction factor. This matrix of correction factors is (optionally) subsequently multiplied by every frame (in the relevant channel) across input images.

If input images are single images, all frames will be used to estimate the illumination. If input images are time-lapse stacks, a single frame is taken from each instead. All frames are corrected depending on options chosen.

## Caveats

This method will likely never be as good as testing and correcting for for uneven illumination before performing the experiment (e.g. by capturing images of a fluorescent dye). However, it may be a viable option under some circumstances, particularly where there is no inherent structure in the input images e.g. microfluidic channels or cell clustering locations across input files. Sparser images, ones where the majority of each image is 'background', will also likely be handled better.

In images where some pixels are close to the maximum value which can be encoded by a 16 bit unsigned integer (65,535), corrected pixel values can overflow to very low values. This is usually very obvious from dark pixels/patches appearing in regions which are bright in the source images. Unfortunately I don't know of a good way to handle this, although images can be output as 32bit instead which avoids the overflow issue.

## Usage

`flatfield_correction.py` is a command line program. It takes two required arguments and several optional ones which will otherwise use default values. Input images are assumed to be 16bit and must be uncorrected! 

Required arguments:
- `input_folder` : absolute or relative path to a folder with input images in .tif or .tiff format. These can be single frames, time-lapse stacks of frames, or multichannel time-lapse stacks. All image frames must have the same dimensions, however time-lapses of different durations are supported. The caveat is that the `--frame` argument must not exceed the number of frames in the shortest time-lapse. Any files without the .tif or .tiff extension will be ignored.

- `output_folder` : absolute or relative path to a folder where output should be saved. If the folder does not exist it will be created. It's recommended to use a new or empty folder, existing output will be overwritten in the event of name clashes.

Optional arguments:
- `--as_single` : treat input images as single time points. If not specified, the default behaviour is to treat input images as time-lapse stacks. In that case the first dimension index is treated as the frame index in a time-lapse series. Any fourth dimension will be treated as an imaging channel index.

- `--frame` : the frame index in time-lapse images which will be used to estimate the illumination distribution. Ignored if `--as_single` is used. Defaults to 0, i.e. the first frame in the stack.

- `--channel` : the channel index which will be used for estimating the illumination distribution and potentially correcting images. Ignored if `--as_single` is used or the first input image only has 3 dimensions. Default is 0, i.e. the first imaging channel.

- `--baseline_value` : the camera baseline intensity (i.e. the value which pixels would have if an image were taken with the shutter closed). Default is 500.

- `--dont_blur` : do not blur the estimated illumination distribution image. If not specified, the image will be blurred before using it for flatfield correction.

- `--blur_method` : the algorithm used to blur the estimated illumination distribution image. Ignored if `--dont_blur` is specified. Possible values are 'gaussian' and 'uniform'. Default is 'uniform'.

- `--uniform_size`: controls the width of the uniform filter. Larger values will result in a blurrier estimated illumination image. Ignored if `--dont_blur` is specified or `--blur_method` is 'gaussian'. Default is 32 pixels.

- `--gaussian_sigma` : controls the width of the gaussian blur filter. Larger values will result in a blurrier estimated illumination image. Ignored if `--dont_blur` is specified or `--blur_method` is 'uniform'. Default is 10.

- `--edge_mode` : determines the mode for handling blurring at the edges of images. Ignored if `--dont_blur` is specified. Default is 'reflect'.
    - 'reflect' : the input is extended by reflecting about the edge of the last pixel.
    - 'nearest' : the input is extended by replicating the last pixel.
    - 'mirror' : the input is extended by reflecting about the center of the last pixel.

- `--dont_correct` : if specified, flatfield correction will not be performed on the input images after generating the estimated illumination image. By default, correction will be performed.

- `--output_format` : determines the bit depth of output estimated illumination and corrected images. Possible values are '16bit' and '32bit'. Default is '16bit'.