from pylab import *
from copy import deepcopy
from scipy.optimize import curve_fit, leastsq
from scipy.signal import fftconvolve
from numpy import fft
from scipy.ndimage import gaussian_filter
   
### library from smith 2019
import skimage.measure as skmeas
from skimage.morphology import watershed, medial_axis, skeletonize
from skimage.filters import sobel
import skimage.filters as skfilt
from skimage.measure import regionprops
import skimage.morphology as skmorph
import skimage.segmentation as skseg
import scipy.ndimage as ndi
from functools import reduce
from skimage.measure import regionprops
import numpy as np

import tempfile
import os
import itertools
from skimage.feature import (
    match_descriptors,
    plot_matches,
    ORB,
    match_template,
)
import skimage.util as skutil
from skimage.measure import ransac
from skimage.transform import AffineTransform
from skimage.measure import regionprops
import matplotlib.pyplot as plt


def remove_background(profiles, radius=20, light_background=True):
    """
    Uses port of ImageJ rolling ball background subtraction
    to estimate background and removes the background from the image

    Parameters
    ------
    profiles : Dictionary
        Key is the well number and the value is a ndarray (2D) of the well
    radius : float, optional
        The radius of the rolling ball (default : 20)
    light_background : Boolean
        Whether the background is light or not (default : True)

    Returns
    ------
    newprofiles : Dictionary
        Key is the well number and the value is a ndarray (2D) of the
        background subtracted well
    """
    # Make "spherical" structuring element
    sz_ = 2 * radius + (radius + 1) % 2
    xco, yco = np.meshgrid(range(sz_), range(sz_))
    ballheight = float(radius**2) - (xco - radius)**2 - (yco - radius)**2
    ballheight[ballheight < 0] = 0
    ballheight = np.ma.masked_where(ballheight < 0, ballheight)
    ballheight = np.sqrt(ballheight)
    newprofiles = {}
    for k, im1 in profiles.items():
        imin= im1.min()
        im2 = im1 - imin
        bg1 = ndi.grey_opening(im2, structure=ballheight, mode="reflect")
        im2 -= bg1
        newprofiles[k] = im2 - im2.min()
    return newprofiles
        
#         im1=-im1;
#         # Run background subtraction
#         if light_background:
#             imax = im1.max()
#             im2 = imax - im1
#             bg1 = ndi.grey_opening(im2, structure=ballheight, mode="reflect")
#             im2 -= bg1
#             newprofiles[k] = im2 - imax
#         else:
#             bg1 = ndi.grey_opening(im1, structure=ballheight, mode="reflect")
#             newprofiles[k] = bg1 - im1
#     return newprofiles


def channel_gaussian_laplace(channel_images):
    '''scale-space filter of the images in the channel.
    using a Laplace of Gaussian convolution at multiple scales, and maximum-projected along the scale axis
    '''
#    sigma_list=(2., 6.),
    sigma_list = np.arange(2.,6.)
    segs = {}  # here images after hessian filter

    for ch_n,im_channel in channel_images.items():
        gl_images = [-(ndi.gaussian_laplace(im_channel, ss, mode="nearest")) * ss ** 2
                     for ss in sigma_list]
        newwell = np.max(gl_images, axis=0)
        segs[ch_n] = newwell
    return segs
