from pylab import *
from copy import deepcopy
from scipy.optimize import curve_fit, leastsq
from scipy.signal import fftconvolve
from numpy import fft
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk
import cv2 as cv
   
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



def bacteria_watershed(
                        segs,
                        maxsize=1000, #uint16((5*2)/(pix2mic**2))  ### max area
                        minsize=200, #uint16((0.5*0.1)/(pix2mic**2))    ### min area
                        absolwidth=1,
                        thr_strength=2.
                        ):
    min_av_width=1
    segs2 = {} # here images after watershed
    thresh_array=[];
#    for ch_n, img0 in segs.items():
#        thresh_array.append(skfilt.threshold_minimum(img0.flatten()))
    med, mad = get_med_and_mad(thresh_array)
    threshold_int = med+3*mad 
    
    for ch_n, img0 in segs.items():
        factor = 1.
        img=-image2gray(segs[0])
        med,mad=get_med_and_mad(img)
        th3 =cv.adaptiveThreshold(img,255,
                                  cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY,
                                  101,  # block size of the adaptive thr
                                  thr_strength*mad) # constant value subtracted 
        #threshold_int = skfilt.threshold_minimum(img0.flatten())
        #if skfilt.threshold_minimum(img0.flatten())> threshold_int:
        #bw0 = img0 > threshold_int+10*factor
        bw0 = (th3==0)
        #else:
        #    bw0 = img0 > threshold_int;

        bw1 = ndi.morphology.binary_erosion(bw0, structure=ones((5,5)), iterations=1)
        bw2 = ndi.morphology.binary_dilation(bw0)
        bw3 = bw2 ^ bw1; ### only the borders
        # perform distance transform on filtered image (distance from the nearest 0)
        dist = ndi.distance_transform_edt(bw1)
        markers = np.zeros(img0.shape, dtype=bool)
        markers[dist >= absolwidth] = True
        markers = ndi.label(markers)[0]
        markers = markers + 1
#    bw3 = bw2 ^ (dist >= absolwidth); ### only the borders
        markers[bw3] = 0

        # Perform watershed
        # edit by Nicola
        segmentation = watershed(-img0, markers)
    #    segmentation = watershed(img0, markers)
        segmentation = ndi.binary_fill_holes(segmentation != 1)

        # label image
        labeled_bacteria, nbac = ndi.label(segmentation)

        ### now filter bacteria for size and width
        dist = ndi.distance_transform_edt(labeled_bacteria > 0)

        newbac = np.zeros(labeled_bacteria.shape, dtype=labeled_bacteria.dtype)

        stats = dict(
            num_rejected_av_width=0,
            num_rejected_area=0,
        )

        label = 0
        for region in regionprops(labeled_bacteria):
            masked_bac = labeled_bacteria == region.label
            skel = skeletonize(masked_bac)
            av_width_skel = np.mean(dist[skel])
            if av_width_skel < min_av_width:
                stats["num_rejected_av_width"] += 1
                continue
            if maxsize > region.area > minsize:
                label += 1
                newbac[labeled_bacteria == region.label] = label
            if  region.area > maxsize:
                stats["num_rejected_area"] += 1
#        if stats["num_rejected_area"]>0:            
        segs2[ch_n] = newbac
    return segs2,thresh_array

def threshold_mad(data, factor):
    """
    Determine Median Absolute Deviation based threshold
    """
    med, mad = get_med_and_mad(data)
    threshold = med - factor * mad
    return threshold

def get_med_and_mad(data):
    """
    Return median and MAD of data
    """
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    return med, mad

def image2gray(img):
    img=uint8((img-img.min())/(img.max()-img.min())*255)
    return img