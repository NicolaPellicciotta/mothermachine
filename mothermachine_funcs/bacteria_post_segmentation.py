from pylab import *
from copy import deepcopy
from scipy.optimize import curve_fit, leastsq
from scipy.signal import fftconvolve
from numpy import fft
from scipy.ndimage import gaussian_filter
   
### library from smith 2019
import skimage.measure as skmeas
from skimage.morphology import medial_axis, skeletonize
from skimage.filters import sobel
import skimage.filters as skfilt
from skimage.measure import regionprops
import skimage.morphology as skmorph
import skimage.segmentation as skseg
import scipy.ndimage as ndi
from functools import reduce
from skimage.measure import regionprops
import numpy as np
from scipy.signal import find_peaks

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
from skimage.segmentation import watershed
import numpy as np
import matplotlib.pyplot as plt



def split_bacteria_in_all_wells(
        bacteria,
        well_images,
        min_skel_length=50,
        strong=False,
    ):
    """
    Takes a dictionary containing a labelled image of detected bacteria
    and attempts to 'split' any labels which may have been detected as
    multiple bacteria instead of one

    Parameters
    ------
    bacteria : Dictionary
        The key is the well coordinates and the value is a labelled image of detected bacteria
    well_images : Dictionary
        Key is the well number and the value is a ndarray (2D) of the well

    returns
    ------
    split_bac : Dictionary
        The key is the well coordinates and the value is a labelled image of detected bacteria
    """
    dists = []
    intensities = []
    for well_label, bacteria_label_image in bacteria.items():
        # this function return the intensity and distances(form background) of pixels of the skeleton of long bacteria
        distnow, intnow = get_bacteria_stats_in_well(
            bacteria_label_image,
            well_images[well_label],
            min_skel_length=min_skel_length
        )
        dists.extend(distnow) # append all from all the wells
        intensities.extend(intnow)
    if dists:
        dists = np.array(dists)
        intensities = np.array(intensities)
        import matplotlib.pyplot as plt

        iqr25, iqr75 = np.percentile(dists, [25, 75])
        dists2 = dists[(dists>=iqr25) & (dists<=iqr75)]  # remove skel distances outside the percentile
        iqr25, iqr75 = np.percentile(intensities, [25, 75])
        intensities2 = intensities[(intensities>=iqr25) & (intensities<=iqr75)] # remove skel intesity outside the percentile

        medd, madd = get_med_and_mad(dists)
        medd2, madd2 = get_med_and_mad(dists2)
        medi, madi = get_med_and_mad(intensities)
        medi2, madi2 = get_med_and_mad(intensities2)

        plt.show()

        for well_label, bacteria_label_image in bacteria.items():
            if strong == False:
                newbacteria = split_bacteria_in_well(
                    bacteria_label_image,
                    well_images[well_label],
                    min_skel_length=min_skel_length,
                    threshold_stat=medd2-madd2,     # this is the threshold distance obtained from all the wells
                    debug=dict(
                        med_dist=medd, mad_dist=madd,
                        med_dist2=medd2, mad_dist2=madd2,
                        med_int=medi, mad_int=madi,
                        med_int2=medi2, mad_int2=madi2,
                    )
                )
            else:
                newbacteria = split_bacteria_in_well_strong(
                    bacteria_label_image,
                    well_images[well_label],
                    min_skel_length=min_skel_length,
                )
            bacteria[well_label] = newbacteria
    return bacteria


def get_med_and_mad(data):
    """
    Return median and MAD of data
    """
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    return med, mad


def threshold_mad(data, factor):
    """
    Determine Median Absolute Deviation based threshold
    """
    med, mad = get_med_and_mad(data)
    threshold = med - factor * mad
    return threshold


def get_bacteria_stats_in_well(
        bacteria_label_image,
        intensity_image,
        min_skel_length=50,
    ):
    """
    Get bacteria width stats in a single well image using
    distance transform and skeletonization

    Parameters
    ------
    bacteria_label_image : ndarray (2D)
        Labelled image of detected bacteria
    intensity_image : ndarray (2D)
        Intensity image for the current well
    min_skel_length : float (optional)
        Minimum skeleton length for a region to be considered for splitting
    returns
    ------
    split_bac : Dictionary
        The key is the well coordinates and the value is a labelled image of detected bacteria
    """
    mask = bacteria_label_image > 0
    dist = ndi.distance_transform_edt(mask)
    skel = skmorph.skeletonize(mask)
    skellabel = ndi.label(skel)[0]
    for prop in regionprops(skellabel):
        if prop.area < min_skel_length:
            skel[skellabel == prop.label] = False
        #skelmask = skellabel == prop.label

    return dist[skel], intensity_image[skel]


def split_bacteria_in_well_strong(
        bacteria_label_image,
        intensity_image,
        min_skel_length=50,
    ):
    """
    Split bacteria in a single well image based on
    peaks intensity along the skeleton

    Parameters
    ------
    bacteria_label_image : ndarray (2D)
        Labelled image of detected bacteria
    intensity_image : ndarray (2d)
        Intensity image of current well
    min_skel_length : float (optional)
        Minimum skeleton length for a region to be considered for splitting
    returns
    ------
    split_bac : Dictionary
        The key is the well coordinates and the value is a labelled image of detected bacteria
    """

    max_label = 0
    stats = []
    dist = ndi.distance_transform_edt(bacteria_label_image > 0)
    output_labels = bacteria_label_image.copy()
    for prop in regionprops(bacteria_label_image):
        region_mask = bacteria_label_image == prop.label
        #skel, dist = skmorph.medial_axis(region_mask, return_distance=True)
        skel = skmorph.skeletonize(region_mask)
        if skel.sum() < min_skel_length:
            max_label += 1
            output_labels[region_mask] = max_label
            continue

        skelprop = regionprops(skel.astype(int))[0]
        posy, posx = skelprop.coords.T
        
        skel_x = uint8(posx.mean())
        # find intensity profile along bacteria axis
        image_skel= intensity_image[posy[0]:posy[-1]+1,skel_x-2:skel_x+2]
        #temp_image[~thinned]= nan
        skel_intensity = nanmean(image_skel,axis=1)

        # find distance profile along bacteria axis
        dist = ndi.distance_transform_edt(bacteria_label_image > 0)
        image_skel_dist= dist[posy[0]:posy[-1],skel_x-2:skel_x+2]
        skel_dist = nanmean(image_skel_dist,axis=1)

        skel_intensity_gf=gaussian_filter(skel_intensity, 2)
        skel_dist_gf=gaussian_filter(skel_dist, 2)
                
        xs_int,ph=find_peaks(-skel_intensity_gf, prominence=0.10, distance=min_skel_length)
        xs_dist,ph=find_peaks(-skel_dist_gf, prominence=0.10, distance=min_skel_length)

        joined_conditions = abs(xs_dist- xs_int[:,newaxis])<3 ## 
        breaks_point= repeat(xs_dist[ np.newaxis,:], len(xs_int), axis=0)[joined_conditions]
        breaks= zeros_like(skel_intensity)
        breaks[breaks_point]= 1 

        skel_breaks = skel.copy()
        #remove the point in the skel 
        skel_breaks[skel] = breaks  # this is the sure background

        break_labels, n_breaks = ndi.label(skel_breaks)

        sizes = ndi.sum(
            skel_breaks, break_labels,
            index=np.arange(1, n_breaks + 1)
        )
        for label, break_size in enumerate(sizes, start=1):
            # if break_size < min_break_size:
            if break_size < 0:
                skel_breaks[break_labels == label] = False

        bacteria_markers, n_bacteria_markers = ndi.label(
            skel ^ skel_breaks,
            structure=np.ones((3, 3), dtype=bool)
            )

        num_removed = 0
        for label in range(1, n_bacteria_markers+1):
            bwnow = bacteria_markers == label
            if bwnow.sum() >= min_skel_length:
                continue
            bacteria_markers[bwnow] = 0
            num_removed += 1

        if num_removed == n_bacteria_markers:
            max_label += 1
            output_labels[region_mask] = max_label
            continue


        bacteria_markers, num_final = ndi.label(
            bacteria_markers>0,
            structure=np.ones((3,3), dtype=bool)
        )

        bacterianow = watershed(-dist, bacteria_markers, mask=region_mask)
        bacterianow[bacterianow > 0] += max_label
        output_labels[region_mask] = bacterianow[region_mask]
        max_label += num_final                                                
                        
    return output_labels





def split_bacteria_in_well(
        bacteria_label_image,
        intensity_image,
        min_skel_length=50,
        threshold_stat=None,
        debug=None,
    ):
    """
    Split bacteria in a single well image based on
    statistical thresholding of width and intensity along
    the skeleton

    Parameters
    ------
    bacteria_label_image : ndarray (2D)
        Labelled image of detected bacteria
    intensity_image : ndarray (2d)
        Intensity image of current well
    min_skel_length : float (optional)
        Minimum skeleton length for a region to be considered for splitting
    returns
    ------
    split_bac : Dictionary
        The key is the well coordinates and the value is a labelled image of detected bacteria
    """

    max_label = 0
    stats = []
    dist = ndi.distance_transform_edt(bacteria_label_image > 0)
    output_labels = bacteria_label_image.copy()
    for prop in regionprops(bacteria_label_image):
        region_mask = bacteria_label_image == prop.label
        #skel, dist = skmorph.medial_axis(region_mask, return_distance=True)
        skel = skmorph.skeletonize(region_mask)
        if skel.sum() < min_skel_length:
            max_label += 1
            output_labels[region_mask] = max_label
            continue

        skelprop = regionprops(skel.astype(int))[0]
        posy, posx = skelprop.coords.T
        
        skel_intensity = intensity_image[skel]
        skel_dist = dist[skel]

        threshold_int = threshold_mad(skel_intensity, 1)
        threshold_dist = threshold_mad(skel_dist, 1)


        breaks = skel_intensity < threshold_int #Nicola: 9.5.20 (debug["med_int"] - debug["mad_int"])

        breaks_dist = skel_dist < threshold_dist# Nicola 9.5.20 debug["med_dist"]

        breaks[~breaks_dist] = False


        skel_breaks = skel.copy()
        skel_breaks[skel] = breaks

        break_labels, n_breaks = ndi.label(skel_breaks)

        sizes = ndi.sum(
            skel_breaks, break_labels,
            index=np.arange(1, n_breaks + 1)
        )
        for label, break_size in enumerate(sizes, start=1):
            # if break_size < min_break_size:
            if break_size < 0:
                skel_breaks[break_labels == label] = False
        bacteria_markers, n_bacteria_markers = ndi.label(
            skel ^ skel_breaks,
            structure=np.ones((3, 3), dtype=bool)
            )

        num_removed = 0
        for label in range(1, n_bacteria_markers+1):
            bwnow = bacteria_markers == label
            if bwnow.sum() >= min_skel_length:
                continue
            bacteria_markers[bwnow] = 0
            num_removed += 1

        if num_removed == n_bacteria_markers:
            max_label += 1
            output_labels[region_mask] = max_label
            continue


        bacteria_markers, num_final = ndi.label(
            bacteria_markers>0,
            structure=np.ones((3,3), dtype=bool)
        )

        bacterianow = watershed(-dist, bacteria_markers, mask=region_mask)
        bacterianow[bacterianow > 0] += max_label
        output_labels[region_mask] = bacterianow[region_mask]
        max_label += num_final

                                             
                        
    return output_labels







# def split_bacteria_in_well(
#         bacteria_label_image,
#         intensity_image,
#         min_skel_length=50,
#         threshold_stat=None,
#         debug=None,
#     ):
#     """
#     Split bacteria in a single well image based on
#     statistical thresholding of width and intensity along
#     the skeleton

#     Parameters
#     ------
#     bacteria_label_image : ndarray (2D)
#         Labelled image of detected bacteria
#     intensity_image : ndarray (2d)
#         Intensity image of current well
#     min_skel_length : float (optional)
#         Minimum skeleton length for a region to be considered for splitting
#     returns
#     ------
#     split_bac : Dictionary
#         The key is the well coordinates and the value is a labelled image of detected bacteria
#     """

#     max_label = 0
#     stats = []
#     dist = ndi.distance_transform_edt(bacteria_label_image > 0)
#     output_labels = bacteria_label_image.copy()
#     for prop in regionprops(bacteria_label_image):
#         region_mask = bacteria_label_image == prop.label
#         #skel, dist = skmorph.medial_axis(region_mask, return_distance=True)
#         skel = skmorph.skeletonize(region_mask)
#         if skel.sum() < min_skel_length:
#             max_label += 1
#             output_labels[region_mask] = max_label
#             continue

#         skelprop = regionprops(skel.astype(int))[0]
#         posy, posx = skelprop.coords.T

#         skel_intensity = intensity_image[skel]
#         skel_dist = dist[skel]

#         threshold_int = threshold_mad(skel_intensity, 1)
#         threshold_dist = threshold_mad(skel_dist, 1)

# #         inds = np.argsort(posy)
# #         for i, (name, vals, name2) in enumerate((
# #                 ("Distance", skel_dist, "dist"),
# #                 ("Intensity", skel_intensity, "int"),
# #                 ("Combined", skel_dist*(-1*skel_intensity), None),
# #                 )):
# #             med, mad = get_med_and_mad(vals)
# #         med_int,mad_int = get_med_and_mad(skel_intensity)
#         breaks = skel_intensity < threshold_int #Nicola: 9.5.20 (debug["med_int"] - debug["mad_int"])

#         breaks_dist = skel_dist < threshold_dist# Nicola 9.5.20 debug["med_dist"]
#         #breaks_dist = skel_dist < np.median(skel_dist) #debug["med_dist"] #threshold_dist

# #        skel_dist_breaks = skel.copy()
# #        skel_dist_breaks[skel] = breaks_dist

#         breaks[~breaks_dist] = False

#         ### TODO: added this in as potential width ratio method
#         ### neeed to rethink with Jeremy
#         #width_ratio = 0.5
#         #threshold_dist_ratio = width_ratio * np.max(skel_dist)
#         #threshold_dist_ratio = np.median(skel_dist) - 1*threshold_stat
# #        threshold_dist_ratio = threshold_stat


# #        breaks_dist_hard = skel_dist < threshold_dist_ratio # it seems useless
#         #breaks[breaks_dist_hard] = True

#         skel_breaks = skel.copy()
#         skel_breaks[skel] = breaks

#         break_labels, n_breaks = ndi.label(skel_breaks)

#         sizes = ndi.sum(
#             skel_breaks, break_labels,
#             index=np.arange(1, n_breaks + 1)
#         )
#         for label, break_size in enumerate(sizes, start=1):
#             # if break_size < min_break_size:
#             if break_size < 0:
#                 skel_breaks[break_labels == label] = False
#         bacteria_markers, n_bacteria_markers = ndi.label(
#             skel ^ skel_breaks,
#             structure=np.ones((3, 3), dtype=bool)
#             )

#         num_removed = 0
#         for label in range(1, n_bacteria_markers+1):
#             bwnow = bacteria_markers == label
#             if bwnow.sum() >= min_skel_length:
#                 continue
#             bacteria_markers[bwnow] = 0
#             num_removed += 1

#         if num_removed == n_bacteria_markers:
#             max_label += 1
#             output_labels[region_mask] = max_label
#             continue


#         bacteria_markers, num_final = ndi.label(
#             bacteria_markers>0,
#             structure=np.ones((3,3), dtype=bool)
#         )

#         bacterianow = watershed(-dist, bacteria_markers, mask=region_mask)
#         bacterianow[bacterianow > 0] += max_label
#         output_labels[region_mask] = bacterianow[region_mask]
#         max_label += num_final

#     return output_labels


def relabel_bacteria(bacteria):
    """
    Relabel bacteria in dictionary to be
    sequential
    """
    relabelled = {}
    maxlabel = 0
    for well_label, bacnow in bacteria.items():
        bacnow[bacnow > 0] += maxlabel
        relabelled[well_label] = bacnow.copy()
        maxlabel = max(maxlabel, bacnow.max())
    return relabelled
