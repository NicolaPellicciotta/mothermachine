"""
A custom class to contain the bacteria measurements, we just use a little part of it.... 
"""

from pylab import *
from copy import deepcopy
from scipy.optimize import curve_fit, leastsq
from scipy.signal import fftconvolve
from numpy import fft
from scipy.ndimage import gaussian_filter
   
### library from smith 2019
import skimage.measure as skmeas
from skimage.morphology import medial_axis, skeletonize
from skimage.segmentation import watershed
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



class BacteriaData():
    """
    A class that containts all the data for all the bacteria

    Attributes
    ------
    bacteria : dictionary
        A dictionary where the key is the bacteria number and the
        value is an instance of an `IndividualBacteria` containing
        its respective information
    """
    def __init__(self):
        self.bacteria = {}

    def add_bac_data(
            self, bac_num, bacteria_lineage, region, tpoint, well_label=None,fluo=None):
        """
        Creates a new instance of the IndividualBacteria class for a new
        bacteria, or updates it if it already exists

        Parameters
        ------
        bac_num : int
            The bacteria number that links to the unique label of the bacteria
        bacteria_lineage : dictionary
            A dictionary that links the physical unique label of a bacteria
            to one which shows information on its lineage
        region : list of RegionProperties
            Each item describes one labeled region, and can be accessed
            using the attributes listed below
        tpoint : int
            The timepoint for the measurement
        """
        # checks to see if the bacteria exists, adds data if it does and
        # records the timepoint
        if bac_num not in self.bacteria.keys():
            self.bacteria[bac_num] = IndividualBacteria(bac_num)
            self.bacteria[bac_num].add_string(bacteria_lineage[bac_num])
        self.bacteria[bac_num].well_label = well_label
        self.bacteria[region.label].add_bf_values(region, tpoint)
        if fluo is not None:
            self.bacteria[region.label].fluorescence["Fluo"].append(fluo)

    def measure_fluo(self, region, fluorescence_data, bkg_values, timepoint):
        """
        Adds fluorescent data for a bacteria

        Parameters
        ------
        region : list of RegionProperties
            Each item describes one labeled region, and can be accessed
            using the attributes listed below
        fluorescence_data : ndarray
            Array containing all the original fluorescent data (2D
            if just one frame)
        bkg_values : tuple
            tuple in the format (background fluorescence, background
            SEM) for the
            respective image
        timepoint : int
            The timepoint for the measurement
        """
        # adds fluorescence measurements
        self.bacteria[region.label].add_fluo_values(
            region, fluorescence_data, bkg_values, timepoint)

    def compile_results(self, max_tpoint=0):
        """
        Compiles all of the measurements

        Parameters
        ------
        max_tpoint : int
            The last timepoint for the measurements
        """
        # compiles the results for each bacteria into a simple list
        # which can easily be written to a CSV
        for bac in self.bacteria.values():
            bac.compile_data(max_tpoint)


class IndividualBacteria():
    """
    A custom class which contains all the required information for
    an individual bacterium

    Attributes
    ------
    bacteria_number : int
        The unique number of the bactera
    bacteria_label : str
        The unique label with the bacterias lineage information
    well_label : int
        Integer identifying the well the bacteria belongs to
    bf_measurements : dictionary
        The keys ("Area","Width","Length") can be used to access chronological
        lists of these measurements
    raw_fluorescence : dictionary
        Keys are a tuple (timepoint, fluorescence channel) which can be used
        to access the raw fluorescence values
    actual_fluorescence : dictionary
        Keys are a tuple (timepoint, fluorescence channel) which can be used
        to access the background subtracted fluorescence values
    integrated_fluorescence : dictionary
        Keys are a tuple (timepoint, fluorescence channel) which can be used
        to access the integrated (actual x area) fluorescence values
    headings_line : list
        A list of the types of measurments, repeated for each timepoint,
        that can easily be written to a csv
    measurements_output : list
        A list of the measurments, repeated for each timepoint,
        that can easily be written to a csv
    timepoints : list
        A list of the timepoints that information is held for
    num_fluo : int
        The number of fluorescent channels information is held for
    """

    def __init__(self, bac_num):
        self.bacteria_number = bac_num
        self.bacteria_label = None
        self.well_label = None
        self.bf_measurements = {
            "Area": [],
            "Width": [],
            "Length": [],
        }
        self.fluorescence = {'Fluo':[]}
        self.raw_fluorescence = {}
        self.actual_fluorescence = {}
        self.integrated_fluorescence = {}
        self.headings_line = []
        self.measurements_output = []
        self.timepoints = []
        self.num_fluo = 0

    def add_string(self, label):
        """
        Adds a readable label

        Parameters
        ------
        Label : str
            The label to be added
        """
        self.bacteria_label = label

    def add_bf_values(self, region, tpoint):
        """
        Updates the brightfield information

        Parameters
        ------
        region : list of RegionProperties
            Each item describes one labeled region, and can be accessed
            using the attributes listed below
        tpoint : int
            The timepoint the data corresponds to
        """
        self.bf_measurements["Area"].append(region.area)
        self.bf_measurements["Width"].append(region.minor_axis_length)
        self.bf_measurements["Length"].append(region.major_axis_length)
        self.timepoints.append(tpoint)


    
    '''this is the original one that i don't understand
    def add_fluo_values(self, region, fluorescence_data,
                        bkg_values, timepoint):
        """
        Updates the fluorescence information

        Parameters
        ------
        region : list of RegionProperties
            Each item describes one labeled region, and can be accessed
            using the attributes listed below
        fluorescence_data : ndarray
            Array containing all the original fluorescent data (2D if
            just one frame)
        bkg_values : tuple
            tuple in the format (background fluorescence, background
            SEM) for the respective image
        timepoint : int
            The timepoint the data corresponds to
        """
        self.num_fluo = len(fluorescence_data)
        import mmhelper.measurements as mmeas
        for num, (fluo_im, bkg) in enumerate(
                zip(fluorescence_data, bkg_values)):
            fluo, fluo_bg, int_fluo = mmeas.fluorescence_measurements(
                region, fluo_im, bkg)
            self.raw_fluorescence[(timepoint, num)] = fluo
            self.actual_fluorescence[(timepoint, num)] = fluo_bg
            self.integrated_fluorescence[(timepoint, num)] = int_fluo
    '''
    def compile_data(self, max_tpoint):
        """
        Compiles all of the data into a readable output

        Parameters
        ------
        max_tpoint : int
            The final timepoint for the analysis
        """
        if not self.raw_fluorescence:
            fluo_values = 0
        else:
            fluo_values = 3
        missed_detection = self.set_missed_detection_line(
            fluo_values=fluo_values)
        data_line = []
        for tindex in range(0, max_tpoint):
            data_line.append([self.well_label])
            data_line.append([self.bacteria_label])
            if tindex not in self.timepoints:
                data_line.append(missed_detection)
                continue
            bf_data_index = self.timepoints.index(tindex)
            data_line.append([self.bf_measurements[key][bf_data_index]
                              for key in sorted(self.bf_measurements.keys())])
            for num in range(0, self.num_fluo):
                data_line.append([self.raw_fluorescence[(tindex, num)],
                                  self.actual_fluorescence[(tindex, num)],
                                  self.integrated_fluorescence[(tindex, num)]])
        self.set_heading_line(max_tpoint)
        self.measurements_output = [
            item for sublist in data_line for item in sublist]

    # Fluo_values is the number of measurements from each fluo image
    def set_missed_detection_line(self, fluo_values=3):
        """
        Creates a line of hyphens that will be used if the detection is misssed

        Parameters
        ------
        fluo_values : int
            The number of fluorescent measurements that are included
            in the results
        """
        return (["-"] * (len(self.bf_measurements) +
                         (self.num_fluo * fluo_values)))

    def set_lysed_bacteria_line(self, fluo_values=3):
        """
        Creates a line of 0's that will be used if the bacteria has lysed

        Parameters
        ------
        fluo_values : int
            The number of fluorescent measurements that are included
            in the results
        """
        return ([0] * (len(self.bf_measurements) +
                       (self.num_fluo * fluo_values)))

    def set_heading_line(self, max_tpoint):
        """
        Sets the heading line that will be used at the top of the
        measurements CSV

        Parameters
        ------
        max_tpoint : int
            The final timepoint for the analysis
        """
        headings = [["well label",
                     "lineage",
                     "area",
                     "length",
                     "width"],
                    (["raw_fluorescence",
                      "fluorescence",
                      "integrated_fluorescence"] * self.num_fluo)]
        headings = [
            [item for sublist in headings for item in sublist] * max_tpoint]
        self.headings_line = [item for sublist in headings for item in sublist]
