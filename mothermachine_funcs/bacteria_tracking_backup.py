from __future__ import print_function

### functions from Smith2019Scientific_Reports
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

def bacteria_tracking(last_wells, current_wells, bacteria_lineage,
                    prob_div=0.3,
                    prob_death=0.01,
                    prob_no_change=0.7,
                    av_bac_length=40,#18,
                    important_bac=None
                    ):
    """
    Takes a dictionary of wells from the previous frame and a second
    dictionary with the corresponding wells for the new frame and
    determines the most likely way the bacteria within them have
    moved, died, divided then relabels them accordingly

    Parameters
    ------
    last_wells : Dictionary
        The previous timepoint. The key is the well coordinates and the value
        is a labelled image of detected bacteria
    current_wells : Dictionary
        The current timepoint. The key is the well coordinates and the value
        is a labelled image of detected bacteria
    bacteria_lineage : dictionary
        A dictionary that links the physical unique label of a bacteria
        to one which shows information on its lineage

    Returns
    ------
    out_wells : Dictionary
        The current timepoint. The key is the well coordinates and the value
        is a labelled image of tracked bacteria
    bacteria_lineage : dictionary
        Updated dictionary that links the physical unique label of a bacteria
        to one which shows information on its lineage
    check_prob : need to check what is it
    """
    out_wells = {}
    check_probs ={}
    best_options={}
    if important_bac==None:
        important_bac= 8
        
    # create an in_list of the bacteria in the last well (ordered by position along the channel)
    for num, well in last_wells.items():
        if num not in current_wells.keys():
            continue
        new_well = (current_wells[num]).astype(uint16)
        end_channel= shape(new_well)[0]
        in_list = []
        centroid_pos =[]
        #option_list = []
        for region in regionprops(well):
            # list the bacteria labels from the current frame
            in_list.append(region.label)
            centroid_pos.append(region.centroid[0])
        ind_sort_y = argsort(centroid_pos)  # nicola: important to sort the list
#        print unique(well)
        in_list = array(in_list)
        in_list= in_list[ind_sort_y]
        in_list = ndarray.tolist(in_list)
        

        if not in_list: # in the last frame there were no bacteria
            check_prob=[]
            if len(regionprops(new_well)) > 0:
                # if there is now a new bacteria we don't want to ignore as it may have just
                # been missed in previous frame
                if bool(bacteria_lineage):
                    smax = max(bacteria_lineage, key=int)
                else: 
                    smax=0
                newwell = np.zeros(new_well.shape, dtype=uint16)
                for new_bac in regionprops(new_well):
                    # so lets give each "new" bacteria a new label
                    smax += 1
                    newwell[new_well == new_bac.label] = smax
                    bacteria_lineage[smax] = str(smax)
            elif not next(options, None):
                # if the out well is also empty then there is nothing to track
                # and simply return an empty well
                newwell = np.zeros(new_well.shape, dtype=new_well.dtype)
            bacteria_lineage_new=deepcopy(bacteria_lineage)

        # if in_list is not empty    
        else:  # determine probabilities and label matching/new bacteria

            # create a list of all the possible combinations of the in_list with the current number of bacteria
            options = itertools.combinations_with_replacement(
                in_list, len(regionprops(new_well)))
            
            # for each option it gives the change in centroid position, area for each bacteria 
            
            
            change_options = find_changes(
                in_list, options, well, new_well)
            
            # from this calculate the most likely option (calculated only on important bacteria)
            best_option = None
            best_prob = 0
            check_prob=[]
            for option, probs in change_options:
                probs_,probs = find_probs(probs,
                                    prob_div=prob_div,
                                    prob_death=prob_death,
                                    prob_no_change=prob_no_change,
                                    av_bac_length=av_bac_length,#18,
                                    end_channel=end_channel,
                                    important_bac=important_bac
                                    )
#                pdb.set_trace()
                if probs_ > best_prob:
                    best_prob = probs_
                    best_option = option
                    check_prob = probs
            # Nicola: decomment for printing useful staff
            
            #find the new number of relevant bacteria (not consider the one coming from not important bacteria)
            important_bac_final=0
            for cjj, cj in enumerate(check_prob.values()):
                if cjj< important_bac:
                    important_bac_final+=(cj[0]+1)
#            
#           now we use the best option to label the bacteria matching the previous frame (only the ones taken in consideration) 
            newwell, bacteria_lineage_new = label_most_likely(
                best_option, new_well, bacteria_lineage, important_bac=important_bac_final)
        check_probs[num] = check_prob
        out_wells[num] = newwell
        best_options[num]=best_option
        
        
    return out_wells, bacteria_lineage_new, check_probs,best_options


#### Nicola: in this function I had to make arrays of all the list in the sum function
def find_changes(in_list, option_list, well, new_well):
    """
    Takes a list

    Parameters
    ------
    in_list : list
        A list of labels from the current well
    option_list : list
        A list of all the possible combinations possible of how the bacteria
        in the previous well could be in the new well
    well : ndarray (2D) of dtype int
        A labelled image showing the detected bacteria in the old well
    new_well : ndarray (2D) of dtype int
        A labelled image showing the detected bacteria in the new well

    Yields
    ------
    option : list
        Containing the potential output combination
    in_options_dict : dictionary
        where the key is one of the input bacteria labels and the values
        is a list of the number of divisions, area change, centroid change, input centroid position and input region length
        for that respective bacteria for that potential output combination
    """
    measurements_in = {}  # need to sort them in the same order as list
    measurements_out = {}
    
    # below shift account for the bacteria growth in the channel that moves bacteria
    shift_centroid = 0# 10./400 # (50 pixels shift for the end channel )

    # nicola major change 8.5.2020
    # we sort always the previous data in centroid order
    pos_centroids_sort=[]
    
    for i, region in enumerate(regionprops(well)):
        pos_centroids_sort.append(region.centroid[0])
    ind_centroids= argsort(pos_centroids_sort)

    ind_cc =0
    for jj in ind_centroids:
        for i, region in enumerate(regionprops(well)):
            if i==jj:

                measurements_in[ind_cc] = [region.centroid[0]+shift_centroid*region.centroid[0], region.major_axis_length,region.area] # i change area and major axis
                ind_cc+=1
                
    # end nicola major change 8.5.2020

    for j, region2 in enumerate(regionprops(new_well)):

        measurements_out[j] = [region2.centroid[0], region2.major_axis_length,region.major_axis_length,region.area]  # i change area and major axis
        
    for option in option_list:  # each option is a potential combination of bacteria lineage/death
        in_options_dict = {}
        
        for in_num, in_options in enumerate(in_list):
            out_bac_area = []
            out_bac_centr = []
            
            # determine the number of divisions/deaths
            num_divs = (option.count(in_options)) - 1
            
            for lst, opt in enumerate(option):
                if opt == in_options:  # if the values match append the new centr/areas
                    
                    out_bac_area.append(measurements_out[lst][1])
                    out_bac_centr.append(measurements_out[lst][0])
                    
            # need to divide by biggest number (so prob < 1)
            if sum(array(out_bac_area)) < (measurements_in[in_num][1]):
                # find relative change in area compared to original
                area_chan = float(sum(array(out_bac_area))) / (measurements_in[in_num][1])
            else:
                # find relative change in area compared to original
                area_chan = (measurements_in[in_num][1]) / float(sum(array(out_bac_area)))
            if len(out_bac_centr) != 0:
                # find the average new centroid, Nicola added centroid shift estimate
                centr_chan = ((float((sum(array(out_bac_centr)))) / (len(out_bac_centr)))
                                 - (measurements_in[in_num][0]))
            else:
                centr_chan = 0
            # assign the values to the correct 'in' label
            in_options_dict[in_num] = [num_divs, area_chan, centr_chan, measurements_in[in_num][0],measurements_in[in_num][2]]

        # change_dict[option] = in_options_dict #assign the changes to the
        # respective option
        yield option, in_options_dict  # assign the changes to the respective option
        # return change_dict


def find_probs(
                probs,
                prob_div=0.3,
                prob_death=0.01,
                prob_no_change=0.7,
                av_bac_length=40,#18,
                end_channel=400.,
                important_bac=None
                ):
    """
    Takes a dictionary of information for a potential combination
    and returns an overall probability

    Parameters
    ------
    probs : dictionary
        Key is a unique number of an input bacteria and the value is a
        list of the number of divisions, area change and centroid change
        for that respective bacteria
    prob_div : float, optional
        Probability a bacteria divides between consecutive timepoints (default : 0.01)
    prob_death : float, optional
        Probability a bacteria lyses between consecutive timepoints (default : 0.5)
    prob_no_change : float, optional
        Probability there is no change between consecutive timepoints (default : 0.95)
    av_bac_length : float, optional
        The average bacteria length in pixels (default : 18)

    Returns
    ------
    combined_prob : float
        The overall probability for this combination of events
    """
    ## Nicola: I introduced an higher probability for dyeing for cells close to the end of the channel
    out_prob_last = 0.8;
    out_prob_secondlast = out_prob_last/50
    m_out=1/(float(end_channel))#1/(float(av_bac_length)*2)
    c_out= 0#1 - (end_channel/(float(av_bac_length)*2))
    # Nicola: I removed the possibility of more than a division
    prob_multiple_division = 1e-12
    # Nicola: I sharpened the importance or relative area change and relative centroid chenge 
    change_crit = 0.1 # a change of more than 10% in the area result in a very low probability
    change_centr_crit = float(av_bac_length/20)
    #
    if important_bac==None:
        max_bacnum = 100
    else:
        max_bacnum=deepcopy(important_bac)
    
    probslist = []
    ccc=0
    death_control =0 
    for pro in probs:
        ccc+=1     # ccc is the bacteria id number starting from the top
        # find the potential number of deaths/divisions for each bac
        divs_deaths = probs[pro][0]
        relative_area = probs[pro][1]  # find the relative area change
        # find the number of pixels the centroid has moved by
        change_centr = abs(probs[pro][2])
        position_centroid = probs[pro][3]
        in_length = probs[pro][4]
        
        adj_change_centr_crit = float(change_centr_crit*ccc)
        

        
        if pro>= max_bacnum:
            probslist.append(1) # we don't care about them
            probs[pro].append(1)
        
        if pro< max_bacnum:
            if divs_deaths < 0:  # if the bacteria has died:
                # Nicola: if they are close to the end of the channel, it is likely they go out
                if ccc==len(probs):# and position_centroid > (end_channel-av_bac_length*2):
                    prob_divis = out_prob_last*(m_out*position_centroid+c_out) ### prob linearly incresing clos to the end
                    death_control+=1
                elif ccc==(len(probs)-1):
                    prob_divis = out_prob_secondlast*(m_out*position_centroid+c_out) ### prob linearly incresing clos to the end
                    death_control+=2
                else:
                    prob_divis = prob_death  # probability simply equals that of death
                 # the change in centroid is set as the prob for the bacteria to go out
                prob_centr = exp(-(abs(end_channel-position_centroid)/(adj_change_centr_crit))) 
                prob_area = 1  # the change in area is irrelevant so set probability as 1

            if divs_deaths == 0:  # if the bacteria hasn't died/or divided
                # probability of division simply equals probability of no change
                prob_divis = prob_no_change
                # the area will be equal to the relative area change - may need
                # adjusting
                prob_area = exp(-((1-relative_area)/change_crit)) #relative_area
                # if there is no change then set prob to 1 (0 will cause div error)
                if change_centr == 0:
                    prob_centr = 1
                if change_centr > 0:
                    # the greater the change the less likely
    #                prob_centr = 1 / (abs(change_centr))
                    prob_centr = exp(-(abs(change_centr)/adj_change_centr_crit)) # Nicola
                if change_centr < 0: # this is less likely 
                    prob_centr = exp(-(abs(change_centr)/(change_centr_crit/3.))) # Nicola

            if divs_deaths == 1:  # if bacteria have divided:
                # need to make sure we divide by biggest number to keep prob < 1
                prob_area = exp(-((1-relative_area)/change_crit)) #relative_area
    #             if relative_area < divs_deaths:
    #                 # normalise relative area to the number of divisions
    #                 prob_area = relative_area / divs_deaths
    #             else:
    #                 # normalise relative area to the number of divisions
    #                 prob_area = divs_deaths / relative_area
    #             # each division becomes more likely - need to think about it
                if in_length > av_bac_length*1.5:
                    prob_divis = prob_div#            (1- exp(-in_length/(2*av_bac_length)))
                else:
                    prob_divis = 1e-24
    #            prob_divis = prob_div**(divs_deaths * divs_deaths)
                # for each division the bacteria centroid is expected to move half
                # the bac length
    #             prob_centr = 1 / \   # removed by Nicola
    #                abs(((divs_deaths * (av_bac_length / 2)) - (change_centr)))
                if change_centr == 0:
                    prob_centr = 1
                if change_centr > 0: # this is likely due to the cell growth
                    prob_centr = exp(-(abs(change_centr)/adj_change_centr_crit)) # Nicola
                if change_centr < 0: # this is less likely 
                    prob_centr = exp(-(abs(change_centr)/(change_centr_crit/3))) # Nicola

            if divs_deaths >1: # very unlikely, this pro is killed by prob_multiple_division

                prob_centr = 1 / \
                    abs(((divs_deaths * (av_bac_length / 2)) - (change_centr)))
                prob_area = 1
                prob_divis= prob_multiple_division**(divs_deaths)
        # combine the probabilities for division, area and centroid 
            probslist.append(prob_area * prob_divis * prob_centr)
            probs[pro].append(prob_area * prob_divis * prob_centr)
            
    # multiply the probabilities across all bacteria
    if (death_control !=2) : ### if the second last goes out, also the last one has to go out
        combined_prob = reduce(lambda x, y: x * y, probslist)
    else:
        combined_prob = 1e-24
    return combined_prob, probs


def label_all_new(new_well, label_dict_string,important_bac=15):
    """
    This is in the emergency situation, label the new bacteria as all new, like we restarted from 0, all these bacteria have no mother

    Parameters
    ------
    new_well : ndarray (2D) of dtype int
        A labelled image showing the detected bacteria in the new well
    label_dict_string : dictionary
        Each key is a unique label of a bacteria, each value is
        a string containing its lineage information
    Returns
    ------
    out_well : ndarray (2D) of dtype int
        A labelled image showing the tracked bacteria in the new well
    label_dict_string : dictionary
        Updated dictionary where each key is a unique label of a bacteria,
        each value is a string containing its lineage information
    """
    relabelled_outwell = {}
    out_well = np.zeros(new_well.shape, dtype=new_well.dtype)    
    if bool(label_dict_string):
        smax = max(label_dict_string, key=int);
    else:
        smax = 0

    for i, region in enumerate(regionprops(new_well)):
        if i<important_bac:
            smax += 1
            out_well[new_well == region.label] = smax
            label_dict_string[smax] = '%d'%smax
            relabelled_outwell[0]=out_well
    return relabelled_outwell, label_dict_string



def label_most_likely(most_likely, new_well, label_dict_string,important_bac=None):
    """
    Takes the most likely combination of how the bacteria may have
    divided/died or moved around and re-labels them accordingly

    Parameters
    ------
    most_likely : list
        Containing the most likely output combination
    new_well : ndarray (2D) of dtype int
        A labelled image showing the detected bacteria in the new well
    label_dict_string : dictionary
        Each key is a unique label of a bacteria, each value is
        a string containing its lineage information
    important_bac: only the bacteria born or deriving from the importnat bacteria are kept

    Returns
    ------
    out_well : ndarray (2D) of dtype int
        A labelled image showing the tracked bacteria in the new well
    label_dict_string : dictionary
        Updated dictionary where each key is a unique label of a bacteria,
        each value is a string containing its lineage information
    """
    out_well = np.zeros(new_well.shape, dtype=new_well.dtype)
    if most_likely is None:
        # if there is no likely option return an empty well
        return out_well, label_dict_string

    if important_bac is not None:
        most_likely=most_likely[:important_bac]
    
    new_label_string = 0
    smax = 0
    smax = max(label_dict_string, key=int);
#    print 'smax %0.2f'%smax
    for i, region in enumerate(regionprops(new_well)):
        if i< len(most_likely):
            if most_likely.count(most_likely[i]) == 1:
                out_well[new_well == region.label] = most_likely[i]
            else:
                smax += 1;# print ' a division smax %0.2f'%smax
    #            print 'which_label %0.2f'%i
                out_well[new_well == region.label] = smax
                if i > 0:
                    last_label_start = label_dict_string[most_likely[i - 1]]
                else:
                    last_label_start = label_dict_string[most_likely[i]]
                new_label_start = label_dict_string[most_likely[i]]
                if new_label_start != last_label_start:
                    new_label_string = 0
                new_label_string += 1
                add_string = "_%s" % (new_label_string)
                label_dict_string[smax] = new_label_start + add_string
# this part gives a random number to the one who we did not find a match (not important bacteria)
        else:
            smax+=1
            out_well[new_well == region.label]=smax # if the most_likely element do not exist
            label_dict_string[smax] = '%d'%smax
    return out_well, label_dict_string




def find_problems(
                check_probs,
                av_bac_length=50,#18,
                important_bac=None
                ):
    """
    Takes a dictionary of information for a potential combination
    and returns a flag == False if the potential combination is not possible

    Parameters
    ------
    chck_probs : dictionary
        Key is a unique number of an input bacteria and the value is a
        list of the number of divisions, area change and centroid change (and centroid position and lenght)
        for that respective bacteria
    av_bac_length : float, optional
        The average bacteria length in pixels (default : 50)

    Returns
    ------
    flags : arrays of booleans, one element for each input channel. the element is False if there are problems
    """
    if important_bac==None:
        max_bacnum = 100
    else:
        max_bacnum=deepcopy(important_bac)-1

    
    av_bac_length=50
    flags=[]
    tot_deaths_count=[]
    tot_divisions_count=[]
    for chan_num in arange(len(check_probs)): # cycle on channels 
        flag= True
        divisions_count = 0
        deaths_count = 0
        ccc=0
        for bac_num in arange(len(check_probs[chan_num])):
            divisions = check_probs[chan_num][bac_num][0]
            if divisions>0:
                divisions_count = divisions_count+divisions
            if divisions<0:
                deaths_count = deaths_count + abs(divisions)
            if divisions>=0:
                relative_area_change = 1./check_probs[chan_num][bac_num][1]  
                centroid_change = check_probs[chan_num][bac_num][2]
                if (relative_area_change > 1.4):#(new area change more than the 50%) 
                    if ccc<=max_bacnum:
                        flag = False
                if (centroid_change > (av_bac_length)) | (centroid_change < -(av_bac_length/2)):
                    if ccc<=max_bacnum:
                        flag = False
            ccc+=1
        flags.append(flag)
        tot_deaths_count.append(deaths_count)
        tot_divisions_count.append(divisions_count)
    return array(flags)

def gaussian_prob(x0=0,sigma=1,value=0):
    sigma=float(sigma)
    x0=float(x0)
    value=float(value)
    
    n_gp=1./sqrt(2*pi*sigma**2)
    return n_gp*exp( -( ((value-x0)**2) / (2*(sigma**2)) ))

def remove_notimportant_bac(bacteriaim,important_bac=10):
        in_list=[]
        centroid_pos=[]
        for region in regionprops(bacteriaim):
                # list the bacteria labels from the current frame
                in_list.append(region.label)
                centroid_pos.append(region.centroid[0])

        ind_sort_y = argsort(centroid_pos)  # nicola: important to sort the list
        in_list = array(in_list)
        print(in_list)
        in_list= in_list[ind_sort_y]

        for j, label in enumerate(in_list):
            if j> important_bac-1:  # still need to think about this
                bacteriaim[bacteriaim==label]= bacteriaim.min()
        return bacteriaim


def bacteria_tracking(last_wells, current_wells, bacteria_lineage,
                    prob_div=0.3,
                    prob_death=0.01,
                    prob_no_change=0.7,
                    av_bac_length=40,#18,
                    important_bac=None
                    ):
    """
    Takes a dictionary of wells from the previous frame and a second
    dictionary with the corresponding wells for the new frame and
    determines the most likely way the bacteria within them have
    moved, died, divided then relabels them accordingly

    Parameters
    ------
    last_wells : Dictionary
        The previous timepoint. The key is the well coordinates and the value
        is a labelled image of detected bacteria
    current_wells : Dictionary
        The current timepoint. The key is the well coordinates and the value
        is a labelled image of detected bacteria
    bacteria_lineage : dictionary
        A dictionary that links the physical unique label of a bacteria
        to one which shows information on its lineage

    Returns
    ------
    out_wells : Dictionary
        The current timepoint. The key is the well coordinates and the value
        is a labelled image of tracked bacteria
    bacteria_lineage : dictionary
        Updated dictionary that links the physical unique label of a bacteria
        to one which shows information on its lineage
    check_prob : need to check what is it
    """
    out_wells = {}
    check_probs ={}
    best_options={}
    if important_bac==None:
        important_bac= 8
        
    # create an in_list of the bacteria in the last well (ordered by position along the channel)
    for num, well in last_wells.items():
        if num not in current_wells.keys():
            continue
        new_well = (current_wells[num]).astype(uint16)
        end_channel= shape(new_well)[0]
        in_list = []
        centroid_pos =[]
        #option_list = []
        for region in regionprops(well):
            # list the bacteria labels from the current frame
            in_list.append(region.label)
            centroid_pos.append(region.centroid[0])
        ind_sort_y = argsort(centroid_pos)  # nicola: important to sort the list
#        print unique(well)
        in_list = array(in_list)
        in_list= in_list[ind_sort_y]
        in_list = ndarray.tolist(in_list)
        

        if not in_list: # in the last frame there were no bacteria
            check_prob=[]
            if len(regionprops(new_well)) > 0:
                # if there is now a new bacteria we don't want to ignore as it may have just
                # been missed in previous frame
                if bool(bacteria_lineage):
                    smax = max(bacteria_lineage, key=int)
                else: 
                    smax=0
                newwell = np.zeros(new_well.shape, dtype=uint16)
                for new_bac in regionprops(new_well):
                    # so lets give each "new" bacteria a new label
                    smax += 1
                    newwell[new_well == new_bac.label] = smax
                    bacteria_lineage[smax] = str(smax)
            elif not next(options, None):
                # if the out well is also empty then there is nothing to track
                # and simply return an empty well
                newwell = np.zeros(new_well.shape, dtype=new_well.dtype)
            bacteria_lineage_new=deepcopy(bacteria_lineage)

        # if in_list is not empty    
        else:  # determine probabilities and label matching/new bacteria

            # create a list of all the possible combinations of the in_list with the current number of bacteria
            options = itertools.combinations_with_replacement(
                in_list, len(regionprops(new_well)))
            
            # for each option it gives the change in centroid position, area for each bacteria 
            
            
            change_options = find_changes(
                in_list, options, well, new_well)
            
            # from this calculate the most likely option (calculated only on important bacteria)
            best_option = None
            best_prob = 0
            check_prob=[]
            for option, probs in change_options:
                probs_,probs = find_probs(probs,
                                    prob_div=prob_div,
                                    prob_death=prob_death,
                                    prob_no_change=prob_no_change,
                                    av_bac_length=av_bac_length,#18,
                                    end_channel=end_channel,
                                    important_bac=important_bac
                                    )
#                pdb.set_trace()
                if probs_ > best_prob:
                    best_prob = probs_
                    best_option = option
                    check_prob = probs
            # Nicola: decomment for printing useful staff
            
            #find the new number of relevant bacteria (not consider the one coming from not important bacteria)
            important_bac_final=0
            for cjj, cj in enumerate(check_prob.values()):
                if cjj< important_bac:
                    important_bac_final+=(cj[0]+1)
#            
#           now we use the best option to label the bacteria matching the previous frame (only the ones taken in consideration) 
            newwell, bacteria_lineage_new = label_most_likely(
                best_option, new_well, bacteria_lineage, important_bac=important_bac_final)
        check_probs[num] = check_prob
        out_wells[num] = newwell
        best_options[num]=best_option
        
        
    return out_wells, bacteria_lineage_new, check_probs,best_options