from scipy.optimize import curve_fit, leastsq
from scipy.signal import * 
from scipy.signal import fftconvolve
from scipy.ndimage import *
from numpy import fft
from scipy.ndimage import gaussian_filter


def getdata(i):
    return loadframe(folder+'/BF%d.tiff'%i).data.astype(float)

def myconvolve(im0, im1):
    im0r=(im0-im0.mean())
    im1r=(im1-im1.mean())[::-1, ::-1]
    return fftconvolve(im0r, im1r, mode='same')

def get_shift(im, ref):
    cc = myconvolve(ref, im)
    h,w=im.shape
    yc, xc = h/2.-0.5, w/2.-0.5    
    x,y = np.unravel_index(np.argmax(cc), cc.shape) # find the match
    return x-xc, y-yc

def correct_shift(im, ref):
    dx,dy = get_shift(im, ref)      
    sim=shift(im, (dx,dy), mode='wrap')
    return sim, array([dx,dy])

def get_allshift(fname, imax, s,skip=1):