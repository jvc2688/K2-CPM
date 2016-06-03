from astropy.io import fits as pyfits
from astropy import wcs
import random
import numpy as np
import matplotlib.pyplot as plt
import leastSquareSolver as lss
import threading
import os
import math
import h5py
import sys
import glob
#from images2gif import writeGif
#from PIL import Image
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, FK4, FK5
from astropy.coordinates import Angle, Latitude, Longitude
from sklearn import linear_model
from sklearn.datasets import load_iris
import gc
from scipy.optimize import minimize
from sklearn.decomposition import PCA

#from sklearn.feature_selection import SelectFromModel

sap_style = dict(color='w', linestyle='', marker='.', markersize=2, markerfacecolor='k', markeredgecolor='k', markevery=None)
cpm_prediction_style = dict(color='r', ls='-', lw=1, alpha=0.8)
cpm_style = dict(color='w', linestyle='', marker='.', markersize=2, markerfacecolor='r', markeredgecolor='r', markevery=None)
fit_prediction_style = dict(color='g', linestyle='-', lw=1, markersize=2, markerfacecolor='g', markeredgecolor='g', markevery=None)
fit_style = dict(color='w', linestyle='', marker='+', markersize=2, markerfacecolor='g', markeredgecolor='g', markevery=None)
best_prediction_style = dict(color='g', linestyle='-', marker='+', lw=1, markersize=1.5, markerfacecolor='g', markeredgecolor='g', markevery=None)
best_style = dict(color='w', linestyle='', marker='.', markersize=2, markerfacecolor='g', markeredgecolor='g', markevery=None)


#help function to find the pixel mask
def get_pixel_mask(flux, kplr_mask):
    pixel_mask = np.zeros(flux.shape)
    pixel_mask[np.isfinite(flux)] = 1 # okay if finite
    pixel_mask[:, (kplr_mask < 1)] = 0 # unless masked by kplr
    pixel_mask[flux==0] = 0
    return pixel_mask

#help function to find the epoch mask
def get_epoch_mask(pixel_mask):
    foo = np.sum(np.sum((pixel_mask > 0), axis=2), axis=1)
    epoch_mask = np.zeros_like(foo)
    epoch_mask[(foo > 0)] = 1
    return epoch_mask

#help function to load header from tpf
def load_header(tpf):
    with pyfits.open(tpf) as file:
        meta = file[1].header
        column = meta['1CRV4P']
        row = meta['2CRV4P']
    return row,column

#help function to load data from tpf
def load_data(tpf):
    kplr_mask, time, flux, flux_err = [], [], [], []
    with pyfits.open(tpf) as file:
        hdu_data = file[1].data
        kplr_mask = file[2].data
        meta = file[1].header
        column = meta['1CRV4P']
        row = meta['2CRV4P']
        time = hdu_data["time"]
        flux = hdu_data["flux"]#hdu_data['raw_cnts'] #hdu_data["flux"]
        flux_err = hdu_data["flux_err"]

    length = time.shape[0]
    time = time[length/2.+100:]
    flux = flux[length/2.+100:]
    flux_err = flux_err[length/2.+100:]
    
    pixel_mask = get_pixel_mask(flux, kplr_mask)
    epoch_mask = get_epoch_mask(pixel_mask)
    flux = flux[:, kplr_mask>0]
    flux_err = flux_err[:, kplr_mask>0]
    shape = flux.shape

    flux = flux.reshape((flux.shape[0], -1))
    flux_err = flux_err.reshape((flux.shape[0], -1))

    #interpolate the bad points
    index = np.arange(flux.shape[0])
    for i in range(flux.shape[1]):
        interMask = np.isfinite(flux[:,i])
        flux[~interMask,i] = np.interp(index[~interMask], index[interMask], flux[interMask,i]) #flux[~interMask,i] = np.interp(time[~interMask], time[interMask], flux[interMask,i])
        flux_err[~interMask,i] = np.inf

    return time, flux, pixel_mask, kplr_mask, epoch_mask, flux_err, column, row

#get predictor matrix randomly from a set of tpfs
def get_predictor_matrix(tpfs, num=None):
    predictor_flux = []
    predictor_epoch_mask=1.
    for tpf in tpfs:
        time, flux, pixel_mask, kplr_mask, epoch_mask, flux_err, column, row = load_data(tpf)
        predictor_epoch_mask *= epoch_mask
        predictor_flux.append(flux)
    predictor_matrix =  np.concatenate(predictor_flux, axis=1)
    size = predictor_matrix.shape[1]
    if num is not None:
        sample = np.random.choice(size,num,replace=False)
        predictor_matrix = predictor_matrix[:,sample]
    print('predictor matrix:')
    print(predictor_matrix.shape)

    return predictor_matrix, predictor_epoch_mask

def get_predictor_matrix_nearby(target_tpf, target_x, target_y):
    time, flux, pixel_mask, target_kplr_mask, epoch_mask, flux_err, column, row = load_data(target_tpf)
    x_lim = target_kplr_mask.shape[0]
    y_lim = target_kplr_mask.shape[1]

    flux = flux.reshape((flux.shape[0],50,50))
    flux_err = flux_err.reshape((flux_err.shape[0],50,50))

    predictor_fluxes, neighbor_pixel_maskes, neighbor_kplr_maskes = [], [], []

    #construt the neighbor flux matrix
    target_flux = np.float64(flux)[:,target_x,target_y]
    target_flux_err = np.float64(flux_err)[:,target_x,target_y]
    target_flux_err[epoch_mask==0] = np.inf

    excl_x = range(target_x-5, target_x+5+1)
    excl_y = range(target_y-5, target_y+5+1)
    #print excl_x
    #print excl_y
    #print x_lim, y_lim
    predictor_mask = np.ones((x_lim, y_lim))
    for i in range(flux.shape[1]):
        for j in range(flux.shape[2]):
            if i in excl_x and j in excl_y:#if i in excl_x or j in excl_y:
                predictor_mask[i,j] = 0
    predictor_mask = predictor_mask.flatten()
    predictor_mask = predictor_mask[target_kplr_mask.flatten()>0]
    flux = flux.reshape((flux.shape[0], -1))
    predictor_flux_matrix = flux[:, predictor_mask>0]

    return predictor_flux_matrix

def get_fit_matrix(target_flux, target_flux_err, target_epoch_mask, predictor_matrix, predictor_epoch_mask, l2, time, poly=0, prefix='lightcurve'):
    """
    ## inputs:
    - `target_flux` - target flux
    - `target_flux_err` - error of the target flux
    - `target_epoch_mask`- target epoch mask
    - `predictor_matrix` - matrix of predictor fluxes
    - `predictor_epoch_mask` - predictor epoch mask
    - `l2` - strength of l2 regularization
    - `time` - one dimension array of BKJD time
    - `poly` - number of orders of polynomials of time need to be added
    - `prefix` - prefix of output file
    
    ## outputs:
    - `target_flux` - target flux
    - `predictor_matrix` - matrix of predictor fluxes
    - `target_flux_err` - error of the target flux
    - `l2_vector` - vector of the l2 regularization 
    - `time` - one dimension array of BKJD time
    - `epoch_mask` - epoch mask
    - `data_mask` - data_mask
    """

    epoch_mask = target_epoch_mask*predictor_epoch_mask

    epoch_len = epoch_mask.shape[0]
    data_mask = np.ones(epoch_len, dtype=int)

    #remove bad time point based on simulteanous epoch mask
    co_mask = data_mask*epoch_mask
    time = time[co_mask>0]
    target_flux = target_flux[co_mask>0]
    target_flux_err = target_flux_err[co_mask>0]
    predictor_matrix = predictor_matrix[co_mask>0, :]

    #add polynomial terms
    if poly is not None:
        time_mean = np.mean(time)
        time_std = np.std(time)
        nor_time = (time-time_mean)/time_std
        p = np.polynomial.polynomial.polyvander(nor_time, poly)
        predictor_matrix = np.concatenate((predictor_matrix, p), axis=1)
    
    #construct l2 vectors
    predictor_num = predictor_matrix.shape[1]
    print predictor_num
    auto_pixel_num = 0
    l2_vector = np.ones(predictor_num, dtype=float)*l2

    print('load matrix successfully')

    return target_flux, predictor_matrix, target_flux_err, l2_vector, time, epoch_mask, data_mask



def get_fit_matrix_kepler(kid, campaign, target_tpf, target_pixel, l2,  poly=0, auto=False, offset=0, window=0, auto_l2=0, part=None, filter=False, prefix='lightcurve'):
    """
    ## inputs:
    - `target_tpf` - target tpf
    - `auto` - if autorgression
    - `poly` - number of orders of polynomials of time need to be added
    
    ## outputs:
    - `neighbor_flux_matrix` - fitting matrix of neighbor flux
    - `target_flux` - target flux
    - `covar_list` - covariance matrix for every pixel
    - `time` - one dimension array of BKJD time
    - `neighbor_kid` - KIC number of the neighbor stars in the fitting matrix
    - `neighbor_kplr_maskes` - kepler maskes of the neighbor stars in the fitting matrix
    - `target_kplr_mask` - kepler mask of the target star
    - `epoch_mask` - epoch mask
    - `l2_vector` - array of L2 regularization strength
    """
    
    filename = "./%s"%prefix
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    f = h5py.File('%s.hdf5'%prefix, 'w')
    
    time, flux, pixel_mask, target_kplr_mask, epoch_mask, flux_err, column, row = load_data(target_tpf)

    flux = flux.reshape((flux.shape[0],50,50))
    flux_err = flux_err.reshape((flux_err.shape[0],50,50))

    predictor_fluxes, neighbor_pixel_maskes, neighbor_kplr_maskes = [], [], []

    #construt the neighbor flux matrix
    target_x = target_pixel[0]
    target_y = target_pixel[1]
    target_flux = np.float64(flux)[:,target_x,target_y]
    target_flux_err = np.float64(flux_err)[:,target_x,target_y]
    target_flux_err[epoch_mask==0] = np.inf
    print(np.min(target_flux_err))

    print(target_flux.shape)

    excl_x = range(target_x-5, target_x+5+1)
    excl_y = range(target_y-5, target_y+5+1)
    print excl_x
    print excl_y
    predictor_mask = np.ones((flux.shape[1], flux.shape[2]))
    for i in range(flux.shape[1]):
        for j in range(flux.shape[2]):
            if i in excl_x and j in excl_y:#if i in excl_x or j in excl_y:
                predictor_mask[i,j] = 0
    predictor_mask = predictor_mask.flatten()
    predictor_flux_matrix = flux.reshape((flux.shape[0], -1))[:, predictor_mask>0]
    print('predictor matrix shape:', predictor_flux_matrix.shape)

    epoch_len = epoch_mask.shape[0]
    data_mask = np.zeros(epoch_len, dtype=int)

    #construct l2 vectors
    predictor_num = predictor_flux_matrix.shape[1]
    auto_pixel_num = 0
    l2_vector = np.ones(predictor_num, dtype=float)*l2
    '''
    l2_vector = np.array([])
    if neighbor_flux_matrix.size != 0:
        pixel_num = neighbor_flux_matrix.shape[1]
        l2_vector = np.ones(pixel_num, dtype=float)*l2
    else:
        pixel_num = 0
    '''

    #spilt lightcurve
    if part is not None:
        epoch_len = epoch_mask.shape[0]
        split_point = []
        split_point_forward = []
        split_point.append(0)
        i = 0
        while i <epoch_len:
            if epoch_mask[i] == 0:
                i += 1
            else:
                j = i+1
                while j<epoch_len and epoch_mask[j] == 0:
                    j += 1
                if j-i > 30 and j-split_point[-1] > 1200:
                    split_point.append(i)
                    split_point.append(j)
                i = j
        split_point.append(epoch_len)
        #print split_point
        #data_mask[split_point[part-1]:split_point[part]] = 1
        start = split_point[2*(part-1)]+offset+window
        end = split_point[2*(part-1)+1]-offset-window+1
        data_mask[start:end] = 1
        '''
        fit_epoch_mask =  np.split(fit_epoch_mask, split_point)[part-1]
        fit_time = np.split(time, split_point)[part-1]
        fit_target_flux = np.split(target_flux, split_point, axis=0)[part-1]
        flux_err = np.split(flux_err, split_point, axis=0)[part-1]
        neighbor_flux_matrix = np.split(neighbor_flux_matrix, split_point, axis=0)[part-1]

        print (fit_epoch_mask.shape, fit_time.shape, fit_target_flux.shape, flux_err.shape, neighbor_flux_matrix.shape)
        '''
    else:
        data_mask[:] = 1

    #add auto-regression terms
    if auto and (window != 0):
        #print 'auto'
        tmp_target_kplr_mask = target_kplr_mask.flatten()
        tmp_target_kplr_mask = tmp_target_kplr_mask[tmp_target_kplr_mask>0]
        auto_flux = target_flux[:, tmp_target_kplr_mask==3]
        for i in range(offset+1, offset+window+1):
            if neighbor_flux_matrix.size != 0:
                neighbor_flux_matrix = np.concatenate((neighbor_flux_matrix, np.roll(auto_flux, i, axis=0)), axis=1)
                neighbor_flux_matrix = np.concatenate((neighbor_flux_matrix, np.roll(auto_flux, -i, axis=0)), axis=1)
            else:
                neighbor_flux_matrix = np.roll(auto_flux, i, axis=0)
                neighbor_flux_matrix = np.concatenate((neighbor_flux_matrix, np.roll(auto_flux, -i, axis=0)), axis=1)
        data_mask[0:offset+window] = 0
        data_mask[-offset-window:] = 0
        auto_pixel_num = neighbor_flux_matrix.shape[1] - pixel_num
        l2_vector = np.concatenate((l2_vector, np.ones(auto_pixel_num, dtype=float)*auto_l2), axis=0)
        '''
        other_pixel_num = pixel_num
        pixel_num = neighbor_flux_matrix.shape[1]
        if l2_vector.size != 0:
            l2_vector = np.concatenate((l2_vector, np.ones(pixel_num-other_pixel_num, dtype=float)*auto_l2), axis=0)
        else:
            l2_vector = np.ones(pixel_num-other_pixel_num, dtype=float)*auto_l2
        '''
    else:
        auto_pixel_num = 0

    #remove bad time point based on simulteanous epoch mask
    co_mask = data_mask*epoch_mask
    time = time[co_mask>0]
    target_flux = target_flux[co_mask>0]
    target_flux_err = target_flux_err[co_mask>0]
    predictor_flux_matrix = predictor_flux_matrix[co_mask>0, :]

    #print neighbor_flux_matrix.shape

    #add polynomial terms
    if poly is not None:
        time_mean = np.mean(time)
        time_std = np.std(time)
        nor_time = (time-time_mean)/time_std
        p = np.polynomial.polynomial.polyvander(nor_time, poly)
        predictor_flux_matrix = np.concatenate((predictor_flux_matrix, p), axis=1)
        l2_vector = np.concatenate((l2_vector, np.zeros(poly+1)), axis=0)
        '''
        median = np.median(target_flux)
        fourier_flux = []
        period = 30.44
        print('month:%f, time difference:%f'%(time[-1]-time[0], time[1]-time[0]))
        for i in range(1,33):
            fourier_flux.append(median*np.sin(2*np.pi*i*(time-time[0])/2./30.44).reshape(time.shape[0],1))
            fourier_flux.append(median*np.cos(2*np.pi*i*(time-time[0])/2./30.44).reshape(time.shape[0],1))
        print fourier_flux[0].shape
        fourier_components = np.concatenate(fourier_flux, axis=1)
        print fourier_components.shape
        neighbor_flux_matrix = np.concatenate((neighbor_flux_matrix, fourier_components), axis=1)
        l2_vector = np.concatenate((l2_vector, np.ones(fourier_components.shape[1])), axis=0)
        '''
    #print neighbor_flux_matrix.shape

    f.attrs['kid'] = kid
    f.attrs['campaign'] = campaign
    if part is not None:
        f.attrs['part'] = part

    data_group = f.create_group('data')
    cpm_info = f.create_group('cpm_info')
    
    data_group['target_flux'] = target_flux
    data_group['time'] = time
    data_group['target_kplr_mask'] = target_kplr_mask
    data_group['epoch_mask'] = epoch_mask
    data_group['data_mask'] = data_mask

    cpm_info['predictor_num'] = predictor_num
    cpm_info['l2'] = l2
    if auto:
        cpm_info['auto'] = 1
        cpm_info['auto_pixel_num'] = auto_pixel_num
        cpm_info['auto_l2'] = auto_l2
        cpm_info['auto_window'] = window
        cpm_info['auto_offset'] = offset
    else:
        cpm_info['auto'] = 0
    cpm_info['poly'] = poly

    print('kic%d load matrix successfully'%kid)

    f.close()

    return predictor_flux_matrix, target_flux, target_flux_err, time, target_kplr_mask, epoch_mask, data_mask, l2_vector, predictor_num, auto_pixel_num


def fit_target(target_flux, target_kplr_mask, predictor_flux_matrix, time, epoch_mask, covar_list, margin, l2_vector=None, thread_num=1, prefix="lightcurve", transit_mask=None):
    """
    ## inputs:
    - `target_flux` - target flux
    - `target_kplr_mask` - kepler mask of the target star
    - `neighbor_flux_matrix` - fitting matrix of neighbor flux
    - `time` - array of time 
    - `epoch_mask` - epoch mask
    - `covar_list` - covariance list
    - `margin` - size of the test region
    - `poly` - number of orders of polynomials of time(zero order is the constant level)
    - `l2_vector` - array of L2 regularization strength
    - `thread_num` - thread number
    - `prefix` - output file's prefix
    
    ## outputs:
    - prefix.npy file - fitting fluxes of pixels
    """
    filename = "./%s"%prefix
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    f = h5py.File('%s.hdf5'%prefix, 'a')
    cpm_info = f['/cpm_info']
    data_group = f['/data']
    cpm_info['margin'] = margin
    
    print covar_list.shape
    covar = covar_list**2
    fit_flux = []
    fit_coe = []
    length = target_flux.shape[0]
    total_length = epoch_mask.shape[0]
    
    thread_len = total_length//thread_num
    last_len = total_length - (thread_num-1)*thread_len
    
    class fit_epoch(threading.Thread):
        def __init__(self, thread_id, initial, len, time_initial, time_len):
            threading.Thread.__init__(self)
            self.thread_id = thread_id
            self.initial = initial
            self.len = len
            self.time_initial = time_initial
            self.time_len = time_len
        def run(self):
            print('Starting%d'%self.thread_id)
            print (self.thread_id , self.time_initial, self.time_len)
            tmp_fit_flux = np.empty(self.time_len)
            time_stp = 0
            for i in range(self.initial, self.initial+self.len):
                if epoch_mask[i] == 0:
                    continue
                train_mask = np.ones(total_length)
                if i<margin:
                    train_mask[0:i+margin+1] = 0
                elif i > total_length-margin-1:
                    train_mask[i-margin:] = 0
                else:
                    train_mask[i-margin:i+margin+1] = 0
                train_mask = train_mask[epoch_mask>0]
                
                tmp_covar = covar[train_mask>0]
                print predictor_flux_matrix.shape, target_flux.shape, tmp_covar.shape, l2_vector.shape
                result = lss.linear_least_squares(predictor_flux_matrix[train_mask>0], target_flux[train_mask>0], tmp_covar, l2_vector)
                tmp_flux = np.dot(predictor_flux_matrix[time_stp+self.time_initial, :], result)
                print result.shape
                print(tmp_flux.shape)
                tmp_fit_flux[time_stp] = np.dot(predictor_flux_matrix[time_stp+self.time_initial, :], result)
                #np.save('./%stmp%d.npy'%(prefix, self.thread_id), tmp_fit_flux)
                time_stp += 1
                #print('done%d'%i)
            print('Exiting%d'%self.thread_id)
            np.save('./%stmp%d.npy'%(prefix, self.thread_id), tmp_fit_flux)
    thread_list = []
    time_initial = 0
    for i in range(0, thread_num-1):
        initial = i*thread_len
        thread_epoch = epoch_mask[initial:initial+thread_len]
        time_len = np.sum(thread_epoch)
        thread = fit_epoch(i, initial, thread_len, time_initial, time_len)
        thread.start()
        thread_list.append(thread)
        time_initial += time_len
    
    initial = (thread_num-1)*thread_len
    thread_epoch = epoch_mask[initial:initial+last_len]
    time_len = np.sum(thread_epoch)
    thread = fit_epoch(thread_num-1, initial, last_len, time_initial, time_len)
    thread.start()
    thread_list.append(thread)
    
    for t in thread_list:
        t.join()
    print 'all done'
    
    offset = 0
    window = 0
    
    for i in range(0, thread_num):
        tmp_fit_flux = np.load('./%stmp%d.npy'%(prefix, i))
        if i==0:
            fit_flux = tmp_fit_flux
        else:
            fit_flux = np.concatenate((fit_flux, tmp_fit_flux), axis=0)

    data_group['fit_flux'] = fit_flux
    f.close()

    for i in range(0, thread_num):
        os.remove('./%stmp%d.npy'%(prefix, i))

def fit_target_no_train(target_flux, target_kplr_mask, predictor_flux_matrix, time, epoch_mask, covar_list, l2_vector=None, thread_num=1, train_mask=None):
    """
    ## inputs:
    - `target_flux` - target flux
    - `target_kplr_mask` - kepler mask of the target star
    - `neighbor_flux_matrix` - fitting matrix of neighbor flux
    - `time` - array of time 
    - `epoch_mask` - epoch mask
    - `covar_list` - covariance list
    - `margin` - size of the test region
    - `poly` - number of orders of polynomials of time(zero order is the constant level)
    - `l2_vector` - array of L2 regularization strength
    - `thread_num` - thread number
    - `prefix` - output file's prefix
    
    ## outputs:
    - prefix.npy file - fitting fluxes of pixels
    """
    '''
    filename = "./%s"%prefix
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    f = h5py.File('%s.hdf5'%prefix, 'a')
    cpm_info = f['/cpm_info']
    data_group = f['/data']
    cpm_info['margin'] = margin
    '''
    if train_mask is not None:
        predictor_flux_matrix = predictor_flux_matrix[train_mask>0,:]
        target_flux = target_flux[train_mask>0]
        if covar_list is not None:
            covar_list = covar_list[train_mask>0]
        #print predictor_matrix.shape
    if covar_list is not None:
        covar = covar_list**2
    else:
        covar = None
    fit_flux = []
    fit_coe = []
    length = target_flux.shape[0]
    total_length = epoch_mask.shape[0]
    
    thread_len = total_length//thread_num
    last_len = total_length - (thread_num-1)*thread_len
    
    result = lss.linear_least_squares(predictor_flux_matrix, target_flux, covar, l2_vector)
    #fit_flux = np.dot(predictor_flux_matrix, result)
    #data_group['fit_flux'] = fit_flux
    #f.close()
    return result


def pixel_plot(time, flux, name, size=None):
    shape = flux.shape
    if size==None:
        x = range(0, shape[1])
        y = range(0, shape[2])
    else:
        x = range(size[0][0], size[0][1])
        y = range(size[1][0], size[1][1])
    # Plot the data
    x0 = x[0]
    y0 = y[0]
    td = shape[0]
    f, axes = plt.subplots(len(x), len(y))

    for i in x:
        for j in y:
            pi=i-x0
            pj=j-y0
            print(pi,pj)
            axes[pi,pj].plot(time,flux[0:td:1,i,j], '.k', markersize = 1)
            plt.setp( axes[pi,pj].get_xticklabels(), visible=False)
            plt.setp( axes[pi,pj].get_yticklabels(), visible=False)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0, hspace=0)
    plt.suptitle('%s'%(name))
    plt.savefig('../plots/%s.png'%(name), dpi=190)
    plt.clf()

def load_var(filename):
    #dtype = [('ID', int), ('ra', float), ('dec', float), ('p', float), ('type', np.unicode)]
    var = np.loadtxt(filename, skiprows=1, usecols=[0,1,2,3,4,5,6,7])
    types = np.loadtxt(filename, skiprows=1, usecols=[11], dtype=np.str)
    return var,types

def load_k2_var(filename):
    var = np.loadtxt(filename, usecols=[0,1,2,3,4,5,6,7,8,9])
    return var

def fold_lc(t0, p, flux, time, epoch_mask):
    flux = flux[epoch_mask>0]
    length = flux.shape[0]
    fold_flux = np.zeros(length)
    fold_time = np.zeros(length)
    for i in range(length):
        fold_time[i] = (time[i]+t0)-int((time[i]+t0)/p)*p
    return fold_time

def get_xy(i, kplr_mask):
    print i
    index_matrix = np.arange(kplr_mask.flatten().shape[0])
    masked_index = index_matrix[kplr_mask.flatten()>0]
    index = masked_index[i]
    x = index/kplr_mask.shape[1]
    y = index-x*kplr_mask.shape[1]
    return x,y

def load_var(filename):
    #dtype = [('ID', int), ('ra', float), ('dec', float), ('p', float), ('type', np.unicode)]
    var = np.loadtxt(filename, skiprows=1, usecols=[0,1,2,3,4,5,6,7])
    types = np.loadtxt(filename, skiprows=1, usecols=[11], dtype=np.str)
    return var,types

def fold_lc(t0, p, flux, time, epoch_mask):
    flux = flux[epoch_mask>0]
    length = flux.shape[0]
    fold_flux = np.zeros(length)
    fold_time = np.zeros(length)
    for i in range(length):
        fold_time[i] = (time[i]+t0)-int((time[i]+t0)/p)*p
    return fold_time

def get_pixel_mask_ffi(ffi):
    pixel_mask = np.zeros(ffi.shape, dtype=int)
    pixel_mask[np.isfinite(ffi)] = 1 # okay if finite
    pixel_mask[np.logical_or(ffi==0, ffi==-1)] = 0
    return pixel_mask

def get_epoch_mask_ffi(pixel_mask):
    foo = np.sum(np.sum((pixel_mask > 0), axis=2), axis=1)
    epoch_mask = np.zeros_like(foo, dtype=int)
    epoch_mask[(foo > 0)] = 1
    return epoch_mask

def get_kplr_mask_ffi(pixel_mask):
    foo = np.sum(pixel_mask>0, axis=0)
    kplr_mask = np.zeros_like(foo, dtype=int)
    kplr_mask[foo>0] = 1
    return kplr_mask

def load_ffi(name):
    hdu_list = pyfits.open(name)
    ffi = hdu_list[0].data
    #ffi = np.load(name)
    pixel_mask = get_pixel_mask_ffi(ffi)
    epoch_mask = get_epoch_mask_ffi(pixel_mask)
    kplr_mask = get_kplr_mask_ffi(pixel_mask)

    print pixel_mask.shape, epoch_mask.shape, kplr_mask.shape
    pixel_mask = None

    ffi = ffi[:,kplr_mask>0]
    ffi = ffi.reshape((ffi.shape[0], -1))
    gc.collect()

    return ffi, kplr_mask, epoch_mask

def get_predictor_matrix_ffi(ffi, x, y, kplr_mask, num, var_mask=None):
    x_lim = kplr_mask.shape[0]
    y_lim = kplr_mask.shape[1]

    predictor_mask = np.ones_like(kplr_mask, dtype=int)

    if x<=5:
        predictor_mask[:x+5+1,:] = 0
    elif x>=x_lim-5-1:
        predictor_mask[x-5:,:] = 0
    else:
        predictor_mask[x-5:x+5+1,:] = 0

    if y<=5:
        predictor_mask[:,:y+5+1] = 0
    elif y>=y_lim-5-1:
        predictor_mask[:,y-5:] = 0
    else:
        predictor_mask[:,y-5:y+5+1] = 0

    if var_mask is None:
        pixels = np.where(np.logical_and(kplr_mask>0, predictor_mask>0))
    else:
        pixels = np.where(np.logical_and(np.logical_and(kplr_mask>0, predictor_mask>0), var_mask<1))

    dis = np.ones_like(kplr_mask)*99999999

    dis[pixels[0], pixels[1]] = np.square(pixels[0]-x)+np.square(pixels[1]-y)

    dis = dis.flatten()
    index = np.argsort(dis)

    predictor_mask = predictor_mask.flatten()
    predictor_mask[index[num:]] = 0

    #plt.imshow(predictor_mask.reshape((1024,1100)), interpolation='None', cmap=plt.get_cmap('Greys'))
    #plt.show()


    predictor_mask = predictor_mask[kplr_mask.flatten()>0]

    predictor_matrix = ffi[:, predictor_mask>0].astype(float)

    return predictor_matrix


def get_predictor_matrix_pca(num):
    predictor_matrix = np.load('/scratch/dw1519/k2c9/data/ffi/kplr.geert.io/k2-c9a-quicklook-v20160521/pca.npy')[:,:num]

    return predictor_matrix

def get_fit_matrix_ffi(target_flux, target_epoch_mask, predictor_matrix, predictor_epoch_mask, l2, poly=0, prefix='lightcurve', ml=None):
    """
    ## inputs:
    - `target_flux` - target flux
    - `target_flux_err` - error of the target flux
    - `target_epoch_mask`- target epoch mask
    - `predictor_matrix` - matrix of predictor fluxes
    - `predictor_epoch_mask` - predictor epoch mask
    - `l2` - strength of l2 regularization
    - `time` - one dimension array of BKJD time
    - `poly` - number of orders of polynomials of time need to be added
    - `prefix` - prefix of output file
    
    ## outputs:
    - `target_flux` - target flux
    - `predictor_matrix` - matrix of predictor fluxes
    - `target_flux_err` - error of the target flux
    - `l2_vector` - vector of the l2 regularization 
    - `time` - one dimension array of BKJD time
    - `epoch_mask` - epoch mask
    - `data_mask` - data_mask
    """

    epoch_mask = target_epoch_mask*predictor_epoch_mask

    epoch_len = epoch_mask.shape[0]
    data_mask = np.ones(epoch_len, dtype=int)

    #remove bad time point based on simulteanous epoch mask
    co_mask = data_mask*epoch_mask
    target_flux = target_flux[co_mask>0]
    predictor_matrix = predictor_matrix[co_mask>0, :]

    #add polynomial terms
    if poly is not None:
        nor_time = np.arange(1287)
        #time_mean = np.mean(time)
        #time_std = np.std(time)
        #nor_time = (time-time_mean)/time_std
        p = np.polynomial.polynomial.polyvander(nor_time, poly)
        predictor_matrix = np.concatenate((predictor_matrix, p), axis=1)

    if ml is not None:
        predictor_matrix = np.concatenate((predictor_matrix, ml), axis=1)

    #construct l2 vectors
    predictor_num = predictor_matrix.shape[1]
    #print predictor_num
    auto_pixel_num = 0
    l2_vector = np.ones(predictor_num, dtype=float)*l2

    predictor_num = predictor_matrix.shape[1]
    #print predictor_num


    #print('load matrix successfully')

    return target_flux, predictor_matrix, None, l2_vector, epoch_mask, data_mask

def V(u_min, t_0, t_E, t):
    u = np.sqrt(u_min * u_min + ((t - t_0) / t_E) ** 2)
    V = ((u * u + 2.) / (u * np.sqrt(u * u + 4.)) - 1.)
    return V

def objective(para, t, kplr_mask, target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2):
    u_min, t_0, t_E = para[0], para[1], para[2]
    #print u_min, t_0, t_E
    thread_num = 1
    ml = np.array([V(u_min, t_0, t_E, t)]).T

    flux, predictor_matrix, flux_err, l2_vector, target_epoch_mask, data_mask \
                    = get_fit_matrix_ffi(target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2, 0, 'lightcurve', ml)

    #print flux.shape
    result = fit_target_no_train(flux, kplr_mask, np.copy(predictor_matrix), None, target_epoch_mask[data_mask>0], None, l2_vector, thread_num)
    #print result[-1]
    fit_flux = np.dot(predictor_matrix, result)[:,0]
    #print fit_flux.shape
    cpm = np.dot(predictor_matrix[:,:-1], result[:-1])[:,0]
    dif = flux-fit_flux
    dif_cpm = flux - cpm

    '''
    print dif_cpm.shape
    plt.plot(dif_cpm, '.k')
    plt.show()
    '''

    error = np.sum(np.square(flux-fit_flux))
    #print error

    return error

def objective_coadd(para, t, kplr_mask, target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2):
    u_min, t_0, t_E = para[0], para[1], para[2]
    #print u_min, t_0, t_E
    thread_num = 1
    ml = np.array([V(u_min, t_0, t_E, t)]).T

    flux, predictor_matrix, flux_err, l2_vector, target_epoch_mask, data_mask \
                    = get_fit_matrix_ffi(target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2, 0, 'lightcurve', ml)

    #print flux.shape
    result = fit_target_no_train(flux, kplr_mask, np.copy(predictor_matrix), None, target_epoch_mask[data_mask>0], None, l2_vector, thread_num)
    #print result[-1]
    fit_flux = np.sum(np.dot(predictor_matrix, result), axis=1)
    #print fit_flux.shape
    #cpm = np.dot(predictor_matrix[:,:-1], result[:-1])[:,0]
    #dif = flux-fit_flux
    #dif_cpm = flux - cpm

    '''
    print dif_cpm.shape
    plt.plot(dif_cpm, '.k')
    plt.show()
    '''

    error = np.sum(np.square(np.sum(flux, axis=1)-fit_flux))
    #print error

    return error

class cpm:
    def __init__(self, data_file, pixel, var_file=None, train_start=0, train_end=0):
        if var_file is not None:
            var_mask = np.load(var_file)

        #pixel_list = np.loadtxt(pixel_file, dtype=int)#[12:13]
        self.l2 = 0#1e5#1e6
        self.num_predictor = 1600
        thread_num = 1

        ffi, self.kplr_mask, self.epoch_mask = load_ffi(data_file)

        self.train_mask = np.ones(ffi.shape[0], dtype=int)
        self.train_mask[train_start:train_end] = 0 

        self.target_x, self.target_y = pixel
        pixel_mask = np.zeros((1024,1100), dtype=int)
        pixel_mask[self.target_x, self.target_y] = 1
        pixel_mask = pixel_mask[self.kplr_mask>0]
        self.target_flux = ffi[:,pixel_mask>0][:,0].astype(float)
        self.predictor_epoch_mask = np.ones(self.epoch_mask.shape[0])
        transit_mask = None
        predictor_matrix = get_predictor_matrix_ffi(ffi, self.target_x, self.target_y, self.kplr_mask, self.num_predictor, var_mask)
        pca = PCA(n_components=200)
        pca.fit(predictor_matrix)
        self.predictor_matrix = pca.transform(predictor_matrix)

    def fit_lc(self, ml):
        ml = np.array([ml]).T
        flux, predictor_matrix, flux_err, l2_vector, target_epoch_mask, data_mask \
                        = get_fit_matrix_ffi(self.target_flux, self.epoch_mask, self.predictor_matrix, self.predictor_epoch_mask, self.l2, 0, 'lightcurve', ml)

        #fit_flux = np.dot(predictor_matrix, result)[:,0]
        cpm_fit = np.dot(predictor_matrix[:,0:-1], result[:-1])[:,0]
        ml_fit = (flux - cpm_fit)/result[-1]

        return ml_fit


if __name__ == "__main__":

#nearby
    if False:
        channel = int(sys.argv[1])
        pixel_file = sys.argv[2]
        train_start = int(sys.argv[3])
        train_end = int(sys.argv[4])
        output_name = sys.argv[5]

        name = '/scratch/dw1519/k2c9/data/ffi/kplr.geert.io/k2-c9a-quicklook-v20160521/ch%d.npy'%channel
        out_name = '/scratch/dw1519/k2c9/data/ffi/kplr.geert.io/k2-c9a-quicklook-v20160521/new_pixel/fit_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)
        out_name_dif = '/scratch/dw1519/k2c9/data/ffi/kplr.geert.io/k2-c9a-quicklook-v20160521/new_pixel/dif_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)

        print 'pixel file:\n%s'%pixel_file
        print 'train:%d-%d'%(train_start,train_end)

        pixel_list = np.loadtxt(pixel_file, dtype=int)
        l2 = 1e5#1e6
        num_predictor = 1600
        thread_num = 1

        ffi, kplr_mask, epoch_mask = load_ffi(name)
        print ffi.shape
        #start = np.argwhere(idx[kplr_mask.flatten()>0]==start)
        #end = np.argwhere(idx[kplr_mask.flatten()>0]==end)
        '''
        plt.imshow(kplr_mask, interpolation='None', cmap=plt.get_cmap('Greys'))
        plt.show()


        target_x,target_y = get_xy(10000, kplr_mask)
        print target_x, target_y
        predictor_matrix = get_predictor_matrix_ffi(ffi, target_x, target_y, kplr_mask, num_predictor)
        '''
        train_mask = np.ones(ffi.shape[0], dtype=int)
        train_mask[train_start:train_end] = 0 
        fit_image = np.zeros((ffi.shape[0], pixel_list.shape[0]), dtype=float)
        dif = np.zeros((ffi.shape[0], pixel_list.shape[0]), dtype=float)
        i = 0
        for pixel in pixel_list:
            target_x,target_y = pixel[0], pixel[1]#get_xy(i, kplr_mask)
            print target_x, target_y
            if kplr_mask[target_x, target_y] > 0:
                pixel_mask = np.zeros((1024,1100), dtype=int)
                pixel_mask[target_x, target_y] = 1
                pixel_mask = pixel_mask[kplr_mask>0]
                target_flux = ffi[:,pixel_mask>0][:,0].astype(float)
                predictor_epoch_mask = np.ones(epoch_mask.shape[0])
                transit_mask = None
                predictor_matrix = get_predictor_matrix_ffi(ffi, target_x, target_y, kplr_mask, num_predictor)
                flux, predictor_matrix, flux_err, l2_vector, target_epoch_mask, data_mask \
                    = get_fit_matrix_ffi(target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2, 0)
                fit_flux, result = fit_target_no_train(flux, kplr_mask, np.copy(predictor_matrix), None, target_epoch_mask[data_mask>0], None, l2_vector, thread_num, train_mask)
                fit_flux = np.dot(predictor_matrix, result)
                fit_image[target_epoch_mask>0,i] = fit_flux[:,0]
                dif[target_epoch_mask>0, i] = flux- fit_flux[:,0]
                #gc.collect()
            else:
                fit_image[epoch_mask>0,i] = -1
                dif[epoch_mask>0,i] = np.nan
            if i%2000 == 0:
                np.save(out_name, fit_image)
                np.save(out_name_dif, dif)
            i+=1
        np.save(out_name, fit_image)
        np.save(out_name_dif, dif)

#pca
    if True:
        channel = int(sys.argv[1])
        pixel_file = sys.argv[2]
        train_start = int(sys.argv[3])
        train_end = int(sys.argv[4])
        output_name = sys.argv[5]
        if len(sys.argv)>=6:
            pixel_file = sys.argv[5]        
            print 'pixel file:\n%s'%pixel_file
        else:
            pixel_file = None
            print 'run full image'
        name = '/Users/dunwang/Desktop/fits/ch%d_image.fits'%channel
        out_name_fit = '/Users/dunwang/Desktop/new_pixel/pca_fit/fit_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)
        out_name_dif = '/Users/dunwang/Desktop/new_pixel/pca_fit/dif_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)

        print 'train:%d-%d'%(train_start,train_end)

        var_mask = np.load('/Users/dunwang/Desktop/new_pixel/var_mask_ch%d.npy'%channel)

        l2 = 0#1e6
        num_predictor = 1600#1600
        thread_num = 1

        ffi, kplr_mask, epoch_mask = load_ffi(name)
        print ffi.shape
        #start = np.argwhere(idx[kplr_mask.flatten()>0]==start)
        #end = np.argwhere(idx[kplr_mask.flatten()>0]==end)
        '''
        plt.imshow(kplr_mask, interpolation='None', cmap=plt.get_cmap('Greys'))
        plt.show()


        target_x,target_y = get_xy(10000, kplr_mask)
        print target_x, target_y
        predictor_matrix = get_predictor_matrix_ffi(ffi, target_x, target_y, kplr_mask, num_predictor)
        '''

        train_mask = np.ones(ffi.shape[0], dtype=int)
        train_mask[train_start:train_end] = 0 

        gain_factor = 115.49
        if pixel_file is not None:
            pixel_list = np.loadtxt(pixel_file, dtype=int)


            fit_image = np.zeros((ffi.shape[0], pixel_list.shape[0]), dtype=float)
            dif = np.zeros((ffi.shape[0], pixel_list.shape[0]), dtype=float)
            i = 0
            for pixel in pixel_list:
                target_x,target_y = pixel[0], pixel[1]#get_xy(i, kplr_mask)
                print target_x, target_y
                if kplr_mask[target_x, target_y] > 0:
                    pixel_mask = np.zeros((1024,1100), dtype=int)
                    pixel_mask[target_x, target_y] = 1
                    pixel_mask = pixel_mask[kplr_mask>0]
                    target_flux = ffi[:,pixel_mask>0][:,0].astype(float)*gain_factor
                    predictor_epoch_mask = np.ones(epoch_mask.shape[0])
                    transit_mask = None
                    #predictor_matrix = get_predictor_matrix_ffi(ffi, target_x, target_y, kplr_mask, num_predictor)
                    predictor_matrix = get_predictor_matrix_ffi(ffi, target_x, target_y, kplr_mask, num_predictor, var_mask)*gain_factor
                    pca = PCA(n_components=200)
                    pca.fit(predictor_matrix)
                    predictor_matrix = pca.transform(predictor_matrix)

                    flux, predictor_matrix, flux_err, l2_vector, target_epoch_mask, data_mask \
                        = get_fit_matrix_ffi(target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2, 0)
                    result = fit_target_no_train(flux, kplr_mask, np.copy(predictor_matrix), None, target_epoch_mask[data_mask>0], None, l2_vector, thread_num, train_mask)
                    print result[-1]
                    fit_flux = np.dot(predictor_matrix, result)
                    fit_image[target_epoch_mask>0,i] = fit_flux[:,0]
                    dif[target_epoch_mask>0, i] = flux- fit_flux[:,0]
                    #gc.collect()
                else:
                    fit_image[epoch_mask>0,i] = -1
                    dif[epoch_mask>0,i] = np.nan
                if i%2000 == 0:
                    np.save(out_name_fit, fit_image)
                    np.save(out_name_dif, dif)
                i+=1
        else:
            fit_image = np.zeros((ffi.shape[0], kplr_mask.shape[0], kplr_mask.shape[1]), dtype=np.float32)-1
            predictor_epoch_mask = np.ones(epoch_mask.shape[0])
            transit_mask = None
            predictor_matrix = get_predictor_matrix_pca(num_predictor)

            flux, predictor_matrix, flux_err, l2_vector, target_epoch_mask, data_mask \
                = get_fit_matrix_ffi(ffi, epoch_mask, predictor_matrix, predictor_epoch_mask, l2, 0)
            result = fit_target_no_train(flux, kplr_mask, np.copy(predictor_matrix), None, target_epoch_mask[data_mask>0], None, l2_vector, thread_num, train_mask)
            fit_image[:,kplr_mask>0] = np.float32(np.dot(predictor_matrix, result))
            np.save(out_name, fit_image)
            gc.collect()
            dif = np.zeros((ffi.shape[0], kplr_mask.shape[0], kplr_mask.shape[1]), dtype=np.float32)+np.nan
            dif[:,:,:] = np.nan
            dif[:,kplr_mask>0] = np.float32(ffi-fit_image[:,kplr_mask>0])
            np.save(out_name_dif, dif)

        np.save(out_name_fit, fit_image)
        np.save(out_name_dif, dif)

#simulteanous
    if False:
        channel = int(sys.argv[1])
        pixel_file = sys.argv[2]
        train_start = int(sys.argv[3])
        train_end = int(sys.argv[4])
        output_name = sys.argv[5]
        u_min = float(sys.argv[6])
        t_0 = float(sys.argv[7])
        t_E = float(sys.argv[8])

        name = '/Users/dunwang/Desktop/fits/ch%d_image.fits'%channel
        out_name_error = '/Users/dunwang/Desktop/new_pixel/pca_fit/error_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)
        out_name_dif = '/Users/dunwang/Desktop/new_pixel/pca_fit/dif_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)
        out_name_para = '/Users/dunwang/Desktop/new_pixel/pca_fit/para_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)
        out_name_coe = '/Users/dunwang/Desktop/new_pixel/pca_fit/coe_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)

        initial = [u_min, t_0, t_E]
        print 'pixel file:\n%s'%pixel_file
        print 'train:%d-%d'%(train_start,train_end)
        print 'initial:%f %f %f'%(u_min, t_0, t_E)

        var_mask = np.load('/Users/dunwang/Desktop/new_pixel/var_mask_ch%d.npy'%channel)

        pixel_list = np.loadtxt(pixel_file, dtype=int)#[12:13]
        print pixel_list
        l2 = 0#1e5#1e6
        num_predictor = 1600
        thread_num = 1

        ffi, kplr_mask, epoch_mask = load_ffi(name)
        print ffi.shape
        #start = np.argwhere(idx[kplr_mask.flatten()>0]==start)
        #end = np.argwhere(idx[kplr_mask.flatten()>0]==end)
        '''
        plt.imshow(kplr_mask, interpolation='None', cmap=plt.get_cmap('Greys'))
        plt.show()


        target_x,target_y = get_xy(10000, kplr_mask)
        print target_x, target_y
        predictor_matrix = get_predictor_matrix_ffi(ffi, target_x, target_y, kplr_mask, num_predictor)
        '''

        t = np.load('/Users/dunwang/Desktop/new_pixel/time.npy')

        train_mask = np.ones(ffi.shape[0], dtype=int)
        train_mask[train_start:train_end] = 0 
        fit_image = np.zeros((ffi.shape[0], pixel_list.shape[0]), dtype=float)
        dif = np.zeros((ffi.shape[0], pixel_list.shape[0]), dtype=float)
        error = np.zeros((ffi.shape[0], pixel_list.shape[0]), dtype=float)
        para = np.zeros((pixel_list.shape[0], 3), dtype=float)
        coe = np.zeros(pixel_list.shape[0], dtype=float)

        gain_factor = 115.49

        i = 0
        for pixel in pixel_list:
            target_x,target_y = pixel[0], pixel[1]#get_xy(i, kplr_mask)
            print target_x, target_y
            if kplr_mask[target_x, target_y] > 0:
                pixel_mask = np.zeros((1024,1100), dtype=int)
                pixel_mask[target_x, target_y] = 1
                pixel_mask = pixel_mask[kplr_mask>0]
                target_flux = ffi[:,pixel_mask>0][:,0].astype(float)*gain_factor
                predictor_epoch_mask = np.ones(epoch_mask.shape[0])
                transit_mask = None
                predictor_matrix = get_predictor_matrix_ffi(ffi, target_x, target_y, kplr_mask, num_predictor, var_mask)*gain_factor
                pca = PCA(n_components=200)
                pca.fit(predictor_matrix)
                predictor_matrix = pca.transform(predictor_matrix)
                print initial
                res = minimize(objective, initial, args=(t, kplr_mask, target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2), bounds=[(0.0001, 3.), (2457500., 2457530.), (0.1, 1000.)])
                print res.message
                #np.save('/scratch/dw1519/k2c9/data/ffi/kplr.geert.io/k2-c9a-quicklook-v20160521/new_pixel/res_%s_%d-%d.npy'%(output_name, target_x, target_y), res.x)
                u_min, t_0, t_E = res.x[0], res.x[1], res.x[2]
                print u_min, t_0, t_E
                ml = np.array([V(u_min, t_0, t_E, t)]).T

                flux, predictor_matrix, flux_err, l2_vector, target_epoch_mask, data_mask \
                        = get_fit_matrix_ffi(target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2, 0, 'lightcurve', ml)
                result = fit_target_no_train(flux, kplr_mask, np.copy(predictor_matrix), None, target_epoch_mask[data_mask>0], None, l2_vector, thread_num)
                fit_flux = np.dot(predictor_matrix, result)[:,0]
                cpm_flux = np.dot(predictor_matrix[:,:-1], result[:-1])[:,0]
                dif_cpm = flux-cpm_flux
                dif[target_epoch_mask>0,i] = dif_cpm
                error[target_epoch_mask>0,i] = flux-fit_flux
                print np.sum(np.square(flux-fit_flux))
                para[i] = res.x
                coe[i] = result[-1]
            else:
                fit_image[epoch_mask>0,i] = -1
                dif[epoch_mask>0,i] = np.nan
            if i%2000 == 0:
                #np.save(out_name, fit_image)
                np.save(out_name_dif, dif)
            i+=1
        np.save(out_name_error, error)
        np.save(out_name_dif, dif)
        np.save(out_name_para, para)
        np.save(out_name_coe, coe)


#simulteanous co-add
    if False:
        channel = int(sys.argv[1])
        pixel_file = sys.argv[2]
        train_start = int(sys.argv[3])
        train_end = int(sys.argv[4])
        output_name = sys.argv[5]
        u_min = float(sys.argv[6])
        t_0 = float(sys.argv[7])
        t_E = float(sys.argv[8])

        name = '/Users/dunwang/Desktop/fits/ch%d_image.fits'%channel
        out_name_error = '/Users/dunwang/Desktop/new_pixel/pca_fit/error_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)
        out_name_dif = '/Users/dunwang/Desktop/new_pixel/pca_fit/dif_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)
        out_name_para = '/Users/dunwang/Desktop/new_pixel/pca_fit/para_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)
        out_name_coe = '/Users/dunwang/Desktop/new_pixel/pca_fit/coe_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)

        initial = [u_min, t_0, t_E]
        print 'pixel file:\n%s'%pixel_file
        print 'train:%d-%d'%(train_start,train_end)
        print 'initial:%f %f %f'%(u_min, t_0, t_E)

        var_mask = np.load('/Users/dunwang/Desktop/new_pixel/var_mask_ch%d.npy'%channel)

        pixel_list = np.loadtxt(pixel_file, dtype=int)[12:13]
        #aperture = np.zeros((5,5))
        #aperture[1:4,1:4] = 1
        #aperture = aperture.flatten()
        #pixel_list = pixel_list[aperture>0] 
        print pixel_list
        l2 = 0#1e5#1e6
        num_predictor = 1600
        thread_num = 1

        ffi, kplr_mask, epoch_mask = load_ffi(name)
        print ffi.shape
        #start = np.argwhere(idx[kplr_mask.flatten()>0]==start)
        #end = np.argwhere(idx[kplr_mask.flatten()>0]==end)
        '''
        plt.imshow(kplr_mask, interpolation='None', cmap=plt.get_cmap('Greys'))
        plt.show()


        target_x,target_y = get_xy(10000, kplr_mask)
        print target_x, target_y
        predictor_matrix = get_predictor_matrix_ffi(ffi, target_x, target_y, kplr_mask, num_predictor)
        '''

        t = np.load('/Users/dunwang/Desktop/new_pixel/time.npy')

        train_mask = np.ones(ffi.shape[0], dtype=int)
        train_mask[train_start:train_end] = 0 
        fit_image = np.zeros((ffi.shape[0], pixel_list.shape[0]), dtype=float)
        dif = np.zeros((ffi.shape[0], pixel_list.shape[0]), dtype=float)
        error = np.zeros((ffi.shape[0], pixel_list.shape[0]), dtype=float)
        para = np.zeros((pixel_list.shape[0], 3), dtype=float)
        coe = np.zeros(pixel_list.shape[0], dtype=float)
        i = 0
        total_flux = np.zeros(ffi.shape[0])
        for pixel in pixel_list:
            target_x,target_y = pixel[0], pixel[1]#get_xy(i, kplr_mask)
            print target_x, target_y
            if kplr_mask[target_x, target_y] > 0:
                pixel_mask = np.zeros((1024,1100), dtype=int)
                pixel_mask[target_x-1:target_x+2, target_y-1:target_y+2] = 1
                pixel_mask = pixel_mask[kplr_mask>0]
                target_flux = ffi[:,pixel_mask>0].astype(float)
                target_flux = np.sum(target_flux, axis=1)
                '''
                plt.plot(target_flux, '.k')
                plt.show()
                '''
                predictor_epoch_mask = np.ones(epoch_mask.shape[0])
                transit_mask = None
                predictor_matrix = get_predictor_matrix_ffi(ffi, target_x, target_y, kplr_mask, num_predictor, var_mask)
                pca = PCA(n_components=100)
                pca.fit(predictor_matrix)
                predictor_matrix = pca.transform(predictor_matrix)
                print initial
                res = minimize(objective, initial, args=(t, kplr_mask, target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2), bounds=[(0.0001, 3.), (2457500., 2457530.), (0.1, 1000.)])
                print res.message
                #np.save('/scratch/dw1519/k2c9/data/ffi/kplr.geert.io/k2-c9a-quicklook-v20160521/new_pixel/res_%s_%d-%d.npy'%(output_name, target_x, target_y), res.x)
                u_min, t_0, t_E = res.x[0], res.x[1], res.x[2]
                print u_min, t_0, t_E
                ml = np.array([V(u_min, t_0, t_E, t)]).T

                flux, predictor_matrix, flux_err, l2_vector, target_epoch_mask, data_mask \
                        = get_fit_matrix_ffi(target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2, 0, 'lightcurve', ml)
                result = fit_target_no_train(flux, kplr_mask, np.copy(predictor_matrix), None, target_epoch_mask[data_mask>0], None, l2_vector, thread_num)

                fit_flux = np.dot(predictor_matrix, result)[:,0]
                cpm_flux = np.dot(predictor_matrix[:,:-1], result[:-1])[:,0]
                dif_cpm = flux-cpm_flux
                dif[target_epoch_mask>0,i] = dif_cpm
                error[target_epoch_mask>0,i] = flux-fit_flux
                print np.sum(np.square(flux-fit_flux))

                #fit_flux = np.sum(np.dot(predictor_matrix, result), axis=1)
                #print np.sum(np.square(np.sum(flux, axis=1)-fit_flux))
                para[i] = res.x
                coe[i] = result[-1]
            else:
                fit_image[epoch_mask>0,i] = -1
                dif[epoch_mask>0,i] = np.nan
            if i%2000 == 0:
                #np.save(out_name, fit_image)
                np.save(out_name_dif, dif)
            i+=1
        np.save(out_name_error, error)
        np.save(out_name_dif, dif)
        np.save(out_name_para, para)
        np.save(out_name_coe, coe)

    if False:
        channel = int(sys.argv[1])
        train_start = int(sys.argv[2])
        train_end = int(sys.argv[3])
        output_name = sys.argv[4]
        if len(sys.argv)>=6:
            pixel_file = sys.argv[5]        
            print 'pixel file:\n%s'%pixel_file
        else:
            pixel_file = None
            print 'run full image'
        name = '/scratch/dw1519/k2c9/data/ffi/kplr.geert.io/k2-c9a-quicklook-v20160521/ch%d.npy'%channel
        out_name = '/scratch/dw1519/k2c9/data/ffi/kplr.geert.io/k2-c9a-quicklook-v20160521/new_pixel/fit_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)
        out_name_dif = '/scratch/dw1519/k2c9/data/ffi/kplr.geert.io/k2-c9a-quicklook-v20160521/new_pixel/dif_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)

        print 'train:%d-%d'%(train_start,train_end)

        l2 = 0#1e6
        num_predictor = 400#1600
        thread_num = 1

        ffi, kplr_mask, epoch_mask = load_ffi(name)
        print ffi.shape
        #start = np.argwhere(idx[kplr_mask.flatten()>0]==start)
        #end = np.argwhere(idx[kplr_mask.flatten()>0]==end)
        '''
        plt.imshow(kplr_mask, interpolation='None', cmap=plt.get_cmap('Greys'))
        plt.show()


        target_x,target_y = get_xy(10000, kplr_mask)
        print target_x, target_y
        predictor_matrix = get_predictor_matrix_ffi(ffi, target_x, target_y, kplr_mask, num_predictor)
        '''

        train_mask = np.ones(ffi.shape[0], dtype=int)
        train_mask[train_start:train_end] = 0 

        t = np.load('/scratch/dw1519/k2c9/data/ffi/kplr.geert.io/k2-c9a-quicklook-v20160521/time.npy')

        if pixel_file is not None:
            pixel_list = np.loadtxt(pixel_file, dtype=int)


            fit_image = np.zeros((ffi.shape[0], pixel_list.shape[0]), dtype=float)
            dif = np.zeros((ffi.shape[0], pixel_list.shape[0]), dtype=float)
            i = 0
            for pixel in pixel_list:
                target_x,target_y = pixel[0], pixel[1]#get_xy(i, kplr_mask)
                print target_x, target_y
                if kplr_mask[target_x, target_y] > 0:
                    pixel_mask = np.zeros((1024,1100), dtype=int)
                    pixel_mask[target_x, target_y] = 1
                    pixel_mask = pixel_mask[kplr_mask>0]
                    target_flux = ffi[:,pixel_mask>0][:,0].astype(float)
                    predictor_epoch_mask = np.ones(epoch_mask.shape[0])
                    transit_mask = None
                    #predictor_matrix = get_predictor_matrix_ffi(ffi, target_x, target_y, kplr_mask, num_predictor)
                    predictor_matrix = get_predictor_matrix_pca(num_predictor)

                    res = minimize(objective, [0.1,2457512.5,6.], args=(t, kplr_mask, target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2), method='L-BFGS-B', bounds=[(0., 1.), (2457500., 2457525.), (0.1, 100.)])
                    print res.message
                    print res.success
                    np.save('/scratch/dw1519/k2c9/data/ffi/kplr.geert.io/k2-c9a-quicklook-v20160521/new_pixel/res_moa_%d-%d.npy'%(target_x, target_y), res.x)
                    u_min, t_0, t_E = res.x[0], res.x[1], res.x[2]
                    print u_min, t_0, t_E
                    ml = np.array([V(u_min, t_0, t_E, t)]).T*10.

                    flux, predictor_matrix, flux_err, l2_vector, target_epoch_mask, data_mask \
                            = get_fit_matrix_ffi(target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2, 0, 'lightcurve', ml)

                    result = fit_target_no_train(flux, kplr_mask, np.copy(predictor_matrix), None, target_epoch_mask[data_mask>0], None, l2_vector, thread_num)
                    cpm_flux = np.dot(predictor_matrix[:,:-1], result[:-1])[:,0]
                    dif_cpm = flux-cpm_flux
                    dif[target_epoch_mask>0,i] = dif_cpm
                i+=1
        np.save(out_name_dif, dif)
