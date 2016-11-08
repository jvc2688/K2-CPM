from astropy.io import fits as pyfits
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
import argparse
#from images2gif import writeGif
#from PIL import Image
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, FK4, FK5
from astropy.coordinates import Angle, Latitude, Longitude
from astropy.wcs import WCS
from sklearn import linear_model
from sklearn.datasets import load_iris
import gc
from scipy.optimize import minimize
from sklearn.decomposition import PCA


sap_style = dict(color='w', linestyle='', marker='.', markersize=2, markerfacecolor='k', markeredgecolor='k', markevery=None)
cpm_prediction_style = dict(color='r', ls='-', lw=1, alpha=0.8)
cpm_style = dict(color='w', linestyle='', marker='.', markersize=2, markerfacecolor='r', markeredgecolor='r', markevery=None)
fit_prediction_style = dict(color='g', linestyle='-', lw=1, markersize=2, markerfacecolor='g', markeredgecolor='g', markevery=None)
fit_style = dict(color='w', linestyle='', marker='+', markersize=2, markerfacecolor='g', markeredgecolor='g', markevery=None)
best_prediction_style = dict(color='g', linestyle='-', marker='+', lw=1, markersize=1.5, markerfacecolor='g', markeredgecolor='g', markevery=None)
best_style = dict(color='w', linestyle='', marker='.', markersize=2, markerfacecolor='g', markeredgecolor='g', markevery=None)


#help function to find the pixel mask
def get_pixel_mask(flux, kplr_mask):
    pixel_mask = np.isfinite(flux) & (flux!=0)
    pixel_mask[:, kplr_mask < 1] = False
    '''
    pixel_mask = np.zeros(flux.shape)
    pixel_mask[np.isfinite(flux)] = 1 # okay if finite
    pixel_mask[:, (kplr_mask < 1)] = 0 # unless masked by kplr
    pixel_mask[flux==0] = 0
    '''
    return pixel_mask

#help function to find the epoch mask
def get_epoch_mask(pixel_mask, time, q):
    foo = np.sum(np.sum((pixel_mask > 0), axis=2), axis=1)
    epoch_mask = (foo>0) & np.isfinite(time) & q 
    return epoch_mask

#help function to load header from tpf
def load_header(tpf):
    with pyfits.open(tpf) as file:
        meta = file[1].header
        column = meta['1CRV4P']
        row = meta['2CRV4P']
    return row,column

#help function to load data from tpf
class tpf():
    def __init__(self, fn):
        # Load the data.
        hdu_list = pyfits.open(fn)
        data = hdu_list[1].data
        self.kid = hdu_list[0].header['KEPLERID']
        self.campaign = hdu_list[0].header['CAMPAIGN']
        self.channel = hdu_list[0].header['CHANNEL']
        self.kplr_mask = hdu_list[2].data
        self.wcs = WCS(hdu_list[2].header)
        self.ra = hdu_list[2].header['RA_OBJ']
        self.dec = hdu_list[2].header['DEC_OBJ']        
        self.ref_col = hdu_list[2].header['CRVAL1P']
        self.ref_row = hdu_list[2].header['CRVAL2P']
        self.col = np.tile(np.arange(self.kplr_mask.shape[1], dtype=int), self.kplr_mask.shape[0]) + self.ref_col
        self.col = self.col[self.kplr_mask.flatten()>0]
        self.row = np.repeat(np.arange(self.kplr_mask.shape[0], dtype=int), self.kplr_mask.shape[1]) + self.ref_row
        self.row = self.row[self.kplr_mask.flatten()>0]
        self.time = data["time"]+2454833
        self.quality = np.array(data["quality"], dtype=int)

        flux = data["flux"]
        flux_err = data["flux_err"]
        
        q = data["quality"]
        q = ((q == 0) | (q == 16384) | (q == 8192) | (q == 24576))
        self.pixel_mask = get_pixel_mask(flux, self.kplr_mask)
        self.epoch_mask = get_epoch_mask(self.pixel_mask, self.time, q)

        flux = flux[:, self.kplr_mask>0]
        flux_err = flux_err[:, self.kplr_mask>0]
        self.median = np.median(flux, axis=0)
        index = np.arange(flux.shape[0])
        for i in range(flux.shape[1]):
            interMask = np.isfinite(flux[:,i])
            flux[~interMask,i] = np.interp(index[~interMask], index[interMask], flux[interMask,i])
            flux_err[~interMask,i] = np.inf

        self.flux = flux#.astype(float)
        self.flux_err = flux_err#.astype(float)

        self.tpfs = None

    def get_index(self, x, y):
        index = np.arange(self.row.shape[0])
        index_mask = ((self.row-self.ref_row)==x) & ((self.col-self.ref_col)==y)
        try:
            return index[index_mask][0]
        except IndexError:
            print("No data for (%d,%d)"%(x,y))

    def get_xy(self, i):
        try:
            xy = (self.row[i]-self.ref_row, self.col[i]-self.ref_col)
        except IndexError:
            print("index out of range")
        return xy


    def __eq__(self, other): 
        return (self.kid == other.kid) & (self.campaign == other.campaign)

    def __hash__(self):
        return hash((self.kid, self.campaign))

    def get_predictor_matrix(self, target_x, target_y, num, dis=16, excl=5, flux_lim=(0.8, 1.2), tpfs=None, var_mask=None):

        if tpfs==None:
            tpfs = set([self])
        else:
            tpfs = set(tpfs)
            tpfs.add(self)

        if self.tpfs == tpfs:
            print 'the same'
        else:
            print 'different'
            self.tpfs = tpfs
            pixel_row = []
            pixel_col = []
            pixel_median = []
            pixel_flux = []
            self.predictor_epoch_mask = np.ones_like(self.epoch_mask, dtype=bool)
            for tpf in tpfs:
                pixel_row.append(tpf.row)
                pixel_col.append(tpf.col)
                pixel_median.append(tpf.median)
                pixel_flux.append(tpf.flux)
                self.predictor_epoch_mask &= tpf.epoch_mask
            self.pixel_row = np.concatenate(pixel_row, axis=0).astype(int)
            self.pixel_col = np.concatenate(pixel_col, axis=0).astype(int)
            self.pixel_median = np.concatenate(pixel_median, axis=0)
            self.pixel_flux = np.concatenate(pixel_flux, axis=1)

        target_index = self.get_index(target_x, target_y)
        pixel_mask = np.ones_like(self.pixel_row, dtype=bool)

        target_col = self.col[target_index]
        target_row = self.row[target_index]
        for pix in range(-excl, excl+1):
            pixel_mask &= (self.pixel_row != (target_row+pix))
            pixel_mask &= (self.pixel_col != (target_col+pix))

        pixel_mask &= ((self.median[target_index]*flux_lim[0]<=self.pixel_median) & (self.median[target_index]*flux_lim[1]>=self.pixel_median))

        distance = np.square(self.pixel_row[pixel_mask]-target_row)+np.square(self.pixel_col[pixel_mask]-target_col)
        dis_mask = distance>dis**2
        distance = distance[dis_mask]

        index = np.argsort(distance)

        pixel_flux = self.pixel_flux[:,pixel_mask][:,dis_mask]
        predictor_flux = pixel_flux[:,index[:num]].astype(float)
        #print predictor_flux.shape

        return predictor_flux, self.predictor_epoch_mask

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


def get_predictor_matrix_ffi(ffi, x, y, kplr_mask, num, excl=16, var_mask=None):
    x_lim = kplr_mask.shape[0]
    y_lim = kplr_mask.shape[1]

    predictor_mask = np.ones_like(kplr_mask, dtype=int)

    if x<=excl:
        predictor_mask[:x+excl+1,:] = 0
    elif x>=x_lim-excl-1:
        predictor_mask[x-excl:,:] = 0
    else:
        predictor_mask[x-excl:x+excl+1,:] = 0

    if y<=excl:
        predictor_mask[:,:y+excl+1] = 0
    elif y>=y_lim-excl-1:
        predictor_mask[:,y-excl:] = 0
    else:
        predictor_mask[:,y-excl:y+excl+1] = 0

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
    #predictor_mask[index[0:num]] = 0
    #predictor_mask[index[2*num:]] = 0

    #plt.imshow(predictor_mask.reshape((1024,1100)), interpolation='None', cmap=plt.get_cmap('Greys'))
    #plt.show()

    predictor_mask = predictor_mask[kplr_mask.flatten()>0]

    predictor_matrix = ffi[:, predictor_mask>0].astype(float)

    return predictor_matrix

def get_predictor_matrix_ffi_bright(ffi, x, y, kplr_mask, num, excl=16, var_mask=None):
    x_lim = kplr_mask.shape[0]
    y_lim = kplr_mask.shape[1]
    predictor_mask = np.ones_like(kplr_mask, dtype=int)

    if x<=excl:
        predictor_mask[:x+excl+1,:] = 0
    elif x>=x_lim-excl-1:
        predictor_mask[x-excl:,:] = 0
    else:
        predictor_mask[x-excl:x+excl+1,:] = 0

    if y<=excl:
        predictor_mask[:,:y+excl+1] = 0
    elif y>=y_lim-excl-1:
        predictor_mask[:,y-excl:] = 0
    else:
        predictor_mask[:,y-excl:y+excl+1] = 0

    if var_mask is None:
        pixel_mask = np.logical_and(kplr_mask>0, predictor_mask>0)
    else:
        pixel_mask = np.logical_and(np.logical_and(kplr_mask>0, predictor_mask>0), var_mask<1)

    pixel_mask = pixel_mask[kplr_mask>0]

    target_mask = np.zeros((1024,1100))
    target_mask[x, y] = 1
    target_mask = target_mask[kplr_mask>0]

    med_flux = np.median(ffi, axis=0)
    flux_dif = np.absolute(med_flux-med_flux[target_mask>0])
    flux_dif[pixel_mask] = 99999999

    index = np.argsort(flux_dif)
    predictor_mask = predictor_mask.flatten()
    predictor_mask[index[num:]] = 0

    predictor_mask = predictor_mask[kplr_mask.flatten()>0]

    predictor_matrix = ffi[:, predictor_mask>0].astype(float)

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

    #series = []
    #time = np.arange(predictor_matrix.shape[0]).astype(float)
    #for n in range(1,51):
    #    series.append(np.cos(time/time[-1]*np.pi*2.*n))
    #series= np.array(series).T
    #plt.plot(series[:,0])
    #plt.show()
    #predictor_matrix = np.concatenate((predictor_matrix, series), axis=1)

    #add polynomial terms
    if poly is not None:
        nor_time = np.arange(predictor_matrix.shape[0])
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

class Image:
    def __init__(self, data_file):
        self.ffi, self.kplr_mask, self.epoch_mask = load_ffi(data_file)
        self.med_flux = np.median(self.ffi, axis=0)
        gc.collect()
        self.data_len = self.ffi.shape[0]
        print 'ffi shape:'
        print self.ffi.shape

    def get_predictor_matrix_ffi_bright(self, x, y, num, var_mask=None):
        x_lim = self.kplr_mask.shape[0]
        y_lim = self.kplr_mask.shape[1]
        kplr_mask = self.kplr_mask
        med_flux = self.med_flux
        ffi = self.ffi
        excl = 16
        predictor_mask = np.ones_like(kplr_mask, dtype=int)

        if x<=excl:
            predictor_mask[:x+excl+1,:] = 0
        elif x>=x_lim-excl-1:
            predictor_mask[x-excl:,:] = 0
        else:
            predictor_mask[x-excl:x+excl+1,:] = 0

        if y<=excl:
            predictor_mask[:,:y+excl+1] = 0
        elif y>=y_lim-excl-1:
            predictor_mask[:,y-excl:] = 0
        else:
            predictor_mask[:,y-excl:y+excl+1] = 0

        if var_mask is None:
            pixel_mask = np.logical_and(kplr_mask>0, predictor_mask>0)
        else:
            pixel_mask = np.logical_and(np.logical_and(kplr_mask>0, predictor_mask>0), var_mask<1)

        predictor_mask[np.logical_not(pixel_mask)] = 0
        pixel_mask = pixel_mask[kplr_mask>0]

        target_mask = np.zeros((x_lim,y_lim))
        target_mask[x, y] = 1
        target_mask = target_mask[kplr_mask>0]

        #med_flux = np.median(ffi, axis=0)
        flux_dif = np.absolute(med_flux-med_flux[target_mask>0])
        flux_dif[np.logical_not(pixel_mask)] = 99999999

        index = np.argsort(flux_dif)
        predictor_mask = predictor_mask.flatten()

        '''
        predictor_mask = predictor_mask.reshape((x_lim,y_lim))
        print predictor_mask.shape
        plt.imshow(predictor_mask, interpolation='None', cmap=plt.get_cmap('Greys'))
        plt.colorbar()
        plt.show()
        '''

        predictor_mask = predictor_mask[kplr_mask.flatten()>0]
        predictor_mask[index[num:]] = 0

        predictor_matrix = ffi[:, predictor_mask>0].astype(float)
        print 'predictor matrix:'
        print predictor_matrix.shape

        return predictor_matrix

    def get_predictor_matrix_ffi(self, x, y, num, var_mask=None):
        kplr_mask = self.kplr_mask
        ffi = self.ffi
        x_lim = kplr_mask.shape[0]
        y_lim = kplr_mask.shape[1]
        excl = 16
        predictor_mask = np.ones_like(kplr_mask, dtype=int)

        if x<=excl:
            predictor_mask[:x+excl+1,:] = 0
        elif x>=x_lim-excl-1:
            predictor_mask[x-excl:,:] = 0
        else:
            predictor_mask[x-excl:x+excl+1,:] = 0

        if y<=excl:
            predictor_mask[:,:y+excl+1] = 0
        elif y>=y_lim-excl-1:
            predictor_mask[:,y-excl:] = 0
        else:
            predictor_mask[:,y-excl:y+excl+1] = 0

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

        predictor_mask = predictor_mask[kplr_mask.flatten()>0]
        predictor_matrix = ffi[:, predictor_mask>0].astype(float)

        return predictor_matrix

def fit(pixel_file, out_name_fit, out_name_dif, image, ffi, kplr_mask, epoch_mask, l2, num_predictor, use_pca, thread_num=1, train_mask=None, covar=None):
    pixel_list = np.loadtxt(pixel_file, dtype=int)
    var_mask = None
    gain_factor = 1.
    fit_image = np.zeros((ffi.shape[0], pixel_list.shape[0]), dtype=float)
    dif = np.zeros((ffi.shape[0], pixel_list.shape[0]), dtype=float)
    i = 0
    for pixel in pixel_list:
        target_x,target_y = pixel[0], pixel[1]#get_xy(i, kplr_mask)
        #print target_x, target_y
        if kplr_mask[target_x, target_y] > 0:
            pixel_mask = np.zeros((1024,1100), dtype=int)
            pixel_mask[target_x, target_y] = 1
            pixel_mask = pixel_mask[kplr_mask>0]
            target_flux = ffi[:,pixel_mask>0][:,0].astype(float)*gain_factor
            predictor_epoch_mask = np.ones(epoch_mask.shape[0])
            transit_mask = None
            #predictor_matrix = get_predictor_matrix_ffi(ffi, target_x, target_y, kplr_mask, num_predictor)
            predictor_matrix = image.get_predictor_matrix_ffi(target_x, target_y, num_predictor, var_mask)*gain_factor
            #predictor_matrix = np.concatenate([predictor_matrix, image.get_predictor_matrix_ffi_bright(target_x, target_y, num_predictor, var_mask)*gain_factor], axis=1)
            #predictor_matrix = image.get_predictor_matrix_ffi_bright(target_x, target_y, num_predictor, var_mask)*gain_factor
            
            if use_pca==True:
                pca = PCA(n_components=50)
                pca.fit(predictor_matrix)
                predictor_matrix = pca.transform(predictor_matrix)
            step = np.zeros((image.data_len, 1))
            step[300:950,0] = 1.
            step = None

            flux, predictor_matrix, flux_err, l2_vector, target_epoch_mask, data_mask \
                = get_fit_matrix_ffi(target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2, 0, 'lightcurve', step)
            #print predictor_matrix.shape
            result = fit_target_no_train(flux, kplr_mask, np.copy(predictor_matrix), None, target_epoch_mask[data_mask>0], covar, l2_vector, thread_num, train_mask)
            #print result.shape
            #print result[-1], result[-2]

            fit_flux = np.dot(predictor_matrix, result)
            fit_image[target_epoch_mask>0,i] = fit_flux[:,0]
            dif[target_epoch_mask>0, i] = flux- fit_flux[:,0]

            #cpm_fit = np.dot(predictor_matrix[:,:-1], result[:-1])
            #dif[target_epoch_mask>0, i] = flux-cpm_fit[:,0]
            #gc.collect()
        else:
            fit_image[epoch_mask>0,i] = -1
            dif[epoch_mask>0,i] = np.nan
        if i%2000 == 0:
            np.save(out_name_fit, fit_image)
            np.save(out_name_dif, dif)
        i+=1

    np.save(out_name_fit, fit_image)
    np.save(out_name_dif, dif)
    return dif

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
        '''
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("in_file", help="Path to the fits file contains the raw data")
        parser.add_argument("pixel_file", help="Path to the file contains the coordinates of the pxiels need to be processed")
        parser.add_argument("-t", "--test", type=int, nargs=2, help="begin and end of the test window")
        parser.add_argument("-o","--output", help="Path to the output file")
        parser.parse_args()
        '''
        channel = int(sys.argv[1])
        pixel_file = sys.argv[2]
        train_start = int(sys.argv[3])
        train_end = int(sys.argv[4])
        output_name = sys.argv[5]

        name = '/Users/dunwang/Desktop/k2c9b/ch%d_image.fits'%channel
        out_name_fit = '/Users/dunwang/Desktop/k2c9b/pca_fit/fit_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)
        out_name_dif = '/Users/dunwang/Desktop/k2c9b/pca_fit/dif_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)

        print 'train:%d-%d'%(train_start,train_end)

        if os.path.isfile('/Users/dunwang/Desktop/k2c9b/var_mask_ch%d.npy'%channel):
            var_mask = np.load('/Users/dunwang/Desktop/k2c9b/var_mask_ch%d.npy'%channel)
        else:
            var_mask = None

        use_pca = True#False
        l2 = 0
        num_predictor = 800
        thread_num = 1

        image = Image(name)
        ffi = image.ffi
        kplr_mask = image.kplr_mask
        epoch_mask = image.epoch_mask

        train_list = [[0,80],[80,160],[160,240],[240,320],[320,400],[400,480],[480,560],[560,640],[640,720],[720,800],\
                    [800,880],[880,960],[960,1040],[1040,1120],[1120,1200],[1200,1287]]
        #train_list = [[0,160],[160,320],[320,480],[480,640],[640,800],\
        #    [800,960],[960,1120],[1120,1287]]
        #train_list = [[800,1200]]
        train_list = [[0,0]]

        for train_lim in train_list:
            train_start = train_lim[0]
            train_end = train_lim[1]
            out_name_fit = '/Users/dunwang/Desktop/k2c9b/pca_fit/fit_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)
            out_name_dif = '/Users/dunwang/Desktop/k2c9b/pca_fit/dif_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)
            train_mask = np.ones(ffi.shape[0], dtype=int)
            train_mask[train_start:train_end] = 0 

            gain_factor = 1#115.49
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
                        predictor_matrix = image.get_predictor_matrix_ffi(target_x, target_y, num_predictor, var_mask)*gain_factor
                        #predictor_matrix = np.concatenate([predictor_matrix, image.get_predictor_matrix_ffi_bright(target_x, target_y, num_predictor, var_mask)*gain_factor], axis=1)
                        #predictor_matrix = image.get_predictor_matrix_ffi_bright(target_x, target_y, num_predictor, var_mask)*gain_factor
                        
                        if use_pca==True:
                            pca = PCA(n_components=50)
                            pca.fit(predictor_matrix)
                            predictor_matrix = pca.transform(predictor_matrix)
                        step = np.zeros((image.data_len, 1))
                        step[300:950,0] = 1.
                        step = None

                        flux, predictor_matrix, flux_err, l2_vector, target_epoch_mask, data_mask \
                            = get_fit_matrix_ffi(target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2, 0, 'lightcurve', step)
                        print predictor_matrix.shape
                        result = fit_target_no_train(flux, kplr_mask, np.copy(predictor_matrix), None, target_epoch_mask[data_mask>0], None, l2_vector, thread_num, train_mask)
                        print result.shape
                        print result[-1], result[-2]

                        fit_flux = np.dot(predictor_matrix, result)
                        fit_image[target_epoch_mask>0,i] = fit_flux[:,0]
                        dif[target_epoch_mask>0, i] = flux- fit_flux[:,0]

                        #cpm_fit = np.dot(predictor_matrix[:,:-1], result[:-1])
                        #dif[target_epoch_mask>0, i] = flux-cpm_fit[:,0]
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
        out_name_cpm = '/Users/dunwang/Desktop/new_pixel/pca_fit/cpm_ch%d_%d-%d_%s.npy'%(channel, train_start, train_end, output_name)

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
        cpm = np.zeros((ffi.shape[0], pixel_list.shape[0]), dtype=float)

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
                res = minimize(objective, initial, args=(t, kplr_mask, target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2), bounds=[(0.0001, 3.), (2457490., 2457530.), (0.1, 1000.)])
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
                cpm[target_epoch_mask>0,i] = cpm_flux
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
        np.save(out_name_cpm, cpm)
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

        #var_mask = np.load('/Users/dunwang/Desktop/new_pixel/var_mask_ch%d.npy'%channel)
        var_mask = None

        pixel_list = np.loadtxt(pixel_file, dtype=int)[4:5]
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
                pca = PCA(n_components=200)
                pca.fit(predictor_matrix)
                predictor_matrix = pca.transform(predictor_matrix)
                print initial
                res = minimize(objective, initial, args=(t, kplr_mask, target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2), bounds=[(0.0001, 3.), (2457490., 2457530.), (0.1, 1000.)])
                print res.message
                #np.save('/scratch/dw1519/k2c9/data/ffi/kplr.geert.io/k2-c9a-quicklook-v20160521/new_pixel/res_%s_%d-%d.npy'%(output_name, target_x, target_y), res.x)
                u_min, t_0, t_E = res.x[0], res.x[1], res.x[2]
                print u_min, t_0, t_E
                ml = np.array([V(u_min, t_0, t_E, t)]).T

                flux, predictor_matrix, flux_err, l2_vector, target_epoch_mask, data_mask \
                        = get_fit_matrix_ffi(target_flux, epoch_mask, predictor_matrix, predictor_epoch_mask, l2, 0, 'lightcurve', ml)
                result = fit_target_no_train(flux, kplr_mask, np.copy(predictor_matrix), None, target_epoch_mask[data_mask>0], None, l2_vector, thread_num)
                print result[-2], result[-1]
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
