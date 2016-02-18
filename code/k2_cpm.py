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
from images2gif import writeGif
from PIL import Image
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy.coordinates import Angle, Latitude, Longitude


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
        flux = hdu_data["flux"]
        flux_err = hdu_data["flux_err"]

    length = time.shape[0]
    time = time[length/2.:]
    flux = flux[length/2.:]
    flux_err = flux_err[length/2.:]
    
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
def get_predictor_matrix(tpfs, num):
    predictor_flux = []
    predictor_epoch_mask=1.
    for tpf in tpfs:
        time, flux, pixel_mask, kplr_mask, epoch_mask, flux_err, column, row = load_data(tpf)
        predictor_epoch_mask *= epoch_mask
        predictor_flux.append(flux)
    predictor_matrix =  np.concatenate(predictor_flux, axis=1)
    size = predictor_matrix.shape[1]
    sample = np.random.choice(size,num,replace=False)
    predictor_matrix = predictor_matrix[:,sample]
    print('predictor matrix:')
    print(predictor_matrix.shape)

    return predictor_matrix, predictor_epoch_mask

def get_fit_matrix(kid, campaign, target_tpf, target_pixel, l2,  poly=0, auto=False, offset=0, window=0, auto_l2=0, part=None, filter=False, prefix='lightcurve'):
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

def fit_target_no_train(target_flux, target_kplr_mask, predictor_flux_matrix, time, epoch_mask, covar_list, margin, l2_vector=None, thread_num=1, prefix="lightcurve", transit_mask=None):
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
    
    result = lss.linear_least_squares(predictor_flux_matrix, target_flux, covar, l2_vector)
    fit_flux = np.dot(predictor_flux_matrix, result)
    data_group['fit_flux'] = fit_flux
    f.close()


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

if __name__ == "__main__":
    if False:
        tpf = '../data/ktwo200000862-c00_lpd-targ.fits'
        time, flux, pixel_mask, kplr_mask, epoch_mask, flux_err, column, row = load_data(tpf)
        flux = flux.reshape((flux.shape[0],50,50))
        plt.plot(np.arange(epoch_mask.shape[0]), flux[:,35,26],'.k')
        plt.show()
        pixel_plot(time, flux, 'ktwo200000862-c00_lpd-targ', [[20,25],[20,25]])

#train and test
    if False:
        kid = 200000862
        campaign = 0
        l2 = 1e5
        auto_l2 = 0
        auto = False
        poly = 0
        auto_offset = 18
        auto_window = 3
        margin = 12
        thread_num = 3
        part = None
        target_list = [[26,12], [26,14], [28,12], [28,13], [28,14],[25,12],[25,13],[25,14],[35,27], [35,28], [35,26], [34,27], [34,28], [34,26], [36,27], [36,28], [36,26]]
        target_pixel = [27,12]#[25,15]

        for x in range(41,46):
            for y in range(37,42):
                target_pixel = [x,y]
                prefix = '../hdf5/cut/ktwo%d_c%d_pixel(%d,%d)_%.0e'%(kid, campaign, target_pixel[0], target_pixel[1], l2)

                tpf = '../data/ktwo200000862-c00_lpd-targ.fits'

                predictor_flux_matrix, target_flux, covar_list, time, target_kplr_mask, epoch_mask, data_mask, l2_vector, predictor_num, auto_pixel_num = get_fit_matrix(kid, campaign, tpf, target_pixel, l2, poly, auto, auto_offset, auto_window, auto_l2, part, False, prefix)

                transit_mask = None

                fit_target(target_flux, target_kplr_mask, predictor_flux_matrix, time, epoch_mask[data_mask>0], covar_list, margin, l2_vector, thread_num, prefix, transit_mask)


#no train and test
    if False:
        kid = 200000862
        campaign = 0
        l2 = 1e5
        auto_l2 = 0
        auto = False
        poly = 0
        auto_offset = 18
        auto_window = 3
        margin = 36
        thread_num = 3
        part = None
        target_list = [[26,12], [26,14], [28,12], [28,13], [28,14],[25,12],[25,13],[25,14],[35,27], [35,28], [35,26], [34,27], [34,28], [34,26], [36,27], [36,28], [36,26]]
        target_pixel = [27,12]#[25,15]

        for x in range(0,50):
            for y in range(0,50):
                target_pixel = [x,y]
                prefix = '../hdf5/notrain/test/ktwo%d_c%d_pixel(%d,%d)_%.0e'%(kid, campaign, target_pixel[0], target_pixel[1], l2)

                tpf = '../data/ktwo200000862-c00_lpd-targ.fits'

                predictor_flux_matrix, target_flux, covar_list, time, target_kplr_mask, epoch_mask, data_mask, l2_vector, predictor_num, auto_pixel_num = get_fit_matrix(kid, campaign, tpf, target_pixel, l2, poly, auto, auto_offset, auto_window, auto_l2, part, False, prefix)

                transit_mask = None

                #fit_target_no_train(target_flux, target_kplr_mask, predictor_flux_matrix, time, epoch_mask[data_mask>0], covar_list, margin, l2_vector, thread_num, prefix, transit_mask)


#train and test 
    if False:
        kid = 200000862
        campaign = 0
        l2 = 1e5
        auto_l2 = 0
        auto = False
        poly = 0
        auto_offset = 18
        auto_window = 3
        margin = 18
        thread_num = 2
        part = None
        target_pixel = [27,14]#[25,15]

        for i in range(41,46):
            for j in range(37,42):
                target_pixel = [i,j]

                prefix = '../hdf5/cut/ktwo%d_c%d_pixel(%d,%d)_%.0e'%(kid, campaign, target_pixel[0], target_pixel[1], l2)


                f = h5py.File('%s.hdf5'%prefix, 'r')
                cpm_info = f['/cpm_info']
                data_group = f['/data']
                
                kid = f.attrs['kid'][()]
                quarter = f.attrs['campaign'][()]
                target_flux = data_group['target_flux'][:]
                target_kplr_mask = data_group['target_kplr_mask'][:,:]
                epoch_mask = data_group['epoch_mask'][:]
                data_mask = data_group['data_mask'][:]
                time = data_group['time'][:]
                fit_flux = data_group['fit_flux']

                res = (target_flux-fit_flux)
                res_rms = np.sqrt(np.median(res**2))

                f, axes = plt.subplots(2, 1)
                axes[0].plot(time, target_flux, '.k')
                axes[0].plot(time, fit_flux, '-r', linewidth=0.5)
                #axes[3].legend(loc=1, ncol=3, prop={'size':8})
                axes[0].set_ylabel("flux")
                plt.setp( axes[0].get_xticklabels(), visible=False)
                axes[1].plot(time, res, '.k')
                axes[1].set_ylabel("residual")
                axes[1].set_xlabel("time [BKJD]")
                axes[1].axhline(y=0,xmin=0,xmax=5, ls='--', c="r",linewidth=1,zorder=10, alpha=0.25)
                axes[1].axhline(y=res_rms,xmin=0,xmax=5, ls='--', c="r",linewidth=1,zorder=10, alpha=0.25)
                axes[1].axhline(y=-res_rms,xmin=0,xmax=5, ls='--', c="r",linewidth=1,zorder=10, alpha=0.25)
                axes[1].set_ylim(-res_rms*14, res_rms*14)
                plt.tight_layout()
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0, hspace=0)
                plt.suptitle('ktwo%d_c0%d pixel (%d, %d)'%(kid, campaign, target_pixel[0], target_pixel[1]))
                plt.savefig('../plots/cut/ktwo%d_c%d_pixel(%d,%d)_%.0e_res.png'%(kid, campaign, target_pixel[0], target_pixel[1], l2), dpi=190)

                '''
                f, axes = plt.subplots(2, 1)
                axes[0].plot(time, (target_flux/median-1.)*10**6, '.k')
                axes[0].plot(time, (fit_flux/median-1.)*10**6, '-r', linewidth=0.5)
                #axes[3].legend(loc=1, ncol=3, prop={'size':8})
                axes[0].set_ylabel("relative flux [ppm]")
                plt.setp( axes[0].get_xticklabels(), visible=False)
                axes[1].plot(time, (target_flux/fit_flux-1.)*10**6, '.k')
                axes[1].set_ylabel("cpm flux [ppm]")
                axes[1].set_xlabel("time [BKJD]")
                axes[1].set_ylim(-9999, 9900)
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0, hspace=0)

                plt.suptitle('ktwo%d_c0%d pixel (%d, %d)'%(kid, campaign, target_pixel[0], target_pixel[1]))
                plt.savefig('%s_ratio.png'%prefix, dpi=190)
                '''

#no train and test
    if False:
        kid = 200000862
        campaign = 0
        l2 = 1e5
        auto_l2 = 0
        auto = False
        poly = 0
        auto_offset = 18
        auto_window = 3
        margin = 18
        thread_num = 2
        part = None
        target_pixel = [27,14]#[25,15]

        for i in range(0,50):
            for j in range(0,50):
                target_pixel = [i,j]

                prefix = '../hdf5/notrain/ktwo%d_c%d_pixel(%d,%d)_%.0e'%(kid, campaign, target_pixel[0], target_pixel[1], l2)


                f = h5py.File('%s.hdf5'%prefix, 'r')
                cpm_info = f['/cpm_info']
                data_group = f['/data']
                
                kid = f.attrs['kid'][()]
                quarter = f.attrs['campaign'][()]
                target_flux = data_group['target_flux'][:]
                target_kplr_mask = data_group['target_kplr_mask'][:,:]
                epoch_mask = data_group['epoch_mask'][:]
                data_mask = data_group['data_mask'][:]
                time = data_group['time'][:]
                fit_flux = data_group['fit_flux']

                print fit_flux.shape
                print target_flux.shape
                res = (target_flux-fit_flux[:,0])
                res_rms = np.sqrt(np.median(res**2))
                print res.shape

                frame, axes = plt.subplots(2, 1)
                axes[0].plot(time, target_flux, '.k')
                axes[0].plot(time, fit_flux, '-r', linewidth=0.5)
                #axes[3].legend(loc=1, ncol=3, prop={'size':8})
                axes[0].set_ylabel("flux")
                plt.setp( axes[0].get_xticklabels(), visible=False)
                axes[1].plot(time, res, '.k')
                axes[1].set_ylabel("residual")
                axes[1].set_xlabel("time [BKJD]")
                axes[1].axhline(y=0,xmin=0,xmax=5, ls='--', c="r",linewidth=1,zorder=10, alpha=0.25)
                axes[1].axhline(y=res_rms,xmin=0,xmax=5, ls='--', c="r",linewidth=1,zorder=10, alpha=0.25)
                axes[1].axhline(y=-res_rms,xmin=0,xmax=5, ls='--', c="r",linewidth=1,zorder=10, alpha=0.25)
                axes[1].set_ylim(-res_rms*14, res_rms*14)
                plt.tight_layout()
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0, hspace=0)
                plt.suptitle('ktwo%d_c0%d pixel (%d, %d)'%(kid, campaign, target_pixel[0], target_pixel[1]))
                plt.savefig('../plots/notrain/ktwo%d_c%d_pixel(%d,%d)_%.0e_res.png'%(kid, campaign, target_pixel[0], target_pixel[1], l2), dpi=190)


                '''
                f, axes = plt.subplots(2, 1)
                axes[0].plot(time, (target_flux/median-1.)*10**6, '.k')
                axes[0].plot(time, (fit_flux/median-1.)*10**6, '-r', linewidth=0.5)
                #axes[3].legend(loc=1, ncol=3, prop={'size':8})
                axes[0].set_ylabel("relative flux [ppm]")
                plt.setp( axes[0].get_xticklabels(), visible=False)
                axes[1].plot(time, (target_flux/fit_flux-1.)*10**6, '.k')
                axes[1].set_ylabel("cpm flux [ppm]")
                axes[1].set_xlabel("time [BKJD]")
                axes[1].set_ylim(-9999, 9900)
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0, hspace=0)

                plt.suptitle('ktwo%d_c0%d pixel (%d, %d)'%(kid, campaign, target_pixel[0], target_pixel[1]))
                plt.savefig('%s_ratio.png'%prefix, dpi=190)
                '''

                f.close()


    if False:
        kid = 200000862
        campaign = 0
        l2 = 1e5
        auto_l2 = 0
        auto = False
        poly = 0
        auto_offset = 18
        auto_window = 3
        margin = 18
        thread_num = 2
        part = None
        target_pixel = [41,37]#[25,15]

        prefix = '../hdf5/cut/ktwo%d_c%d_pixel(%d,%d)_%.0e'%(kid, campaign, target_pixel[0], target_pixel[1], l2)
        f = h5py.File('%s.hdf5'%prefix, 'r')
        data_group = f['/data']
        epoch_mask = data_group['epoch_mask'][:]
        f.close()

        length = epoch_mask.shape[0]

        image_size = 5
        image = np.zeros((length,image_size,image_size))
        image_data = np.zeros((length,image_size,image_size))
        image_ratio = np.zeros((length,image_size,image_size))
        x0 = 41
        y0 = 37
        for i in range(41,46):
            for j in range(37,42):
                target_pixel = [i,j]

                prefix = '../hdf5/cut/ktwo%d_c%d_pixel(%d,%d)_%.0e'%(kid, campaign, target_pixel[0], target_pixel[1], l2)


                f = h5py.File('%s.hdf5'%prefix, 'r')
                cpm_info = f['/cpm_info']
                data_group = f['/data']
                
                kid = f.attrs['kid'][()]
                quarter = f.attrs['campaign'][()]
                target_flux = data_group['target_flux'][:]
                target_kplr_mask = data_group['target_kplr_mask'][:,:]
                epoch_mask = data_group['epoch_mask'][:]
                data_mask = data_group['data_mask'][:]
                time = data_group['time'][:]
                fit_flux = data_group['fit_flux']

                ratio = (target_flux/fit_flux-1.)*10**6
                res_rms = np.sqrt(np.median(ratio**2))

                res = (target_flux-fit_flux)
                res_rms = np.sqrt(np.median(res**2))

                image[:,i-x0,j-y0][epoch_mask>0] = fit_flux
                image_ratio[:,i-x0,j-y0][epoch_mask>0] = res
                image_data[:,i-x0,j-y0][epoch_mask>0] = target_flux


        np.save('../hdf5/cut/ktwo%d_c%d_p(%d,%d)_pre.npy'%(kid, campaign, x0, y0), image)
        np.save('../hdf5/cut/ktwo%d_c%d_p(%d,%d)_dif_res.npy'%(kid, campaign, x0, y0), image_ratio)
        np.save('../hdf5/cut/ktwo%d_c%d_p(%d,%d)_data.npy'%(kid, campaign, x0, y0), image_data)

#no train
    if False:
        kid = 200000862
        campaign = 0
        l2 = 1e5
        auto_l2 = 0
        auto = False
        poly = 0
        auto_offset = 18
        auto_window = 3
        margin = 18
        thread_num = 2
        part = None
        target_pixel = [27,14]#[25,15]

        prefix = '../hdf5/notrain/row/ktwo%d_c%d_pixel(%d,%d)_%.0e'%(kid, campaign, target_pixel[0], target_pixel[1], l2)
        f = h5py.File('%s.hdf5'%prefix, 'r')
        data_group = f['/data']
        epoch_mask = data_group['epoch_mask'][:]
        f.close()

        length = epoch_mask.shape[0]

        image = np.zeros((length,50,50))
        image_data = np.zeros((length,50,50))
        image_ratio = np.zeros((length,50,50))
        x0 = 0
        y0 = 0
        for i in range(0,50):# range(42,47):
            for j in range(0,50):# range(38,43):
                target_pixel = [i,j]

                prefix = '../hdf5/notrain/row/ktwo%d_c%d_pixel(%d,%d)_%.0e'%(kid, campaign, target_pixel[0], target_pixel[1], l2)


                f = h5py.File('%s.hdf5'%prefix, 'r')
                cpm_info = f['/cpm_info']
                data_group = f['/data']
                
                kid = f.attrs['kid'][()]
                quarter = f.attrs['campaign'][()]
                target_flux = data_group['target_flux'][:]
                target_kplr_mask = data_group['target_kplr_mask'][:,:]
                epoch_mask = data_group['epoch_mask'][:]
                data_mask = data_group['data_mask'][:]
                time = data_group['time'][:]
                fit_flux = data_group['fit_flux'][:,0]

                ratio = (target_flux/fit_flux-1.)*10**6
                res_rms = np.sqrt(np.median(ratio**2))

                res = (target_flux-fit_flux)
                res_rms = np.sqrt(np.median(res**2))


                image[:,i-x0,j-y0][epoch_mask>0] = fit_flux
                image_ratio[:,i-x0,j-y0][epoch_mask>0] = res
                image_data[:,i-x0,j-y0][epoch_mask>0] = target_flux

                f.close()

        np.save('../hdf5/notrain/row/ktwo%d_c%d_p(%d,%d)_pre.npy'%(kid, campaign, x0, y0), image)
        np.save('../hdf5/notrain/row/ktwo%d_c%d_p(%d,%d)_dif_res.npy'%(kid, campaign, x0, y0), image_ratio)
        np.save('../hdf5/notrain/row/ktwo%d_c%d_p(%d,%d)_data.npy'%(kid, campaign, x0, y0), image_data)

    #movie
    if False:
        kid = 200000862
        campaign = 0
        l2 = 1e5
        auto_l2 = 0
        auto = False
        poly = 0
        auto_offset = 18
        auto_window = 3
        margin = 18
        thread_num = 2
        part = None
        x0 = 41
        y0 = 37
        dir_list = ['data', 'pre', 'dif_res']
        #dir_list = [ 'dif']

        for direc in dir_list: 
            filename = '../movie/cut/ktwo%d_c%d_p(%d,%d)/new/frame.png'%(kid, campaign, x0, y0)
            dir = os.path.dirname(filename)
            if not os.path.exists(dir):
                os.makedirs(dir)

        fluxes=[]
        titles = ['data', 'prediction', 'difference']
        i=0
        limit=[0,0]
        for direc in dir_list:
            flux = np.load('../hdf5/cut/ktwo%d_c%d_p(%d,%d)_%s.npy'%(kid, campaign, x0, y0, direc))
            '''
            rms = np.sqrt(np.median(flux**2, axis=0))
            rms = np.max(rms)
            if i==2:
                limit=[0,0]
                limit[0] = -10*rms
                limit[1] = 10*rms
            elif i==0:
                limit[0] = np.min(flux)
                limit[1] = np.max(flux)
            '''
            rms = np.sqrt(np.median(flux**2, axis=0))

            rms = np.percentile(rms, 95)#np.max(rms)
            if i==2:
                limit=[0,0]
                if 10*rms > np.percentile(flux, 99):
                    limit[0] = -np.percentile(flux, 99)
                    limit[1] = np.percentile(flux, 99)
                else:
                    limit[0] = -10*rms
                    limit[1] = 10*rms
            elif i==0:
                limit[0] = np.min(flux)
                limit[1] = np.percentile(flux, 95)#np.max(flux)

            num_frames = flux.shape[0]
            fluxes.append((flux, limit, titles[i]))
            i+=1
        print num_frames
        fig = plt.gcf()
        fig.set_size_inches(12.9,3)
        for i in range(0, num_frames):
            j=1
            for flux,limit,title in fluxes:
                ax = plt.subplot(130+j)
                a = flux[i, np.isfinite(flux[i])]
                if a.shape[0]==0:
                    continue
                mini = limit[0]
                perc = limit[1]
                plt.imshow(flux[i], interpolation='None', cmap=plt.get_cmap('Greys'), vmin=mini, vmax=perc)
                plt.colorbar()
                plt.title(title)
                j+=1
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            fname = '../movie/cut/ktwo%d_c%d_p(%d,%d)/new/_tmp%04d.png' %(kid, campaign, x0, y0, i)
            print('Saving frame', fname)
            plt.savefig(fname)
            plt.clf()


#movie no train 
    if False:
        kid = 200000862
        campaign = 0
        l2 = 1e5
        auto_l2 = 0
        auto = False
        poly = 0
        auto_offset = 18
        auto_window = 3
        margin = 18
        thread_num = 2
        part = None
        x0 = 0
        y0 = 0
        dir_list = ['data', 'pre', 'dif_res']
        #dir_list = [ 'dif']

        for direc in dir_list: 
            filename = '../movie/notrain/row/ktwo%d_c%d_p(%d,%d)/new/frame.png'%(kid, campaign, x0, y0)
            dir = os.path.dirname(filename)
            if not os.path.exists(dir):
                os.makedirs(dir)

        fluxes=[]
        titles = ['data', 'prediction', 'difference']
        i=0
        limit=[0,0]
        for direc in dir_list:
            flux = np.load('../hdf5/notrain/row/ktwo%d_c%d_p(%d,%d)_%s.npy'%(kid, campaign, x0, y0, direc))
            rms = np.sqrt(np.median(flux**2, axis=0))

            rms = np.max(rms) #np.percentile(rms, 95)
            if i==2:
                limit=[0,0]
                '''
                if 10*rms > np.percentile(flux, 99):
                    limit[0] = -np.percentile(flux, 99)
                    limit[1] = np.percentile(flux, 99)
                else:
                '''
                limit[0] = -40 #-10*rms
                limit[1] = 40 #10*rms
            elif i==0:
                limit[0] = np.min(flux)
                limit[1] = np.percentile(flux, 95)#np.max(flux)
            num_frames = flux.shape[0]
            fluxes.append((flux, limit, titles[i]))
            i+=1
        print num_frames
        fig = plt.gcf()
        fig.set_size_inches(12.9,3)
        for i in range(0, num_frames):
            j=1
            for flux,limit,title in fluxes:
                ax = plt.subplot(130+j)
                a = flux[i, np.isfinite(flux[i])]
                if a.shape[0]==0:
                    continue
                mini = limit[0]
                perc = limit[1]
                plt.imshow(flux[i], interpolation='None', cmap=plt.get_cmap('Greys'), vmin=mini, vmax=perc)
                plt.colorbar()
                plt.title(title)
                j+=1
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            fname = '../movie/notrain/row/ktwo%d_c%d_p(%d,%d)/new/_tmp%04d.png' %(kid, campaign, x0, y0, i)
            print('Saving frame', fname)
            plt.savefig(fname)
            plt.clf()



    if False:
        kid = 200000862
        campaign = 0
        l2 = 1e5
        auto_l2 = 0
        auto = False
        poly = 0
        auto_offset = 18
        auto_window = 3
        margin = 18
        thread_num = 2
        part = None
        x0 = 41
        y0 = 37
        dir_list = ['data', 'pre', 'dif']
        dir_list = [ 'dif']
        for direc in dir_list:
            folder = '../movie/cut/ktwo%d_c%d_p(%d,%d)/new/' %(kid, campaign, x0, y0)
            file_names = sorted((fn for fn in os.listdir(folder) if fn.endswith('.png')))[100:300]
            #['animationframa.png', 'animationframb.png', ...] "

            images = [Image.open(folder+fn) for fn in file_names]
            '''
            size = (150,150)
            for im in images:
                im.thumbnail(size, Image.ANTIALIAS)
            '''
            #print writeGif.__doc__
            filename = '../movie/cut/ktwo%d_c%d_p(%d,%d)/moive.gif' %(kid, campaign, x0, y0)
            writeGif(filename, images, duration=0.1)

#no train
    if False:
        kid = 200000862
        campaign = 0
        l2 = 1e5
        auto_l2 = 0
        auto = False
        poly = 0
        auto_offset = 18
        auto_window = 3
        margin = 18
        thread_num = 2
        part = None
        x0 = 0
        y0 = 0
        dir_list = ['data', 'pre', 'dif']
        dir_list = [ 'dif']
        for direc in dir_list:
            folder = '../movie/notrain/row/ktwo%d_c%d_p(%d,%d)/new/' %(kid, campaign, x0, y0)
            file_names = sorted((fn for fn in os.listdir(folder) if fn.endswith('.png')))[500:700]
            #['animationframa.png', 'animationframb.png', ...] "

            images = [Image.open(folder+fn) for fn in file_names]
            '''
            size = (150,150)
            for im in images:
                im.thumbnail(size, Image.ANTIALIAS)
            '''
            #print writeGif.__doc__
            filename = '../movie/notrain/row/ktwo%d_c%d_p(%d,%d)/moive.gif' %(kid, campaign, x0, y0)
            writeGif(filename, images, duration=0.1)


#no train and test, photometry
    if False:
        kid = 200000862
        campaign = 0
        l2 = 1e5
        auto_l2 = 0
        auto = False
        poly = 0
        auto_offset = 18
        auto_window = 3
        margin = 18
        thread_num = 2
        part = None
        target_pixel = [27,14]#[25,15]

        prefix = '../hdf5/notrain/ktwo%d_c%d_pixel(%d,%d)_%.0e'%(kid, campaign, target_pixel[0], target_pixel[1], l2)
        f = h5py.File('%s.hdf5'%prefix, 'r')
        data_group = f['/data']
        time = data_group['time'][:]
        epoch_mask = data_group['epoch_mask'][:]
        f.close()

        length = epoch_mask.shape[0]

        total_flux = np.zeros(length)
        total_pre = np.zeros(length)
        total_origin = np.zeros(length)

        for i in range(27,32):
            for j in range(8,13):
                target_pixel = [i,j]

                prefix = '../hdf5/notrain/ktwo%d_c%d_pixel(%d,%d)_%.0e'%(kid, campaign, target_pixel[0], target_pixel[1], l2)


                f = h5py.File('%s.hdf5'%prefix, 'r')
                cpm_info = f['/cpm_info']
                data_group = f['/data']
                
                kid = f.attrs['kid'][()]
                quarter = f.attrs['campaign'][()]
                target_flux = data_group['target_flux'][:]
                target_kplr_mask = data_group['target_kplr_mask'][:,:]
                epoch_mask = data_group['epoch_mask'][:]
                data_mask = data_group['data_mask'][:]
                time = data_group['time'][:]
                fit_flux = data_group['fit_flux'][:]

                print fit_flux.shape
                print target_flux.shape
                res = (target_flux-fit_flux[:,0])
                print res.shape

                total_flux[epoch_mask>0] += res
                total_pre[epoch_mask>0] += fit_flux[:,0]
                total_origin[epoch_mask>0] += target_flux

                f.close()

        print total_flux.shape
        print time.shape

        total_flux = total_flux/total_pre
        res_rms = np.sqrt(np.median(total_flux**2))
        frame, axes = plt.subplots(1, 1)

        '''
        total_flux = total_origin/total_pre
        total_flux = (total_flux/np.median(total_flux))-1.
        '''

        plt.plot(time, total_flux[epoch_mask>0])
        plt.ylabel("residual")
        plt.xlabel("time [BKJD]")
        plt.axhline(y=0,xmin=0,xmax=5, ls='--', c="r",linewidth=1,zorder=10, alpha=0.25)
        plt.axhline(y=res_rms,xmin=0,xmax=5, ls='--', c="r",linewidth=1,zorder=10, alpha=0.25)
        plt.axhline(y=-res_rms,xmin=0,xmax=5, ls='--', c="r",linewidth=1,zorder=10, alpha=0.25)
        #plt.ylim(-res_rms*6, res_rms*6)
        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0, hspace=0)
        #plt.suptitle('ktwo%d_c0%d v1'%(kid, campaign))
        plt.show()
        #plt.savefig('../plots/notrain/v1photo9p_ktwo%d_c%d_pixel(%d,%d)_%.0e_res.png'%(kid, campaign, target_pixel[0], target_pixel[1], l2), dpi=190)

    if False:
        c = SkyCoord('06h07m35.4s', '+24d05m40.8s', frame=FK5, equinox='J2000.0')

        var, types = load_var('../data/variable.dat')

        #var = np.array([[0, c.ra.deg, c.dec.deg,  2.3629, 0, 0,  19.218,0]])
        #types = np.array(['TR'])
        print var, types

        w = wcs.WCS(naxis=2)
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        w.wcs.crpix = [  25.500000000028194,    25.49999999985539]
        w.wcs.cdelt = [  -0.001105321813883, 0.001105321813883168]
        w.wcs.crval=[   91.87697836784632,  24.093880962956327]
        w.wcs.pc[0,0]= -0.22441822041889725
        w.wcs.pc[0,1]= 0.9752573507734456
        w.wcs.pc[1,0]= -0.9738349091773285
        w.wcs.pc[1,1]= -0.2239584036055725

        pix_coo = w.all_world2pix(var[:,1:3],0)
        pix_coo[:,0] = pix_coo[:,0]-1
        first = np.logical_and(pix_coo[:,0]>0, pix_coo[:,0]<48.5)
        sec = np.logical_and(pix_coo[:,1]>0, pix_coo[:,1]<48.5)
        tot = np.logical_and(first,sec)
        pix_coo = pix_coo[tot]
        types = types[tot]
        var = var[tot]
        print pix_coo

        kid = 200000862
        campaign = 0
        l2 = 1e5
        auto_l2 = 0
        auto = False
        poly = 0
        auto_offset = 18
        auto_window = 3
        margin = 18
        thread_num = 2
        part = None

        lim_list = [[-400,400],[-450,200],[-1000,400],[-100,150],[-400,200], [-500,200],[-200,200],[-150,150],[-400,200],[-200,200],[-200,100],[-400,200],[-1000,1000]]

        index=0
        for source in pix_coo:
            s_type = types[index]
            print source
            mid_pixel = np.around([source[1],source[0]])
            print mid_pixel

            prefix = '../hdf5/notrain/ktwo%d_c%d_pixel(%d,%d)_%.0e'%(kid, campaign, mid_pixel[0], mid_pixel[1], l2)
            f = h5py.File('%s.hdf5'%prefix, 'r')
            data_group = f['/data']
            time = data_group['time'][:]
            epoch_mask = data_group['epoch_mask'][:]
            f.close()

            length = epoch_mask.shape[0]

            total_flux = np.zeros(length)
            total_pre = np.zeros(length)
            total_origin = np.zeros(length)

            ap_size=1
            for i in range(int(mid_pixel[0])-ap_size,int(mid_pixel[0])+ap_size+1):
                for j in range(int(mid_pixel[1])-ap_size,int(mid_pixel[1])+ap_size+1):
                    target_pixel = [i,j]

                    prefix = '../hdf5/notrain/ktwo%d_c%d_pixel(%d,%d)_%.0e'%(kid, campaign, target_pixel[0], target_pixel[1], l2)

                    f = h5py.File('%s.hdf5'%prefix, 'r')
                    cpm_info = f['/cpm_info']
                    data_group = f['/data']
                    
                    kid = f.attrs['kid'][()]
                    quarter = f.attrs['campaign'][()]
                    target_flux = data_group['target_flux'][:]
                    target_kplr_mask = data_group['target_kplr_mask'][:,:]
                    epoch_mask = data_group['epoch_mask'][:]
                    data_mask = data_group['data_mask'][:]
                    time = data_group['time'][:]
                    fit_flux = data_group['fit_flux'][:]

                    #print fit_flux.shape
                    #print target_flux.shape
                    res = (target_flux-fit_flux[:,0])
                    #print res.shape

                    total_flux[epoch_mask>0] += res
                    total_pre[epoch_mask>0] += fit_flux[:,0]
                    total_origin[epoch_mask>0] += target_flux

                    f.close()

            #print total_flux.shape
            #print time.shape

            #total_flux = total_flux/total_pre
            res_rms = np.sqrt(np.median(total_flux**2))
            frame, axes = plt.subplots(1, 1)

            p = var[index,3]

            fold_time = fold_lc(0.5, p, total_flux, time, epoch_mask)
            '''
            total_flux = total_origin/total_pre
            total_flux = (total_flux/np.median(total_flux))-1.
            '''
            mask = np.logical_and(fold_time>0.5, fold_time<1.2)
            plt.plot(fold_time, total_flux[epoch_mask>0], '.k')
            plt.ylabel("Difference")
            plt.xlabel("time [BKJD]")
            plt.title('ID:%d    '%var[index,0]+'type:'+s_type+'\n'+'p=%f   '%var[index,3]+'R:%.3f'%var[index,6])
            plt.axhline(y=0,xmin=0,xmax=5, ls='--', c="r",linewidth=1,zorder=10, alpha=0.25)
            plt.axhline(y=res_rms,xmin=0,xmax=5, ls='--', c="r",linewidth=1,zorder=10, alpha=0.25)
            plt.axhline(y=-res_rms,xmin=0,xmax=5, ls='--', c="r",linewidth=1,zorder=10, alpha=0.25)
            #plt.ylim(-res_rms*10, res_rms*10)
            lim = lim_list[index]
            plt.ylim(lim[0],lim[1])
            plt.tight_layout()
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0)
            #plt.suptitle('ktwo%d_c0%d v1'%(kid, campaign))
            #plt.show()
            plt.savefig('../plots/notrain/shift_new0_v%dphoto9p_ktwo%d_c%d_pixel(%d,%d)_%.0e_res_fold.png'%(index,kid, campaign, target_pixel[0], target_pixel[1], l2), dpi=190)
            index += 1

    if False:
        var = load_k2_var('../data/k2_var.dat')

        w = wcs.WCS(naxis=2)
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        w.wcs.crpix = [  25.500000000028194,    25.49999999985539]
        w.wcs.cdelt = [  -0.001105321813883, 0.001105321813883168]
        w.wcs.crval=[   91.87697836784632,  24.093880962956327]
        w.wcs.pc[0,0]= -0.22441822041889725
        w.wcs.pc[0,1]= 0.9752573507734456
        w.wcs.pc[1,0]= -0.9738349091773285
        w.wcs.pc[1,1]= -0.2239584036055725

        pix_coo = w.all_world2pix(var[:,1:3],0)
        first = np.logical_and(pix_coo[:,0]>0, pix_coo[:,0]<48.5)
        sec = np.logical_and(pix_coo[:,1]>0, pix_coo[:,1]<48.5)
        tot = np.logical_and(first,sec)
        pix_coo = pix_coo[tot]
        var = var[tot]
        print pix_coo

        kid = 200000862
        campaign = 0
        l2 = 1e5
        auto_l2 = 0
        auto = False
        poly = 0
        auto_offset = 18
        auto_window = 3
        margin = 18
        thread_num = 2
        part = None

        index=0
        for source in pix_coo:
            print source
            mid_pixel = np.around([source[1],source[0]])
            print mid_pixel

            prefix = '../hdf5/notrain/ktwo%d_c%d_pixel(%d,%d)_%.0e'%(kid, campaign, mid_pixel[0], mid_pixel[1], l2)
            f = h5py.File('%s.hdf5'%prefix, 'r')
            data_group = f['/data']
            time = data_group['time'][:]
            epoch_mask = data_group['epoch_mask'][:]
            f.close()

            length = epoch_mask.shape[0]

            total_flux = np.zeros(length)
            total_pre = np.zeros(length)
            total_origin = np.zeros(length)

            ap_size=1
            for i in range(int(mid_pixel[0])-ap_size,int(mid_pixel[0])+ap_size+1):
                for j in range(int(mid_pixel[1])-ap_size,int(mid_pixel[1])+ap_size+1):
                    target_pixel = [i,j]
                    print target_pixel

                    prefix = '../hdf5/notrain/ktwo%d_c%d_pixel(%d,%d)_%.0e'%(kid, campaign, target_pixel[0], target_pixel[1], l2)

                    f = h5py.File('%s.hdf5'%prefix, 'r')
                    cpm_info = f['/cpm_info']
                    data_group = f['/data']
                    
                    kid = f.attrs['kid'][()]
                    quarter = f.attrs['campaign'][()]
                    target_flux = data_group['target_flux'][:]
                    target_kplr_mask = data_group['target_kplr_mask'][:,:]
                    epoch_mask = data_group['epoch_mask'][:]
                    data_mask = data_group['data_mask'][:]
                    time = data_group['time'][:]
                    fit_flux = data_group['fit_flux'][:]

                    #print fit_flux.shape
                    #print target_flux.shape
                    res = (target_flux-fit_flux[:,0])
                    #print res.shape

                    total_flux[epoch_mask>0] += res
                    total_pre[epoch_mask>0] += fit_flux[:,0]
                    total_origin[epoch_mask>0] += target_flux

                    f.close()

            #print total_flux.shape
            #print time.shape

            total_flux = total_flux/total_pre
            res_rms = np.sqrt(np.median(total_flux**2))
            frame, axes = plt.subplots(1, 1)

            '''
            total_flux = total_origin/total_pre
            total_flux = (total_flux/np.median(total_flux))-1.
            '''

            plt.plot(time, total_flux[epoch_mask>0])
            plt.ylabel("residual")
            plt.xlabel("time [BKJD]")
            plt.title('%d %d %f'%(var[index,0], var[index,8], var[index,9]))
            plt.axhline(y=0,xmin=0,xmax=5, ls='--', c="r",linewidth=1,zorder=10, alpha=0.25)
            plt.axhline(y=res_rms,xmin=0,xmax=5, ls='--', c="r",linewidth=1,zorder=10, alpha=0.25)
            plt.axhline(y=-res_rms,xmin=0,xmax=5, ls='--', c="r",linewidth=1,zorder=10, alpha=0.25)
            #plt.ylim(-res_rms*6, res_rms*6)
            plt.tight_layout()
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0)
            #plt.suptitle('ktwo%d_c0%d v1'%(kid, campaign))
            #plt.show()
            plt.savefig('../plots/notrain/new_v%dphoto9p_ktwo%d_c%d_pixel(%d,%d)_%.0e_res.png'%(index,kid, campaign, target_pixel[0], target_pixel[1], l2), dpi=190)
            index += 1


#movie no train 
    if False:
        kid = 200000862
        campaign = 0
        l2 = 1e5
        auto_l2 = 0
        auto = False
        poly = 0
        auto_offset = 18
        auto_window = 3
        margin = 18
        thread_num = 2
        part = None
        x0 = 0
        y0 = 0

        var, types = load_var('../data/variable.dat')
        

        #c = SkyCoord('06h07m35.4s', '+24d05m40.8s', frame=FK5, equinox='J2000.0')
        #var = np.array([[0, c.ra.deg, c.dec.deg,  2.3629, 0, 0,  19.218,0]])
        #types = np.array(['TR'])

        print var, types

        w = wcs.WCS(naxis=2)
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        w.wcs.crpix = [  25.500000000028194,    25.49999999985539]
        w.wcs.cdelt = [  -0.001105321813883, 0.001105321813883168]
        w.wcs.crval=[   91.87697836784632,  24.093880962956327]
        w.wcs.pc[0,0]= -0.22441822041889725
        w.wcs.pc[0,1]= 0.9752573507734456
        w.wcs.pc[1,0]= -0.9738349091773285
        w.wcs.pc[1,1]= -0.2239584036055725

        pix_coo = w.all_world2pix(var[:,1:3],0)
        first = np.logical_and(pix_coo[:,0]>0, pix_coo[:,0]<48.5)
        sec = np.logical_and(pix_coo[:,1]>0, pix_coo[:,1]<48.5)
        tot = np.logical_and(first,sec)
        pix_coo = pix_coo[tot]
        types = types[tot]
        var = var[tot]
        print pix_coo

        fluxes=[]
        dir_list = ['dif_res']
        titles = ['difference']
        i=0
        limit=[0,0]

        direc='pre'
        pre = np.load('../hdf5/notrain/ktwo%d_c%d_p(%d,%d)_%s.npy'%(kid, campaign, x0, y0, direc))[200:]
        pre = np.median(pre, axis=0)

        direc = 'dif_res'
        flux = np.load('../hdf5/notrain/ktwo%d_c%d_p(%d,%d)_%s.npy'%(kid, campaign, x0, y0, direc))[200:]
        print np.max(flux), np.min(flux)
        print flux.shape
        var = np.std(flux, axis=0)
        print var.shape
        print np.median(var)

        plt.imshow(var, interpolation='None', cmap=plt.get_cmap('Greys'), vmin=0, vmax=40)
        plt.colorbar()

        for pix in pix_coo:
            x=np.round(pix[0])
            y=np.round(pix[1])
            plt.axhline(y=y-0.5,xmin=(x)/50.,xmax=(x+1)/50., ls='-', c="b",linewidth=1,zorder=10, alpha=1)
            plt.axhline(y=y+0.5,xmin=(x)/50.,xmax=(x+1)/50., ls='-', c="b",linewidth=1,zorder=10, alpha=1)

            plt.axvline(x=x-0.5,ymin=(50-y)/50.,ymax=(50-y-1.)/50., ls='-', c="b",linewidth=1,zorder=10, alpha=1)
            plt.axvline(x=x+0.5,ymin=(50-y)/50.,ymax=(50-y-1.)/50., ls='-', c="b",linewidth=1,zorder=10, alpha=1)

        plt.show()
