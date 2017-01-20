from __future__ import print_function

import epic
from astropy.io import fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import threading
import os
import math
import h5py
import sys
import glob
from astropy.wcs import WCS
import gc
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures


def get_pixel_mask(flux, kplr_mask):
    '''help function to find the pixel mask'''
    pixel_mask = np.isfinite(flux) & (flux!=0)
    pixel_mask[:, kplr_mask < 1] = False
    '''
    pixel_mask = np.zeros(flux.shape)
    pixel_mask[np.isfinite(flux)] = 1 # okay if finite
    pixel_mask[:, (kplr_mask < 1)] = 0 # unless masked by kplr
    pixel_mask[flux==0] = 0
    '''
    return pixel_mask

def get_epoch_mask(pixel_mask, time, q):
    '''help function to find the epoch mask'''
    foo = np.sum(np.sum((pixel_mask > 0), axis=2), axis=1)
    epoch_mask = (foo>0) & np.isfinite(time) & q 
    return epoch_mask

class Tpf():
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

    def get_pixels(self, coo):
        pixels = self.wcs.all_world2pix(coo,0)
        return pixels

    def get_index(self, x, y):
        index = np.arange(self.row.shape[0])
        index_mask = ((self.row-self.ref_row)==x) & ((self.col-self.ref_col)==y)
        try:
            return index[index_mask][0]
        except IndexError:
            print("No data for ({0:d},{1:d})".format(x, y))

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
        print('new')
        if tpfs==None:
            tpfs = set([self])
        else:
            tpfs = set(tpfs)
            tpfs.add(self)

        if self.tpfs == tpfs:
            print('the same')
        else:
            print('different')
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
                #normalize the data
                #norm =((tpf.flux/tpf.median)-1.)
                #norm = norm/np.std(norm, axis=0)
                pixel_flux.append(tpf.flux)
                #print(norm)
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
        #print(predictor_flux.shape)

        return predictor_flux, self.predictor_epoch_mask


def get_fit_matrix_ffi(target_flux, target_err, target_epoch_mask, predictor_matrix, predictor_epoch_mask, l2, time, num_pca, poly=0):
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
    target_err = target_err[co_mask>0]
    predictor_matrix = predictor_matrix[co_mask>0, :]
    time = time[co_mask>0]

    #normalize the data
    '''
    predictor_matrix = ((predictor_matrix/np.median(predictor_matrix, axis=0))-1.)
    predictor_matrix = predictor_matrix/np.std(predictor_matrix, axis=0)

    target_med = np.median(target_flux, axis=0)
    target_flux = ((target_flux/np.median(target_flux, axis=0))-1.)
    target_std = np.std(target_flux, axis=0)
    target_flux = target_flux/np.std(target_flux, axis=0)
    '''
    target_med = 1. 
    target_std = 1.

    if num_pca>0:
        pca = PCA(n_components=num_pca)
        pca.fit(predictor_matrix)
        predictor_matrix = pca.transform(predictor_matrix)
        #np.save('/Users/dunwang/Desktop/old/k2c9a/pca_train_more_out'+'/'+'OGLE-2016-BLG-0884new6-pca.npy', predictor_matrix)
        #p_feature = PolynomialFeatures(2, include_bias=False)
        #predictor_matrix = p_feature.fit_transform(predictor_matrix)
    #p_feature = PolynomialFeatures(2, include_bias=False)
    #predictor_matrix = p_feature.fit_transform(predictor_matrix)

    #add polynomial terms
    if poly is not None:
        nor_time = np.arange(predictor_matrix.shape[0])
        #time_mean = np.mean(time)
        #time_std = np.std(time)
        #nor_time = (time-time_mean)/time_std
        p = np.polynomial.polynomial.polyvander(nor_time, poly)
        predictor_matrix = np.concatenate((predictor_matrix, p), axis=1)

    return target_flux, predictor_matrix, target_err, time, epoch_mask, data_mask, target_med, target_std

def get_data(target_epic_num, camp, num_predictor, num_pca, dis, excl, flux_lim, input_dir, output_dir, pixel_list=None):
    epic.load_tpf(target_epic_num, camp, input_dir)
    file_name = input_dir+'/'+'ktwo{0:d}-c{1:d}_lpd-targ.fits.gz'.format(int(target_epic_num), camp)
    tpf = Tpf(file_name)
    shape = tpf.kplr_mask.shape
    ra = tpf.ra
    dec = tpf.dec
    r = 0
    tpfs = []
    epic_list = []
    while len(epic_list)<=5:
        r+=6
        epic_list = epic.get_tpfs(ra,dec,r,camp, tpf.channel)

    for epic_num in epic_list:
        if epic_num == 200070874 or epic_num == 200070438:
            continue
        epic.load_tpf(int(epic_num), camp, input_dir)
        try:
            tpfs.append(Tpf(input_dir+'/'+'ktwo{0}-c{1}_lpd-targ.fits.gz'.format(int(epic_num), camp)))
        except IOError:
            os.remove(input_dir+'/'+'ktwo{0}-c{1}_lpd-targ.fits.gz'.format(int(epic_num), camp))
            epic.load_tpf(int(epic_num), camp, input_dir)
            tpfs.append(Tpf(input_dir+'/'+'ktwo{0}-c{1}_lpd-targ.fits.gz'.format(int(epic_num), camp)))

    if pixel_list == None:
        print('no pixel list, run cpm on full tpf')
        pixel_list = np.array([np.repeat(np.arange(shape[0]), shape[1]), np.tile(np.arange(shape[1]), shape[0])], dtype=int).T
    data_list = []
    for pixel in pixel_list:
        x = pixel[0]
        y = pixel[1]
        print(x, y)
        if (x<0) or (x>=tpf.kplr_mask.shape[0]) or (y<0) or (y>=tpf.kplr_mask.shape[1]):
            print('pixel out of range')
            data = None
        elif (tpf.kplr_mask[x,y])>0:
            print(len(tpfs))
            predictor_matrix, predictor_epoch_mask = tpf.get_predictor_matrix(x, y, num_predictor, dis=dis, excl=excl, flux_lim=flux_lim, tpfs=tpfs, var_mask=None)
            while predictor_matrix.shape[1]<num_predictor:
                r+=6
                epic_list_new = set(epic.get_tpfs(ra,dec,r,camp, tpf.channel))
                more = epic_list_new.difference(epic_list)
                if len(more) == 0:
                    low_lim = flux_lim[0]-0.1
                    up_lim = flux_lim[1]+0.1
                    while predictor_matrix.shape[1]<num_predictor:
                        old_num = predictor_matrix.shape[1]
                        predictor_matrix, predictor_epoch_mask = tpf.get_predictor_matrix(x, y, num_predictor, dis=dis, excl=excl, flux_lim=(low_lim,up_lim), tpfs=tpfs, var_mask=None)
                        low_lim = np.max(low_lim-0.1,0.1)
                        up_lim = up_lim+0.1
                        difference = predictor_matrix.shape[1] - old_num
                        if difference == 0:
                            print('no more pixel at all')
                            break
                    break
                elif len(more)>0:
                    for epic_num in more:
                        if epic_num == 200070874 or epic_num == 200070438:
                            continue
                        epic.load_tpf(int(epic_num), camp, input_dir)
                        print(epic_num)
                        tpfs.append(Tpf(input_dir+'/'+'ktwo{0:d}-c{1:d}_lpd-targ.fits.gz'.format(int(epic_num), camp)))
                    predictor_matrix, predictor_epoch_mask = tpf.get_predictor_matrix(x, y, num_predictor, dis=dis, excl=excl, flux_lim=flux_lim, tpfs=tpfs, var_mask=None)
                    print(predictor_matrix.shape)
                    epic_list = epic_list_new
            index = tpf.get_index(x,y)
            flux, predictor_matrix, flux_err, time, target_epoch_mask, data_mask, target_med, target_std \
                = get_fit_matrix_ffi(tpf.flux[:,index], tpf.flux_err[:,index], tpf.epoch_mask, predictor_matrix, predictor_epoch_mask, l2, tpf.time, num_pca, 0, None)
            data = [flux, predictor_matrix, flux_err, time, target_epoch_mask, data_mask, target_med, target_std] 
        data_list.append(data)
    return data_list

if __name__ == "__main__":
    pass

