from __future__ import print_function

import numpy as np
import leastSquareSolver as lss

def CPM():
    def __init__(self, target_flux, target_kplr_mask, predictor_matrix, time, epoch_mask, covar_list, l2, target_med, target_std, thread_num=1):
        self.target_flux = target_flux
        self.target_kplr_mask = target_kplr_mask
        self.predictor_matrix = predictor_matrix
        self.time = time
        self.epoch_mask = epoch_mask
        self.covar_list = covar_list
        self.l2 = l2
        self.target_med = target_med
        self.target_std = target_std
        self.thread_num = thread_num
        self.result = None
        self.fit_flux = None
        self.dif = None
        self.chi2 = None
        self.cpm_flux = None

    def fit(train_lim=None, ml=None):
        target_flux = self.target_flux
        predictor_matrix = self.predictor_matrix
        covar_list = self.covar_list
        time = self.time
        epoch_mask = self.epoch_mask

        if ml is not None:
            predictor_matrix = np.concatenate((predictor_matrix, ml[epoch_mask]), axis=1)

        if train_lim is not None:
            train_mask = (time<train_lim[0]) | (time>train_lim[1])
            predictor_matrix = predictor_matrix[train_mask,:]
            target_flux = target_flux[train_mask]
            if covar_list is not None:
                covar_list = covar_list[train_mask]
        print(predictor_matrix.shape)

        if covar_list is not None:
            covar = covar_list**2
        else:
            covar = None
        fit_flux = []
        fit_coe = []
        length = target_flux.shape[0]
        total_length = epoch_mask.shape[0]
        predictor_num = predictor_matrix
        l2_vector = np.ones(predictor_num, dtype=float)*l2
        
        self.result = lss.linear_least_squares(predictor_matrix, target_flux, covar, l2_vector)
        self.fit_flux = np.dot(self.predictor_matrix, result)
        self.dif = (self.target_flux-self.fit_flux[:,0])
        self.chi2 = np.sum(np.square((target.flux-np.dot(predictor_matrix, result)))/covar)
        if ml is not None:
            self.cpm_flux = self.target_flux - np.dot(self.predictor_matrix[:,:-1], self.result[:-1])
        else:
            self.cpm_flux = self.dif            

if __name__ == "__main__":
    pass

