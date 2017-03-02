import numpy as np

#import leastSquareSolver as lss
from code import leastSquareSolver as lss


def get_fit_matrix_ffi(target_flux, target_epoch_mask, predictor_matrix, l2, time, poly=0, ml=None):
    """
    ## inputs:
    - `predictor_matrix` - matrix of predictor fluxes
    - `l2` - strength of l2 regularization
    - `time` - one dimension array of BKJD time
    - `poly` - number of orders of polynomials of time need to be added

    ## outputs:
    - `target_flux` - target flux
    - `predictor_matrix` - matrix of predictor fluxes
    - `target_flux_err` - error of the target flux
    - `l2_vector` - vector of the l2 regularization 
    """

    epoch_mask = target_epoch_mask

    #remove bad time point based on simulteanous epoch mask
    target_flux = target_flux[epoch_mask]
#    predictor_matrix = predictor_matrix[epoch_mask]
    time = time[epoch_mask]

    l2_length_of_ones = predictor_matrix.shape[0]

    #add polynomial terms
    if poly is not None:
        nor_time = np.arange(predictor_matrix.shape[0]) # Note that this assumes exactly equal time differences and no missing data.
        p = np.polynomial.polynomial.polyvander(nor_time, poly)
        predictor_matrix = np.concatenate((predictor_matrix, p), axis=1)

    if ml is not None:
        predictor_matrix = np.concatenate((predictor_matrix, ml), axis=1)

    #construct l2 vectors
    l2_vector = np.ones(predictor_matrix.shape[1], dtype=float) * l2

    l2_vector[l2_length_of_ones:] = 0. # This ensures that there's no reguralization on concatenated models and polynomials. 

    return target_flux, predictor_matrix, None, l2_vector, time

def fit_target(target_flux, target_kplr_mask, predictor_flux_matrix, time, covar_list, l2_vector=None, train_lim=None):
    """
    TO DO - remove target_kplr_mask because not used !!!

    ## inputs:
    - `target_kplr_mask` - kepler mask of the target star
    - `predictor_flux_matrix` - fitting matrix of neighbor flux
    - `l2_vector` - array of L2 regularization strength
    ## outputs:
    """
    if train_lim is not None:
        train_mask = (time<train_lim[0]) | (time>train_lim[1])
        predictor_flux_matrix = predictor_flux_matrix[train_mask,:]
        target_flux = target_flux[train_mask]
        if covar_list is not None:
            covar_list = covar_list[train_mask]

    if covar_list is not None:
        covar = covar_list**2
    else:
        covar = None

    result = lss.linear_least_squares(predictor_flux_matrix, target_flux, covar, l2_vector)
    return result

