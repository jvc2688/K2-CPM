import sys
import numpy as np

#import k2_cpm
#import matrix_xy
from code import k2_cpm
from code import matrix_xy


def read_true_false_file(file_name):
    """reads file with values True or False"""
    parser = {'TRUE': True, 'FALSE': False}
    out = []
    with open(file_name) as in_file:
        for line in in_file.readlines():
            out.append(parser[line[:-1].upper()])
    return np.array(out)


def execute_cpm_part2(n_test=1):
    # settings:
    l2 = 1000.
    campaign = 92 
    pixel = [883, 670]
    in_directory = "../test/intermediate/"
    out_directory = "../test/output/"
    
    pre_matrix_file = "{:}{:}-pre_matrix_xy.dat".format(in_directory, n_test)
    predictor_epoch_mask_file = "{:}{:}-predictor_epoch_mask.dat".format(in_directory, n_test)
    pixel_flux_file_name = '{:}{:}-pixel_flux.dat'.format(in_directory, n_test)
    pixel_mask_file_name = '{:}{:}-mask.dat'.format(in_directory, n_test)
    epoch_mask_file_name = '{:}{:}-epoch_mask.dat'.format(in_directory, n_test)

    # output files:
    result_file_name = '{:}{:}-result.dat'.format(out_directory, n_test)
    dif_file_name = '{:}{:}-dif.dat'.format(out_directory, n_test)
    # Settings end here.
    
    # Load all required files:
    pre_matrix = matrix_xy.load_matrix_xy(pre_matrix_file)
    predictor_epoch_mask = read_true_false_file(predictor_epoch_mask_file)
    (tpf_time, tpf_flux) = np.loadtxt(pixel_flux_file_name, unpack=True)
    pixel_mask = matrix_xy.load_matrix_xy(pixel_mask_file_name, data_type='boolean')
    tpf_epoch_mask = read_true_false_file(epoch_mask_file_name)
    
    # Calculations:
    fit_matrix_results = k2_cpm.get_fit_matrix_ffi(tpf_flux, tpf_epoch_mask, pre_matrix, predictor_epoch_mask, l2, tpf_time, 0, None)
    (target_flux, predictor_matrix, none_none, l2_vector, time, target_epoch_mask, data_mask) = fit_matrix_results
    result = k2_cpm.fit_target_no_train(target_flux, pixel_mask, np.copy(predictor_matrix), time, target_epoch_mask[data_mask>0], None, l2_vector, 1, None)    
    fit_flux = np.dot(predictor_matrix, result)
    dif = target_flux - fit_flux[:,0]

    # Save results:
    np.savetxt(result_file_name, result, fmt='%.8f')
    np.savetxt(dif_file_name, dif, fmt='%.8f')
    

if __name__ == '__main__':
    # case dependent settings:    
    n_test = 1
    #n_test = 2
    #n_test = 3
    #n_test = 4
    
    execute_cpm_part2(n_test)
    
