import sys
import numpy as np

import k2_cpm
import epic
import matrix_xy


# This could be moved to other file.
def get_tpf_file_name(directory, epic_id, campaign):
    """get the name of the TPF file"""
    return '{:}/ktwo{:}-c{:}_lpd-targ.fits.gz'.format(directory, epic_id, campaign)

# This could be moved to other file.
def get_epoch_mask_for_epic_id(directory, epic_id, campaign):
    """reads the file and gets epoch_mask from it"""
    file_name = get_tpf_file_name(directory, epic_id, campaign)
    #print(epic_id)
    sys.stdout.flush()
    epic.load_tpf(epic_id, campaign, directory)
    tpf = k2_cpm.Tpf(file_name)
    return tpf.epoch_mask

# This could be moved to other file.
def get_predictor_epoch_mask(tpfs_epic_list_file, directory, campaign):
    """take a list of tpf files and construc predictor_epoch_mask using them"""
    predictor_epoch_mask = None
    with open(tpfs_epic_list_file) as epic_list:
        for line in epic_list.readlines():
            epoch_mask = get_epoch_mask_for_epic_id(directory, int(line), campaign)
            if predictor_epoch_mask is None:
                predictor_epoch_mask = np.ones(epoch_mask.shape, dtype=bool)
            predictor_epoch_mask &= epoch_mask
    return predictor_epoch_mask
    
def read_true_false_file(file_name):
    """reads file with values True or False"""
    parser = {'TRUE': True, 'FALSE': False}
    out = []
    with open(file_name) as in_file:
        for line in in_file.readlines():
            out.append(parser[line[:-1].upper()])
    return np.array(out)


if __name__ == '__main__':
    # case dependent settings:    
    n_test = 1
    #n_test = 2
    #n_test = 3
    #n_test = 4
    
    # common settings:
    l2 = 1000.
    epic_id = "200071074"
    campaign = 92 
    pixel = [883, 670]
    pre_matrix_file = "../test/output/{:}-pre_matrix.dat".format(n_test)
    tpfs_epic_list_file = "../test/output/1-epic.dat" # Content is the same for each file (1,2,3,4).
    predictor_epoch_mask_file = "1-predictor_epoch_mask.dat" # Content is the same for each file (1,2,3,4).
    pixel_flux_file_name = '{:}-pixel_flux.dat'.format(n_test)
    pixel_mask_file_name = '{:}-mask.dat'.format(n_test)
    epoch_mask_file_name = '{:}-epoch_mask.dat'.format(n_test)
    
    # output files:
    fit_matrix_file_name = '{:}-fit_matrix_comp.dat'.format(n_test)
    result_file_name = '{:}-result_comp.dat'.format(n_test)
    dif_file_name = '{:}-dif_comp.dat'.format(n_test)
    # Settings end here.
    
    # Load all required files:
    pre_matrix = np.loadtxt(pre_matrix_file)
    predictor_epoch_mask = read_true_false_file(predictor_epoch_mask_file)
    (tpf_time, tpf_flux) = np.loadtxt(pixel_flux_file_name, unpack=True)
    pixel_mask = matrix_xy.load_matrix_xy(pixel_mask_file_name, data_type='boolean')
    tpf_epoch_mask = read_true_false_file(epoch_mask_file_name)
    
    # Calculations:
    fit_matrix_results = k2_cpm.get_fit_matrix_ffi(tpf_flux, tpf_epoch_mask, pre_matrix, predictor_epoch_mask, l2, tpf_time, 0, None)
    (target_flux, predictor_matrix, none_none, l2_vector, time, target_epoch_mask, data_mask) = fit_matrix_results
    result = k2_cpm.fit_target_no_train(target_flux, pixel_mask, np.copy(predictor_matrix), time, target_epoch_mask[data_mask>0], None, l2_vector, 1, None)    
    fit_flux = np.dot(predictor_matrix, result)
    dif = target_flux-fit_flux[:,0]

    # Save results:
    matrix_xy.save_matrix_xy(predictor_matrix, fit_matrix_file_name)
    np.savetxt(result_file_name, result, fmt='%.8f')
    np.savetxt(dif_file_name, dif, fmt='%.8f')
    