import numpy as np

import check_get_fit_matrix


if __name__ == '__main__':
    directory = 'tpf/'
    campaign = 92 
    tpfs_epic_list_file = "../test/output/1-epic.dat"
    out_file = '1-predictor_epoch_mask.dat'
    
    predictor_epoch_mask = check_get_fit_matrix.get_predictor_epoch_mask(tpfs_epic_list_file, directory, campaign)

    np.savetxt(out_file, predictor_epoch_mask, fmt='%r')
