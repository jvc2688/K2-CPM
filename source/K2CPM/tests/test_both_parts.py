import numpy as np

from K2CPM import cpm_part1
from K2CPM import cpm_part2
from K2CPM import matrix_xy
from K2CPM import tpfdata


def do_test_cpm_part1_and_part2(n_test_compare, 
        target_epic_num=200071074, channel=52, campaign=92, 
        n_predictor=400, n_pca=0, distance=16, exclusion=5, 
        flux_lim=(0.2, 1.5), input_dir="tpf", 
        pixel_list=np.array([[883, 670]]), 
        train_lim=None, l2=1000.):
    """test both cpm_part1 and cpm_part2"""

    file_expect_result = 'output/expected/{:}-result.dat'.format(n_test_compare)
    expect_result = np.loadtxt(file_expect_result)
    file_expect_dif = 'output/expected/{:}-dif.dat'.format(n_test_compare)
    expect_dif = np.loadtxt(file_expect_dif)

    # run cpm_part1
    (predictor_matrixes, predictor_masks) = cpm_part1.run_cpm_part1(
                            channel, campaign, n_predictor, n_pca, 
                            distance, exclusion, flux_lim, input_dir, 
                            pixel_list, train_lim, 
                            return_predictor_epoch_masks=True)

    # open TPF file to get additional information
    tpfdata.TpfData.directory = input_dir
    tpf = tpfdata.TpfData(epic_id=target_epic_num, campaign=campaign)
    assert pixel_list.shape[0] == 1, 'this version accepts only single pixel in pixel_list'
    assert pixel_list.shape[1] == 2, 'exactly 2 coordinates of pixel required'
    tpf_flux = tpf.get_flux_for_pixel(row=pixel_list[0][0], column=pixel_list[0][1])
    tpf_flux_err = tpf.get_flux_err_for_pixel(row=pixel_list[0][0], column=pixel_list[0][1])

    (result, fit_flux, dif, time) = cpm_part2.cpm_part2(tpf.jd_short, 
                                    tpf_flux, tpf_flux_err, 
                                    tpf_epoch_mask=tpf.epoch_mask, 
                                    predictor_matrix=predictor_matrixes[0], 
                                    predictor_mask=predictor_masks[0],
                                    l2=l2)
            
    np.testing.assert_almost_equal(result[:,0], expect_result)
    np.testing.assert_almost_equal(dif, expect_dif, decimal=5)
    
def test_cpm_both_parts():
    """same settings as in Dun's test #1"""
    do_test_cpm_part1_and_part2(n_test_compare=1)
   