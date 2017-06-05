import os
import numpy as np

from K2CPM import cpm_part1
from K2CPM import cpm_part2 ##
from K2CPM import matrix_xy


def do_test_cpm_part1(n_test, channel=52, 
        campaign=92, n_predictor=400, n_pca=0, distance=16, exclusion=5, 
        flux_lim=(0.2, 1.5), input_dir="tpf", 
        pixel_list=np.array([[883, 670]]), train_lim=None):
    """basic function that runs run_cpm_part1() and checks results
    default settings are for Dun's unit test #1"""
    file_name = "{:}-pre_matrix_xy.dat".format(n_test)     
    output_file = "intermediate/" + file_name
    output_file_mask = "intermediate/{:}-predictor_epoch_mask.dat".format(n_test)
    expected_file = "intermediate/expected/" + file_name
    for file_name in [output_file, output_file_mask]:
        if os.path.isfile(file_name):
            os.remove(file_name)

    cpm_part1.run_cpm_part1(channel, campaign, n_predictor, n_pca, 
                            distance, exclusion, flux_lim, input_dir, 
                            pixel_list, train_lim, output_file, output_file_mask)
                            
    mask = cpm_part2.read_true_false_file(output_file_mask)
    predictor_matrix = matrix_xy.load_matrix_xy(output_file)
    predictor_matrix_expected = matrix_xy.load_matrix_xy(expected_file)
    
    np.testing.assert_almost_equal(predictor_matrix[mask], 
                                    predictor_matrix_expected, decimal=2)

def test_cpm_part1_1():
    do_test_cpm_part1(n_test=1)

def test_cpm_part1_2():
    do_test_cpm_part1(n_test=2, n_predictor=800)

def test_cpm_part1_3():
    do_test_cpm_part1(n_test=3, n_predictor=1600, n_pca=200)
    
def test_cpm_part1_4():
    do_test_cpm_part1(n_test=4, distance=8, exclusion=3)
    