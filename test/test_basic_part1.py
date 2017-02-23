import numpy as np

from code import cpm_part1
from code import matrix_xy


def test_cpm_part1():
    target_epic_num = 200071074
    campaign = 92
    n_predictor = 400
    n_pca = 0
    distance = 16
    exclusion = 5
    flux_lim = (0.2, 1.5)
    input_dir = "tpf"
    pixel_list = np.array([[883, 670]])
    train_lim = None
    output_file = "intermediate/1-pre_matrix_xy.dat"
    expected_file = "intermediate/expected/1-pre_matrix_xy.dat"

    cpm_part1.run_cpm_part1(target_epic_num, campaign, n_predictor, n_pca, 
                            distance, exclusion, flux_lim, input_dir, 
                            pixel_list, train_lim, output_file)

    predictor_matrix = matrix_xy.load_matrix_xy(output_file)
    predictor_matrix_expected = matrix_xy.load_matrix_xy(expected_file)
    np.testing.assert_almost_equal(predictor_matrix / predictor_matrix_expected, 1., decimal=3)
    np.testing.assert_almost_equal(predictor_matrix, predictor_matrix_expected, decimal=2)

