import numpy as np

from code.tpfdata import TpfData
from code import cpm_part2


def single_test(n_test):
    cpm_part2.execute_cpm_part2(n_test)
    
    file_out_result = 'output/{:}-result.dat'.format(n_test)
    file_expect_result = 'output/expected/{:}-result.dat'.format(n_test)
    file_out_dif = 'output/{:}-dif.dat'.format(n_test)
    file_expect_dif = 'output/expected/{:}-dif.dat'.format(n_test)

    out_result = np.loadtxt(file_out_result)
    expect_result = np.loadtxt(file_expect_result)
    out_dif = np.loadtxt(file_out_dif)
    expect_dif = np.loadtxt(file_expect_dif)

    np.testing.assert_almost_equal(out_result, expect_result)
    #np.testing.assert_almost_equal(out_dif, expect_dif, decimal=3)

def test_1():
    single_test(1)

def test_2():
    single_test(2)

def test_3():
    single_test(3)

def test_4():
    single_test(4)

