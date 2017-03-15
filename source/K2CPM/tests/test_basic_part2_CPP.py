# -*-coding:Utf-8 -*
# ====================================================================
# Tests of the C++ version of the CPM code.
# File mostly similar to `test_basic_part2.py`.
# ====================================================================
# Standard packages
# ====================================================================
import os
import subprocess
import numpy as np

# ====================================================================
# Functions
# ====================================================================
def single_test(n_test=1):

    # Change current path
    # -------------------
    path_code = "../code/"
    #os.chdir(path_code)

    print "coucou"

    # Execute the C++ routine
    # -----------------------
    # cmd_list = ["./libcpm", "{:d}".format(n_test)]
    # output = subprocess.Popen(cmd_list, stdout=subprocess.PIPE).communicate()[0]
    # # print output
    #
    # # Change current path
    # # -------------------
    # path_code = "../test/"
    # os.chdir(path_code)
    #
    # # Define file names
    # # -----------------
    # file_out_result = 'output/{:}-result.dat'.format(n_test)
    # file_expect_result = 'output/expected/{:}-result.dat'.format(n_test)
    # file_out_dif = 'output/{:}-dif.dat'.format(n_test)
    # file_expect_dif = 'output/expected/{:}-dif.dat'.format(n_test)
    #
    # # Load result files
    # # -----------------
    # out_result = np.loadtxt(file_out_result)
    # expect_result = np.loadtxt(file_expect_result)
    # out_dif = np.loadtxt(file_out_dif)
    # expect_dif = np.loadtxt(file_expect_dif)
    #
    # # Test the differences
    # # --------------------
    # np.testing.assert_almost_equal(out_result, expect_result, decimal=4)
    # np.testing.assert_almost_equal(out_dif, expect_dif, decimal=4)
# --------------------------------------------------------------------
def test_1():
    single_test(1)
# --------------------------------------------------------------------
def test_2():
    single_test(2)
# --------------------------------------------------------------------
def test_3():
    single_test(3)
# --------------------------------------------------------------------
def test_4():
    single_test(4)
# --------------------------------------------------------------------
if __name__ == '__main__':
    single_test(n_test=1)
# --------------------------------------------------------------------