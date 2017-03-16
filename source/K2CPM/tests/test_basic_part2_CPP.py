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
def check_slash(path):
    if len(path) > 0:
        if path[-1] != '/':
            path = '{:s}/'.format(path)
    return path
# --------------------------------------------------------------------
def getpath_this():
    """Return the path of the event directory.

    :return: path of event directory
    :rtype: string
    """
    path = os.path.realpath(__file__)
    return '{:s}/'.format('/'.join(path.split('/')[:-1]))
# --------------------------------------------------------------------
def single_test(ref=1, l2=1000):

    # Change current path
    # -------------------
    path_this = getpath_this()
    path_this = check_slash(path_this)

    path_run = os.getcwd()
    path_run = check_slash(path_run)

    path_code = "{:s}../".format(path_this)
    os.chdir(path_code)

    # Prepare files
    # -------------
    cmd_list = ["python", "conversion2cpp.py", "-p", "tests/intermediate/", "-r", "{:d}-".format(ref)]
    output = subprocess.Popen(cmd_list, stdout=subprocess.PIPE).communicate()[0]
    if output!="": print output

    # Execute the C++ routine
    # -----------------------
    cmd_list = ["./libcpm", "tests/intermediate/", "{:d}-".format(ref), "{:.1f}".format(l2)]
    output = subprocess.Popen(cmd_list, stdout=subprocess.PIPE).communicate()[0]
    if output != "": print output

    # Move the output files
    # ---------------------
    cmd_list = ["mv", "tests/intermediate/{:d}-cpmflux.dat".format(ref), "tests/output/"]
    output = subprocess.Popen(cmd_list, stdout=subprocess.PIPE).communicate()[0]
    if output != "": print output

    cmd_list = ["mv", "tests/intermediate/{:d}-result.dat".format(ref), "tests/output/"]
    output = subprocess.Popen(cmd_list, stdout=subprocess.PIPE).communicate()[0]
    if output != "": print output

    cmd_list = ["rm", "tests/intermediate/{:d}-epoch_mask.cpp.dat".format(ref)]
    output = subprocess.Popen(cmd_list, stdout=subprocess.PIPE).communicate()[0]
    if output != "": print output

    cmd_list = ["rm", "tests/intermediate/{:d}-pixel_flux.cpp.dat".format(ref)]
    output = subprocess.Popen(cmd_list, stdout=subprocess.PIPE).communicate()[0]
    if output != "": print output

    cmd_list = ["rm", "tests/intermediate/{:d}-pre_matrix_xy.cpp.dat".format(ref)]
    output = subprocess.Popen(cmd_list, stdout=subprocess.PIPE).communicate()[0]
    if output != "": print output

    cmd_list = ["rm", "tests/intermediate/{:d}-predictor_epoch_mask.cpp.dat".format(ref)]
    output = subprocess.Popen(cmd_list, stdout=subprocess.PIPE).communicate()[0]
    if output != "": print output

    # Change current path
    # -------------------
    path_test = "tests/"
    os.chdir(path_test)

    # Define file names
    # -----------------
    file_out_result = 'output/{:}-result.dat'.format(ref)
    file_expect_result = 'output/expected/{:}-result.dat'.format(ref)
    file_out_dif = 'output/{:}-cpmflux.dat'.format(ref)
    file_expect_dif = 'output/expected/{:}-dif.dat'.format(ref)

    # Load result files
    # -----------------
    out_result = np.loadtxt(file_out_result)
    expect_result = np.loadtxt(file_expect_result)
    out_dif = np.loadtxt(file_out_dif)
    expect_dif = np.loadtxt(file_expect_dif)

    # Test the differences
    # --------------------
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
    single_test(ref=1, l2=1000)
# --------------------------------------------------------------------