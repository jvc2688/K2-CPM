# -*-coding:Utf-8 -*
# ================================================================== #
# Copyright 2017 Clement Ranc

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ================================================================== #
# This routine converts files to be read by the C++ version of CPM
# code.
# ================================================================== #

import os
import sys
import glob
import argparse
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
# ====================================================================
# Main
# ====================================================================
if (__name__ == "__main__"):

    # Find paths
    # ----------
    path_this = getpath_this()
    path_this = check_slash(path_this)

    # Command line options
    # --------------------
    text = 'Convert files to be read by C++ version of the CPM code.'
    parser = argparse.ArgumentParser(prog='python convert', description=text)
    parser.add_argument('-p', '--path', nargs=1, type=str, default=['./'], help='Path to the input files.')
    parser.add_argument('-r', '--ref', nargs=1, type=str, default=[''], help='Reference.')
    opts = parser.parse_args()
    options = dict()
    [options.update({key: getattr(opts, key)}) for key in vars(opts)]
    [options.update({key: np.atleast_1d(options[key])[0]}) for key in options if len(np.atleast_1d(options[key]))==1]

    options['path'] = check_slash(options['path'])
    if options['path'][0] != "/":
        options['path'] = "{:s}{:s}".format(path_this, options['path'])

    # Save data for C++ version
    # -------------------------
    if (options['ref'] != ""):
        fname = options['path'] + options['ref'] + "predictor_epoch_mask.dat"
        if not os.path.exists(fname):
            text = "File not found:\n{:s}".format(fname)
            sys.exit(text)
    else:
        fname = glob.glob(options['path'] + "*predictor_epoch_mask.dat")
        if len(fname) > 1:
            text = "Several files with different references are found."
            text += "\nPlease use the option -r to give the reference."
            sys.exit(text)
        if len(fname) == 1:
            fname = fname[0]
        if (len(fname) == 0) or (not os.path.exists(fname)):
            text = "File not found:\n{:s}".format(options['path'] + "*predictor_epoch_mask.dat")
            sys.exit(text)

    file = open(fname, 'r')
    fullfile = ""
    nb_line = 0
    for line in file:
        fullfile += line
        nb_line = nb_line + 1
    file.close()
    fullfile = fullfile.replace('\n', ' ')
    if fullfile[-1]==' ': fullfile = fullfile[:-1]
    fullfile = fullfile.upper()
    fullfile = fullfile.replace('TRUE', '1')
    fullfile = fullfile.replace('FALSE', '0')
    pre_epoch_mask = np.array(fullfile.replace('\n', '').split(' '))
    pre_epoch_mask = pre_epoch_mask.astype(np.int)
    fullfile = "{:d} 1 1\n".format(nb_line) + fullfile + "\n"
    fname = fname[:-3] + "cpp.dat"
    file = open(fname, 'w')
    file.write(fullfile)
    file.close()

    if (options['ref'] != ""):
        fname = options['path'] + options['ref'] + "pre_matrix_xy.dat"
        if not os.path.exists(fname):
            text = "File not found:\n{:s}".format(fname)
            sys.exit(text)
    else:
        fname = glob.glob(options['path'] + "*pre_matrix_xy.dat")
        if len(fname) > 1:
            text = "Several files with different references are found."
            text += "\nPlease use the option -r to give the reference."
            sys.exit(text)
        if len(fname) == 1:
            fname = fname[0]
        if (len(fname) == 0) or (not os.path.exists(fname)):
            text = "File not found:\n{:s}".format(options['path'] + "*pre_matrix_xy.dat")
            sys.exit(text)

    file = open(fname, 'r')
    fullfile = ""
    nb_line = 0
    for line in file:
        fullfile += line.split(' ')[2]
        nb_line = nb_line + 1
    file.close()
    fullfile = fullfile.replace('\n', ' ')
    if fullfile[-1]==' ': fullfile = fullfile[:-1]
    n1 = int(line.split(' ')[0]) + 1
    n2 = int(line.split(' ')[1]) + 1
    fullfile = "{:d} {:d} 1\n".format(n1, n2) + fullfile + "\n"
    fname = fname[:-3] + "cpp.dat"
    file = open(fname, 'w')
    file.write(fullfile)
    file.close()

    if (options['ref'] != ""):
        fname = options['path'] + options['ref'] + "pixel_flux.dat"
        if not os.path.exists(fname):
            text = "File not found:\n{:s}".format(fname)
            sys.exit(text)
    else:
        fname = glob.glob(options['path'] + "*pixel_flux.dat")
        if len(fname) > 1:
            text = "Several files with different references are found."
            text += "\nPlease use the option -r to give the reference."
            sys.exit(text)
        if len(fname) == 1:
            fname = fname[0]
        if (len(fname) == 0) or (not os.path.exists(fname)):
            text = "File not found:\n{:s}".format(options['path'] + "*pixel_flux.dat")
            sys.exit(text)

    file = open(fname, 'r')
    fullfile = ""
    nb_line = 0
    for line in file:
        fullfile += line
        nb_line = nb_line + 1
    file.close()
    epochs = [a.split(' ')[0] for a in fullfile.split('\n') if a != '']
    fullfile = fullfile.replace('\n', ' ')
    if fullfile[-1]==' ': fullfile = fullfile[:-1]
    fullfile = "{:d} 3 1\n".format(nb_line) + fullfile + "\n"
    fname = fname[:-3] + "cpp.dat"
    file = open(fname, 'w')
    file.write(fullfile)
    file.close()

    epochs = np.array([epochs[i] for i in xrange(len(epochs)) if pre_epoch_mask[i]])
    if fname[-1] == "/": fname = fname[:-1]
    fname = fname.split("/")[-1]
    fname = fname.split("pixel_flux.cpp.dat")[0]
    fname += "epochs_ml.dat"
    fname = options['path'] + fname
    np.savetxt(fname, epochs, fmt='%s')

    if (options['ref'] != ""):
        fname = options['path'] + options['ref'] + "*epoch_mask.dat"
        if not os.path.exists(fname):
            text = "File not found:\n{:s}".format(fname)
            sys.exit(text)
    else:
        fname = np.array(glob.glob(options['path'] + "*epoch_mask.dat"))
        fname_no = np.array(glob.glob(options['path'] + "*predictor_epoch_mask.dat"))
        fname = [a for a in fname if len(np.where(a == np.intersect1d(fname, fname_no))[0]) == 0]
        if len(fname) > 1:
            text = "Several files with different references are found."
            text += "\nPlease use the option -r to give the reference."
            sys.exit(text)
        if len(fname) == 1:
            fname = fname[0]
        if (len(fname) == 0) or (not os.path.exists(fname)):
            text = "File not found:\n{:s}".format(options['path'] + "*epoch_mask.dat")
            sys.exit(text)

    file = open(fname, 'r')
    fullfile = ""
    nb_line = 0
    for line in file:
        fullfile += line
        nb_line = nb_line + 1
    file.close()
    fullfile = fullfile.replace('\n', ' ')
    if fullfile[-1]==' ': fullfile = fullfile[:-1]
    fullfile = fullfile.upper()
    fullfile = fullfile.replace('TRUE', '1')
    fullfile = fullfile.replace('FALSE', '0')
    fullfile = "{:d} 1 1\n".format(nb_line) + fullfile + "\n"
    fname = fname[:-3] + "cpp.dat"
    file = open(fname, 'w')
    file.write(fullfile)
    file.close()
