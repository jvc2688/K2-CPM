from __future__ import print_function

import sys
import numpy as np
from sklearn.decomposition import PCA
import argparse

from K2CPM import k2_cpm as k2cpm
from K2CPM import epic
from K2CPM import matrix_xy
from K2CPM import multipletpf
from K2CPM import tpfdata


def run_cpm_part1(target_epic_num, camp, num_predictor, num_pca, dis, excl, 
                    flux_lim, input_dir, pixel_list=None, train_lim=None, 
                    output_file=None, 
                    return_predictor_epoch_masks=False):
# REMOVED: l2, output_dir
# ADDED: output_file, return_predictor_epoch_masks
#def run_cpm_part1(target_epic_num, camp, num_predictor, l2, num_pca, dis, excl, flux_lim, input_dir, output_dir, pixel_list=None, train_lim=None):

    if pixel_list is not None:
        if pixel_list.shape[0] != 1 and output_file is not None:
            raise ValueError('\n\nCurrently we can deal with only a single pixel at a time if the output file is specified')
        
    epic.load_tpf(target_epic_num, camp, input_dir)
    file_name = input_dir+'/'+'ktwo{0:d}-c{1:d}_lpd-targ.fits.gz'.format(target_epic_num, camp)
    tpf = k2cpm.Tpf(file_name)
    shape = tpf.kplr_mask.shape
    ra = tpf.ra
    dec = tpf.dec
    r = 0
    tpfs = []
    epic_list = []
    while len(epic_list)<=5:
        r+=6
        epic_list = epic.get_tpfs(ra,dec,r,camp, tpf.channel)

    for epic_num in epic_list:
        if epic_num == 200070874 or epic_num == 200070438:
            continue
        epic.load_tpf(int(epic_num), camp, input_dir)
        try:
            tpfs.append(k2cpm.Tpf(input_dir+'/'+'ktwo{0}-c{1}_lpd-targ.fits.gz'.format(int(epic_num), camp)))
        except IOError:
            os.remove(input_dir+'/'+'ktwo{0}-c{1}_lpd-targ.fits.gz'.format(int(epic_num), camp))
            epic.load_tpf(int(epic_num), camp, input_dir)
            tpfs.append(k2cpm.Tpf(input_dir+'/'+'ktwo{0}-c{1}_lpd-targ.fits.gz'.format(int(epic_num), camp)))

    if pixel_list is None:
        print('no pixel list, run cpm on full tpf')
        pixel_list = np.array([np.repeat(np.arange(shape[0]), shape[1]), np.tile(np.arange(shape[1]), shape[0])], dtype=int).T
        pixel_list += np.array([tpf.ref_row, tpf.ref_col])
    data_len = pixel_list.shape[0]

    out_predictor_matrixes = []
    out_predictor_epoch_masks = []

    for pixel in pixel_list:
        x = pixel[0]-tpf.ref_row
        y = pixel[1]-tpf.ref_col
        print(x, y)
        if (x<0) or (x>=tpf.kplr_mask.shape[0]) or (y<0) or (y>=tpf.kplr_mask.shape[1]):
            print('pixel out of range')
        elif (tpf.kplr_mask[x,y])>0:
            print(len(tpfs))
            predictor_matrix, _ = tpf.get_predictor_matrix(x, y, num_predictor, dis=dis, excl=excl, flux_lim=flux_lim, tpfs=tpfs, var_mask=None)
            while predictor_matrix.shape[1]<num_predictor:
                r+=6
                epic_list_new = set(epic.get_tpfs(ra,dec,r,camp, tpf.channel))
                more = epic_list_new.difference(epic_list)
                if len(more) == 0:
                    low_lim = flux_lim[0]-0.1
                    up_lim = flux_lim[1]+0.1
                    while predictor_matrix.shape[1]<num_predictor:
                        old_num = predictor_matrix.shape[1]
                        predictor_matrix, _ = tpf.get_predictor_matrix(x, y, num_predictor, dis=dis, excl=excl, flux_lim=(low_lim,up_lim), tpfs=tpfs, var_mask=None)
                        low_lim = np.max(low_lim-0.1,0.1)
                        up_lim = up_lim+0.1
                        difference = predictor_matrix.shape[1] - old_num
                        if difference == 0:
                            print('no more pixel at all')
                            break
                    break
                elif len(more)>0:
                    for epic_num in more:
                        if epic_num == 200070874 or epic_num == 200070438:
                            continue
                        epic.load_tpf(int(epic_num), camp, input_dir)
                        print(epic_num)
                        tpfs.append(k2cpm.Tpf(input_dir+'/'+'ktwo{0:d}-c{1:d}_lpd-targ.fits.gz'.format(int(epic_num), camp)))
                    predictor_matrix, _ = tpf.get_predictor_matrix(x, y, num_predictor, dis=dis, excl=excl, flux_lim=flux_lim, tpfs=tpfs, var_mask=None)
                    print(predictor_matrix.shape)
                    epic_list = epic_list_new

            if num_pca>0:
                pca = PCA(n_components=num_pca)
                pca.fit(predictor_matrix)
                predictor_matrix = pca.transform(predictor_matrix)
                
            tpf_set = multipletpf.MultipleTpf()
            tpfdata.TpfData.directory = input_dir
            for tpf in tpfs:
                new_tpf = tpfdata.TpfData(epic_id=tpf.kid, campaign=tpf.campaign)
                tpf_set.add_tpf_data(new_tpf)
            out_predictor_epoch_masks.append(tpf_set.predictor_epoch_mask)
            # np.savetxt(output_file+"_epoch_mask", tpf_set.predictor_epoch_mask, fmt='%r')
            
            predictor_matrix = predictor_matrix[tpf_set.predictor_epoch_mask]
            out_predictor_matrixes.append(predictor_matrix)

            if output_file is not None: 
                matrix_xy.save_matrix_xy(predictor_matrix, output_file)

    if return_predictor_epoch_masks:
        return (out_predictor_matrixes, out_predictor_epoch_masks)
    else:
        return out_predictor_matrixes


def main():
    parser = argparse.ArgumentParser(description='k2 CPM')
    parser.add_argument('epic', nargs=1, type=int, help="epic number")
    parser.add_argument('campaign', nargs=1, type=int, help="campaign number, 91 for phase a, 92 for phase b")
    parser.add_argument('n_predictor', nargs=1, type=int, help="number of predictor pixels")
    parser.add_argument('n_pca', nargs=1, type=int, help="number of the PCA components to use, if 0, no PCA")
    parser.add_argument('distance', nargs=1, type=int, help="distance between target pixel and predictor pixels")
    parser.add_argument('exclusion', nargs=1, type=int, help="how many rows and columns that are excluded around the target pixel")
    parser.add_argument('input_dir', nargs=1, help="directory to store the output file")
    parser.add_argument('output_file', nargs=1, help="output predicotr matrix file")
    parser.add_argument('-p', '--pixel', metavar='pixel_list', help="path to the pixel list file that specify list of pixels to be modelled." 
                                                                    "If not provided, the whole target pixel file will be modelled")
    parser.add_argument('-t', '--train', nargs=2, metavar='train_lim', help="lower and upper limit defining the training data set")

    args = parser.parse_args()
    print("epic number: {0}".format(args.epic[0]))
    print("campaign: {0}".format(args.campaign[0]))
    print("number of predictors: {0}".format(args.n_predictor[0]))
    print("number of PCA components: {0}".format(args.n_pca[0]))
    print("distance: {0}".format(args.distance[0]))
    print("exclusion: {0}".format(args.exclusion[0]))
    print("directory of TPFs: {0}".format(args.input_dir[0]))
    print("output predictor_matrix file: {0}".format(args.output_file[0]))
    # Variables flux_lim, pixel_list, amd train_lim and used later but not printed here.

    if args.pixel is not None:
        pixel_list = np.loadtxt(args.pixel, dtype=int, ndmin=2)
        print("pixel list: {0}".format(args.pixel))
    else:
        pixel_list = None
        print("full image")
    flux_lim = (0.2, 1.5)

    if args.train is not None:
        train_lim = (float(args.train[0]), float(args.train[1]))
        print("train limit: {0}".format(train_lim)) 
    else:
        train_lim = None
        print("all data used")
        
    run_cpm_part1(args.epic[0], args.campaign[0], args.n_predictor[0], 
                    args.n_pca[0], args.distance[0], args.exclusion[0], 
                    flux_lim, args.input_dir[0], pixel_list, train_lim,
                    output_file=args.output_file[0]) 

if __name__ == '__main__':
    main()
