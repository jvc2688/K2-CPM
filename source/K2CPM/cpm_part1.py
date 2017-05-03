from __future__ import print_function

import sys
import numpy as np
from sklearn.decomposition import PCA
import argparse

from K2CPM import matrix_xy, multipletpf, tpfdata
from K2CPM import wcsfromtpf


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
    tpfdata.TpfData.directory = input_dir

    flux_lim_step_down = 0.1
    flux_lim_step_up = 0.1
    min_flux_lim = 0.1
    n_use = 15 # THIS HAS TO BE CHANGED. !!!

    tpf_data = tpfdata.TpfData(epic_id=target_epic_num, campaign=camp)
    wcs = wcsfromtpf.WcsFromTpf(tpf_data.channel, camp)
    m_tpfs = multipletpf.MultipleTpf()
    m_tpfs.add_tpf_data(tpf_data)
    
    #for epic_num in epic_list:
    #    if epic_num == 200070874 or epic_num == 200070438:
    #        continue
    # XXX THE ABOVE LOOP SHOULD BE SOMEHOW TRANSLATED.

    if pixel_list is None:
        print('no pixel list, run cpm on full tpf')
        pixel_list = tpf_data.pixel_list

    out_predictor_matrixes = []
    out_predictor_epoch_masks = []

    for pixel in pixel_list:
        print(pixel[0], pixel[1])
        if not tpf_data.check_pixel_in_tpf(column=pixel[1], row=pixel[0]):
            print('pixel out of range')
        elif tpf_data.check_pixel_covered(column=pixel[1], row=pixel[0]):
            (ra, dec) = wcs.radec_for_pixel(column=pixel[1], row=pixel[0])
            (epics_to_use_all, _, _) = wcs.get_epics_around_radec(ra, dec)
            epics_to_use = epics_to_use_all[:n_use]
            m_tpfs.add_tpf_data_from_epic_list(epics_to_use, camp)
            
            predictor_matrix = tpf_data.get_predictor_matrix(pixel[0], pixel[1], num_predictor, dis=dis, excl=excl,
                                                        flux_lim=flux_lim, 
                                                        multiple_tpfs=m_tpfs, tpfs_epics=epics_to_use)
            while predictor_matrix.shape[1] < num_predictor:
                n_use += 3
                epics_to_use = epics_to_use_all[:n_use]
                m_tpfs.add_tpf_data_from_epic_list(epics_to_use[-3:], camp)
# Re-code stuff below ???
                    #low_lim = flux_lim[0] - flux_lim_step_down
                    #up_lim = flux_lim[1] + flux_lim_step_up
                    #while predictor_matrix.shape[1] < num_predictor:
                    #    old_num = predictor_matrix.shape[1]
                    #    predictor_matrix = tpf_data.get_predictor_matrix(pixel[0], pixel[1], num_predictor, dis=dis, excl=excl,
                    #                                    flux_lim=(low_lim,up_lim), 
                    #                                    multiple_tpfs=m_tpfs, tpfs_epics=epics_to_use)
                    #    low_lim = np.max(low_lim-flux_lim_step_down, min_flux_lim)
                    #    up_lim = up_lim + flux_lim_step_up
                    #    difference = predictor_matrix.shape[1] - old_num
                    #    if difference == 0:
                    #        print('no more pixel at all')
                    #        break
                    #break
                predictor_matrix = tpf_data.get_predictor_matrix(pixel[0], pixel[1], num_predictor, dis=dis, excl=excl,
                                                        flux_lim=(low_lim, up_lim),
                                                        multiple_tpfs=m_tpfs, tpfs_epics=epics_to_use)

            if num_pca>0:
                pca = PCA(n_components=num_pca, svd_solver='full')
                pca.fit(predictor_matrix)
                predictor_matrix = pca.transform(predictor_matrix)
                
            out_predictor_epoch_masks.append(m_tpfs.predictor_epoch_mask)

            predictor_matrix = predictor_matrix[m_tpfs.predictor_epoch_mask]
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
