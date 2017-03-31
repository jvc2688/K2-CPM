from __future__ import print_function

import sys
import numpy as np
from sklearn.decomposition import PCA
import argparse

from K2CPM import k2_cpm as k2cpm
from K2CPM import epic, matrix_xy, multipletpf, tpfdata


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

    epic.load_tpf(target_epic_num, camp, input_dir)
    file_name = epic.path_for_epic(input_dir, target_epic_num, camp)
    tpf = k2cpm.Tpf(file_name) # XXX
    tpf_data = tpfdata.TpfData(epic_id=target_epic_num, campaign=camp)
    shape = tpf.kplr_mask.shape
    ra = tpf_data.ra_object
    dec = tpf_data.dec_object
    r = 0
    tpfs = []
    epic_list = []
    while len(epic_list)<=5:
        r+=6
        epic_list = epic.get_tpfs(ra,dec,r,camp, tpf_data.channel)

    for epic_num in epic_list:
        if epic_num == 200070874 or epic_num == 200070438:
            continue
        epic.load_tpf(int(epic_num), camp, input_dir)
        tpf_file_name = epic.path_for_epic(input_dir, epic_num, camp)
        try:
            tpfs.append(k2cpm.Tpf(tpf_file_name))
        except IOError:
            os.remove(tpf_file_name)
            epic.load_tpf(int(epic_num), camp, input_dir)
            tpfs.append(k2cpm.Tpf(tpf_file_name))

    if pixel_list is None:
        print('no pixel list, run cpm on full tpf')
        pixel_list = tpf_data.pixel_list

    out_predictor_matrixes = []
    out_predictor_epoch_masks = []

    for pixel in pixel_list:
        dx = pixel[0] - tpf_data.reference_row
        dy = pixel[1] - tpf_data.reference_column
        print(dx, dy)
        if not tpf_data.check_pixel_in_tpf(column=pixel[1], row=pixel[0]):
            print('pixel out of range')
        elif tpf_data.check_pixel_covered(column=pixel[1], row=pixel[0]):
            print(len(tpfs))
            # NEW:
            predictor_matrix = tpf_data.get_predictor_matrix(dx, dy, num_predictor, dis=dis, excl=excl, 
                                                        flux_lim=flux_lim, tpfs=tpfs, var_mask=None)
            # OLD XXX :
            #(predictor_matrix, _) = tpf.get_predictor_matrix(dx, dy, 
                                                        #num_predictor, dis=dis, excl=excl, 
                                                        #flux_lim=flux_lim, tpfs=tpfs, var_mask=None)
            while predictor_matrix.shape[1]<num_predictor:
                r+=6
                epic_list_new = set(epic.get_tpfs(ra, dec, r, camp, tpf_data.channel))
                more = epic_list_new.difference(epic_list)
                if len(more) == 0:
                    low_lim = flux_lim[0] - flux_lim_step_down
                    up_lim = flux_lim[1] + flux_lim_step_up
                    while predictor_matrix.shape[1]<num_predictor:
                        old_num = predictor_matrix.shape[1]
                        # NEW XXX :
                        predictor_matrix = tpf_data.get_predictor_matrix(dx, dy, num_predictor, dis=dis, excl=excl, 
                                                        flux_lim=(low_lim,up_lim), tpfs=tpfs, var_mask=None)
                        # OLD XXX :
                        #(predictor_matrix, _) = tpf.get_predictor_matrix(dx, dy, 
                                                        #num_predictor, dis=dis, excl=excl, 
                                                        #flux_lim=(low_lim,up_lim), tpfs=tpfs, var_mask=None)
                        low_lim = np.max(low_lim-flux_lim_step_down, min_flux_lim)
                        up_lim = up_lim + flux_lim_step_up
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
                        tpfs.append(k2cpm.Tpf(epic.path_for_epic(input_dir, epic_num, camp)))
                    # OLD XXX :
                    predictor_matrix = tpf_data.get_predictor_matrix(dx, dy, num_predictor, dis=dis, excl=excl, 
                                                        flux_lim=[low_lim, up_lim], tpfs=tpfs, var_mask=None)
                    # NEW XXX :
                    #(predictor_matrix, _) = tpf.get_predictor_matrix(dx, dy, 
                                                        #num_predictor, dis=dis, excl=excl, 
                                                        #flux_lim=flux_lim, tpfs=tpfs, var_mask=None)
                    print(predictor_matrix.shape)
                    epic_list = epic_list_new

            if num_pca>0:
                pca = PCA(n_components=num_pca)
                pca.fit(predictor_matrix)
                predictor_matrix = pca.transform(predictor_matrix)
                
            tpf_set = multipletpf.MultipleTpf()
            for tpf in tpfs:
                new_tpf = tpfdata.TpfData(epic_id=target_epic_num, campaign=camp)
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
