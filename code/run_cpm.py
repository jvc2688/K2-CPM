from __future__ import print_function

import k2_cpm as k2cpm
import sys
import epic
import numpy as np
from sklearn.decomposition import PCA
import argparse


def run(target_epic_num, camp, num_predictor, l2, num_pca, dis, excl, flux_lim, input_dir, output_dir, pixel_list=None):
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
        epic.load_tpf(int(epic_num), camp, input_dir)
        tpfs.append(k2cpm.Tpf(input_dir+'/'+'ktwo{0:d}-c{1:d}_lpd-targ.fits.gz'.format(int(epic_num), camp)))

    if pixel_list == None:
        print('no pixel list, run cpm on full tpf')
        pixel_list = np.array([np.repeat(np.arange(shape[0]), shape[1]), np.tile(np.arange(shape[1]), shape[0])], dtype=int).T
    data_len = pixel_list.shape[0]
    dif_file = np.zeros([tpf.flux.shape[0], data_len])+np.nan
    fit_file = np.zeros([tpf.flux.shape[0], data_len])
    pixel_idx = 0
    for pixel in pixel_list:
        x = pixel[0]
        y = pixel[1]
        print(x, y)
        if tpf.kplr_mask[x,y]>0:
            print(len(tpfs))
            predictor_matrix, predictor_epoch_mask = tpf.get_predictor_matrix(x, y, num_predictor, dis=dis, excl=excl, flux_lim=flux_lim, tpfs=tpfs, var_mask=None)
            while predictor_matrix.shape[1]<num_predictor:
                r+=6
                epic_list_new = set(epic.get_tpfs(ra,dec,r,camp, tpf.channel))
                more = epic_list_new.difference(epic_list)
                if len(more) == 0:
                    low_lim = flux_lim[0]-0.1
                    up_lim = flux_lim[1]+0.1
                    while predictor_matrix.shape[1]<num_predictor:
                        predictor_matrix, predictor_epoch_mask = tpf.get_predictor_matrix(x, y, num_predictor, dis=dis, excl=excl, flux_lim=(low_lim,up_lim), tpfs=tpfs, var_mask=None)
                        low_lim = np.max(low_lim-0.1,0.1)
                        up_lim = up_lim+0.1
                else:
                    for epic_num in more:
                        epic.load_tpf(int(epic_num), camp, input_dir)
                        print(epic_num)
                        tpfs.append(k2cpm.Tpf(input_dir+'/'+'ktwo{0:d}-c{1:d}_lpd-targ.fits.gz'.format(int(epic_num), camp)))
                    predictor_matrix, predictor_epoch_mask = tpf.get_predictor_matrix(x, y, num_predictor, dis=dis, excl=excl, flux_lim=flux_lim, tpfs=tpfs, var_mask=None)
                    print(predictor_matrix.shape)
                    epic_list = epic_list_new

            if num_pca>0:
                pca = PCA(n_components=num_pca)
                pca.fit(predictor_matrix)
                predictor_matrix = pca.transform(predictor_matrix)
            index = tpf.get_index(x,y)
            flux, predictor_matrix, flux_err, l2_vector, target_epoch_mask, data_mask \
                = k2cpm.get_fit_matrix_ffi(tpf.flux[:,index], tpf.epoch_mask, predictor_matrix, predictor_epoch_mask, l2, 0, 'lightcurve', None)

            thread_num = 1
            train_mask = None
            result = k2cpm.fit_target_no_train(flux, tpf.kplr_mask, np.copy(predictor_matrix), None, target_epoch_mask[data_mask>0], None, l2_vector, thread_num, train_mask)
            fit_flux = np.dot(predictor_matrix, result)
            dif = flux-fit_flux[:,0]
            fit_file[target_epoch_mask, pixel_idx] = fit_flux[:,0]
            dif_file[target_epoch_mask, pixel_idx] = dif
        pixel_idx += 1
    np.save(output_dir+'/'+'{0:d}-c{1:d}_fit.npy'.format(target_epic_num, camp), fit_file)
    np.save(output_dir+'/'+'{0:d}-c{1:d}_dif.npy'.format(target_epic_num, camp), dif_file)

def main():
    parser = argparse.ArgumentParser(description='k2 CPM')
    parser.add_argument('epic', nargs=1, type=int, help="epic number")
    parser.add_argument('campaign', nargs=1, type=int, help="campaign number, 91 for phase a, 92 for phase b")
    parser.add_argument('n_predictor', nargs=1, type=int, help="number of predictor pixels")
    parser.add_argument('l2', nargs=1, type=float, help="strength of l2 regularization")
    parser.add_argument('n_pca', nargs=1, type=int, help="number of the PCA components to use, if 0, no PCA")
    parser.add_argument('distance', nargs=1, type=int, help="distance between target pixel and predictor pixels")
    parser.add_argument('exclusion', nargs=1, type=int, help="how many rows and columns that are excluded around the target pixel")
    parser.add_argument('input_dir', nargs=1, help="directory to store the output file")
    parser.add_argument('output_dir', nargs=1, help="directory to the target pixel files")
    parser.add_argument('-p', '--pixel', metavar='pixel_list', help="path to the pixel list file that specify list of pixels to be modelled." 
                                                                    "If not provided, the whole target pixel file will be modelled")
    args = parser.parse_args()
    print("epic number: {0}".format(args.epic[0]))
    print("campaign: {0}".format(args.campaign[0]))
    print("number of predictors: {0}".format(args.n_predictor[0]))
    print("l2 regularization: {0}".format(args.l2[0]))
    print("number of PCA components: {0}".format(args.n_pca[0]))
    print("distance: {0}".format(args.distance[0]))
    print("exclusion: {0}".format(args.exclusion[0]))
    print("directory of TPFs: {0}".format(args.input_dir[0]))
    print("output directory: {0}".format(args.output_dir[0]))
    '''
    epic_num = int(sys.argv[1])
    campaign = int(sys.argv[2])
    n_predictor = int(sys.argv[3])
    l2 = float(sys.argv[4])
    n_pca = int(sys.argv[5])
    distance = int(sys.argv[6])
    exclusion = int(sys.argv[7])
    input_dir = sys.argv[8] #'/Users/dunwang/Desktop/k2c9b'
    output_dir = sys.argv[9] #'/Users/dunwang/Desktop/k2c9b/tpf'
    pixel_list =None
    if len(sys.argv)>10:
        pixel_file = sys.argv[10]
        pixel_list = np.loadtxt(pixel_file, dtype=int)
    '''
    if args.pixel is not None:
        pixel_list = np.loadtxt(args.pixel, dtype=int)
        print("pixel list: {0}".format(args.pixel))
    else:
        pixel_list = None
        print("full image")
    flux_lim = (0.2, 1.5)
    #run(epic_num, campaign, n_predictor, l2, n_pca, distance, exclusion, flux_lim, input_dir, output_dir, pixel_list)
    run(args.epic[0], args.campaign[0], args.n_predictor[0], args.l2[0], args.n_pca[0], args.distance[0], args.exclusion[0], flux_lim, args.input_dir[0], args.output_dir[0], pixel_list)

if __name__ == '__main__':
    main()

