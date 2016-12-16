from __future__ import print_function

import k2_cpm as k2cpm
import sys
import epic
import numpy as np
from sklearn.decomposition import PCA
import argparse
from astropy.io import fits as pyfits

def write2fits(time, data, tpf, out_file, cpm_set):
	table = np.core.records.fromarrays([time, data], dtype=[('TIME', '>f8'), ('FLUX_DIF', '>f4', (data.shape[1],data.shape[2]))])
	hdub = pyfits.BinTableHDU.from_columns([pyfits.Column(name='TIME', format='D',unit='JD', array=time), pyfits.Column(name='FLUX_DIF',format='{0}E'.format(data.shape[1]*data.shape[2]), unit='e-/s', dim='({0},{1})'.format(data.shape[1], data.shape[2]), array=data)])

	tpf_hdu_list = pyfits.open(tpf)
	tpf_header = tpf_hdu_list[1].header
	
	prihdu = pyfits.PrimaryHDU()
	hdulist = pyfits.HDUList([prihdu, hdub, tpf_hdu_list[2]])
	hdulist.writeto(out_file)
	
	with pyfits.open(out_file, mode='update') as hdul:
		header = hdul[1].header                                                                                                                                                                                                      
		header.comments['TTYPE1'] = 'column title: data time stamps'
		header.comments['TFORM1'] = 'column format: 64-bit floating point' 
		header.comments['TTYPE1'] = 'column units: JD'

		header.comments['TTYPE2'] = 'column title: CPM difference pixel flux'
		header.comments['TTYPE2'] = 'column units: electrons per second'
		header['WCSN2P'] = 'PHYSICAL'
		header.comments['WCSN2P'] = 'table column WCS name'
		header['WCAX2P'] = 2 
		header.comments['WCAX2P'] = 'table column physical WCS dimensions'


		header['1CTYP2'] = tpf_header['1CTYP5']
		header['2CTYP2'] = tpf_header['2CTYP5']
		header['1CRPX2'] = tpf_header['1CRPX5']
		header['2CRPX2'] = tpf_header['2CRPX5']
		header['1CRVL2'] = tpf_header['1CRVL5']
		header['2CRVL2'] = tpf_header['2CRVL5']
		header['1CUNI2'] = tpf_header['1CUNI5']
		header['2CUNI2'] = tpf_header['2CUNI5']
		header['1CDLT2'] = tpf_header['1CDLT5']
		header['2CDLT2'] = tpf_header['2CDLT5']
		header['11PC2'] = tpf_header['11PC5']
		header['12PC2'] = tpf_header['12PC5']
		header['21PC2'] = tpf_header['21PC5']
		header['22PC2'] = tpf_header['22PC5']
		header['TELESCOP'] = 'Kepler'                                                                                
		header['OBJECT'] = tpf_header['OBJECT']                  
		header['KEPLERID'] = tpf_header['KEPLERID']

		for key in cpm_set.keys():
			header[key] = cpm_set[key]


def run(target_epic_num, camp, num_predictor, l2, num_pca, dis, excl, flux_lim, input_dir, output_dir, pixel_list=None, train_lim=None):

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
		if (x<0) or (x>=tpf.kplr_mask.shape[0]) or (y<0) or (y>=tpf.kplr_mask.shape[1]):
			print('pixel out of range')
		elif (tpf.kplr_mask[x,y])>0:
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
						old_num = predictor_matrix.shape[1]
						predictor_matrix, predictor_epoch_mask = tpf.get_predictor_matrix(x, y, num_predictor, dis=dis, excl=excl, flux_lim=(low_lim,up_lim), tpfs=tpfs, var_mask=None)
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
					predictor_matrix, predictor_epoch_mask = tpf.get_predictor_matrix(x, y, num_predictor, dis=dis, excl=excl, flux_lim=flux_lim, tpfs=tpfs, var_mask=None)
					print(predictor_matrix.shape)
					epic_list = epic_list_new

			if num_pca>0:
				pca = PCA(n_components=num_pca)
				pca.fit(predictor_matrix)
				predictor_matrix = pca.transform(predictor_matrix)
			index = tpf.get_index(x,y)
			flux, predictor_matrix, flux_err, l2_vector, time, target_epoch_mask, data_mask \
			    = k2cpm.get_fit_matrix_ffi(tpf.flux[:,index], tpf.epoch_mask, predictor_matrix, predictor_epoch_mask, l2, tpf.time, 0, None)

			thread_num = 1
			result = k2cpm.fit_target_no_train(flux, tpf.kplr_mask, np.copy(predictor_matrix), time, target_epoch_mask[data_mask>0], None, l2_vector, thread_num, train_lim)
			fit_flux = np.dot(predictor_matrix, result)
			dif = flux-fit_flux[:,0]
			fit_file[target_epoch_mask, pixel_idx] = fit_flux[:,0]
			dif_file[target_epoch_mask, pixel_idx] = dif
		pixel_idx += 1
	#np.save(output_dir+'-fit.npy', fit_file)
	#np.save(output_dir+'-dif.npy', dif_file)
	if data_len == kplr_mask.shape[0]*kplr_mask.shape[1]:
		dif_file = def_file.reshape((tpf.flux.shape[0], kplr_mask.shape[0], kplr_mask.shape[1]))
	else:
		dif_file = def_file.reshape((tpf.flux.shape[0], data_len, 1))
	cpm_set = {'Np': num_predictor, 'l2':l2, 'num_pca': num_pca, 'dis':dis, 'excl': excl, 'flux_lim':'{0}'.format(flux_lim), 'Tlim':'{0}'.format(train_lim)}
	write2fits(time, dif_file, input_dir+'/'+'ktwo{0:d}-c{1:d}_lpd-targ.fits.gz'.format(target_epic_num, camp), output_dir+'-dif.fits', cpm_set)

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
	parser.add_argument('-t', '--train', nargs=2, metavar='train_lim', help="lower and upper limit defining the training data set")

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

	if args.pixel is not None:
		pixel_list = np.loadtxt(args.pixel, dtype=int)
		print("pixel list: {0}".format(args.pixel))
	else:
		pixel_list = None
		print("full image")
	flux_lim = (0.2, 1.5)

	if args.train is not None:
		train_lim = (float(args.train[0]), float(args.train[1]))
		print("train limit: {0}".format(train_lim)) 
	else:
		trian_lim = None
		print("all data used")
	run(args.epic[0], args.campaign[0], args.n_predictor[0], args.l2[0], args.n_pca[0], args.distance[0], args.exclusion[0], flux_lim, args.input_dir[0], args.output_dir[0], pixel_list, train_lim)

if __name__ == '__main__':
	main()

