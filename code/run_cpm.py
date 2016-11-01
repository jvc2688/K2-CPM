import k2_cpm as k2cpm
import sys
import epic
import numpy as np
from sklearn.decomposition import PCA


def run(target_epic_num, camp, num_predictor, l2, num_pca, dis, excl, flux_lim, input_dir, output_dir, pixel_list=None):
	epic.load_tpf(target_epic_num, camp, input_dir)
	file_name = input_dir+'/'+'ktwo%d-c%d_lpd-targ.fits.gz'%(target_epic_num, camp)
	tpf = k2cpm.tpf(file_name)
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
	    tpfs.append(k2cpm.tpf(input_dir+'/'+'ktwo%d-c%d_lpd-targ.fits.gz'%(int(epic_num), camp)))

	if pixel_list == None:
		print 'no pixel list, run cpm on full tpf'
		pixel_list = np.array([np.repeat(np.arange(shape[0]), shape[1]), np.tile(np.arange(shape[1]), shape[0])], dtype=int).T
	data_len = pixel_list.shape[0]
	dif_file = np.zeros([tpf.flux.shape[0], data_len])+np.nan
	fit_file = np.zeros([tpf.flux.shape[0], data_len])
	pixel_idx = 0
	for pixel in pixel_list:
		x = pixel[0]
		y = pixel[1]
		print(x,y)
		if tpf.kplr_mask[x,y]>0:
			print len(tpfs)
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
						print epic_num
						tpfs.append(k2cpm.tpf(input_dir+'/'+'ktwo%d-c%d_lpd-targ.fits.gz'%(int(epic_num), camp)))
					predictor_matrix, predictor_epoch_mask = tpf.get_predictor_matrix(x, y, num_predictor, dis=dis, excl=excl, flux_lim=flux_lim, tpfs=tpfs, var_mask=None)
					print predictor_matrix.shape
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
	np.save(output_dir+'/'+'%d-c%d_fit.npy'%(target_epic_num, camp), fit_file)
	np.save(output_dir+'/'+'%d-c%d_dif.npy'%(target_epic_num, camp), dif_file)

def main():
	epic_num = int(sys.argv[1])
	camp = int(sys.argv[2])
	num_predictor = int(sys.argv[3])
	l2 = float(sys.argv[4])
	num_pca = int(sys.argv[5])
	dis = int(sys.argv[6])
	excl = int(sys.argv[7])
	input_dir = sys.argv[8] #'/Users/dunwang/Desktop/k2c9b'
	output_dir = sys.argv[9] #'/Users/dunwang/Desktop/k2c9b/tpf'
	pixel_list =None
	if len(sys.argv)>10:
		pixel_file = sys.argv[10]
		pixel_list = np.loadtxt(pixel_file, dtype=int)
	flux_lim = (0.2, 1.5)
	run(epic_num, camp, num_predictor, l2, num_pca, dis, excl, flux_lim, input_dir, output_dir, pixel_list)

if __name__ == '__main__':
	main()