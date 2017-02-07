
import os
import numpy as np
import urllib
from sklearn.decomposition import PCA

from astropy.io import fits as pyfits

import channelinfo


class TpfData(object):
    """handles data read from TPF file"""

    directory = None

    def __init__(self, epic_id=None, campaign=None, file_name=None, load_data=True):
        if (epic_id is None) != (campaign is None):
            raise ValueError('wrong parameters epic_id and campaign in TpfData.__init__()')
        if (file_name is not None) and (epic_id is not None):
            raise ValueError('in TpfData.__init__() you cannot specify file_name and epic_id at the same time')
        self.epic_id = epic_id
        self.campaign = campaign
        if file_name is None:
            file_name = self._guess_file_name()
        if load_data:
            self._load_data(file_name)  
        self._column = None
        self._row = None

    def _guess_file_name(self):
        """guesses file name based on epic_id and campaign"""
        return 'ktwo{:}-c{:}_lpd-targ.fits.gz'.format(self.epic_id, self.campaign)

    def _load_data(self, file_name):
        """loads header information and data from given file"""
        hdu_list = pyfits.open(file_name)
        self.ra_object = hdu_list[2].header['RA_OBJ']
        self.dec_object = hdu_list[2].header['DEC_OBJ']
        self.channel = hdu_list[0].header['CHANNEL']
        self.reference_column = hdu_list[2].header['CRVAL1P']
        self.reference_row = hdu_list[2].header['CRVAL2P']
        self.pixel_mask = hdu_list[2].data
        
        data = hdu_list[1].data
        self.jd_short = data["time"] + 4833. # is it HJD, BJD, JD?
        self.quality_flags = data["quality"].astype(dtype=int)  
        flux = data["flux"]
        pixel_mask = np.isfinite(flux) & (flux != 0)
        pixel_mask[:, self.pixel_mask < 1] = False
        self.pixel_mask = pixel_mask 
        quality_flags = data["quality"]
        quality_flags_ok = ((quality_flags == 0) | (quality_flags == 8192) | (quality_flags == 16384) | (quality_flags == 24576)) # TO BE DONE - can someone check if these are the only flags we should remove? Should we change it to a parameter? 
        foo = np.sum(np.sum((self.pixel_mask > 0), axis=2), axis=1) # Does anybody understand what is happening here?
        self.epoch_mask = (foo > 0) & np.isfinite(self.jd_short) & quality_flags_ok
        flux = flux[:, self.pixel_mask>0]
        if not all(np.isfinite(flux)):
            raise ValueError('non-finite value in flux table of {:} - feature not done yet'.format(file_name))
            # TO BE DONE - code interpolation using e.g. k2_cpm.py lines: 89-92
        self.flux = flux
        self.median = np.median(flux, axis=0)

        hdu_list.close()
# ra -> ra_object
# dec -> dec_object
# kplr_mask -> pixel_mask 
# ref_row -> reference_row
# ref_col -> reference_column
# time -> jd_short

    @property
    def _path(self):
        """path to the TPF file"""
        return TpfData.directory + '/' + self.file_name

    def verify_and_download(self):
        """check if file is where it should and download if not"""
        if os.path.isfile(self._path):
            return
        # File does not exist, so we download it
        d1 = self.epic_id - self.epic_id % 100000
        d2 = self.epic_id % 100000 - self.epic_id % 1000
        url_template = 'http://archive.stsci.edu/missions/k2/target_pixel_files/c{0:d}/{1:d}/{2:05d}/{3}'
        url_to_load = url_template.format(self.campaign, d1, d2, self.file_name)
        
        url_retriver = urllib.URLopener()
        url_retriver.retrieve(url_to_load, self._path)
    
    def get_position_inside(self, row, column):
        """get position of (row, column) inside given TPF file; returns (None, None) if pixel is not in mask range"""
        relative_row = row - self.reference_row # This is X in Dun's code.
        relative_column = column - self.reference_column # This is Y in Dun's code.
        if relative_row < 0 or relative_row >= self.pixel_mask.shape[0]:
            return (None, None)
        if relative_column < 0 or relative_column >= self.pixel_mask.shape[1]:
            return (None, None)
        return (relative_row, relative_column)

    def check_row_column_pixel_mask(self, row, column):
        """check if given (row, column) pixel has data taken (assuming you've choosen right TPF file!)"""
        (relative_row, relative_column) = self.get_position_inside(row, column)
        if relative_row is None:
            return False
        return bool(self.pixel_mask[relative_row, relative_column])

    def _make_column_row_vectors(self):
        """prepare vectors with some numbers"""
        self._column = np.tile(np.arange(self.pixel_mask.shape[1], dtype=int), self.pixel_mask.shape[0]) + self.reference_column
        self._column = self._column[self.pixel_mask.flatten()>0]
        self._row = np.repeat(np.arange(self.pixel_mask.shape[0], dtype=int), self.pixel_mask.shape[1]) + self.reference_row
        self._row = self._row[self.pixel_mask.flatten()>0]

    @property
    def pixel_column(self):
        """return pixel_column"""
        if self._column is None:
            self._make_column_row_vectors()
        return self._column

    @property
    def pixel_row(self):
        """return pixel_row"""
        if self._row is None:
            self._make_column_row_vectors()
        return self._row

    def _get_indexes_for_pixel(self, x, y):
        """find indexes in main tables that give all the epochs for given pixel"""
        index = np.arange(self.pixel_row.shape[0])
        index_mask = ((self.pixel_row - self.reference_row) == x) & ((self.pixel_column - self.reference_column) == y)
        try:
            return index[index_mask][0]
        except IndexError:
            raise IndexError('No data for ({0:d},{1:d})'.format(x, y)) # I'm not sure how we would like to handle this type of an error
    
    def get_predictor_matrix(self, target_x, target_y, multiple_tpf, n_predictor_pixel=100, exclude_columns=1, exclude_rows=None, distance_limit=16, flux_lim_ratio_median=(0.8, 1.2)):
        """prepare predictor matrix"""
        target_index = self._get_indexes_for_pixel(target_x, target_y)
        column_data = self.pixel_column[target_index]
        row_data = self.pixel_row[target_index]

        pixel_mask = np.ones_like(self.pixel_row, dtype=bool)
        if exclude_columns is not None: # exclude_columns = 0 would still remove a single column.
            for delta_pixel in range(-exclude_columns, exclude_columns+1):
                pixel_mask &= (self.pixel_column != (column_data + delta_pixel))
        if exclude_rows is not None: # exclude_rows = 0 would still remove a single row.
            for delta_pixel in range(-exclude_rows, exclude_rows+1):
                pixel_mask &= (self.pixel_row != (row_data + delta_pixel))    
        if flux_lim_ratio_median[0] is not None:
            pixel_mask &= (self.median[target_index] * flux_lim_ratio_median[0] <= multiple_tpf.pixel_median)
        if flux_lim_ratio_median[1] is not None:
            pixel_mask &= (self.median[target_index] * flux_lim_ratio_median[1] <= multiple_tpf.pixel_median)
        
        distance_square = np.square(self.pixel_row[pixel_mask]-row_data) + np.square(self.pixel_column[pixel_mask]-column_data)
        distance_mask = (distance_square > distance_limit**2)  
        distance_square = distance_square[distance_mask]
        index = np.argsort(distance_square)
        pixel_flux = multiple_tpf.pixel_flux[:,pixel_mask][:,distance_mask]
        self.predictor_matrix = pixel_flux[:,index[:n_predictor_pixel]].astype(float)
        # predictor_matrix was previously named predictor_flux

    def get_predictor_matrix_adjust_median_limits(self, target_x, target_y, multiple_tpf, n_predictor_pixel=100, exclude_columns=1, exclude_rows=None, distance_limit=16, flux_lim_ratio_median_start=(0.8, 1.2), step_lower_limit=0.1, minimum_lower_limit=0.1, step_upper_limit=0.1):
        """expands median flux ratio limits and calls get_predictor_matrix() until it gets enough stars"""
        lower_limit = flux_lim_ratio_median_start[0]
        upper_limit = flux_lim_ratio_median_start[1]
        # TO DE DONE - The loop below should be modified - the condition (difference != 0) may be true at some point, but not true later i.e. with smaller lower_limit and higher upper_limit.
        while True:
            old_n_pixels_ok = self.predictor_matrix.shape[1]
            lower_limit = np.max(lower_limit - step_lower_limit, minimum_lower_limit)
            upper_limit = upper_limit + step_upper_limit
            
            self.get_predictor_matrix(target_x=target_x, target_y=target_y, multiple_tpf=multiple_tpf, n_predictor_pixel=n_predictor_pixel, exclude_columns=exclude_columns, exclude_rows=exclude_rows, distance_limit=distance_limit, flux_lim_ratio_median=(lower_limit, upper_limit))
            
            n_pixels_ok = self.predictor_matrix.shape[1]
            if ((n_pixels_ok >= n_predictor_pixel) 
               or (n_pixels_ok == old_n_pixels_ok)):
                break

    def remove_largest_from_set(self, input_set, except_largest):
        """if set contains ids of very large TPF files, then remove those"""
        huge_tpf = hugetpf.HugeTpf(n_huge=except_largest, campaign=self.campaign)
        for epic_id in huge_tpf.huge_ids:
            if epic_id in input_set:
                input_set.remove(epic_id)

    def get_predictor_matrix_expand(self, target_x, target_y, target_ra, target_dec, radius_start, except_largest, epic_list_start, multiple_tpf, n_predictor_pixel=100, exclude_columns=1, exclude_rows=None, distance_limit=16, flux_lim_ratio_median_start=(0.8, 1.2), step_radius=6, step_lower_limit=0.1, minimum_lower_limit=0.1, step_upper_limit=0.1):
        """increase radius of search and also change median ratio flux limits trying to get to n_predictor_pixel pixels in predictor matrix"""
        n_pixels_ok = self.predictor_matrix.shape[1]
        channel_info = channelinfo.ChannelInfo(campaign=self.campaign, channel=int(self.channel))
        channel_info.radius = radius_start
        epic_list_old = epic_list_start
        # TO BE DONE - The output of the loop below depends on many settings: exclude_columns, exclude_rows, step_radius, step_lower_limit, minimum_lower_limit, step_upper_limit... - THIS HAS TO BE CHANGED 
        while n_pixels_ok < n_predictor_pixel:
            channel_info.radius += step_radius
            channel_info.epic_ids_near_ra_dec_fixed_radius(ra=target_ra, dec=target_dec, except_largest=except_largest)
            epic_set_new = set(channel_info.epic_list)
            new_epic_ids = epic_set_new.difference(epic_list_old)
            if not new_epic_ids:
                self.get_predictor_matrix_adjust_median_limits(self, target_x=target_x, target_y=target_y, multiple_tpf=multiple_tpf, n_predictor_pixel=n_predictor_pixel, exclude_columns=exclude_columns, exclude_rows=exclude_rows, distance_limit=distance_limit, flux_lim_ratio_median_start=flux_lim_ratio_median_start, step_lower_limit=step_lower_limit, minimum_lower_limit=minimum_lower_limit, step_upper_limit=step_upper_limit)
                break
            else: # When new_epic_ids is not empty.
                self.remove_largest_from_set(new_epic_ids, except_largest=except_largest)
                for epic_id in new_epic_ids:
                    new_tpf_data = TpfData(epic_id=epic_id, campaign=self.campaign)
                    multiple_tpf.add_tpf_data(new_tpf_data)
                self.get_predictor_matrix(target_x=target_x, target_y=target_y, multiple_tpf=multiple_tpf, n_predictor_pixel=n_predictor_pixel, exclude_columns=exclude_columns, exclude_rows=exclude_rows, distance_limit=distance_limit, flux_lim_ratio_median=flux_lim_ratio_median_start)

    def apply_pca_to_predictor_matrix(n_pca_components):
        """apply Principal Component Analysis to predictor matrix"""
        pca = PCA(n_components=n_pca_components)
        pca.fit(self.predictor_matrix)
        self.predictor_matrix = pca.transform(self.predictor_matrix)

