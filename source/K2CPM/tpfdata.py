
import os
import sys
import warnings
import numpy as np
import urllib
from sklearn.decomposition import PCA
if sys.version_info[0] > 2:
    from urllib.request import URLopener
else:
    from urllib import URLopener

from astropy.io import fits as pyfits

from K2CPM import matrix_xy


class TpfData(object):
    """handles data read from TPF file"""

    directory = None # The directory where TPF files are stored.

    def __init__(self, epic_id=None, campaign=None, file_name=None):
        if (epic_id is None) != (campaign is None):
            raise ValueError('wrong parameters epic_id and campaign in TpfData.__init__()')
        if (file_name is not None) and (epic_id is not None):
            raise ValueError('in TpfData.__init__() you cannot specify file_name and epic_id at the same time')
        self.epic_id = epic_id
        self.campaign = campaign
        if file_name is None:
            file_name = self._guess_file_name()
        self.file_name = file_name
        self.verify_and_download()
        self._load_data(self._path)
        self._column = None
        self._row = None
        self._pixel_list = None

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
        self.mask = hdu_list[2].data 
        self.n_rows = self.mask.shape[0]
        self.n_columns = self.mask.shape[1]
        self.n_pixels = self.n_rows * self.n_columns
        
        data = hdu_list[1].data
        self.jd_short = data["time"] + 4833. # is it HJD, BJD, JD?
        self.quality_flags = data["quality"].astype(dtype=int)  
        flux = data["flux"]
        pixel_mask = np.isfinite(flux) & (flux != 0)
        pixel_mask[:, self.mask < 1] = False
        self.pixel_mask = pixel_mask 
        
        quality_flags = data["quality"]
        # TO_BE_DONE - can someone check if these are the only flags we should remove? Should we change it to a parameter? 
        quality_flags_ok = ((quality_flags == 0) | (quality_flags == 8192) | (quality_flags == 16384) | (quality_flags == 24576)) 
        foo = np.sum(np.sum((self.pixel_mask > 0), axis=2), axis=1) # Does anybody understand what is happening here?
        self.epoch_mask = (foo > 0) & np.isfinite(self.jd_short) & quality_flags_ok
        self.jd_short_masked = self.jd_short[self.epoch_mask]
        flux = flux[:, self.mask>0]
        if not np.isfinite(flux).all():
            raise ValueError('non-finite value in flux table of {:} - feature not done yet'.format(file_name))
            # TO_BE_DONE - code interpolation using e.g. k2_cpm.py lines: 89-92
            # TO_BE_DONE - also checks on flux_err?
        self.flux = flux
        self.median = np.median(flux, axis=0)

        flux_err = data["flux_err"]
        flux_err = flux_err[:, self.mask>0]
        self.flux_err = flux_err

        hdu_list.close()

    @property
    def _path(self):
        """path to the TPF file"""
        if TpfData.directory is None:
            raise ValueError("TpfData.directory value not set")
        return os.path.join(TpfData.directory, self.file_name)

    def verify_and_download(self):
        """check if file is where it should and download if not"""
        if os.path.isfile(self._path):
            return
        # File does not exist, so we download it
        epic_id = int(self.epic_id)
        d1 = epic_id - epic_id % 100000
        d2 = epic_id % 100000 - epic_id % 1000
        url_template = 'http://archive.stsci.edu/missions/k2/target_pixel_files/c{0:d}/{1:d}/{2:05d}/{3}'
        url_to_load = url_template.format(self.campaign, d1, d2, self.file_name)
        
        url_retriver = URLopener()
        url_retriver.retrieve(url_to_load, self._path)
    
    @property
    def reference_pixel(self):
        """return array that gives reference pixel position"""
        return np.array([self.reference_column, self.reference_row])

    @property
    def pixel_list(self):
        """return array with a list of all pixels"""
        if self._pixel_list is None:
            inside_1 = np.repeat(np.arange(self.n_columns), self.n_rows)
            inside_2 = np.tile(np.arange(self.n_rows), self.n_columns)
            inside_coords = np.array([inside_1, inside_2], dtype=int).T
            self._pixel_list = inside_coords + self.reference_pixel
        return self._pixel_list

    def check_pixel_in_tpf(self, column, row):
        """check if given (column,row) pixel is inside the area covered by this TPF file"""
        d_column = column - self.reference_column
        d_row = row - self.reference_row
        if (d_column < 0) or (d_column >= self.n_columns):
            return False
        if (d_row < 0) or (d_row >= self.n_rows):
            return False
        return True

    def check_pixel_covered(self, column, row):
        """check if we have data for given (column,row) pixel"""
        if not self.check_pixel_in_tpf(column, row):
            return False
        mask_value = self.mask[row - self.reference_row, column - self.reference_column]
        return (mask_value > 0)
        
    def _make_column_row_vectors(self):
        """prepare vectors with some numbers"""
        # HERE
        self._column = np.tile(np.arange(self.n_columns, dtype=int), self.n_rows) 
        self._column = self._column[self.mask.flatten()>0] + self.reference_column
        self._row = np.repeat(np.arange(self.n_rows, dtype=int), self.n_columns) 
        self._row = self._row[self.mask.flatten()>0] + self.reference_row

    @property
    def rows(self):
        """gives array that translates index pixel into row number"""
        if self._row is None:
            self._make_column_row_vectors()
        return self._row

    @property
    def columns(self):
        """gives array that translates index pixel into column number"""
        if self._column is None:
            self._make_column_row_vectors()
        return self._column

    def _get_pixel_index(self, row, column):
        """finds index of given (row, column) pixel in given file - information necessary to extract flux"""
        if (self._row is None) or (self._column is None):
            self._make_column_row_vectors()
        index = np.arange(self.n_pixels)
        index_mask = ((self._row == row) & (self._column == column))
        try:
            out = index[index_mask][0]
        except IndexError:
            out = None
        return out

    def get_flux_for_pixel(self, row, column, apply_epoch_mask=False):
        """extracts flux for a single pixel (all epochs) specified as row and column"""
        print("GET_FLUX {:} {:} {:} {:}".format(row, column, self.check_pixel_covered(column, row), self._get_pixel_index(row, column)))
        if not self.check_pixel_covered(column, row):
            return None
        index = self._get_pixel_index(row, column)
        if apply_epoch_mask:
            return self.flux[:,index][self.epoch_mask]
        else:
            return self.flux[:,index]

    def get_flux_err_for_pixel(self, row, column, apply_epoch_mask=False):
        """extracts flux_err for a single pixel (all epochs) specified as row and column"""
        if not self.check_pixel_covered(column, row):
            return None
        index = self._get_pixel_index(row, column)
        if apply_epoch_mask:
            return self.flux_err[:,index][self.epoch_mask]
        else:
            return self.flux_err[:,index]
    
    def get_fluxes_for_square(self, row_center, column_center, half_size, apply_epoch_mask=False):
        """get matrix that gives fluxes for pixels from (center-half_size) to
        (center+half_size) in each axis and including both ends"""
        full_size = 2 * half_size + 1
        if apply_epoch_mask:
            length = sum(self.epoch_mask)
        else:
            length = len(self.jd_short) 
        out = np.zeros((full_size, full_size, length))
        
        for i_row in range(-half_size, half_size+1):
            row = i_row + row_center
            for i_column in range(-half_size, half_size+1):
                column = i_column + column_center
                out[i_row+half_size][i_column+half_size] = self.get_flux_for_pixel(
                                        row, column, apply_epoch_mask=apply_epoch_mask)
        return out
    
    def get_predictor_matrix(self, target_x, target_y, num, dis=16, excl=5, flux_lim=(0.8, 1.2), multiple_tpfs=None, tpfs_epics=None):
        """prepare predictor matrix"""
        (self.pixel_row, self.pixel_col) = multiple_tpfs.get_rows_columns(tpfs_epics)
        self.pixel_flux = multiple_tpfs.get_fluxes(tpfs_epics)
        self.pixel_median = multiple_tpfs.get_median_fluxes(tpfs_epics)

        target_index = self._get_pixel_index(target_x, target_y)
        pixel_mask = np.ones_like(self.pixel_row, dtype=bool)

        target_col = self._column[target_index]
        target_row = self._row[target_index]
        for pix in range(-excl, excl+1):
            pixel_mask &= (self.pixel_row != (target_row+pix))
            pixel_mask &= (self.pixel_col != (target_col+pix))

        mask_1 = (self.median[target_index]*flux_lim[0] <= self.pixel_median)
        mask_2 = (self.median[target_index]*flux_lim[1] >= self.pixel_median)
        pixel_mask &= (mask_1 & mask_2)

        distance2_row = np.square(self.pixel_row[pixel_mask]-target_row)
        distance2_col = np.square(self.pixel_col[pixel_mask]-target_col)
        distance2 = distance2_row + distance2_col
        dis_mask = (distance2 > dis**2)
        distance2 = distance2[dis_mask]

        index = np.argsort(distance2, kind="mergesort")

        pixel_flux = self.pixel_flux[:,pixel_mask][:,dis_mask]
        predictor_flux = pixel_flux[:,index[:num]].astype(float)

        return predictor_flux

    def save_pixel_curve(self, row, column, file_name, full_time=True):
        """saves the time vector and the flux for a single pixel into a file"""
        flux = self.get_flux_for_pixel(row=row, column=column)
        if flux is None:
            msg = "wrong call to save_pixel_curve():\nrow = {:}\ncolumn={:}"
            warnings.warn(msg.format(row, column))
            return
        time = np.copy(self.jd_short)
        if full_time:
            time += 2450000.
        np.savetxt(file_name, np.array([time, flux]).T, fmt="%.5f %.8f")

    def save_pixel_curve_with_err(self, row, column, file_name, 
                                                        full_time=True):
        """saves: 
        the time vector, flux vector, and flux_err vector 
        for a single pixel into a file"""
        flux = self.get_flux_for_pixel(row=row, column=column)
        if flux is None:
            msg = "\n\nwrong call to save_pixel_curve_with_err():\nrow = {:}\ncolumn = {:}\n"
            warnings.warn(msg.format(row, column))
            return
        flux_err = self.get_flux_err_for_pixel(row=row, column=column)
        time = np.copy(self.jd_short)
        if full_time:
            time += 2450000.
        np.savetxt(file_name, np.array([time, flux, flux_err]).T, 
                                                fmt="%.5f %.8f %.8f")


if __name__ == '__main__':
    epic_id = "200071074"
    directory = 'tpf/'
    campaign = 92 
    pixel = [883, 670]
    out_file_a = '1-pixel_flux.dat'
    out_file_b = '1-mask.dat'
    out_file_c = '1-epoch_mask.dat'
    
    TpfData.directory = directory
    tpf = TpfData(epic_id=epic_id, campaign=campaign)
    tpf.save_pixel_curve_with_err(pixel[0], pixel[1], file_name=out_file_a)

    matrix_xy.save_matrix_xy(tpf.mask, out_file_b, data_type='boolean') # CHANGE THIS INTO TpfData.save_mask().
    
    np.savetxt(out_file_c, tpf.epoch_mask, fmt="%s") # CHANGE THIS INTO TpfData.save_epoch_mask().
    
