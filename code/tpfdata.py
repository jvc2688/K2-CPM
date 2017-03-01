
import os
import numpy as np
import urllib
from sklearn.decomposition import PCA

from astropy.io import fits as pyfits

#import matrix_xy
from code import matrix_xy


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
        self.mask = hdu_list[2].data
        
        data = hdu_list[1].data
        self.jd_short = data["time"] + 4833. # is it HJD, BJD, JD?
        self.quality_flags = data["quality"].astype(dtype=int)  
        flux = data["flux"]
        pixel_mask = np.isfinite(flux) & (flux != 0)
        pixel_mask[:, self.pixel_mask < 1] = False
        self.pixel_mask = pixel_mask 
        quality_flags = data["quality"]
        # TO BE DONE - can someone check if these are the only flags we should remove? Should we change it to a parameter? 
        quality_flags_ok = ((quality_flags == 0) | (quality_flags == 8192) | (quality_flags == 16384) | (quality_flags == 24576)) 
        foo = np.sum(np.sum((self.pixel_mask > 0), axis=2), axis=1) # Does anybody understand what is happening here?
        self.epoch_mask = (foo > 0) & np.isfinite(self.jd_short) & quality_flags_ok
        flux = flux[:, self.mask>0]
        if not np.isfinite(flux).all():
            raise ValueError('non-finite value in flux table of {:} - feature not done yet'.format(file_name))
            # TO BE DONE - code interpolation using e.g. k2_cpm.py lines: 89-92
        self.flux = flux
        self.median = np.median(flux, axis=0)

        hdu_list.close()

    @property
    def _path(self):
        """path to the TPF file"""
        if TpfData.directory is None:
            raise ValueError("TpfData.directory value not set")
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
    
    def _make_column_row_vectors(self):
        """prepare vectors with some numbers"""
        self._column = np.tile(np.arange(self.mask.shape[1], dtype=int), self.mask.shape[0]) + self.reference_column
        self._column = self._column[self.mask.flatten()>0]
        self._row = np.repeat(np.arange(self.mask.shape[0], dtype=int), self.mask.shape[1]) + self.reference_row
        self._row = self._row[self.mask.flatten()>0]

    def _get_pixel_index(self, row, column):
        """finds index of given (row, column) pixel in given file - information necessary to extract flux"""
        if (self._row is None) or (self._column is None):
            self._make_column_row_vectors()
        index = np.arange(self._row.shape[0])
        index_mask = ((self._row == row) & (self._column == column))
        try:
            out = index[index_mask][0]
        except IndexError:
            out = None
        return out

    def get_flux_for_pixel(self, row, column):
        """extracts flux for a single pixel (all epochs) specified as row and column"""
        index = self._get_pixel_index(row, column)
        return self.flux[:,index]
    
    def save_pixel_curve(self, row, column, file_name, full_time=True):
        """saves the time vector and the flux for a single pixel into a file"""
        flux = self.get_flux_for_pixel(row=row, column=column)
        time = self.jd_short
        if full_time:
            time += 2450000.
        np.savetxt(file_name, np.array([time, flux]).T, fmt="%.5f %.8f")


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
    tpf.save_pixel_curve(pixel[0], pixel[1], file_name=out_file_a)

    matrix_xy.save_matrix_xy(tpf.mask, out_file_b, data_type='boolean') # CHANGE THIS INTO TpfData.save_mask().
    
    np.savetxt(out_file_c, tpf.epoch_mask, fmt="%s") # CHANGE THIS INTO TpfData.save_epoch_mask().
    
