import numpy as np

import gridradec2pix


class CampaignGridRaDec2Pix(object):
    """collects all grids for given (sub-)campaign"""

    def __init__(self, campaign=None, file_name=None):
        if campaign is None:
            self.campaign = None
        else:
            self.campaign = int(campaign)

        self.grids = None

        if file_name is not None:
            self._read_from_file(file_name)

        if self.grids is None:
            raise ValueError('no input data provided')

        self.bjd_array = np.array(self.bjd)

    def _read_from_file(self, file_name):
        """read multiple grids from a single file"""
        with open(file_name) as in_data:
            self.cadence = []
            self.bjd = []
            self.stars_used = []
            self.sigma = []
            self.grids = []
            for line in in_data.readlines():
                if line[0] == "#":
                    continue
                columns = line.split()
                grid = gridradec2pix.GridRaDec2Pix(coefs_x=columns[4:10], 
                                                   coefs_y=columns[10:16])
                self.cadence.append(int(columns[0]))
                self.bjd.append(float(columns[1]))
                self.stars_used.append(int(columns[2]))
                self.sigma.append(float(columns[3]))
                self.grids.append(grid)

    def index_for_bjd(self, bjd):
        """find index that is closest to given BJD"""
        index = (np.abs(self.bjd_array-bjd)).argmin()
        if np.abs(self.bjd_array[index]-bjd) > 2.e-4:
            text = 'No close BJD found for {:}, the closest differs by {:}'
            message = text.format(bjd, np.abs(self.bjd_array[index]-bjd))
            raise ValueError(message)
        return index

    def apply_grids(self, ra, dec):
        """Calculate pixel coordinates for given (RA,Dec) for all epochs. 
        (RA,Dec) can be floats, lists, or numpy.arrays.
        
        For input of length n_stars the output is 2 arrays (first for X, 
        second for Y), each of shape (n_epochs, n_stars).
        If inputs are floats than output is 2 1D arrays.
        """
        out_1 = []
        out_2 = []
        for grid in self.grids:
            out = grid.apply_grid(ra=ra, dec=dec)
            out_1.append(out[0])
            out_2.append(out[1])
        out_1 = np.array(out_1)
        out_2 = np.array(out_2)
        if isinstance(ra, float):
            out_1 = out_1[:,0]
            out_2 = out_2[:,0]
        return (out_1, out_2)


# Example usage:
if __name__ == "__main__":

    file_name = '../../data_K2C9/grids_RADEC2pix_91_30.data'

    grids = CampaignGridRaDec2Pix(campaign=91, file_name=file_name)

    ra_list = [269.9634930295, 269.5086016757]
    dec_list = [-27.5082020827, -27.4774965784]

    # Print Y positions of 0-th star on the list at all epochs:
    print(grids.apply_grids(ra_list, dec_list)[1][:,0])

