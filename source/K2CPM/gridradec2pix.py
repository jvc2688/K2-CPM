import numpy as np

import poly2d


class GridRaDec2Pix(object):
    """transformation from (RA, Dec) to (x,y)"""

    def __init__(self, coefs_x, coefs_y):
        """coefs_x and coefs_y has to be of the same type: np.array, list, string"""
        if type(coefs_x) != type(coefs_y):
            raise TypeError('different types of input data')

        if isinstance(coefs_x, np.ndarray):
            coefs_x_in = coefs_x
            coefs_y_in = coefs_y
        elif isinstance(coefs_x, list):
            if isinstance(coefs_x[0], str):
                coefs_x_in = np.array([float(value) for value in coefs_x])
                coefs_y_in = np.array([float(value) for value in coefs_y])
            else:
                raise TypeError('unrecognized type in input list: {:}'.format(type(coefs_x[0])))
        else:
            raise TypeError('unrecognized input type: {:}'.format(type(coefs_x)))

        self.coefs_x = coefs_x_in
        self.coefs_y = coefs_y_in

    def apply_grid(self, ra, dec):
        """calculate pixel coordinates for given (RA,Dec) which can be floats, lists, or numpy.arrays"""
        x_out = poly2d.eval_poly_2d(ra, dec, self.coefs_x)
        y_out = poly2d.eval_poly_2d(ra, dec, self.coefs_y)
        return (x_out, y_out)


# example usage - prints coordinates for 3 stars for a selected epoch
if __name__ == "__main__":
    file_name = '../../data_K2C9/grids_RADEC2pix_91_30.data'
    selected_cadence = 125243

    # This is a very bright star, we expect (216.396, 834.244)
    ra = 269.7459379642
    dec = -27.5007687679

    # These are very bright stars, we expect (47.557, 880.357) and (397.209, 770.258) for first and second, respectively
    ra_list = [269.9634930295, 269.5086016757]
    dec_list = [-27.5082020827, -27.4774965784]

    with open(file_name) as in_data:
        for line in in_data.readlines():
            if line[0] == "#":
                continue
            columns = line.split()
            cadence = int(columns[0])
            BJD = float(columns[1])
            stars_used = int(columns[2])
            sigma = float(columns[3])
            grid = GridRaDec2Pix(coefs_x=columns[4:10], coefs_y=columns[10:16])
            if cadence == selected_cadence:
                print(grid.apply_grid(ra, dec))
                print(grid.apply_grid(ra_list, dec_list))

