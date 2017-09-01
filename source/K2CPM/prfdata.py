import os
import glob
import numpy as np
from scipy.interpolate import RectBivariateSpline
from math import fabs

from astropy.io import fits


# For K2 channel number give corresponding module and output numbers.
module_output_for_channel = {
1: (2, 1), 2: (2, 2), 3: (2, 3), 4: (2, 4), 5: (3, 1), 
6: (3, 2), 7: (3, 3), 8: (3, 4), 9: (4, 1), 10: (4, 2), 
11: (4, 3), 12: (4, 4), 13: (6, 1), 14: (6, 2), 15: (6, 3), 
16: (6, 4), 17: (7, 1), 18: (7, 2), 19: (7, 3), 20: (7, 4), 
21: (8, 1), 22: (8, 2), 23: (8, 3), 24: (8, 4), 25: (9, 1), 
26: (9, 2), 27: (9, 3), 28: (9, 4), 29: (10, 1), 30: (10, 2), 
31: (10, 3), 32: (10, 4), 33: (11, 1), 34: (11, 2), 35: (11, 3), 
36: (11, 4), 37: (12, 1), 38: (12, 2), 39: (12, 3), 40: (12, 4), 
41: (13, 1), 42: (13, 2), 43: (13, 3), 44: (13, 4), 45: (14, 1), 
46: (14, 2), 47: (14, 3), 48: (14, 4), 49: (15, 1), 50: (15, 2), 
51: (15, 3), 52: (15, 4), 53: (16, 1), 54: (16, 2), 55: (16, 3), 
56: (16, 4), 57: (17, 1), 58: (17, 2), 59: (17, 3), 60: (17, 4), 
61: (18, 1), 62: (18, 2), 63: (18, 3), 64: (18, 4), 65: (19, 1), 
66: (19, 2), 67: (19, 3), 68: (19, 4), 69: (20, 1), 70: (20, 2), 
71: (20, 3), 72: (20, 4), 73: (22, 1), 74: (22, 2), 75: (22, 3), 
76: (22, 4), 77: (23, 1), 78: (23, 2), 79: (23, 3), 80: (23, 4), 
81: (24, 1), 82: (24, 2), 83: (24, 3), 84: (24, 4)
}


class PrfData(object):
    """
    K2 PRF data 
    """

    data_directory = None

    def __init__(self, channel=None, module=None, output=None):
        """
        provide channel or both module and output
        data_directory has to be set
        """
        if self.data_directory is None:
            raise ValueError('PrfData: data_directory not set')
        if (module is None) != (output is None):
            raise ValueError('You must set both module and output options')
        if (channel is None) == (module is None):
            raise ValueError('provide channel or both module and output')

        if channel is not None:
            (module, output) = module_output_for_channel[channel]
        text = "kplr{:02d}.{:}_*_prf.fits".format(int(module), output)
        names = os.path.join(self.data_directory, text)

        try:
            file_name = glob.glob(names)[-1]
        except:
            raise FileNotFoundError('PRF files {:} not found'.format(names))
        
        keys = ['CRPIX1P', 'CRVAL1P', 'CDELT1P', 'CRPIX2P', 'CRVAL2P', 'CDELT2P']
        with fits.open(file_name) as prf_hdus:
            self._data = []
            self._keywords = []
            for hdu in prf_hdus[1:]:
                self._data.append(hdu.data)
                keywords = dict()
                for key in keys:
                    keywords[key] = hdu.header[key]
                self._keywords.append(keywords)

        # make sure last hdu is for central area
        center_x = np.array([value['CRVAL1P'] for value in self._keywords])
        center_y = np.array([value['CRVAL2P'] for value in self._keywords])
        dx = center_x - np.mean(center_x)
        dy = center_y - np.mean(center_y)
        if np.argmin(np.sqrt(dx**2+dy**2)) != len(center_x)-1:
            raise ValueError('The last hdu in PRF file is not the one in ' + 
                            'the center - contarary to what we assumed here!')
        
        # make a list of pairs but exclude the central point
        n = len(center_x)
        self._corners_pairs = [(i, i+1) if i>=0 else (i+n-1, i+1) for i in range(-1, n-2)]

        # make sure that the first four corners are in clockwise, or 
        # anti-clockwise order:
        for (i, j) in self._corners_pairs:
            # We want one coordinate to be equal and other to be different.
            if (fabs(center_x[i] - center_x[j]) < .001 != 
                    fabs(center_y[i] - center_y[j]) < .001): 
                raise ValueError('something wrong with order of centers of hdus')

        # prepare equations to be used for barycentric interpolation
        self._equations = dict()
        for (i, j) in self._corners_pairs:
            xs = [center_x[i], center_x[j], center_x[-1]]
            ys = [center_y[i], center_y[j], center_y[-1]]
            self._equations[(i, j)] = np.array([xs, ys, [1., 1., 1.]])

        # grid on which prf is defined:
        (nx, ny) = self._data[0].shape
        self._prf_grid_x = (np.arange(nx) - nx / 2. + .5) * self._keywords[0]['CDELT1P']
        self._prf_grid_y = (np.arange(ny) - ny / 2. + .5) * self._keywords[0]['CDELT2P']
        # The two lines above should rather use 'CRPIX1P' instead of n/2. etc.

    def _get_barycentric_interpolation_weights(self, x, y):
        """find in which triangle given point is located and 
        calculate weights for barycentric interpolation"""
        for (i, j) in self._corners_pairs:
            weights = np.linalg.solve(self._equations[(i, j)], np.array([x, y, 1.]))
            if np.all(weights >= 0.): # i.e. we found triangle in which point is located
                return (np.array([i, j, -1]), weights)
        raise ValueError("Point doesn't lie in any of the triangles")

    def _interpolate_prf(self, x, y):
        """barycentric interpolation on a traiangle grid"""
        (indexes, weights) = self._get_barycentric_interpolation_weights(x=x, y=y)
        prf = (self._data[indexes[0]] * weights[0] 
                    + self._data[indexes[1]] * weights[1] 
                    + self._data[indexes[2]] * weights[2])
        return prf

    def get_interpolated_prf(self, star_x, star_y, pixels_list):
        """
        For star centered at given position calculate PRF for list of pixels.
        Example:    star_x=100.5, 
                    star_y=200.5, 
                    pixels_list=[[100., 200.], [101., 200.], [102., 200.]]
        """
        prf = self._interpolate_prf(star_x, star_y)
        
        spline_function = RectBivariateSpline(x=self._prf_grid_x, 
                                                y=self._prf_grid_y, z=prf)
        
        out = [spline_function(x-star_x, y-star_y)[0][0] for (x, y) in pixels_list]
        
        return out

# example usage of the class:
if __name__ == '__main__':

    channel = 51

    x_star = 549.49
    y_star = 511.49

    point_1 = [x_star-0.23, y_star-0.03]
    point_2 = [x_star-1.03, y_star+1.39]
    points = [point_1, point_2]

    # Give path to Kepler PRF data. You can download the archive from:
    # http://archive.stsci.edu/missions/kepler/fpc/prf/
    PrfData.data_directory = "PATH/TO/PRF/DATA"

    prf_template = PrfData(channel=channel)

    values = prf_template.get_interpolated_prf(x_star, y_star, points)

    # We expect [0.397551, 0.0181485]
    print(values)

