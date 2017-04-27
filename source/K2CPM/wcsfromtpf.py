#! /usr/bin/env python

import os
import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord


dir_here = os.path.abspath(__file__)
for i in range(3):
    dir_here = os.path.dirname(dir_here)


class WcsFromTpf(object):
    def __init__(self, channel, subcampaign):
        if channel not in [30, 31, 32, 49, 52]:
            raise ValueError('WcsFromTpf channel value must be one of: 30, 31, 32, 49, or 52')
        if subcampaign not in [91, 92]:
            raise ValueError('WcsFromTpf subcampain value must be one either 91 or 92')
        file_name = dir_here + "/data_K2C9/WCS_xy_radec_epic_{:}_{:}.fits.gz".format(subcampaign, channel)
        hdulist = fits.open(file_name)
        self.pix_x = hdulist[1].data.field(0) # column
        self.pix_y = hdulist[1].data.field(1) # row
        self.ra = hdulist[1].data.field(2)
        self.dec = hdulist[1].data.field(3)
        self.epic = hdulist[1].data.field(4)
        hdulist.close
        self._skycoord = SkyCoord(self.ra*u.deg, self.dec*u.deg)
        self._sorted_epics = None

    @property
    def sorted_epics(self):
        """get list of EPIC IDs in numerical order"""
        if self._sorted_epics is None:
            self._sorted_epics = sorted(set(self.epic))
        return self._sorted_epics

    def _get_mask_around_radec(self, ra_deg, dec_deg, radius_arcmin):
        """search for all pixels within given radius around RA/Dec position and store resulting mask mask"""
        position = SkyCoord(ra_deg*u.deg, dec_deg*u.deg)
        distances = position.separation(self._skycoord)
        self._mask = (distances < radius_arcmin * u.arcmin)
        
    def get_epic_around_radec(self, ra_deg, dec_deg, radius_arcmin):
        """returns unique EPIC ids found around given sky position"""
        self._get_mask_around_radec(ra_deg, dec_deg, radius_arcmin)
        return [int(i) for i in set(self.epic[self._mask])]
    
    def get_nearest_pixel_radec(self, ra_deg, dec_deg):
        """find pixel nearest to given RA/Dec"""
        position = SkyCoord(ra_deg*u.deg, dec_deg*u.deg)
        distances = position.separation(self._skycoord)
        i = np.argmin(distances)
        return (self.pix_x[i], self.pix_y[i], self.ra[i], self.dec[i], self.epic[i], distances[i].to(u.arcsec))

    def get_epics_around_radec(self, ra_deg, dec_deg):
        """find pixel nearest to given sky coordinates and run get_epics_around_pixel()"""
        nearest_pixel = self.get_nearest_pixel_radec(ra_deg, dec_deg)
        return self.get_epics_around_pixel(column=nearest_pixel[0], row=nearest_pixel[1])

    def get_epics_around_pixel(self, column, row):
        """
        for every EPIC ID take all pixels, calculate distance from given (column, row)
        and get minimum and maximum of these; then sort EPIC IDs according to these minimum
        distances
        """
        distance2_col = np.square(0. + column - self.pix_x)
        distance2_row = np.square(0. + row - self.pix_y)
        distance = np.sqrt(distance2_row + distance2_col)
        
        epic_min = dict()
        epic_max = dict()
        for (ind, val) in enumerate(distance):
            epic = self.epic[ind]
            if epic in epic_min:
                if val < epic_min[epic]:
                    epic_min[epic] = val
                elif val > epic_max[epic]:
                    epic_max[epic] = val
            else:
                epic_min[epic] = val
                epic_max[epic] = val
        
        min_values = np.array([epic_min[epic] for epic in self.sorted_epics])
        max_values = np.array([epic_max[epic] for epic in self.sorted_epics])
        epics = np.array(self.sorted_epics)
        index = np.argsort(min_values, kind="mergesort")
        return (epics[index], min_values[index], max_values[index])


if __name__ == '__main__':
    """example usage"""
    wcs = WcsFromTpf(31, 92)
    list_1 = wcs.get_epic_around_radec(269.013923, -28.227162, 0.12)
    list_2 = wcs.get_epic_around_radec(269.013923, -28.227162, 2.12)
    print("narrow search:")
    print(list_1)
    print("wider search:")
    print(list_2)
    print("difference:")
    print([i for i in set(list_2).difference(list_1)])
    
    results = wcs.get_nearest_pixel_radec(269.013923, -28.227162)
    print(' ')
    print('closest pixel: {:} {:}'.format(results[0], results[1]))
    print('its RA/Dec: {:} {:}'.format(results[2], results[3]))
    print('EPIC ID: {:}'.format(results[4]))
    print('distance: {:}'.format(results[5]))

    print("")
    print("")

    w = WcsFromTpf(52, 92)
    ra = 271.401798
    dec = -27.67367
    print(w.get_nearest_pixel_radec(ra, dec))
    print("")
    (out1, out2, out3) = w.get_epics_around_radec(ra, dec)
    for (i, v) in enumerate(out1):
        if i>4: break
        print(i, v, out2[i], out3[i])


