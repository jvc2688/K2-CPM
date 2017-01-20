#! /usr/bin/env python

import os
import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord


dir_here = os.path.dirname(os.path.abspath(__file__))


class WcsFromTpf(object):
    def __init__(self, channel, subcampaign):
        if channel not in [30, 31, 32, 49, 52]:
            raise ValueError('WcsFromTpf channel value must be one of: 30, 31, 32, 49, or 52')
        if subcampaign not in [91, 92]:
            raise ValueError('WcsFromTpf subcampain value must be one either 91 or 92')
        file_name = dir_here + "/data_K2C9/WCS_xy_radec_epic_{:}_{:}.fits.gz".format(subcampaign, channel)
        hdulist = fits.open(file_name)
        self.pix_x = hdulist[1].data.field(0)
        self.pix_y = hdulist[1].data.field(1)
        self.ra = hdulist[1].data.field(2)
        self.dec = hdulist[1].data.field(3)
        self.epic = hdulist[1].data.field(4)
        hdulist.close
        self._skycoord = SkyCoord(self.ra*u.deg, self.dec*u.deg)

    def _get_mask_around_radec(self, ra_deg, dec_deg, radius_arcmin):
        """search for all pixels within given radius around RA/Dec position and store resulting mask mask"""
        position = SkyCoord(ra_deg*u.deg, dec_deg*u.deg)
        distances = position.separation(self._skycoord)
        self._mask = (distances < radius_arcmin * u.arcmin)
        
    def get_epic_around_radec(self, ra_deg, dec_deg, radius_arcmin):
        """returns unique EPIC ids found around given sky position"""
        self._get_mask_around_radec(ra_deg, dec_deg, radius_arcmin)
        return [int(i) for i in set(self.epic[self._mask])]

if __name__ == '__main__':
    """example usage"""
    wcs = WcsFromTpf(31, 92)
    list_1 = wcs.get_epic_around_radec(269.013923, -28.227162, 0.12)
    list_2 = wcs.get_epic_around_radec(269.013923, -28.227162, 2.12)
    print(list_1)
    print(list_2)
    print(set(list_2).difference(list_1))
