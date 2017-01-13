#! /usr/bin/env python

import os
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord


dir_here = os.path.dirname(os.path.abspath(__file__))


class WcsFromTpf(object):
    def __init__(self, channel=None, subcampaign=None):
        file_name = dir_here + "/data_K2C9/WCS_xy_radec_epic_{:}_{:}.dat.bz2".format(subcampaign, channel)
        x, y, epic = np.loadtxt(file_name, unpack=True, usecols=(0, 1, 4), dtype=int)
        ra, dec = np.loadtxt(file_name, unpack=True, usecols=(2, 3))

    def get_epic_around_radec(self, ra_deg, dec_deg, radius):
        pass

if __name__ == '__main__':
    wcs = WcsFromTpf(49, 91)

