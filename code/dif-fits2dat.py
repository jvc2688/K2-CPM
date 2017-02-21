#! /usr/bin/env python

import sys
from astropy.io import fits


fits_file = fits.open(sys.argv[1])
data = fits_file[1].data
fits_file.close()

out_t = []
out_dif = []
for d in data:
    out_t.append(d.field(0))
    out_dif.append(d.field(1)[0][0])

for i in range(len(out_t)):
    print("{:.5f} {:.5f}".format(out_t[i], out_dif[i]))


