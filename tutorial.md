
# CPM tutorial

I'll show you how we can extract CPM photometry of K2 Campaign 9 data. 
Note that after downloading K2-CPM you have to add source directory to 
your `PYTHONPATH`, e.g., (for tcsh and bash, respectively):
```
setenv PYTHONPATH $PYTHONPATH\:/PATH_TO_K2-CPM/source
```
```
export PYTHONPATH=$PYTHONPATH:/PATH_TO_K2-CPM/source
```
I'm marking in __bold__ the parts that require immediate attention. 

### CPM\_PART1

The event I'm interested in is OGLE-2016-BLG-1043 (ob161043 for short). The 
event coordinates are 18:05:07.84 -26:59:57.6 (or 271.2826667 -26.9993333
if your mind works in degrees). In the ground-based data it 
peaked at HJD'=7549 or on June 9th, 2016. This is after the second part of 
K2 Campaign 9 started, hence we're interested in campaign "92" (as opposed to 
"91"). On some plot I see that these coordinates correspond to channel 49. 
If you don't know which channel your coordinates are and you have 
[K2fov package](https://github.com/KeplerGO/K2fov) installed, then 
you can check it like that (though I'm not sure if it correctly works close to 
boundary between channels 31 and 32): 
```python
import K2fov
c9fov = K2fov.fields.getKeplerFov(9)
print(c9fov.pickAChannel(271.2826667, -26.9993333))
```
```
49.0
```
Now it's time to find 
which TPF and which pixel is closest to the event -- this is only rough estimate 
that is based on WCS information in TPF files. We start python and do:

```python
from K2CPM import wcsfromtpf
import K2CPM

ra = 271.2826667
dec = -26.9993333
campaign = 92
channel = 49

wcs = wcsfromtpf.WcsFromTpf(channel, campaign)
nearest_pixel = wcs.get_nearest_pixel_radec(ra, dec)
(pix_x, pix_y, _, _, epic_id, separation) = nearest_pixel
```
If you want to see raw pixel data you can plot them using, e.g., 
```python
import matplotlib.pyplot as plt
from K2CPM import tpfdata, plot_utils

tpf_dir = 'tpf/'
half_size = 2
file_out = "ob161043_tpf.png"

tpfdata.TpfData.directory = tpf_dir
tpf = tpfdata.TpfData(epic_id=epic_id, campaign=campaign)
flux_matrix = tpf.get_fluxes_for_square(pix_y, pix_x, half_size)

fig = plt.gcf()
fig.set_size_inches(50, 30)
plot_utils.plot_matrix_subplots(fig, tpf.jd_short, flux_matrix)
plt.savefig(file_out)
```
The plot shows light curve for 25 pixels centered on ```(pix_x, pix_y)```. In 
this case you're not able to see it, but we know that the event is shifted by 
1 pixel from the predicted position, hence we apply:
```python
pix_y -= 1
```

Now let's see these values:

```python
msg = "pix_x = {:}\npix_y = {:}\nepic_id = {:}\nseparation = {:.1f} arcsec"
print(msg.format(pix_x, pix_y, epic_id, separation.value))
```
```
pix_x = 741
pix_y = 679
epic_id = 200070496
separation = 1.9 arcsec
```

Opps, I'm not sure what x and y mean here. 
Kepler and K2 documents don't use (x,y) they use (column,row) or (row,column). 
Here x runs from 13 to 1112, and y runs from 21 to 1044 
(__the format of coordinates has to be strictly specified__). Fortunately, the 
separation given above is < 4 arcsec, so the coordinates are from the right 
channel. 

We know which pixel we're interested in so now its time to prepare predictor matrix. Note that first time you run this code for given ```epic_id``` and ```campaign```, it will take some time to download the data. Run:

```python
from K2CPM.cpm_part1 import run_cpm_part1
from K2CPM import matrix_xy
import numpy as np

n_predictor = 400
n_pca = 0
distance = 16
exclusion = 1
flux_lim = (0.1, 2.0)
tpf_dir = 'tpf/'
pixel_list = np.array([[pix_y, pix_x]])

(predictor_matrix_list, predictor_masks) = run_cpm_part1(
		epic_id, campaign, n_predictor, n_pca, distance, exclusion, 
		flux_lim, tpf_dir, pixel_list,
		return_predictor_epoch_masks=True)

stem = "{:}_{:}_{:}_{:}".format(campaign, channel, pix_y, pix_x)
matrix_xy.save_matrix_xy(predictor_matrix_list[0], stem+"_pre_matrix_xy.dat")
np.savetxt(stem+"_predictor_mask.dat", predictor_masks[0], fmt='%r')
```

We asked for 400 pixels in predictor matrix, did not use Principal Component Analysis 
(n\_pca=0), and applied some standard exclusion limits. The tpf\_dir is where we keep 
downloaded TPF files. If you already have some of those files, then just make 
symbolic link. Note that ```pixel_list = np.array([[pix_y, pix_x]])``` reverts 
the order of coordinates. 

At this point we have two files ready, we need two more:

```python
from K2CPM import tpfdata

tpfdata.TpfData.directory = tpf_dir
tpf = tpfdata.TpfData(epic_id=epic_id, campaign=campaign)
tpf.save_pixel_curve_with_err(pixel_list[0][0], pixel_list[0][1], file_name=stem+"_pixel_flux.dat")
np.savetxt(stem+"_epoch_mask.dat", tpf.epoch_mask, fmt='%r')
```

At this point you should have four files saved:
```
92_49_679_741_epoch_mask.dat
92_49_679_741_pixel_flux.dat
92_49_679_741_pre_matrix_xy.dat
92_49_679_741_predictor_mask.dat
```
The largest file is \*\_pre\_matrix\_xy.dat and its size depends on n\_predictor.

You may exit python shell at this point.

### CPM\_PART2

We can proceed to second part of CPM. Open python shell once more and import 
required modules, and load the files:

```python
import numpy as np
from K2CPM import cpm_part2, matrix_xy

stem = "92_49_679_741"

predictor_matrix = matrix_xy.load_matrix_xy(stem+"_pre_matrix_xy.dat")
predictor_mask = cpm_part2.read_true_false_file(stem+"_predictor_mask.dat")
epoch_mask = cpm_part2.read_true_false_file(stem+"_epoch_mask.dat")
(tpf_time, tpf_flux, tpf_flux_err) = np.loadtxt(stem+"_pixel_flux.dat", unpack=True)
```

Next step is just running the cpm\_part2:

```python
l2 = 1000.
result = cpm_part2.cpm_part2(tpf_time, tpf_flux, tpf_flux_err, epoch_mask, predictor_matrix, predictor_mask, l2)
```

At this point tuple ```result``` contains: 

1. solution of the linear equations system,
2. predicted flux for the target pixel,
3. difference flux, i.e., the signal we're interested in (as long as model magnification curve is not provided),
4. time vector.

If you want to plot the light curve use ```result[2]``` vs. ```result[3]```, e.g.,
```python
import matplotlib.pyplot as plt
plt.scatter(result[3]-2450000., result[2])
plt.show()
```
You should see a binary lens light curve that peaks at HJD = 2457548.6.

The cpm_part2() call presented above uses the model that is trained on all 
the data. For the events with signal in well defined window (for ob161043 
it is from 2457547 to 2457550) one can limit the training to the epochs 
outside given range. This can be achieved by providing train_lim keyword 
with a list that has the two limiting values e.g.:

```python
result = cpm_part2.cpm_part2(tpf_time, tpf_flux, tpf_flux_err, epoch_mask, predictor_matrix, predictor_mask, l2, train_lim = [2457547., 2457550.])
```


(C) Radek Poleski, revised May 2017
