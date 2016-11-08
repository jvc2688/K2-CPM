# K2-CPM
Causal Pixel Model for K2 data

# How to use
```
$ python run_cpm.py

positional arguments:
  epic          int k2 epic number
  campaign      int campaign number, 91 for phase a, 92 for phase b
  n_predictors  int number of the predictors pixels
  l2            float strength of l2 regularization
  n_pca         int number of the PCA components to use, if 0, no PCA 
  distance      int distance between target pixel and predictor pixels
  exclusion     int how many rows and columns that are excluded around the target pixel
  input_dir     str directory to the target pixel files
  output_dir    str directory to store the output file

optional arguments:
  -p [pixel_list], --pixel [pixel_list]
                str path to the pixel list file that specify which pixels to be modelled. If not provided, the whole target pixel file will be modeled
```

# Example
```
$ python run_cpm.py 200069974 92 800 1e3 0 16 5 ./output ./tpf -p ./test_pixel.dat
```
