# K2-CPM
Causal Pixel Model for K2 data

# How to use
- epic_number **int** k2 ID
- campaign_number **int** 91 for phase a, 92 for phase b
- number_of_predictors **int** number of the predictors pixels to use
- l2_regularization **float** strength of l2 regularization
- number_of_PCA **int** number of the PCA components to use, if 0, no PCA 
- distance2target_pixel **int** distance between target pixel and predictor pixels
- exclusion_rows_cols **int** how many rows and columns that are excluded around the target pixel
- path2output **str** directory to store the output file
- path2tpf **str** directory to the target pixel files
- path2pixel_list **str** path to the pixel list file that specify which pixels to be modelled. If not provided, the whole target pixel file will be modelled

python run_cpm.py epic_number campaign_number number_of_predictors l2_regularization number_of_PCA distance2target_pixel exclusion_rows_cols path2output path2tpf path2pixel_list
