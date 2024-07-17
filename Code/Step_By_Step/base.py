# Full pipeline for ML modelling of a regression problem
#  when the dataset contains single measurements for the
# same sample/animal
# Expectations:
#    Data cleaning must be performed prior to using this script.
#    The dataset must contain only numerical values, must be cleaned
#    of undesired characters/errors and must not contain any missing values.
#    The last column represents the predictor variable.
# Dan Tulpan, dtulpan@uoguelph.ca
# Last update: June 1, 2024

# How to run:
#   python3  regression_singlemeasurements_pipeline.py  -in  Pig_ImageJ_mments_after_ave.csv
#   python3  regression_singlemeasurements_pipeline.py  -in  ../../Datasets/MarshallEtAl2023/MarshallEtAl2023_selected_measurements.csv

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from statsmodels.graphics.gofplots import qqplot
from numpy import arange
import joblib
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
import numpy as np
import argparse
import sys
import os, errno
from matplotlib import pyplot as plt
import seaborn as sns

# ==================== #
# function definitions #
# ==================== #

# function that calculates Lin's Concordance Correlation Coefficient
#  Lin et al., 1989: https://www.jstor.org/stable/2532051
def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient."""
    """https://rowannicholls.github.io/python/statistics/agreement/concordance_correlation_coefficient.html"""
    # Raw data
    dct = {
        'y_true': y_true,
        'y_pred': y_pred
    }
    df = pd.DataFrame(dct)
    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    y_true = df['y_true']
    y_pred = df['y_pred']
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    if (denominator == 0):
        print('Null denominator when calculating the CCC. Exiting ...')
        exit()
    return numerator / denominator

# custom scoring function using CCC
my_scorer = make_scorer(concordance_correlation_coefficient, greater_is_better=True)

# function to add a list of CV results to the data frame
def add_res_to_cv_results(cvres, alg_name, status):
    global all_cv_results
    for num in cvres:
        all_cv_results = pd.concat([
        all_cv_results,
        pd.Series({
            'algorithm':alg_name,
            'cv_result':num,
            'before_after_optim':status
        }).to_frame().T])

# function to add a list of averaged CV results (or single values)
#   to the combined_results data frame
def add_res_to_combined_avg_results(cvres, alg_name, state, meas):
    global combined_avg_results
    if (type(cvres) is np.ndarray):
        combined_avg_results = pd.concat([
            combined_avg_results,
            pd.Series({
                'algorithm':alg_name,
                'state':state,
                'cv_mean':abs(np.mean(cvres)),
                'cv_std':np.std(cvres),
                'cv_min':abs(np.min(cvres)),
                'cv_max':abs(np.max(cvres)),
                'measure': meas
            }).to_frame().T])
    else:
        combined_avg_results = pd.concat([
        combined_avg_results,
        pd.Series({
        'algorithm':alg_name,
        'state':state,
        'cv_mean':cvres,
        'cv_std':0,
        'cv_min':cvres,
        'cv_max':cvres,
        'measure': meas
        }).to_frame().T])

# ================ #
# global variables #
# ================ #

# create an empty DataFrame to collect all CV results for a comparison boxplot
all_cv_results = pd.DataFrame(columns=['algorithm','cv_result','before_after_optim'])

# create an empty DataFrame to collect all CV results stats for saving into CSV file
combined_avg_results = pd.DataFrame(columns=['algorithm','state','cv_mean', 'cv_std', 'cv_min', 'cv_max', 'measure'])


# ============================= #
# define command line arguments #
# ============================= #
parser = argparse.ArgumentParser(description='regression pipeline')
parser.add_argument('--input_file', '-in', action="store", dest='infile', required=True, help='Name of csv input file. The last column of the file is the desired output. The first column contains the animal id.')
parser.add_argument('--num_splits', '-k', action="store", dest='num_splits', required=False, default=5, help='Number of splits for k-fold cross-validation.')
parser.add_argument('--num_repeats', '-n', action="store", dest='num_repeats', required=False, default=3, help='Number of repeated k-fold cross-validations.')
parser.add_argument('--output_dir', '-o', action="store", dest='outdir', required=False, default='results', help='Name of the output directory. Default name is results/.')

# check and handle user errors
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# read filename
filename = args.infile
num_splits = int(args.num_splits)
num_repeats = int(args.num_repeats)
output_directory = args.outdir

# create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# read input file. Consider first line (line 0) as header line.
dataset = pd.read_csv(filename, header=0)
print('\nDataset size before cleaning (#rows,#columns): ', dataset.shape)
