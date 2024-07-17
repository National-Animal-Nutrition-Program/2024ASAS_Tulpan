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

# ===================== #
# Step 1: Data cleaning #
# ===================== #
print('\n>>> Step 1: Data cleaning\n')

# remove rows with missing values and reset the row index
dataset.dropna(ignore_index=True)

# find duplicate rows
dups = dataset.duplicated()
print("\nDuplicated rows:\n", dups)

# report if there are any duplicates
print("\nAny duplicated rows: ", dups.any())

# list all duplicate rows
print("\nDuplicated rows:\n", dataset[dups])

# delete duplicate rows
#dataset.drop_duplicates(inplace=True)
dataset = dataset[~dataset.index.duplicated()]

# delete duplicate columns
duplicate_cols = dataset.columns[dataset.columns.duplicated()]
dataset.drop(columns=duplicate_cols, inplace=True)

# remove single value columns
# get the number of unique values for each column
counts = dataset.nunique()

# record columns to delete
to_del = [i for i,v in enumerate(counts) if v == 1]
print("Columns to delete: ", to_del)

# drop duplicated columns
if (len(to_del) > 0):
    dataset.drop(dataset.columns[to_del], axis=1, inplace=True)

# reset the index of the dataset
dataset.reset_index(drop=True, inplace=True)

# perform outlier detection via Z-scores
# --------------------------------------
# identify numeric columns
print('\nColumn types:\n', dataset.dtypes)
numeric_cols = dataset.select_dtypes(include=np.number).columns.tolist()
print("Numeric columns: ", numeric_cols)

for cname in numeric_cols:
    meanVal = np.mean(dataset[cname])
    stdVal = np.std(dataset[cname])

    #setting cutoff and lower/upper bounds for data values
    cut_off = stdVal * 4
    lower = meanVal - cut_off
    upper = meanVal + cut_off

    #identifying outliers
    outliers = []
    outlierRows = []
    #print("len of dataset[cname]: ",dataset.at[0,cname])
    for i in range(0,len(dataset[cname])):
        if (dataset.at[i,cname] < lower) or (dataset.at[i,cname] > upper):
            outliers.append(dataset.at[i,cname])
            outlierRows.append(i+1)
    print("Outlier values in column ", cname, ": ", outliers)
    print("Row id corresponding to outlier values in column ",cname , ": ",outlierRows)

#remove outliers from previously identified indicies
if (len(numeric_cols) > 0):
    dataset.drop(outlierRows, inplace=True)

# reset the index of the dataset
dataset.reset_index(drop=True, inplace=True)

# change categorical columns to numeric
le = LabelEncoder()
col_list = dataset.select_dtypes(include = "object").columns
for colsn in col_list:
    dataset[colsn] = le.fit_transform(dataset[colsn].astype(str))

# save clean dataset to a file
dataset.to_csv(output_directory + '/clean_dataset.csv', index=False)

# ====================== #
# Step 2: summarize data #
# ====================== #
print('\n>>> Step 2: Data summarization\n')

# dataset size
print('\nDataset size after cleaning (#rows,#columns): ', dataset.shape)

# first 10 values
print('\nFirst 10 lines of data:\n', dataset.head(10))

# summary statistics
print('\nSummary stats of data:\n', dataset.describe())


# =============================== #
# Step 3: visual data exploration #
# =============================== #
print('\n>>> Step 3: Data visualization\n')

# histograms
fig, ax = plt.subplots()
dataset.hist(figsize=[9, 9], bins=20, rwidth=0.9, color="green", grid=False)
plt.tight_layout()
plt.savefig(output_directory+'/data_histogram.png',dpi=300)
#plt.show()
plt.close()

# scatter plot matrix
axes = scatter_matrix(dataset,figsize=[9, 9])
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')
plt.tight_layout()
plt.savefig(output_directory+'/data_scatter_matrix.png',dpi=300)
#plt.show()
plt.close()

# Pearson product-moment correlation plot
if (len(dataset.columns) < 8):
    plt.figure(figsize=(12,10))
    sns.heatmap(dataset.corr(), cmap="coolwarm",annot=True)
else:
    g = sns.clustermap(dataset.corr(),
                   method = 'complete',
                   cmap   = 'coolwarm',
                   annot  = True,
                   annot_kws = {'size': 8})
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=60);
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=60);
plt.tight_layout()
plt.savefig(output_directory+'/data_correlation_plot.png',dpi=300)
#plt.show()
plt.close()

# ================== #
# Step 4: split data #
# ================== #
print('\n>>> Step 4: Data splitting\n')

# separate data into training/validation and testing datasets
array = dataset.values
X = array[:,:-1]
y = array[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# ================== #
# Step 5: scale data #
# ================== #

# scale input data
print('\n>>> Step 5: Data scaling\n')

scaler = StandardScaler() # sensitive to outliers
#scaler = MinMaxScaler()  # sensitive to outliers
#scaler = RobustScaler()  # resilient to outliers
X_train = scaler.fit_transform(X_train)

# do not use fit_transform on testing data to avoid introduction of bias!!!
X_test = scaler.transform(X_test)

# ========================= #
# Step 6: initialize models #
# ========================= #

# initialize models
print('\n>>> Step 6: Model initilization (default parameters)\n')

models = []
models.append(('LR', LinearRegression()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('DT', DecisionTreeRegressor()))
models.append(('SVM', SVR(gamma='auto')))

# ======================================= #
# Step 7: eval models before optimization #
# ======================================= #

# evaluate models before hyper-parameter optimization
print('\n>>> Step 7: Preliminary model evaluation\n')

results = []
names = []
for name, model in models:
    kfold = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=1)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1, error_score='raise')
    results.append(cv_results)
    add_res_to_cv_results(cv_results, name, 'before')
    add_res_to_combined_avg_results(cv_results, name, 'training_before_optim', 'MAE')
    names.append(name)
    # remove the absolute value function if using measures
    #   that can have meaningful negative values
    print('%s: %f (%f)' % (name, abs(cv_results.mean()), cv_results.std()))

# boxplot to compare models based on training results
plt.boxplot(
	list(map(abs,results)),
	tick_labels=names,
	patch_artist=True,
	showmeans=True,
	meanprops={
        "marker":"o",
        "markerfacecolor":"white",
        "markeredgecolor":"red",
        "markersize":2
    },
    boxprops=dict(facecolor='lightblue'))
plt.title('Algorithm Comparison - before optimization')
plt.ylabel('MAE')
plt.xlabel('Algorithm')
plt.tight_layout()
plt.savefig(output_directory+'/alg_comp_before_optim.png',dpi=300)
#plt.show()
plt.close()

# ======================================== #
# Step 8: Preliminary overfitting analysis #
# ======================================== #

# Ovefitting analysis via learning curves (before hyper-parameter optimization)
print('\n>>> Step 8: Overfitting analysis of default models\n')

for name, model in models:
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, Y_train, scoring='neg_mean_absolute_error')
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.figure()
    plt.plot(train_sizes, abs(train_scores_mean), 'o-', color="r",label="Training score")
    plt.plot(train_sizes, abs(test_scores_mean), 'o-', color="g",label="Cross-validation score")
    plt.xlabel("Number of training samples")
    plt.ylabel("Score (MAE)")
    plt.title(name)
    plt.legend(loc="best")
    plt.savefig(output_directory+'/learning_curve_before_optim_{}.png'.format(name))
    plt.close()

# ==================================== #
# Step 9: hyper-parameter optimization #
# ==================================== #

# hyper-parameter optimization
print('\n>>> Step 9: Hyper-parameter optimizations\n')

# define parameters to be optimized and possible values for each model
model_params = dict()
model_params['LR'] = dict()
model_params['LR']['fit_intercept'] = [True, False]
model_params['KNN'] = dict()
model_params['KNN']['n_neighbors'] = list(range(1,5,1))
model_params['DT'] = dict()
model_params['DT']['criterion'] = ['friedman_mse', 'absolute_error', 'poisson', 'squared_error']
model_params['DT']['max_depth'] = list(range(1,10,1))
model_params['SVM'] = dict()
model_params['SVM']['C'] = list(arange(0.01,1.6,0.2)) + [1]

# find and store best parameters for each model
best_params = dict()
for name, model in models:
    cv = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=1)
    # if time is a concern use RandomizedSearchCV instead of GridSearchCV
    optim_search = GridSearchCV(estimator=model, param_grid=model_params[name], n_jobs=-1, cv=cv, scoring='neg_mean_absolute_error')
    #optim_search = RandomizedSearchCV(estimator=model, param_distributions=model_params[name], n_iter=5, n_jobs=-1, cv=cv, scoring='neg_mean_absolute_error')
    optim_result = optim_search.fit(X_train, Y_train)
    print("Model %s -- Best: %f using %s" % (name, abs(optim_result.best_score_), optim_result.best_params_))
    best_params[name] = optim_result.best_params_

# ====================================== #
# Step 10: update model hyper-parameters #
# ====================================== #

# re-initialize models using best parameter settings
print('\n>>> Step 10: Update model hyper-parameters\n')

optimized_models = []
optimized_models.append(('LR', LinearRegression(fit_intercept=best_params['LR']['fit_intercept'])))
optimized_models.append(('KNN', KNeighborsRegressor(n_neighbors=best_params['KNN']['n_neighbors'])))
optimized_models.append(('DT', DecisionTreeRegressor(criterion=best_params['DT']['criterion'], max_depth=best_params['DT']['max_depth'])))
optimized_models.append(('SVM', SVR(gamma='auto',C=best_params['SVM']['C'])))

# ======================================= #
# Step 11: evaluation of optimized models #
# ======================================= #

# evaluate models after hyper-parameter optimization
print('\n>>> Step 11: Evaluate optimized models\n')

results = []
names = []
for name, model in optimized_models:
	kfold = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=1)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1, error_score='raise')
	results.append(cv_results)
	add_res_to_cv_results(cv_results, name, 'after')
	add_res_to_combined_avg_results(cv_results, name, 'training_after_optim','MAE')
	names.append(name)
	print('%s: %f (%f)' % (name, abs(cv_results.mean()), cv_results.std()))

# compare optimized models based on training results
plt.boxplot(
	list(map(abs,results)),
	tick_labels=names,
	patch_artist=True,
	showmeans=True,
	meanprops={
        "marker":"o",
        "markerfacecolor":"white",
        "markeredgecolor":"red",
        "markersize":2
    },
    boxprops=dict(facecolor='moccasin')
)
plt.title('Algorithm Comparison - after optimization')
plt.ylabel('MAE')
plt.xlabel('Algorithm')
plt.tight_layout()
plt.savefig(output_directory+'/alg_comp_after_optim.png',dpi=300)
#plt.show()
plt.close()

# create a grouped boxplot to compare before and after optimization results
my_pal = ['lightblue','moccasin']
sns.boxplot(data = all_cv_results,
            x = all_cv_results['algorithm'],
            y = abs(all_cv_results['cv_result']),
            hue = all_cv_results['before_after_optim'],
            gap = .15,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker":"o",
                "markerfacecolor":"white",
                "markeredgecolor":"red",
                "markersize":2
            },
            palette=my_pal)

plt.ylabel('MAE')
plt.savefig(output_directory+'/alg_comp_before_and_after_optim.png',dpi=300)
#plt.show()
plt.close()

# ================================================= #
# Step 12: overfitting analysis of optimized models #
# ================================================= #

# Ovefitting analysis via learning curves (after hyper-parameter optimization)
print('\n>>> Step 12: Overfitting analysis of optimized models\n')

for name, model in optimized_models:
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, Y_train, scoring='neg_mean_absolute_error')
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.figure()
    plt.plot(train_sizes, abs(train_scores_mean), 'o-', color="r",label="Training score")
    plt.plot(train_sizes, abs(test_scores_mean), 'o-', color="g",label="Cross-validation score")
    plt.xlabel("Number of training samples")
    plt.ylabel("Score (MAE)")
    plt.title(name)
    plt.legend(loc="best")
    plt.savefig(output_directory+'/learning_curve_after_optim_{}.png'.format(name))
    plt.close()

# ============================== #
# Step 13: save optimized models #
# ============================== #

# fit and save optimized models
print('\n>>> Step 13: Save optimized models\n')

for name, model in optimized_models:
    model.fit(X_train, Y_train)
    filename = output_directory + '/' + name + '_optimized_model.sav'
    joblib.dump(model, filename)

# ====================================== #
# Step 14: feature importance evaluation #
# ====================================== #

# Permutation feature importance
print('\n>>> Step 14: Investigate feature importance of optimized models\n')

for name, model in optimized_models:
	model.fit(X_train, Y_train)
	result = permutation_importance(model,
                                X_train,
                                Y_train,
                                scoring='neg_mean_absolute_error')
	sorted_idx = result.importances_mean.argsort()

	# plot feature importances
	fig, ax = plt.subplots()
	bp = ax.boxplot(result.importances[sorted_idx].T,
           vert=False,
           tick_labels=dataset[dataset.columns[:len(dataset.columns)]].columns[sorted_idx],
           patch_artist=True,
           showmeans=True,
           meanprops={
               "marker":"o",
               "markerfacecolor":"white",
               "markeredgecolor":"blue",
               "markersize":2,
               "label":"mean"
           })
	# add legend without repeated entries
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles[:1], labels[:1], loc='lower right', bbox_to_anchor=(1, 0))

	# customize boxplot
	for box in bp['boxes']:
		# change outline color
		box.set(color='black', linewidth=1)
		# change fill color
		box.set(facecolor = 'lightgreen' )

	ax.set_title("Permutation Feature Importance: " + name)
	plt.xlabel('Feature importance score')
	fig.tight_layout()
	#plt.show()
	plt.savefig(output_directory+'/pfi_barplot_'+name+'.png', dpi=300)
	plt.close()

# ====================================== #
# Step 15: model evaluation on test sets #
# ====================================== #

# testing results
print('\n>>> Step 15: Evaluate models on test sets\n')

for name, model in optimized_models:
	model.fit(X_train, Y_train)
	predicted_results = model.predict(X_test)

	mape_result = mean_absolute_percentage_error(Y_test,predicted_results)
	print('{} Mean Absolute Percenage Error (MAPE): {:.3f}'.format(name, mape_result))
	add_res_to_combined_avg_results(mape_result, name, 'testing', 'MAPE')

	mae_result = mean_absolute_error(Y_test,predicted_results)
	print('{} Mean Absolute Error (MAE): {:.3f}'.format(name, mae_result))
	add_res_to_combined_avg_results(mae_result, name, 'testing', 'MAE')

	mse_result = mean_squared_error(Y_test,predicted_results)
	print('{} Mean Squared Error (MSE): {:.3f}'.format(name, mse_result))
	add_res_to_combined_avg_results(mse_result, name, 'testing', 'MSE')

	rmse_result = root_mean_squared_error(Y_test,predicted_results)
	print('{} Root Mean Squared Error (RMSE): {:.3f}'.format(name, rmse_result))
	add_res_to_combined_avg_results(rmse_result, name, 'testing', 'RMSE')

	r2_result = r2_score(Y_test,predicted_results)
	print('{} R-squared (R2): {:.3f}'.format(name, r2_result))
	add_res_to_combined_avg_results(r2_result, name, 'testing', 'R2')

	pearson_result = pearsonr(Y_test,predicted_results)
	print('{} Pearson correl. coef. (r): {:.3f} (p-value = {:.2E})'.format(name, pearson_result.statistic, pearson_result.pvalue))
	add_res_to_combined_avg_results(pearson_result.statistic, name, 'testing', 'Pearson_CC score')
	add_res_to_combined_avg_results(pearson_result.pvalue, name, 'testing', 'Pearson_CC p-value')

	ccc_result = concordance_correlation_coefficient(Y_test,predicted_results)
	print('{} Concordance Correlation Coefficient (CCC): {:.3f}'.format(name, ccc_result))
	print('')

    # generate a scatter plot for each model
	plt.scatter(Y_test,predicted_results)
	plt.title('Test results for ' + name)
	plt.xlabel('Ground truth')
	plt.ylabel('Predicted results')
	xpoints = ypoints = max(plt.xlim(),plt.ylim())
	plt.plot(xpoints, ypoints, linestyle='--', color='gray', lw=1, scalex=False, scaley=False)
	plt.xlim(xpoints)
	plt.ylim(ypoints)
	plt.tight_layout()
	plt.savefig(output_directory+'/testing_scatterplot_'+name+'.png', dpi=300)
	#plt.show()
	plt.close()

	# generate a QQ plot for each model
	residuals = Y_test - predicted_results
	residuals = np.array(residuals)
	fig = plt.figure()
	qqplot(residuals, fit=True, line="45")
	plt.title("Residuals for ground truth and predicted results")
	plt.tight_layout()
	plt.savefig(output_directory+'/testing_qqplot_'+name+'.png', dpi=300)
	#plt.show()
	plt.close()

# Save DataFrame with combined results to file
combined_avg_results.to_csv(output_directory+'/combined_average_results.csv',index=False)
