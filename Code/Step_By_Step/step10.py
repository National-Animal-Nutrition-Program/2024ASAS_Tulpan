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
