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
