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
