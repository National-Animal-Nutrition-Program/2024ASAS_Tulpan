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
