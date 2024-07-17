# ================== #
# Step 4: split data #
# ================== #
print('\n>>> Step 4: Data splitting\n')

# separate data into training/validation and testing datasets
array = dataset.values
X = array[:,:-1]
y = array[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)
