# ============================== #
# Step 13: save optimized models #
# ============================== #

# fit and save optimized models
print('\n>>> Step 13: Save optimized models\n')

for name, model in optimized_models:
    model.fit(X_train, Y_train)
    filename = output_directory + '/' + name + '_optimized_model.sav'
    joblib.dump(model, filename)
