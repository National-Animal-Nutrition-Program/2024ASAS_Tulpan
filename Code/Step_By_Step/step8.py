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
