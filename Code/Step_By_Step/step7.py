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
