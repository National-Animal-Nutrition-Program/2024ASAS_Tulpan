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
