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
