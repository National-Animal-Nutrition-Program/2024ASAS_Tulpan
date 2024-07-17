# ====================================== #
# Step 15: model evaluation on test sets #
# ====================================== #

# testing results
print('\n>>> Step 15: Evaluate models on test sets\n')

for name, model in optimized_models:
	model.fit(X_train, Y_train)
	predicted_results = model.predict(X_test)

	mape_result = mean_absolute_percentage_error(Y_test,predicted_results)
	print('{} Mean Absolute Percenage Error (MAPE): {:.3f}'.format(name, mape_result))
	add_res_to_combined_avg_results(mape_result, name, 'testing', 'MAPE')

	mae_result = mean_absolute_error(Y_test,predicted_results)
	print('{} Mean Absolute Error (MAE): {:.3f}'.format(name, mae_result))
	add_res_to_combined_avg_results(mae_result, name, 'testing', 'MAE')

	mse_result = mean_squared_error(Y_test,predicted_results)
	print('{} Mean Squared Error (MSE): {:.3f}'.format(name, mse_result))
	add_res_to_combined_avg_results(mse_result, name, 'testing', 'MSE')

	rmse_result = root_mean_squared_error(Y_test,predicted_results)
	print('{} Root Mean Squared Error (RMSE): {:.3f}'.format(name, rmse_result))
	add_res_to_combined_avg_results(rmse_result, name, 'testing', 'RMSE')

	r2_result = r2_score(Y_test,predicted_results)
	print('{} R-squared (R2): {:.3f}'.format(name, r2_result))
	add_res_to_combined_avg_results(r2_result, name, 'testing', 'R2')

	pearson_result = pearsonr(Y_test,predicted_results)
	print('{} Pearson correl. coef. (r): {:.3f} (p-value = {:.2E})'.format(name, pearson_result.statistic, pearson_result.pvalue))
	add_res_to_combined_avg_results(pearson_result.statistic, name, 'testing', 'Pearson_CC score')
	add_res_to_combined_avg_results(pearson_result.pvalue, name, 'testing', 'Pearson_CC p-value')

	ccc_result = concordance_correlation_coefficient(Y_test,predicted_results)
	print('{} Concordance Correlation Coefficient (CCC): {:.3f}'.format(name, ccc_result))
	print('')

    # generate a scatter plot for each model
	plt.scatter(Y_test,predicted_results)
	plt.title('Test results for ' + name)
	plt.xlabel('Ground truth')
	plt.ylabel('Predicted results')
	xpoints = ypoints = max(plt.xlim(),plt.ylim())
	plt.plot(xpoints, ypoints, linestyle='--', color='gray', lw=1, scalex=False, scaley=False)
	plt.xlim(xpoints)
	plt.ylim(ypoints)
	plt.tight_layout()
	plt.savefig(output_directory+'/testing_scatterplot_'+name+'.png', dpi=300)
	#plt.show()
	plt.close()

	# generate a QQ plot for each model
	residuals = Y_test - predicted_results
	residuals = np.array(residuals)
	fig = plt.figure()
	qqplot(residuals, fit=True, line="45")
	plt.title("Residuals for ground truth and predicted results")
	plt.tight_layout()
	plt.savefig(output_directory+'/testing_qqplot_'+name+'.png', dpi=300)
	#plt.show()
	plt.close()

# Save DataFrame with combined results to file
combined_avg_results.to_csv(output_directory+'/combined_average_results.csv',index=False)
