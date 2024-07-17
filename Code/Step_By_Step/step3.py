# =============================== #
# Step 3: visual data exploration #
# =============================== #
print('\n>>> Step 3: Data visualization\n')

# histograms
fig, ax = plt.subplots()
dataset.hist(figsize=[9, 9], bins=20, rwidth=0.9, color="green", grid=False)
plt.tight_layout()
plt.savefig(output_directory+'/data_histogram.png',dpi=300)
#plt.show()
plt.close()

# scatter plot matrix
axes = scatter_matrix(dataset,figsize=[9, 9])
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')
plt.tight_layout()
plt.savefig(output_directory+'/data_scatter_matrix.png',dpi=300)
#plt.show()
plt.close()

# Pearson product-moment correlation plot
if (len(dataset.columns) < 8):
    plt.figure(figsize=(12,10))
    sns.heatmap(dataset.corr(), cmap="coolwarm",annot=True)
else:
    g = sns.clustermap(dataset.corr(),
                   method = 'complete',
                   cmap   = 'coolwarm',
                   annot  = True,
                   annot_kws = {'size': 8})
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=60);
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=60);
plt.tight_layout()
plt.savefig(output_directory+'/data_correlation_plot.png',dpi=300)
#plt.show()
plt.close()
