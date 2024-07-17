# ====================== #
# Step 2: summarize data #
# ====================== #
print('\n>>> Step 2: Data summarization\n')

# dataset size
print('\nDataset size after cleaning (#rows,#columns): ', dataset.shape)

# first 10 values
print('\nFirst 10 lines of data:\n', dataset.head(10))

# summary statistics
print('\nSummary stats of data:\n', dataset.describe())
