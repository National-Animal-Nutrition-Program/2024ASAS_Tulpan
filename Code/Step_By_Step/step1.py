# ===================== #
# Step 1: Data cleaning #
# ===================== #
print('\n>>> Step 1: Data cleaning\n')

# remove rows with missing values and reset the row index
dataset = dataset.dropna(ignore_index=True)

# find duplicate rows
dups = dataset.duplicated()
print("\nDuplicated rows:\n", dups)

# report if there are any duplicates
print("\nAny duplicated rows: ", dups.any())

# list all duplicate rows
print("\nDuplicated rows:\n", dataset[dups])

# delete duplicate rows
#dataset.drop_duplicates(inplace=True)
dataset = dataset[~dataset.index.duplicated()]

# delete duplicate columns
duplicate_cols = dataset.columns[dataset.columns.duplicated()]
dataset.drop(columns=duplicate_cols, inplace=True)

# remove single value columns
# get the number of unique values for each column
counts = dataset.nunique()

# record columns to delete
to_del = [i for i,v in enumerate(counts) if v == 1]
print("Columns to delete: ", to_del)

# drop duplicated columns
if (len(to_del) > 0):
    dataset.drop(dataset.columns[to_del], axis=1, inplace=True)

# reset the index of the dataset
dataset.reset_index(drop=True, inplace=True)

# perform outlier detection via Z-scores
# --------------------------------------
# identify numeric columns
print('\nColumn types:\n', dataset.dtypes)
numeric_cols = dataset.select_dtypes(include=np.number).columns.tolist()
print("Numeric columns: ", numeric_cols)

for cname in numeric_cols:
    meanVal = np.mean(dataset[cname])
    stdVal = np.std(dataset[cname])

    #setting cutoff and lower/upper bounds for data values
    cut_off = stdVal * 4
    lower = meanVal - cut_off
    upper = meanVal + cut_off

    #identifying outliers
    outliers = []
    outlierRows = []
    #print("len of dataset[cname]: ",dataset.at[0,cname])
    for i in range(0,len(dataset[cname])):
        if (dataset.at[i,cname] < lower) or (dataset.at[i,cname] > upper):
            outliers.append(dataset.at[i,cname])
            outlierRows.append(i+1)
    print("Outlier values in column ", cname, ": ", outliers)
    print("Row id corresponding to outlier values in column ",cname , ": ",outlierRows)

#remove outliers from previously identified indicies
if (len(numeric_cols) > 0):
    dataset.drop(outlierRows, inplace=True)

# reset the index of the dataset
dataset.reset_index(drop=True, inplace=True)

# change categorical columns to numeric
le = LabelEncoder()
col_list = dataset.select_dtypes(include = "object").columns
for colsn in col_list:
    dataset[colsn] = le.fit_transform(dataset[colsn].astype(str))

# save clean dataset to a file
dataset.to_csv(output_directory + '/clean_dataset.csv', index=False)
