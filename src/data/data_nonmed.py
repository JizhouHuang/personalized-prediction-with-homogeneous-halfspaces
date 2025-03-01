import pandas as pd

# data_names = ["adultNM", "adultNMNOH", "CSDS1", "CSDS2", "CSDS3", "GiveMe", "lending", "lendingNOH", "UCI_credit"]
data_names = ["CSDS1", "CSDS2", "CSDS3", "GiveMe", "lending", "lendingNOH", "UCI_credit"]
data_types = ["train", "test"]
label_name = "TARGET"

for dn in data_names:
    for dt in data_types:
        file_path = "_".join([dn, dt]) + ".pkl"
        data = pd.read_pickle(file_path)
        print(f"Dataset name: {file_path}")

        # data[label_name] = data[label_name].astype(float)

        print(f"NaN number is {data.isna().sum().sum()}")
        print(data.info())
        print(data.head())
        
        # move label column to the first
        data = data[
            [label_name] + [col for col in data.columns if col != label_name]
        ]
        print(data.info())
        print(data.head())

        # Center the features (excluding the label column)
        col_to_normalize = [col for col in data.columns if col != label_name]
        data[col_to_normalize] = data[col_to_normalize].astype(float)
        data[col_to_normalize] = data[col_to_normalize] - data[col_to_normalize].mean()
        print(data.info())
        print(data.head())

        # Identify columns with only one unique value
        col_to_normalize = data.columns[data.nunique() > 1]
        col_to_normalize = [col for col in col_to_normalize if col != label_name]
        data[col_to_normalize] = data[col_to_normalize] / data[col_to_normalize].std()

        print(f"NaN number is {data.isna().sum().sum()}")

        # Verify there are no columns that are of type str
        str_columns = data.select_dtypes(exclude=['number', 'bool']).columns
        print(f"String columns are: {str_columns.tolist()}")

        print(data.info())
        print(data.head())
        # data.to_pickle(file_path)

# for dn in data_names:
#     for dt in data_types:
#         file_path = "_".join([dn, dt]) + ".pkl"
#         data = pd.read_pickle(file_path)
#         print(f"Dataset name: {file_path}")
#         print(data.info())
        # print(data.head())

# dn = "hypothyroid_train.pkl"
# df = pd.read_pickle(dn)
# print(df['lithium'].unique())