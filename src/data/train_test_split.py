import pandas as pd
import yaml

# for UCI Medical data
data_names = ["diabetes", "haberman", "hepatitis", "hypothyroid", "wdbc"]

for dn in data_names:
    data_path = ".".join([dn, "data"])
    yaml_path = "../config/data/" + ".".join([dn, "yaml"])

    print(f"Dataset name: {data_path}")

    # read yaml config file
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract configuration values
    attribute_names = config['attribute_names']
    label_name = config['label_name']
    categorical_attr_names = config['categorical_attr_names']
    binary_attr_names = config['binary_attr_names']
    sparse_attr_names = config['sparse_attr_names']
    label_true = config['label_true']
    label_false = config['label_false']
    attr_true = config['attr_true']
    attr_false = config['attr_false']

    data = pd.read_csv(
        data_path, 
        header=None, 
        names=attribute_names, 
        na_values="?"
    )

    # drop sparse attributes
    if sparse_attr_names:
        data = data.drop(
            labels=sparse_attr_names,
            axis=1
        )
    
    # move label column to the first
    data = data[
        [label_name] + [col for col in data.columns if col != label_name]
    ]

    # map binary attributes to {0, 1}
    if binary_attr_names:
        data[binary_attr_names] = data[binary_attr_names].replace(
            {
                attr_false: 0, 
                attr_true: 1
            }
        )

    # Map labels to {0, 1}
    data[label_name] = data[label_name].map({label_false: 0, label_true: 1})

    # Display basic info
    # print(data.head())
    # print(data.isnull().sum())  # Check for missing values

    # Convert categorical columns (e.g., "Sex") to one-hot encoding if needed
    if categorical_attr_names:
        data = pd.get_dummies(
            data, 
            columns=categorical_attr_names
        )
    
    # Fill missing values (e.g., with the column mean)
    data.fillna(
        data.mean(), 
        inplace=True
    )

    # Center the features (excluding the label column)
    col_to_normalize = [col for col in data.columns if col != label_name]
    data[col_to_normalize] = data[col_to_normalize].astype(float)
    data[col_to_normalize] = data[col_to_normalize] - data[col_to_normalize].mean()

    # Identify columns with only one unique value
    col_to_normalize = data.columns[data.nunique() > 1]
    col_to_normalize = [col for col in col_to_normalize if col != label_name]
    data[col_to_normalize] = data[col_to_normalize] / data[col_to_normalize].std()

    print(f"Number of NaN: {data.isna().sum().sum()}")

    fraction = 2/3

    # random split data
    data_train = data.sample(frac=fraction, random_state=42)

    print(data_train.info())
    print(data_train.head())

    # store as pkl files
    data_test = data.drop(data_train.index)
    
    print(data_test.info())
    print(data_test.head())

    data_train.to_pickle(dn + "_" + "train.pkl")
    data_test.to_pickle(dn + "_" + "test.pkl")