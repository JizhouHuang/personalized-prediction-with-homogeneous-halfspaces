import pandas as pd

data_names = ["CSDS1", "CSDS2", "CSDS3", "GiveMe", "lending", "lendingNOH", "UCI_credit", "diabetes", "haberman", "hepatitis", "hypothyroid", "wdbc"]
data_types = ["train", "test"]
posfix_src = ".pkl"
posfix_dst = ".csv"

for dn in data_names:
    for dt in data_types:
        dataset_name = "_".join([dn, dt])
        src_file_path = dataset_name + posfix_src
        data = pd.read_pickle(src_file_path)

        dst_file_path = "csv/" + dataset_name + posfix_dst
        # Save the DataFrame to a CSV file
        data.to_csv(dst_file_path, index=False)