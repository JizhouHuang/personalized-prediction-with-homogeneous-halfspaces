import pandas as pd

data_names = ["lending_test.csv", "lending_train.csv", "lendingNOH_train.csv", "lendingNOH_test.csv", "CSDS1_train.csv", "CSDS1_test.csv", "CSDS3_train.csv",]

data_names = ["CSDS1_train.csv"]

for dn in data_names:
    data = pd.read_csv(dn)

    # if len(data) > num:
    #     data_downsampled = data.sample(n=num, random_state=42)
    # else:
    #     data_downsampled = data

    num = int(len(data) * 0.95)
    data_downsampled = data.sample(n=num, random_state=42)
    
    data_downsampled.to_csv(dn, index=False)