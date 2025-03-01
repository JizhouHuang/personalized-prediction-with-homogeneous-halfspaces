import pandas as pd

data_names = ["lending_test.pkl", "lendingNOH_train.pkl", "lending_train.pkl", "CSDS1_train.pkl"]
num_samples = [160000, 450000, 160000, 75000]

for dn, num in zip(data_names, num_samples):
    data = pd.read_pickle(dn)

    if len(data) > num:
        data_downsampled = data.sample(n=num, random_state=42)
    else:
        data_downsampled = data
    
    data_downsampled.to_pickle(dn)