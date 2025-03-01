import torch
import pandas as pd
from ..main import get_statistics
from tabulate import tabulate

data_name = "hepatitis"
file_path = "src/log/raw_" + data_name + ".csv"
device = torch.device('cpu')

dlist = pd.read_csv(file_path).values.tolist()

num_trials = len(dlist[0]) - 1

table = [
    ["Classifier Type", "Data", "Trials", "Min ER", "Min Cover", "Med ER", "Med Cover", "95th ER", "95th Cover", "Avg ER", "Avg Cover", "ER std", "95th Avg ER", "95th Avg ER"],
    get_statistics(dlist[0][0], data_name, num_trials, torch.tensor(dlist[0][1:], dtype=torch.float32, device=device)),
    get_statistics(dlist[1][0], data_name, num_trials, torch.tensor(dlist[1][1:], dtype=torch.float32, device=device)),
    get_statistics(dlist[2][0], data_name, num_trials, torch.tensor(dlist[2][1:], dtype=torch.float32, device=device), torch.tensor(dlist[4][1:], dtype=torch.float32, device=device)),
    get_statistics(dlist[3][0], data_name, num_trials, torch.tensor(dlist[3][1:], dtype=torch.float32, device=device), torch.tensor(dlist[4][1:], dtype=torch.float32, device=device))
]
print(tabulate(table, headers="firstrow", tablefmt="grid"))