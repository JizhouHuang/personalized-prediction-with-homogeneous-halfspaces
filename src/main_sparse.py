import os
import argparse
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import pandas as pd
from tabulate import tabulate
from .models.personalized_learner import SparseClassifierLearner

def main(
        data_name: str,
        num_experiment: int
):

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datasets: diabetes, haberman, hepatitis, hypothyroid, wdbc

    # Construct data file paths
    data_train_path = "".join(["src/data/csv/", data_name, "_train.csv"])
    data_test_path = "".join(["src/data/csv/", data_name, "_test.csv"])

    config_file_path = "".join(["src/config/model/", data_name, ".yaml"])
    # config_file_path = "src/config/model/model_toy.yaml"

    header = "main -"

    # Load the data
    data_train = torch.tensor(
        pd.read_csv(data_train_path).to_numpy(), 
        dtype=torch.float32
    ).to(device)

    data_test = torch.tensor(
        pd.read_csv(data_test_path).to_numpy(), 
        dtype=torch.float32
    ).to(device)

    sparse_predictors = []

    # for eid in tqdm(range(num_experiment), desc=" ".join([header, "running experiments"])):
    for eid in range(num_experiment):
        # Initialize the experiment
        experiment = SparseClassifierLearner(
            prev_header=header + ">",
            experiment_id=eid, 
            config_file_path=config_file_path,
            device=device
        )
        # Run the experiment
        error_rate = experiment(
            data_train=data_train,
            data_test=data_test
        )
        sparse_predictors.append(error_rate)
        print(f"Experiment {eid} completed with error rate: {error_rate}")
    
    print(" ".join([header, "average error rate:", str(sum(sparse_predictors) / len(sparse_predictors))]))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data analysis project.")
    parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset to use.')
    parser.add_argument('--num_exp', type=int, default=1, required=False, help='Name of the dataset to use.')

    args = parser.parse_args()
    main(args.data_name, args.num_exp)
