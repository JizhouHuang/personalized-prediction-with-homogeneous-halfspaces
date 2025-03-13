import os
import argparse
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from typing import List, Union
import torch
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from .models.personalized_predictor import PersonalizedPredictor

def main(data_name: str):

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datasets: diabetes, haberman, hepatitis, hypothyroid, wdbc

    # Construct data file paths
    data_train_path = "".join(["src/data/csv/", data_name, "_train.csv"])
    data_test_path = "".join(["src/data/csv/", data_name, "_test.csv"])

    config_file_path = "".join(["src/config/model/", data_name, ".yaml"])
    config_file_path = "src/config/model/model_toy.yaml"

    num_experiment = 2
    sparse_errs = []
    cond_errs_wo = []
    cond_errs = []
    cond_svm_errs = []
    coverages = []
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
        experiment = PersonalizedPredictor(
            prev_header=header + ">",
            experiment_id=eid, 
            config_file_path=config_file_path,
            device=device
        )

        # Run the experiment
        res, sp = experiment(
            data_train,
            data_test
        )
        sparse_predictors.append(sp)

        # Record the result error measures
        sparse_errs.append(res[0])
        cond_errs_wo.append(res[1])
        cond_errs.append(res[2]) 
        coverages.append(res[3])

        print(f"{header} printing error statistics ...")
        # Print the results in a table format
        table = [
            ["Classifier Type", "Data", "Trials", "Min ER", "Min Cover", "Med ER", "Med Cover", "95th ER", "95th Cover", "Avg ER", "Avg Cover", "ER std", "95th Avg ER", "95th Avg Cover"],
            get_statistics("Classic Sparse", data_name, eid + 1, torch.tensor(sparse_errs, dtype=torch.float32, device=device)),
            get_statistics("Cond Sparse w/o Selector", data_name, eid + 1, torch.tensor(cond_errs_wo, dtype=torch.float32, device=device)),
            get_statistics("Cond Sparse", data_name, eid + 1, torch.tensor(cond_errs, dtype=torch.float32, device=device), torch.tensor(coverages, dtype=torch.float32, device=device))
        ]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

    print(f"{header} sparse predictors sizes are {sparse_predictors[0].size()}, {sparse_predictors[1].size()}")
    print(f"{header} sparse predictors norms are {torch.norm(sparse_predictors[0].weights)}, {torch.norm(sparse_predictors[1].weights)}")
    print(f"{header} sparse predictors diff is {torch.norm(sparse_predictors[0].weights - sparse_predictors[1].weights)}")
    
        # data_store = [sparse_errs, cond_errs_wo, cond_errs, cond_svm_errs, coverages]
        # rows = ["Classic Sparse ER", "Cond Sparse ER w/o Selector", "Cond Sparse ER", "Cond SVM ER", "Coverage"]
        # df = pd.DataFrame(data_store, index=rows)
        # # df.to_csv("src/log/raw_" + data_name + ".csv", index=True)
        # df.to_csv("src/log/raw_" + data_name + "_3" + ".csv", index=True)
        

    
def get_statistics(
        classifier: str,
        data_name:str,
        eid: int,
        errors: torch.Tensor,
        coverage: torch.Tensor = None
) -> List[Union[str, torch.Tensor]]:
    
    # if eid == 1:
    #     return [classifier, data_name, eid, errors, coverage, errors, coverage, errors, coverage, errors, coverage, errors, coverage]
    
    # min err
    min_err, min_ids = torch.min(errors, dim=0)
    
    # median err
    med_err, med_ids = torch.median(errors, dim=0)

    # sorting for computing 95th quantile statistics
    sorted_err, sorted_ids = torch.sort(errors)

    # 95th quantile err
    nfq_err = torch.quantile(sorted_err, q=0.95, interpolation='lower')

    # average err
    avg_err = torch.mean(errors)

    # err std
    err_std = errors.std()

    # 95th quatile average err
    nfq_err_ids = torch.where(sorted_err == nfq_err)[0]
    if nfq_err_ids.size(0) > 1:
        nfq_err_ids = nfq_err_ids[0]

    nf_avg_err = torch.mean(sorted_err[:nfq_err_ids + 1])

    # compute coverages
    min_coverage = 1
    med_coverage = 1
    nfq_coverage = 1
    avg_coverage = 1
    nf_avg_coverage = 1

    if coverage is not None:
        min_coverage = coverage[min_ids]
        med_coverage = coverage[med_ids]
        sorted_cov = coverage[sorted_ids]
        
        nfq_coverage = sorted_cov[nfq_err_ids]
        avg_coverage = torch.mean(coverage)
        nf_avg_coverage = torch.mean(sorted_cov[:nfq_err_ids + 1])

    return [classifier, data_name, eid, min_err, min_coverage, med_err, med_coverage, nfq_err, nfq_coverage, avg_err, avg_coverage, err_std, nf_avg_err, nf_avg_coverage]

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data analysis project.")
    parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset to use.')

    args = parser.parse_args()
    main(args.data_name)