import os
import argparse
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from typing import List, Union
import torch
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score

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
    
    # Load the data
    data_train_np = pd.read_csv(data_train_path).to_numpy()
    # data_train = torch.tensor(
    #     data_train_np, 
    #     dtype=torch.float32
    # ).to(device)

    data_test_np = pd.read_csv(data_test_path).to_numpy()
    # data_test = torch.tensor(
    #     data_test_np, 
    #     dtype=torch.float32
    # ).to(device)

    X_train, y_train = data_train_np[:, 1:], data_train_np[:, 0]
    X_test, y_test = data_test_np[:, 1:], data_test_np[:, 0]

    # for eid in tqdm(range(num_experiment), desc=" ".join([header, "running experiments"])):
    for eid in range(num_experiment):
        ### Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        print("LogReg Accuracy:", 1 - accuracy_score(y_test, lr.predict(X_test)))

        ### SVM (Linear)
        svm_linear = SVC(kernel='linear')
        svm_linear.fit(X_train, y_train)
        print("SVM Linear Accuracy:", 1 - accuracy_score(y_test, svm_linear.predict(X_test)))

        ### SVM (RBF)
        svm_rbf = SVC(kernel='rbf')
        svm_rbf.fit(X_train, y_train)
        print("SVM RBF Accuracy:", 1 - accuracy_score(y_test, svm_rbf.predict(X_test)))

        ### XGBoost
        xgb_model = xgb.XGBClassifier(eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        print("XGBoost Accuracy:", 1 - accuracy_score(y_test, xgb_model.predict(X_test)))

        ### Random Forest
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)
        print("Random Forest Accuracy:", 1 - accuracy_score(y_test, rf.predict(X_test)))
        # sparse_predictors.append(sp)

        # # Record the result error measures
        # sparse_errs.append(res[0])
        # cond_errs_wo.append(res[1])
        # cond_errs.append(res[2]) 

        # print(f"{header} printing error statistics ...")
        # # Print the results in a table format
        # table = [
        #     ["Classifier Type", "Data", "Trials", "Min ER", "Min Cover", "Med ER", "Med Cover", "95th ER", "95th Cover", "Avg ER", "Avg Cover", "ER std", "95th Avg ER"],
        #     get_statistics("Classic Sparse", data_name, eid + 1, torch.tensor(sparse_errs, dtype=torch.float32, device=device)),
        #     get_statistics("Cond Sparse w/o Selector", data_name, eid + 1, torch.tensor(cond_errs_wo, dtype=torch.float32, device=device)),
        #     get_statistics("Cond Sparse", data_name, eid + 1, torch.tensor(cond_errs, dtype=torch.float32, device=device))
        # ]
        # print(tabulate(table, headers="firstrow", tablefmt="grid"))

    # print(f"{header} sparse predictors sizes are {sparse_predictors[0].size()}, {sparse_predictors[1].size()}")
    # print(f"{header} sparse predictors diff is {torch.norm(sparse_predictors[0].weights - sparse_predictors[1].weights)}")
    
        # data_store = [sparse_errs, cond_errs_wo, cond_errs, cond_svm_errs, coverages]
        # rows = ["Classic Sparse ER", "Cond Sparse ER w/o Selector", "Cond Sparse ER", "Cond SVM ER", "Coverage"]
        # df = pd.DataFrame(data_store, index=rows)
        # # df.to_csv("src/log/raw_" + data_name + ".csv", index=True)
        # df.to_csv("src/log/raw_" + data_name + "_3" + ".csv", index=True)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data analysis project.")
    parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset to use.')
    parser.add_argument('--num_exp', type=int, default=1, required=False, help='Name of the dataset to use.')

    args = parser.parse_args()
    main(args.data_name, args.num_exp)
