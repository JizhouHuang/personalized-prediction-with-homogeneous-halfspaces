import torch
import torch.nn as nn
import yaml
from typing import Union, Tuple, List
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from ..utils.data import MultiLabelledDataset
from ..models.conditional_predictor import ReferenceClass
from ..models.robust_predictor import RobustListLearner
from ..models.baseline_learner import SVMLearner
from ..utils.simple_models import LinearModel


class ExperimentCCSC(nn.Module):
    """
    Experiment of Conditional Classification for Sparse Classes.
    """
    def __init__(
            self,
            prev_header: str,
            experiment_id: int,
            config_file_path: str,
            device: torch.device
    ):
        """
        Initialize through reading parameters from YAML file located under src/config/.

        Parameters:
        experiment_id (int): The ID of the experiment.
        config_file_path (str): The path to the configuration file.

        Explanations:
        num_sample_rll:     Number of training data used for Robust List Learning.
        margin:             According to Appendix A, the RHS of the linear system is formed by labels subtracted by the margin.
        sparsity:           Number of non-zero dimensions for the resulting sparse representations.
        cluster_size:       To speed up the computation, instead of iterating only one classifier at a time, 
                            we partition all the sparse classifiers into multiple clusters and run PSGD on a cluster in each iteration.
        data_frac_psgd:     Fraction of training data used for the updating stage of Projected SGD.
        lr_coeff:           A constant to scale the learning rate (beta) of PSGD.
        num_iter:           The number of iteration for PSGD to run.
        batch_size:         Number of example to estimate the expectation of projected gradient in each gradient step.
        """
        super(ExperimentCCSC, self).__init__()
        self.header = " ".join([prev_header, "experiment", str(experiment_id), "-"])
        self.device = device

        # Read the YAML configuration file
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Load configuration values
        # self.data_frac = config['data_frac']
        self.num_sample_rll = config['num_sample_rll']
        self.margin = config['margin']
        self.sparsity = config['sparsity']
        self.cluster_size = config['cluster_size']
        self.train_subset_fracs = config['train_subset_fracs']
        self.lr = config['lr']
        self.num_iter = config['num_iter']
        self.batch_size = config['batch_size']
        self.eid = experiment_id

    def forward(
            self,
            data_train: torch.Tensor,
            data_test: torch.Tensor
    ) -> List[
        Tuple[
            torch.Tensor, 
            Union[
                torch.float, 
                Tuple[torch.float, torch.float]
            ]
        ]
    ]:
        """
        Call Robust List Learner to generate a list of sparse classifiers and input them to the Conditional Learner.

        Parameters:
        data_train:     Training data for both Robust List Learning and Conditional Learning.
        data_test:      Testing data to estimate the error measures of the final classifier-selector pair.
                        Disjoint from data_train.
        """
        # Learn the sparse classifiers
        print(" ".join([self.header, "initializing robust list learner for sparse perceptrons ..."]))

        robust_list_learner = RobustListLearner(
            prev_header=self.header + ">",
            sparsity=self.sparsity, 
            margin=self.margin,
            cluster_size=self.cluster_size,
            device=self.device
        )

        sparse_classifier_clusters: List[LinearModel] = robust_list_learner(
            DataLoader(
                MultiLabelledDataset(
                    data_train, 
                    # random_state=42
                ),
                batch_size=self.num_sample_rll
            )
        )   # List[LinearModel]

        # Perform conditional learning
        print(" ".join([self.header, "initializing conditional classification learner for homogeneous halfspaces ..."]))
        conditional_learner = ReferenceClass(
            prev_header=self.header + ">",
            subset_fracs=self.train_subset_fracs, 
            num_iter=self.num_iter, 
            lr=self.lr, 
            device=self.device
        )

        # learn reference class with label mapping
        eval_errors, eval_ids, selectors = conditional_learner(
            dataset = MultiLabelledDataset(
                data=data_train,
                predictor=sparse_classifier_clusters[0]
            )
        )   # Tuple[torch.Tensor, torch.Tensor, LinearModel]

        print(f"{self.header} learned selectors (after selecting) size: {selectors.size()}\n")

        init_weights_ids = torch.argmin(eval_errors)
        predictor: LinearModel = sparse_classifier_clusters[0].reduce(eval_ids[init_weights_ids])
        selectors: LinearModel = selectors.reduce(init_weights_ids)
        print(f"{self.header} result predictor size: {predictor.size()}\n")
        print(f"{self.header} reuslt selector size: {selectors.size()}\n")

        # map the labels in test data according to the selected predictor
        labels_test, features_test = MultiLabelledDataset(
            data=data_test,
            predictor=predictor
        )[:]

        print(f"{self.header} testing dataset feature size: {features_test.size()}")
        print(f"{self.header} testing dataset label size: {labels_test.size()}")

        # model selection for sparse classifiers based on regular classification error
        print(" ".join([self.header, "finding empirical error minimizer from sparse perceptrons ..."]))
        min_error, _ = torch.min(
            sparse_classifier_clusters[0].error_rate(
                X=features_test,
                y=data_test[:, 0]
            ), 
            dim=0
        )

        print(f"{self.header} best regular sparse classifier error rate: {min_error}")
        
        # Estimate error measures with selectors
        error_wo = predictor.error_rate(
            X=features_test,
            y=data_test[:, 0]
        )

        error = selectors.conditional_one_rate(
            X=features_test,
            y=labels_test
        )     

        coverage = selectors.prediction_rate(X=features_test)
        print(f"{self.header} result conditional error and coverage: {error, coverage}")

        res = [
            float(min_error),
            float(error_wo),
            float(error), 
            float(coverage)
        ]
        
        # Print the results in a table format
        table = [
            ["Classifier Type", "Train Size", "Test Size", "Sample Dim", "Sparsity", "PSGD Iter", "Batch Size", "LR Coeff", "Est ER", "Coverage"],
            ["Classic Sparse", data_train.size(0), data_test.shape[0], data_test.shape[1] - 1, self.sparsity, self.num_iter, self.batch_size, self.lr, min_error, 1],
            ["Cond Sparse w/o Selector", data_train.size(0), data_test.shape[0], data_test.shape[1] - 1, self.sparsity, self.num_iter, self.batch_size, self.lr, error_wo, 1],
            ["Cond Sparse", data_train.size(0), data_test.shape[0], data_test.shape[1] - 1, self.sparsity, self.num_iter, self.batch_size, self.lr, error, coverage]
        ]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

        return res
    
    def normalize(
            self, 
            X: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize the input tensor X.
        
        Parameters:
        X (torch.Tensor): The input tensor to be normalized.
                          [m, d]
        
        Returns:
        X_normalized (torch.Tensor): The normalized tensor.
        """
        return X / torch.norm(X, p=2, dim=1, keepdim=True)

    def random_observation(
            self,
            length: int
        ) -> torch.Tensor:
        """
        Generate two opposite observations on a random direction.
        
        Returns:
        observation (torch.Tensor): The random observation.
                                    [dim_sample]
        """
        X = torch.randn(
            length, 
            device=self.device
        )
        X = torch.stack(
            [X, -X]
        )
        return self.normalize(X)