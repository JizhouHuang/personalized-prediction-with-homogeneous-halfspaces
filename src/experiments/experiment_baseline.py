import torch
import torch.nn as nn
import multiprocessing as mp
import yaml
from joblib import Parallel, delayed
from typing import List, Any, Callable
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader, random_split
from ..utils.data import TransformedDataset
from ..utils.simple_models import ConditionalLinearModel, PredictiveModel
from ..models.projected_sgd import SelectorPerceptron

class ExperimentBaseline(nn.Module):
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
        cluster_size:       To speed up the computation, instead of iterating only one classifier at a time, 
                            we partition all the sparse classifiers into multiple clusters and run PSGD on a cluster in each iteration.
        data_frac_psgd:     Fraction of training data used for the updating stage of Projected SGD.
        lr_coeff:           A constant to scale the learning rate (beta) of PSGD.
        num_iter:           The number of iteration for PSGD to run.
        batch_size:         Number of example to estimate the expectation of projected gradient in each gradient step.
        """
        super(ExperimentBaseline, self).__init__()
        self.header = " ".join([prev_header, "experiment", str(experiment_id), "-"])
        self.device = device

        # Read the YAML configuration file
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Load configuration values
        self.num_sample_rll = config['num_sample_rll']
        self.cluster_size = config['cluster_size']
        self.data_frac_psgd = config['data_frac_psgd']
        self.lr_coeff = config['lr_coeff']
        self.num_iter = config['num_iter']
        self.batch_size = config['batch_size']
        self.eid = experiment_id

    def forward(
            self,
            data_train: torch.Tensor,
            data_test: torch.Tensor,
            predictors: List[Any]
    ) -> List[List[torch.Tensor]]:
        """
        Call Robust List Learner to generate a list of sparse classifiers and input them to the Conditional Learner.

        Parameters:
        data_train:     Training data for both Robust List Learning and Conditional Learning.
        data_test:      Testing data to estimate the error measures of the final classifier-selector pair.
                        Disjoint from data_train.
        """

        # Perform conditional learning
        print(" ".join([self.header, "starting conditional classification for homogeneous halfspaces ..."]))

        cond_classifiers = []
        dim_sample = data_train.size(1) - 1
        dataset = TransformedDataset(data=data_train)
        sample_size_psgd = int(len(dataset) * self.data_frac_psgd)
        init_weight = torch.randn(
            dim_sample, 
            device=self.device
        )
        init_weight = init_weight / torch.norm(init_weight, p=2)
        # print(f"{self.header} learning seletors ...")
        for pred in tqdm(predictors, desc=f"{self.header} learning selectors"):
        # for pred in self.predictors:
            dataset.set_predictor(predictor=pred)
            dataset_train, dataset_val = random_split(
                dataset,
                [sample_size_psgd, len(dataset) - sample_size_psgd]
            )
            
            psgd_optim = SelectorPerceptron(
                prev_header=self.header + ">",
                dim_sample=dim_sample,
                cluster_id=0,
                cluster_size=1,
                num_iter=self.num_iter,
                lr_beta=self.lr_coeff * torch.sqrt(torch.tensor(1 / (self.num_iter * dim_sample))),
                batch_size=self.batch_size,
                device=self.device
            )

            cond_classifiers.append(
                ConditionalLinearModel(
                    seletor_weights=psgd_optim(
                        dataset_train=dataset_train,
                        dataset_val=dataset_val,
                        init_weight=init_weight
                    ),
                    predictor=pred
                )
            )

        errors, error_wo, coverages = torch.zeros(len(cond_classifiers)), torch.zeros(len(cond_classifiers)), torch.zeros(len(cond_classifiers))
        
        for i, cc in enumerate(cond_classifiers):
        # Estimate error measures with selectors
            error_wo[i] = cc.predictor.error_rate(
                X=data_test[:, 1:],
                y=data_test[:, 0]
            )
            tmp = cc.conditional_error_rate(
                X=data_test[:, 1:],
                y=data_test[:, 0]
            )
            errors[i] = tmp

            coverages[i] = cc.selector.prediction_rate(X=data_test[:, 1:])

        res = (error_wo[0],  errors[0], coverages[0])
        
        # Print the results in a table format
        table = [
            ["Classifier Type", "Train Size", "Test Size", "Sample Dim", "SGD Data Size", "PSGD Iter", "Batch Size", "LR Coeff", "Est ER", "Coverage"],
            ["Classic SVM", data_train.size(0), data_test.shape[0], data_test.shape[1] - 1, sample_size_psgd, self.num_iter, self.batch_size, self.lr_coeff, error_wo[0], 1],
            ["Cond SVM", data_train.size(0), data_test.shape[0], data_test.shape[1] - 1, sample_size_psgd, self.num_iter, self.batch_size, self.lr_coeff, errors[0], coverages[0]]
        ]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

        return res