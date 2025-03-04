import torch
import torch.nn as nn
import yaml
from typing import Union, Tuple, List
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from ..utils.data import TransformedDataset
from ..models.personalized_learner import ConditionalLearnerForFiniteClass
from ..models.robust_list_learner import RobustListLearner
from ..models.baseline_learner import SVMLearner
from ..utils.simple_models import ConditionalLinearModel


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
        self.data_frac_psgd = config['data_frac_psgd']
        self.lr_coeff = config['lr_coeff']
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

        sparse_classifier_clusters: List[ConditionalLinearModel] = robust_list_learner(
            DataLoader(
                TransformedDataset(
                    data_train, 
                    # random_state=42
                ),
                batch_size=self.num_sample_rll
            )
        )   # List[ConditionalLinearModel]


        # Perform conditional learning
        print(" ".join([self.header, "initializing conditional classification learner for homogeneous halfspaces ..."]))
        conditional_learner = ConditionalLearnerForFiniteClass(
            prev_header=self.header + ">",
            dim_sample = data_train.shape[1] - 1,
            num_iter=self.num_iter, 
            lr_coeff=self.lr_coeff, 
            sample_size_psgd=int(data_train.shape[0] * self.data_frac_psgd), 
            batch_size=self.batch_size,
            device=self.device
        )

        conditional_classifier: ConditionalLinearModel = conditional_learner(
            dataset = TransformedDataset(data_train),
            classifier_clusters=sparse_classifier_clusters
        )   # Tuple[torch.Tensor, torch.Tensor]

        # fitting a SVM on the selector
        print(f"{self.header} fitting SVM on the best selector ...")
        svm_classifier = ConditionalLinearModel(
            seletor_weights=conditional_classifier.selector.weights.clone(),
            predictor=SVMLearner(device=self.device),
            device=self.device
        )
        try:
            svm_classifier.predictor.train(
                svm_classifier.select_data(
                    X=data_train[:, 1:],
                    y=data_train[:, 0]
                )
            )
            error_svm = svm_classifier.conditional_error_rate(
                X=data_test[:, 1:],
                y=data_test[:, 0]
            )
        except ValueError:
            error_svm = 0

        # model selection for sparse classifiers based on regular classification error
        print(" ".join([self.header, "finding empirical error minimizer from sparse perceptrons ..."]))
        eem_classifier, min_error = None, 1
        # for classifiers in tqdm(
        #     sparse_classifier_clusters, 
        #     total=len(sparse_classifier_clusters), 
        #     desc=f"{self.header} find EEM",
        #     leave=False
        # ):
        for classifiers in sparse_classifier_clusters:
            error_rate, index = torch.min(
                classifiers.predictor.error_rate(
                    X=data_test[:, 1:],
                    y=data_test[:, 0]
                ), 
                dim=0
            )
            if error_rate < min_error:
                min_error = error_rate
                eem_classifier = classifiers.predictor[index].to_dense()
        
        # Estimate error measures with selectors
        error_wo = conditional_classifier.predictor.error_rate(
            X=data_test[:, 1:],
            y=data_test[:, 0]
        )

        error = conditional_classifier.conditional_error_rate(
            X=data_test[:, 1:],
            y=data_test[:, 0]
        )     

        coverage = conditional_classifier.selector.prediction_rate(X=data_test[:, 1:])

        res = [
            (eem_classifier, float(min_error)),
            (conditional_classifier.predictor[...], float(error_wo)),
            (conditional_classifier, (float(error), float(coverage))),
            (svm_classifier, (float(error_svm), float(coverage)))
        ]
        
        # Print the results in a table format
        table = [
            ["Classifier Type", "Train Size", "Test Size", "Sample Dim", "Sparsity", "PSGD Iter", "Batch Size", "LR Coeff", "Est ER", "Coverage"],
            ["Classic Sparse", data_train.size(0), data_test.shape[0], data_test.shape[1] - 1, self.sparsity, self.num_iter, self.batch_size, self.lr_coeff, min_error, 1],
            ["Cond Sparse w/o Selector", data_train.size(0), data_test.shape[0], data_test.shape[1] - 1, self.sparsity, self.num_iter, self.batch_size, self.lr_coeff, error_wo, 1],
            ["Cond Sparse", data_train.size(0), data_test.shape[0], data_test.shape[1] - 1, self.sparsity, self.num_iter, self.batch_size, self.lr_coeff, error, coverage],
            ["Cond SVM", data_train.size(0), data_test.shape[0], data_test.shape[1] - 1, self.sparsity, self.num_iter, self.batch_size, self.lr_coeff, error_svm, coverage]
        ]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

        return res