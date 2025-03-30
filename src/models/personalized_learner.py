import torch
import torch.nn as nn
import itertools
import yaml
import math
from typing import Union, Tuple, Iterable
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import random_split
from ..utils.data import MultiLabeledDataset
from .selector_learner import SelectiveHalfspaceLearner, ReferenceClassLearner
from .predictor_learner import RobustSparseHalfspaceLearner
from ..utils.simple_models import LinearModel


class PersonalizedPredictorLeaner(nn.Module):
    """
    Personalized predictor.
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
        sample_complexity:       To speed up the computation, instead of iterating only one classifier at a time, 
                            we partition all the sparse classifiers into multiple clusters and run PSGD on a cluster in each iteration.
        data_frac_psgd:     Fraction of training data used for the updating stage of Projected SGD.
        lr_coeff:           A constant to scale the learning rate (beta) of PSGD.
        num_iter:           The number of iteration for PSGD to run.
        batch_size:         Number of example to estimate the expectation of projected gradient in each gradient step.
        """
        super(PersonalizedPredictorLeaner, self).__init__()
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
        self.sample_complexity = config['sample_complexity']
        self.train_subset_fracs = config['train_subset_fracs']
        self.lr = config['lr']
        self.num_iter = config['num_iter']
        self.eid = experiment_id

        print(" ".join([self.header, "initializing learners ..."]))

        self.predictor_learner = RobustSparseHalfspaceLearner(
            prev_header=self.header + ">",
            sparsity=self.sparsity, 
            margin=self.margin,
            device=self.device
        )

        self.selector_learner = SelectiveHalfspaceLearner(
            prev_header=self.header + ">",
            subset_fracs=self.train_subset_fracs, 
            num_iter=self.num_iter, 
            lr=self.lr, 
            device=self.device
        )

        self.restricted_selector_learner = ReferenceClassLearner(
            prev_header=self.header + ">",
            subset_fracs=self.train_subset_fracs, 
            num_iter=self.num_iter, 
            lr=self.lr, 
            device=self.device
        )
        
    def forward(
            self,
            data_train: torch.Tensor,
            data_test: torch.Tensor
    ) -> Tuple[torch.Tensor, LinearModel]:
        """
        Call Robust List Learner to generate a list of sparse classifiers and input them to the Conditional Learner.

        Parameters:
        data_train:     Training data for both Robust List Learning and Conditional Learning.
        data_test:      Testing data to estimate the error measures of the final classifier-selector pair.
                        Disjoint from data_train.
        """

        dataset_train = MultiLabeledDataset(data=data_train)
        # learn without observations
        sparse_lm, cond_ers, predictors, selectors = self.subroutine(
            dataset=dataset_train,
            predictor_learner=self.predictor_learner,
            selector_learner=self.selector_learner,
            observations=self.random_weight(
                length=dataset_train.num_features()
            ),
            desc=f"{self.header} (initial) learning predictor-selector pair"
        )
        predictor: LinearModel = predictors[torch.argmin(cond_ers)]
        selector: LinearModel = selectors[torch.argmin(cond_ers)]

        selected_ids = selector.predict(
            X=data_test[:, 1:]
        )
        
        selected_errors = predictor.errors(
            *MultiLabeledDataset(
                data=data_test[selected_ids]
            )[:]
        )

        dataset_test = MultiLabeledDataset(
            data=data_test[~selected_ids]
        )

        _, cond_ers, predictors, selectors = self.subroutine(
            dataset=dataset_train,
            predictor_learner=self.predictor_learner,
            selector_learner=self.restricted_selector_learner,
            observations=self.normalize(dataset_test.features()),
            desc=f"{self.header} (remaining) learning predictor-selector pair"
        )

        errors = torch.cat(
            [
                selected_errors,
                predictors.pointwise_errors(*dataset_test[:])
            ]
        )

        print(f"{self.header} errors shape {errors.size()}")

        error_rates = errors.sum()/errors.size(0)

        test_stats = self.oos_statistics(
            dataset=MultiLabeledDataset(data=data_test),
            sparse_predictor=sparse_lm,
            predictor=predictor,
            selector=selector
        )

        # Print the results in a table format
        table = [
            ["Classifier Type", "Train Size", "Test Size", "Sample Dim", "Sparsity", "PSGD Iter", "LR Coeff", "Est ER", "Coverage"],
            ["Classic Sparse", data_train.size(0), data_test.shape[0], data_test.shape[1] - 1, self.sparsity, self.num_iter, self.lr, test_stats[0], 1],
            ["Cond Sparse w/o Selector", data_train.size(0), data_test.shape[0], data_test.shape[1] - 1, self.sparsity, self.num_iter, self.lr, test_stats[1], 1],
            ["Cond Sparse", data_train.size(0), data_test.shape[0], data_test.shape[1] - 1, self.sparsity, self.num_iter, self.lr, error_rates, test_stats[3]]
        ]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

        return test_stats, sparse_lm
    
    def subroutine(
            self,
            dataset: MultiLabeledDataset,
            predictor_learner: RobustSparseHalfspaceLearner,
            selector_learner: Union[SelectiveHalfspaceLearner, ReferenceClassLearner],
            observations: torch.Tensor,
            desc: str = None
    ) -> Tuple[LinearModel, torch.Tensor, LinearModel, LinearModel]:
        
        # split dataset for training and model selections for standalone sparse predictors
        ds_train_pred, ds_sel_pred = random_split(
            dataset,
            [self.num_sample_rll, len(dataset) - self.num_sample_rll],
            generator=torch.Generator().manual_seed(42)
        )

        # initialize current best
        sparse_er, sparse_lm = 1, None
        min_cond_er = torch.ones(observations.size(0), device=self.device)
        min_predictors = LinearModel(weights=torch.randn_like(observations, device=self.device))
        min_selectors = LinearModel(weights=observations.clone())

        # we split the sparsity into two part, and iterate the combinations generated by the first part
        # while parallely compute the combinations generated by the other part
        for sample_indices, feature_indices in self.prefix_combination_generator(
            num_sample=self.num_sample_rll,
            num_feature=dataset.num_features(),
            num_observation=observations.size(0),
            desc=desc
        ):
            print(f"{self.header} selected sample indices {sample_indices}, selected feature indices {feature_indices}")
            # learning sparse predictors
            predictors: LinearModel = predictor_learner(
                ds_train_pred, 
                sample_indices, 
                feature_indices
            )

            # print(f"{self.header} sparse predictor size: {predictors.size()}")

            # model selection for sparse classifiers based on regular classification error
            sparse_er, sparse_lm = self.model_selection(
                error_rates=predictors.error_rate(*ds_sel_pred[:]),
                predictors=predictors,
                prev_predictor=sparse_lm,
                prev_er=sparse_er
            )

            # learn reference class with label mapping
            eval_cond_errors, eval_ids, selectors = selector_learner(
                dataset.label_with(predictors), 
                observations
            )   # Tuple[torch.Tensor, torch.Tensor, LinearModel]

            # print(f"{self.header} learned selectors (after selecting) size: {selectors.size()}")
            
            # selecting corresponding predictors
            predictors: LinearModel = predictors.reduce(eval_ids)

            # update current best
            ids = eval_cond_errors < min_cond_er
            min_cond_er[ids] = min_cond_er[ids]
            min_predictors.partial_update(
                ids=ids,
                model=predictors
            )
            min_selectors.partial_update(
                ids=ids,
                model=selectors
            )

        print(f"{self.header} result predictor size: {min_predictors.size()}, is sparse? {min_predictors.weights.is_sparse}")
        print(f"{self.header} reuslt selector size: {min_selectors.size()}, is sparse? {min_selectors.weights.is_sparse}")
        print(f"{self.header} sparse predictors size {min_predictors.size()}")

        return sparse_lm, min_cond_er, min_predictors, min_selectors

    def model_selection(
            self,
            error_rates: torch.Tensor,
            predictors: LinearModel,
            prev_predictor: LinearModel,
            prev_er: float
    ) -> Tuple[float, LinearModel]:
        min_er, min_id = torch.min(
            error_rates,
            dim=0
        )
        if min_er < prev_er:
            return min_er, predictors[min_id]
        return prev_er, prev_predictor
        
    def oos_statistics(
            self,
            dataset: MultiLabeledDataset,
            sparse_predictor: LinearModel,
            predictor: LinearModel,
            selector: LinearModel
    ) -> torch.Tensor:
        
        error = sparse_predictor.error_rate(
            *dataset[:]
        )
        # Estimate error measures with selectors
        error_wo = predictor.error_rate(
            *dataset[:]
        )

        # map the labels in test data according to the selected predictor
        cond_error = selector.conditional_one_rate(
            *dataset.label_with(predictor)[:]
        )     

        coverage = selector.prediction_rate(
            X=dataset.features()
        )

        return torch.tensor(
            [
                error,
                error_wo,
                cond_error, 
                coverage
            ]
        )

    def prefix_combination_generator(
            self,
            num_sample: int,
            num_feature: int,
            num_observation: int,
            desc: str = None
    ) -> Iterable:
        '''
        Search for the lengths of sample and feature indices prefix such that, when conditioning on these prefix 
        indices, the number of combinations of the remaining indices is the closest to sample complexity
        
        Returns:
        gen: Iterable           A range-like generator that yields a pair of sample and feature indices based on
                                the closest prefix lengths. If desc is given, then we return a tqdm wrapped iterable.
        '''
        
        closest_num_sample_prefix = 0
        closest_num_feature_prefix = 0
        closest_num = 0

        num_fp_max = min(self.sparsity, num_feature)
        if self.sparsity > num_feature // 2:
            num_fp_max = 1
        for num_fp in range(num_fp_max):
            for num_sp in range(min(self.sparsity, num_sample)):
                num_sample_comb = math.comb(
                    num_sample,
                    self.sparsity - num_sp
                )
                num_feature_comb = math.comb(
                    num_feature,
                    self.sparsity - num_fp
                )
                total = num_sample_comb * num_feature_comb * num_observation
                
                if closest_num < total * num_feature <= self.sample_complexity:
                    closest_num_sample_prefix = num_sp
                    closest_num_feature_prefix = num_fp
                    closest_num = total * num_feature

        gen = self.two_level_combination_generator(
            num_sample=num_sample,
            deg_sample=closest_num_sample_prefix,
            num_feature=num_feature,
            deg_feature=closest_num_feature_prefix
        )
        if desc is not None:
            num_sp_comb = math.comb(num_sample, closest_num_sample_prefix)
            num_fp_comb = math.comb(num_feature, closest_num_feature_prefix)
            return tqdm(
                iterable=gen,
                total=num_sp_comb * num_fp_comb,
                desc=desc
            )
        return gen
    
    def two_level_combination_generator(
            self,
            num_sample: int,
            deg_sample: int,
            num_feature: int,
            deg_feature: int
    ) -> Iterable:
        for sample_comb in itertools.combinations(
            torch.arange(num_sample).tolist(), 
            deg_sample
        ):
            for feature_comb in itertools.combinations(
                torch.arange(num_feature).tolist(),
                deg_feature
            ):
                yield (
                    torch.tensor(
                        sample_comb, 
                        device=self.device,
                        dtype=torch.int
                    ),
                    torch.tensor(
                        feature_comb, 
                        device=self.device,
                        dtype=torch.int
                    )
                )
        
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
        return X / torch.norm(X, p=2, dim=-1, keepdim=True)

    def random_weight(
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
