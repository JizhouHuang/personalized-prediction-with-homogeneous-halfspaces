import torch
import torch.nn as nn
from torch.utils.data import random_split
from tqdm import tqdm
from typing import List, Tuple
from ..utils.data import TransformedDataset
from ..utils.simple_models import LinearModel, ConditionalLinearModel
from .cpgd import SelectorPerceptron

class ConditionalLearnerForFiniteClass(nn.Module):
    """
    Conditional Classification for Any Finite Classes
    """
    def __init__(
            self, 
            prev_header: str,
            dim_sample: int,
            num_iter: int, 
            sample_size_psgd: int,
            lr_coeff: float = 0.5,
            batch_size: int = 32,
            device: torch.device = torch.device('cpu')
    ):
        """
        Initialize the conditional learner for finite class classification.
        Compute the learning rate of PSGD for the given lr coefficient using
        the formula:
            beta = O(sqrt(1/num_iter * dim_sample)).

        Parameters:
        dim_sample (int):             The dimension of the sample features.
        num_iter (int):               The number of iterations for SGD.
        lr_coeff (float):             The learning rate coefficient.
        sample_size_psgd (float):          The ratio of training samples.
        batch_size (int):             The batch size for SGD.
        """
        super(ConditionalLearnerForFiniteClass, self).__init__()
        self.header = " ".join([prev_header, "conditional learner", "-"])
        self.dim_sample = dim_sample
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.sample_size_psgd = sample_size_psgd
        self.device = device

        self.lr_beta = lr_coeff * torch.sqrt(
            torch.tensor(
                1 / (num_iter * dim_sample)
            )
        )
        self.init_weight = torch.randn(
            self.dim_sample,
            generator=torch.Generator().manual_seed(42)
        ).to(device)
        self.init_weight = self.init_weight / torch.norm(self.init_weight, p=2)

    def forward(
            self, 
            dataset: TransformedDataset,
            classifier_clusters: List[ConditionalLinearModel]
    ) -> ConditionalLinearModel:
        """
        Call PSGD optimizer for each cluster of sparse classifiers using all the data given.
        
        Note that the PSGD optimizer runs in parallel for all the sparse classifiers in a 
        cluster. PSGD optimizer will return one selector for each sparse classifier of each 
        cluster.

        For each cluster, we evaluate the best classifier-selector pair using all the data given
        due to insufficient data size.

        At last, we use the same data set to find the best classifier-selector pair across cluster.

        Parameters:
        classifier_clusters (List[ConditionalLinearModel]): The list of sparse classifiers.

        Returns:
        selector_list (torch.Tensor): The list of weights for each classifier.
                                      The weight_list is represented as a sparse tensor.
                                      The order of the weight vectors in the list is the same as the following two loops:
                                      for features in feature_combinations:
                                          for samples in sample_combinations:
                                              ...
        """        
        
        candidate_selectors = torch.zeros(
            [len(classifier_clusters), self.dim_sample]
        ).to(self.device)
        candidate_classifiers = torch.zeros(
            [len(classifier_clusters), self.dim_sample]
        ).to(self.device)

        # initialize evaluation dataset for conditional learner
        labels_eval, features_eval = dataset[:]

        # initialze train dataset for PSGD
        dataset_train, dataset_val = random_split(
            dataset, 
            [self.sample_size_psgd, len(dataset) - self.sample_size_psgd],
            # generator=torch.Generator().manual_seed(42)
        )

        for i, classifiers in enumerate(
            tqdm(
                classifier_clusters, 
                desc=f"{self.header} learning selectors",
                # leave=False
            )
        ):
        # for i, classifiers in enumerate(classifier_clusters):
            dataset.set_predictor(
                predictor=classifiers.predictor
            )

            selector_learner = SelectorPerceptron(
                prev_header=self.header + ">",
                dim_sample=self.dim_sample,
                cluster_id = i + 1,
                cluster_size=classifiers.predictor.size(0),
                num_iter=self.num_iter,
                lr_beta=self.lr_beta,
                batch_size=self.batch_size,
                device=self.device
            )

            selectors = selector_learner(
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                init_weight=self.init_weight.to(self.device)
            )  # [cluster size, dim sample]

            classifiers.set_selector(weights=selectors)
            _, min_ids = torch.min(
                classifiers.conditional_error_rate(
                    X=features_eval,
                    y=labels_eval
                ),
                dim=0
            )
            candidate_classifiers[i, ...], candidate_selectors[i, ...] = classifiers.predictor[min_ids].to_dense(), selectors[min_ids]
            classifiers.set_selector(weights=None)

        classifiers = ConditionalLinearModel(
            seletor_weights=candidate_selectors,
            predictor=LinearModel(weights=candidate_classifiers),
            device=self.device
        )
        _, min_ids = torch.min(
            classifiers.conditional_error_rate(
                X=features_eval,
                y=labels_eval
            ),
            dim=0
        )
        return ConditionalLinearModel(
            seletor_weights=candidate_selectors[min_ids],
            predictor=LinearModel(weights=candidate_classifiers[min_ids]),
            device=self.device
        )