import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from typing import List
from ..utils.simple_models import LinearModel, ConditionalLinearModel
from tqdm import tqdm

class RobustListLearner(nn.Module):
    """
    Robust List Classification for Sparse Classes
    """
    def __init__(
            self, 
            prev_header: str,
            sparsity: int, 
            margin: float,
            cluster_size: int,
            device: torch.device
    ):
        """
        sparsity (int):           The degree for each combination.
        margin (int):             The margin to use for the robust list learning.
        """
        super(RobustListLearner, self).__init__()
        self.header = " ".join([prev_header, "robust list learner", "-"])
        self.sparsity = sparsity
        self.margin = margin
        self.cluster_size = cluster_size
        self.device = device

    def forward(
            self, 
            dataset: DataLoader
    ) -> List[ConditionalLinearModel]:
        """
        Perform robust list learning as specified by the algorithm in Appendix A.
        
        Parameters:
        dataset (DataLoader):      The input dataset. 
                                           The first column is the label, which takes values in {0, 1}.
        
        Returns:
        sparse_weight_list (list[ConditionalLinearModel]): The list of weights for each combination.
                                           The weight_list is represented as a sparse tensor.
                                           The order of the weight vectors in the list is the same as the following two loops:
                                           for features in feature_combinations:
                                               for samples in sample_combinations:
                                                   ...
        """
        

        # Extract features and labels
        # Assume the first column is the label column
        labels, features = next(iter(dataset))
        # Map the labels from {0, 1} to {-1, +1}
        labels = 2 * labels - 1
        
        print(f"{self.header} selecting sub-matrices from data of all combinations ...")
        labeled_features = labels.unsqueeze(1) * features

        sample_size, self.sample_dim = labeled_features.shape[0], labeled_features.shape[1]
        # handle exception that sparsity level exceeds the number of features
        if self.sample_dim < self.sparsity:
            self.sparsity = self.sample_dim
        
        # Generate row indices of all possible combinations of samples
        # print(f"{self.header} computing combinations of sample indices ...")
        sample_indices = torch.combinations(
            torch.arange(sample_size), 
            self.sparsity
        ).to(self.device)   # [sample_size choose sparsity, sparsity]
        
        self.num_sample_combinations = sample_indices.shape[0]

        # Generate column indices of all possible combinations of features
        # print(f"{self.header} computing combinations of feature indices ...")
        feature_indices = torch.combinations(
            torch.arange(self.sample_dim), 
            self.sparsity
        ).to(self.device)   # [sample_dim choose sparsity, sparsity]
        
        self.num_feature_combinations = feature_indices.shape[0]

        # Select the labels for the generated sample combinations
        # print(f"{self.header} selecting labels for different sample combinations ...")
        label_combinations = torch.index_select(
            labels,
            0,
            sample_indices.flatten()
        ).reshape(-1, self.sparsity)    # [sample_size choose sparsity, sparsity]

        # Select the rows for the generated sample combinations while remaining flattened
        # print(f"{self.header} selecting labeled features for different sample combinations ...")
        labeled_feature_combinations = torch.index_select(
            labeled_features,
            0,
            sample_indices.flatten()
        )   # [sample_size choose sparsity * sparsity, sample_dim]

        # Select the columns for the generated feature combinations from the flattened labeled_feature_combinations
        # print(f"{self.header} selecting and reshaping previously resulting labeled features for different feature combinations ...")
        labeled_feature_combinations = torch.t(
            torch.index_select(
            labeled_feature_combinations,
            1,
            feature_indices.flatten()
            )
        ).reshape(
            -1, 
            self.sparsity, 
            self.num_sample_combinations * self.sparsity
        )   # [sample_dim choose sparsity, sparsity, (sample_size choose sparsity) * sparsity]
        # print(f"{self.header} reshaping the resulting labeled features ...")
        labeled_feature_combinations = torch.transpose(
            labeled_feature_combinations, 
            1, 
            2
        ).reshape(
            -1, 
            self.sparsity, 
            self.sparsity
        ) # [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity, sparsity]

        # repeat label_combinations to match the shape of labeled_feature_combinations
        # print(f"{self.header} repeating label combinations to match the shape of labeled features ...")
        label_combinations = label_combinations.repeat(
            self.num_feature_combinations, 
            1
        )   # [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity]

        ### solve the linear system specified in Algorithm 4 in Appendix A ###
        # weight_list = torch.linalg.solve(
        #     labeled_feature_combinations, 
        #     label_combinations - self.margin
        # )
        # print(f"{self.header} solving the linear system for least square solution ...")
        # weight_list = torch.linalg.lstsq(
        #     labeled_feature_combinations,       # [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity, sparsity]
        #     label_combinations - self.margin,   # [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity]
        #     rcond=1e-4
        # ).solution    # [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity]
        print(f"{self.header} solving the linear system using pseudo-inverse...")
        weight_list = torch.matmul(
            torch.linalg.pinv(labeled_feature_combinations),    # [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity, sparsity]
            (label_combinations - self.margin).unsqueeze(-1)    # [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity, 1]
        ).squeeze() # [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity]

        # batch method
        return self.to_batched_sparse_tensor(
            weights=weight_list,
            feature_combinations=feature_indices
        )


    def to_batched_sparse_tensor(
            self,
            weights: torch.Tensor,
            feature_combinations: torch.Tensor
    ) -> List[ConditionalLinearModel]:

        col_indices = feature_combinations.repeat_interleave(
            self.num_sample_combinations,
            dim=0
        )   # [(sample_dim choose sparsity) * (sample_size choose sparsity), sparsity]

        # ensure cluster size not exceed the number of sparse classifiers
        row_indices = torch.arange(
            min(col_indices.size(0), self.cluster_size)
        ).repeat_interleave(self.sparsity).to(self.device)

        # add progress bar to sparse encoding process
        # progress_bar = tqdm(
        #     total=math.ceil(col_indices.shape[0] / self.cluster_size),
        #     desc=f"{self.header} encoding sparse classifiers",
        #     # leave=False
        # )
        list_of_sparse_tensors = []
        pos = 0
        while pos < col_indices.size(0) - self.cluster_size:
            list_of_sparse_tensors.append(
                self.to_sparse_tensor(
                    row_indices=row_indices,
                    col_indices=col_indices[pos : pos + self.cluster_size],
                    weight_slice=weights[pos : pos + self.cluster_size]
                )
            )
            pos += self.cluster_size
            # progress_bar.update(1)

        list_of_sparse_tensors.append(
            self.to_sparse_tensor(
                row_indices=row_indices[:self.sparsity * (col_indices.size(0) - pos)],
                col_indices=col_indices[pos:],
                weight_slice=weights[pos:]
            )
        )
        # progress_bar.update(1)
        # progress_bar.close()

        return list_of_sparse_tensors

    def to_sparse_tensor(
            self,
            row_indices: torch.Tensor,
            col_indices: torch.Tensor,
            weight_slice: torch.Tensor
    ) -> ConditionalLinearModel:

        indices = torch.stack(
            (
                row_indices,
                col_indices.flatten()
            )
        )
        size = torch.Size(
            [weight_slice.shape[0], self.sample_dim]
        )
        return ConditionalLinearModel(
            predictor=LinearModel(
                weights=torch.sparse_coo_tensor(
                    indices,
                    weight_slice.flatten(),
                    size
                )
            ),
            device=self.device
        )

    # verification function
    def forward_verifier(
            self, 
            dataset: DataLoader
    ) -> torch.Tensor:
        """
        Perform robust list learning of sparse linear classifiers for verification purpose.
        
        Parameters:
        dataset (torch.Tensor):       The input dataset.
                                    The first column is the label, which takes values in {0, 1}.
        
        Returns:
        weight_list (torch.Tensor): The list of weights for each combination.
        """

        # Extract features and labels
        # Assume the first column is the label column
        # and the rest are feature columns
        labels = dataset[:, 0] * 2 - 1
        labeled_features = labels.unsqueeze(1) * dataset[:, 1:]

        sample_size, self.sample_dim = labeled_features.shape[0], labeled_features.shape[1]
        
        # Generate row indices of all possible combinations of samples
        # dimension: [sample_size choose sparsity, sparsity]
        sample_indices = torch.combinations(torch.arange(sample_size), self.sparsity)

        # Generate column indices of all possible combinations of features
        # dimension: [sample_dim choose sparsity, sparsity]
        feature_indices = torch.combinations(torch.arange(self.sample_dim), self.sparsity)


        weight_list = torch.zeros(sample_indices.shape[0] * feature_indices.shape[0],  self.sample_dim)
        i = 0
        for fid in feature_indices:
            for sid in sample_indices:
                labeled_feature_combination = torch.index_select(
                    torch.index_select(
                        labeled_features,
                        1,
                        fid
                    ),
                    0, 
                    sid
                )
                labeled_feature_combination_inv = torch.linalg.inv(labeled_feature_combination)
                label_combination = torch.index_select(
                    labels,
                    0,
                    sid
                )
                weight_list[i, fid] = torch.mv(labeled_feature_combination_inv, label_combination - self.margin)
                i += 1

        return weight_list