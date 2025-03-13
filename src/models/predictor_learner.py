import torch
import torch.nn as nn
from torch.utils.data import Subset
import math
from typing import Union
from ..utils.simple_models import LinearModel
from ..utils.data import MultiLabeledDataset

class RobustSparseHalfspaceLearner(nn.Module):
    """
    Robust List Classification for Sparse Classes
    """
    def __init__(
            self, 
            prev_header: str,
            sparsity: int,
            margin: float,
            device: torch.device = torch.device('cpu')
    ):
        """
        Parameters:
        sparsity (int):           The degree for each combination.
        margin (int):             The margin to use for the robust list learning.
        """
        super(RobustSparseHalfspaceLearner, self).__init__()
        self.header = " ".join([prev_header, "robust list learner", "-"])
        self.sparsity = sparsity
        self.margin = margin
        self.device = device
    

    def forward(
            self, 
            dataset: Union[MultiLabeledDataset, Subset],
            prev_sample_indices: torch.Tensor,
            prev_feature_indices: torch.Tensor
    ) -> LinearModel:
        """
        Perform robust list learning as specified by the algorithm in Appendix A.
        
        Parameters:
        dataset (MultiLabeledDataset):      The input dataset. 
                                            The first column is the label, which takes values in {0, 1}.
        
        Returns:
        sparse_predictors (LinearModel):    The list of weights for each combination.
                                            The weight_list is represented as a sparse tensor.
                                            The order of the weight vectors in the list is the same as the following two loops:
                                            for features in feature_combinations:
                                                for samples in sample_combinations:
                                                    ...
        """
        
        # Extract features and labels
        # Assume the first column is the label column
        labels, features = dataset[:]

        # Map the labels from {0, 1} to {-1, +1}
        labels = 2 * labels - 1
        
        # generate combinations of row indices
        sample_indices_combinations = self.indices_combinations(
            degree=self.sparsity - prev_sample_indices.size(0),
            prev_indices=prev_sample_indices,
            num=features.size(0)
        ).flatten()                         # [num sample choose deg sample, sparsity] --> [(num sample choose deg sample) * sparsity]
        

        # Select the labels for the generated sample combinations
        label_combinations = torch.index_select(
            labels,
            0,
            sample_indices_combinations     # [(num sample choose deg sample) * sparsity]
        ).reshape(-1, self.sparsity)        # [num sample choose deg sample, sparsity]

        # Select the rows of labeled samples using the generated sample combinations while remaining flattened
        labeled_feature_combinations = torch.index_select(
            labels.unsqueeze(1) * features, # [num sample, num feature]
            0,
            sample_indices_combinations     # [(num sample choose deg sample) * sparsity]
        )                                   # [(num sample choose deg sample) * sparsity, num feature]

        # Generate combinations of column indices
        feature_indices_combinations = self.indices_combinations(
            degree=self.sparsity - prev_feature_indices.size(0),
            prev_indices=prev_feature_indices,
            num=features.size(1)
        )                                   # [num feature choose deg feature, sparsity]
        
        # Select the columns for the generated feature combinations from the flattened labeled_feature_combinations
        labeled_feature_combinations = torch.index_select(
            labeled_feature_combinations,
            1,
            feature_indices_combinations.flatten()  # [(num feature choose deg feature) * sparsity]
        ).t().reshape(                      # [(num feature choose deg feature) * sparsity, (num sample choose deg sample) * sparsity]
            -1, 
            self.sparsity, 
            sample_indices_combinations.size(0)
        )                                   # [num feature choose deg feature, sparsity, (num sample choose deg sample) * sparsity]

        labeled_feature_combinations = torch.transpose(
            labeled_feature_combinations,   # [num feature choose deg feature, sparsity, (num sample choose deg sample) * sparsity]
            1, 
            2
        ).reshape(                          # [num feature choose deg feature, (num sample choose deg sample) * sparsity, sparsity]
            -1, 
            self.sparsity, 
            self.sparsity
        )                                   # [(num feature choose deg feature) * (num sample choose deg sample), sparsity, sparsity]

        # repeat label_combinations to match the shape of labeled_feature_combinations
        label_combinations = label_combinations.repeat(
            feature_indices_combinations.size(0), 
            1
        )                                   # [(num feature choose deg feature) * (num sample choose deg sample), sparsity]

        ### solve the linear system specified in Algorithm 4 in Appendix A ###
        # weight_list = torch.linalg.solve(
        #     labeled_feature_combinations, 
        #     label_combinations - self.margin
        # )
        # print(f"{self.header} solving the linear system for least square solution ...")
        # weight_list = torch.linalg.lstsq(
        #     labeled_feature_combinations,                         # [(num feature choose deg feature) * (num sample choose deg sample), sparsity, sparsity]
        #     label_combinations - self.margin,                     # [(num feature choose deg feature) * (num sample choose deg sample), sparsity]
        #     rcond=1e-4
        # ).solution                                                # [(num feature choose deg feature) * (num sample choose deg sample), sparsity]
        weight_list = torch.matmul(
            torch.linalg.pinv(labeled_feature_combinations),        # [(num feature choose deg feature) * (num sample choose deg sample), sparsity, sparsity]
            (label_combinations - self.margin).unsqueeze(-1)        # [(num feature choose deg feature) * (num sample choose deg sample), sparsity, 1]
        ).squeeze()                                                 # [(num feature choose deg feature) * (num sample choose deg sample), sparsity]

        # return sparse representations
        return self.to_sparse_tensor(
            weights=weight_list,                                    # [(num feature choose deg feature) * (num sample choose deg sample), sparsity]
            col_indices_combinations=feature_indices_combinations,  # [num feature choose deg feature, sparsity]
            num_features=features.size(1)
        )

    def indices_combinations(
            self,
            degree: int,
            prev_indices: torch.Tensor,
            num: int
    ) -> torch.Tensor:
        # generate a valid range of numbers that excluding those been previously fixed
        valid_indices = torch.arange(num).to(self.device)
        valid_indices = valid_indices[~torch.isin(valid_indices, prev_indices)]

        # check for speical case
        if valid_indices.size(0) <= degree:
            return torch.cat((prev_indices, valid_indices))
        
        # Generate combinations of samples
        partial_combinations = torch.combinations(
            valid_indices, 
            degree
        )   # [num choose degree, degree]

        # if no previous indices, return current combinations
        if prev_indices.size(0) <= 0:
            return partial_combinations

        return torch.cat(
            (
                prev_indices.repeat(partial_combinations.size(0), 1),
                partial_combinations
            ),
            dim=1
        )

    def to_sparse_tensor(
            self,
            weights: torch.Tensor,
            col_indices_combinations: torch.Tensor,
            num_features: int
    ) -> LinearModel:
        # generate col indices for each feature combinations
        col_indices = col_indices_combinations.repeat_interleave(   # [num feature choose sparsity, sparsity]
            weights.size(0) // col_indices_combinations.size(0),    # num sample choose deg sample
            dim=0
        )                                                           # [(num feature choose sparsity) * (num sample choose deg sample), sparsity]

        # generate row indices for each feature combinations
        row_indices = torch.arange(
            col_indices.size(0)
        ).repeat_interleave(self.sparsity).to(self.device)      # [(num feature choose sparsity) * (num sample choose sparsity), sparsity]

        indices = torch.stack(
            (
                row_indices,
                col_indices.flatten()
            )
        )
        size = torch.Size(
            [weights.shape[0], num_features]
        )
        return LinearModel(
            weights=torch.sparse_coo_tensor(
                indices,
                weights.flatten(),
                size
            )
        )

    # verification function
    def forward_verifier(
            self, 
            dataset: MultiLabeledDataset,
            prev_sample_indices: torch.Tensor,
            prev_feature_indices: torch.Tensor,
            predictors: LinearModel
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
        labels, features = dataset[:]
        # Assume the first column is the label column
        # and the rest are feature columns
        labels = labels * 2 - 1

        labeled_features = features * labels.unsqueeze(1)
        
        # Generate row indices of all possible combinations of samples
        # dimension: [num_feature choose sparsity, sparsity]
        sample_indices = self.indices_combinations(
            degree=self.sparsity - prev_sample_indices.size(0),
            prev_indices=prev_sample_indices,
            num=features.size(0)
        )
        # print(f"verification sample indices {sample_indices}")

        # Generate column indices of all possible combinations of features
        # dimension: [sample_dim choose sparsity, sparsity]
        feature_indices = self.indices_combinations(
            degree=self.sparsity - prev_feature_indices.size(0),
            prev_indices=prev_feature_indices,
            num=features.size(1)
        )

        # print(f"verification feature indices {feature_indices}")


        weight_list = torch.zeros(sample_indices.shape[0] * feature_indices.shape[0],  features.size(1)).to(self.device)
        i = 0
        for fid in feature_indices:
            for sid in sample_indices:
                labeled_feature_combinations = torch.index_select(
                    labeled_features[sid],
                    1,
                    fid
                )
                # label_combinations = torch.index_select(
                #     labels,
                #     0,
                #     sid
                # )
                weight_list[i, fid] = torch.matmul(
                    torch.linalg.pinv(labeled_feature_combinations),        # [sparsity, sparsity]
                    (labels[sid] - self.margin).unsqueeze(-1)               # [sparsity, 1]
                ).squeeze()      

                i += 1

        print(f"{self.header} predictor norms are {torch.norm(predictors.weights)} and {torch.norm(weight_list)}, verification error {torch.norm(predictors.to_dense().weights - weight_list)}")