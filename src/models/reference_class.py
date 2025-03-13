import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, Subset
from typing import List, Tuple, Union, Any
from ..utils.data import MultiLabeledDataset
from ..utils.simple_models import LinearModel
from .conditional_predictor import ConditionalPredictor
from tqdm import tqdm

class ReferenceClass(ConditionalPredictor):
    """
    Conditional Classification for Any Finite Classes
    """
    def __init__(
            self,
            prev_header: str,
            subset_fracs: List[float],
            num_iter: int, 
            lr: float,
            device: torch.device = torch.device('cpu')
    ):
        """
        Initialize the conditional learner for finite class classification.
        Compute the learning rate of PSGD for the given lr coefficient using
        the formula:
            beta = O(sqrt(1/num_iter * dim_sample)).

        Parameters:
        prev_header (str):              The header of the previous module.
        num_iter (int):                 The number of iterations for optimizer.
        lr (float):                     The learning rate.
        subset_fracs (List[float]):     The ratio between training data size and validation data size.
        device (torch.device):          The device to be used.
        """
        super().__init__(
            prev_header=prev_header,
            subset_fracs=subset_fracs,
            num_iter=num_iter,
            lr=lr,
            device=device
        )
        self.header = " ".join([prev_header, "learning reference class", "-"])

    def grad_update(
            self,
            lin_model: LinearModel,
            labels: torch.Tensor,       # labels:   [num predictors, num train sample]
            features: torch.Tensor      # features: [num train sample, num features]
    ) -> None:
        """
        Perform the gradient step for weights.
        
        Parameters:
        lin_model (LinearModel):         The linear model to be updated.
        labels (torch.Tensor):           The labels to be used.
        features (torch.Tensor):         The features to be used.
        """
        # compute projected gradients
        proj_grads = lin_model.projected_gradient(
            y=labels,
            X=features
        )

        # gradient step
        lin_model.update(
            weights= - self.lr * proj_grads
        )

        # contractive projection
        lin_model.project_onto(X=self.observations)
        
        # update convergence progress
        self.converged_bar.n = int((torch.norm(proj_grads, p=2, dim=-1) < 0.015).sum())
        self.converged_bar.refresh()