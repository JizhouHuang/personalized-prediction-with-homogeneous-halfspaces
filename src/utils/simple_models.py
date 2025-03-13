from typing import Union, List, Tuple, Any, Self
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from sklearn.metrics import accuracy_score

MIN_BOOL_ERROR = 0.1

class LinearModel(nn.Module):
    def __init__(
            self,
            weights: Union[torch.Tensor, torch.sparse.FloatTensor]  # Float [N1, N2, ..., N(k - 1), d]
    ):
        super(LinearModel, self).__init__()
        
        self.weights = weights.clone()
        self.device: torch.device = weights.device

    def forward(
            self,
            X: torch.Tensor     # Float     [m, d]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1), m]
        if self.weights.is_sparse:
            return torch.sparse.mm(self.weights, X.t())
        else:
            return torch.matmul(self.weights, X.t())
        
    def pointwise_forward(
            self,
            X: torch.Tensor     # Float     [Ni, ..., N(k - 1), d]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1)]
        return (self.weights * X).sum(-1)
    
    def size(
            self,
            dim: int = -1
    ) -> torch.Tensor:
        if dim >= 0:
            return self.weights.size(dim)
        else:
            return self.weights.size()
        
    def to_dense(
            self,
    ) -> Self:
        if self.weights.is_sparse:
            return LinearModel(weights=self.weights.to_dense())
        return self
    
    def reduce(
            self,
            ids: torch.Tensor,  # [N]
            dim: int = 0
    ) -> Self:
        if self.weights.is_sparse:
            weights = self.weights.to_dense()
        else:
            weights = self.weights.clone()
        if dim == 1:
            weights = weights[
                torch.arange(weights.size(0)),
                ids
            ]
        elif dim == 0:
            weights = weights[ids]
        else:
            print(f"Reducing LinearModel dimensions can only be applied on the first two dimensions, but the input dimension is: {dim}")

        return LinearModel(weights=weights)
    
    def __getitem__(
            self,
            idx: int
    ) -> Self:
        if self.weights.is_sparse:
            return LinearModel(weights=self.weights.to_dense()[idx])
        else:
            return LinearModel(weights=self.weights[idx])
    
    def predict(
            self,
            X: torch.Tensor     # Float     [m, d]
    ) -> torch.Tensor:          # Boolean   [N1, N2, ..., N(k - 1), m]
        return self.forward(X=X) >= 0
    
    def pointwise_predict(
            self,
            X: torch.Tensor     # Float     [Ni, ..., N(k - 1), d]
    ) -> torch.Tensor:          # Boolean   [N1, N2, ..., N(k - 1)]
        return self.pointwise_forward(X=X) >= 0
    
    def prediction_rate(
            self,
            X: torch.Tensor     # Float     [m, d]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1)]
        return self.predict(X=X).sum(dim=-1) / X.size(0)
    
    def agreements(
            self,
            y: torch.Tensor,                    # Boolean   [Ni, ..., N(k - 1), m]
            X: torch.Tensor,                    # Float     [m, d]
    ) -> torch.Tensor:                          # Boolean   [N1, ..., Ni, ..., N(k - 1), m]
        return self.predict(X=X) == y.bool()    # [N1, ..., Ni, ..., N(k - 1), m], [Ni, ..., N(k - 1), m]
    
    def accuracy(
            self,
            y: torch.Tensor,    # Boolean   [Ni, ..., N(k - 1), m]
            X: torch.Tensor,    # Float     [m, d]
    ) -> torch.Tensor:          # Boolean   [N1, ..., Ni, ..., N(k - 1)]
        return self.agreements(
            y=y,
            X=X
        ).sum(dim=-1) / y.size(-1)
    
    def errors(
            self,
            y: torch.Tensor,                    # Boolean   [Ni, ..., N(k - 1), m]
            X: torch.Tensor,                    # Float     [m, d]
    ) -> torch.Tensor:                          # Boolean   [N1, ..., Ni, ..., N(k - 1), m]
        return self.predict(X=X) != y.bool()    # [N1, ..., Ni, ..., N(k - 1), m], [Ni, ..., N(k - 1), m]
    
    def pointwise_errors(
            self,
            y: torch.Tensor,                    # Boolean   [Ni, ..., N(k - 1)]
            X: torch.Tensor,                    # Float     [Ni, ..., N(k - 1), d]
    ) -> torch.Tensor:                          # Boolean   [N1, ..., Ni, ..., N(k - 1)]
        return self.pointwise_predict(X=X) != y.bool()
    
    def error_rate(
            self,
            y: torch.Tensor,    # Boolean   [Ni, ..., N(k - 1), m]
            X: torch.Tensor,    # Float     [m, d]
    ) -> torch.Tensor:          # Float     [N1, ..., Ni, ..., N(k - 1)]
        return self.errors(
            y=y,
            X=X
        ).sum(dim=-1) / y.size(-1)
    
    def update(
            self,
            weights: torch.Tensor       # Float [N1, N2, ..., N(k - 1), d]
    ) -> None:
        self.weights += weights         # [N1, N2, ..., N(k - 1), d]
        self.weights /= torch.norm(
            self.weights, 
            p=2,
            dim=-1
        ).unsqueeze(-1)                 # [N1, N2, ..., N(k - 1), 1]

    def project_onto(
            self,
            X: torch.Tensor     # Float     [N(k - 1), d]
    ) -> None:
        # <w, x> in batch
        proj_coeffs = self.pointwise_forward(X=X)    # [N1, N2, ..., N(k - 1)]

        # indices of weights that is out of the designated halfspace
        ids = proj_coeffs < 0

        # update: w = w - <w, x>x in batch
        if ids.any():
            self.weights[ids] = self.weights[ids] - proj_coeffs[ids].unsqueeze(-1) * X[ids]

    def projection_of(
            self,
            X: torch.Tensor     # Float     [m, d]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1), m, d]
        # X - <X, w>w^T
        return X - torch.matmul(
            self.forward(X).unsqueeze(-1),  # [N1, N2, ..., N(k - 1), m, 1]
            self.weights.unsqueeze(-2)      # [N1, N2, ..., N(k - 1), 1, d]
        )      # [N1, N2, ..., N(k - 1), m, d]
    
    def projected_gradient(
            self,
            y: torch.Tensor,    # Boolean   [Ni, ..., N(k - 1), m]
            X: torch.Tensor,    # Float     [m, d]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1), d]                                                                                   
        return torch.mean(
            self.agreements(
                y=y,
                X=X
            ).unsqueeze(-1) * self.projection_of(X=X),       # [N1, N2, ..., N(k - 1), m, d]
            dim=-2
        )
    
    def conditional_one_rate(
            self,
            y: torch.Tensor,    # Boolean   [Ni, ..., N(k - 1), m]
            X: torch.Tensor,    # Float     [m, d]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1)]
        y_sel = y * self.predict(X=X)     # [N1, N2, ..., N(k - 1), m]
        return y_sel.sum(dim=-1) / self.predict(X=X).sum(dim=-1)
    
    def conditional_zero_rate(
            self,
            y: torch.Tensor,    # Boolean   [Ni, ..., N(k - 1), m]
            X: torch.Tensor,    # Float     [m, d]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1)]
        return 1 - self.conditional_accuracy(
            y=y,
            X=X
        )
    
    def model_selection_by_one(
            self,
            dim: int,
            dataset: Union[Dataset, Subset]
    ) -> Tuple[torch.Tensor, torch.Tensor, Self]:                  
        min_vals, min_ids = torch.min(
            self.conditional_one_rate(
                *dataset[:]
            ),              # [N1, ..., N(k - 1)]
            dim=dim
        )                   # [N1, ..., N(k - 2)], [N1, ..., N(k - 2)]

        return min_vals, min_ids, self.reduce(
            ids=min_ids,
            dim=dim
        )
    
    def partial_update(
            self,
            ids: torch.Tensor,  # Boolean
            model: Self
    ) -> None:
        self.weights[ids] = model.weights[ids]
    
    
class PredictiveModel(nn.Module):
    def __init__(
            self,
            model: Any,     # any sklearn model
            max_data_train: int,
            device: torch.device
    ):
        super(PredictiveModel, self).__init__()
        self.model: Any = model
        self.max_data_train: int = max_data_train
        self.device: torch.device = device
    
    def train(
            self,
            data: Tuple[torch.Tensor]
    ) -> None:
        y, X = data
        if hasattr(self.model, "fit"):
            cutoff = min(y.size(0), self.max_data_train)
            self.model.fit(
                X[:cutoff].cpu().numpy(), 
                y[:cutoff].cpu().numpy()
            )

    def predict(
            self,
            X: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(self.model, "predict"):
            return torch.from_numpy(
                self.model.predict(X.cpu().numpy())
            ).to(self.device).bool()
        else:
            print(f"Predictor is empty!")
            return torch.zeros(X.size(0))
    
    def errors(
            self,
            X: torch.Tensor,
            y: torch.Tensor
    ) -> torch.Tensor:
        return self.predict(X=X) != y.bool()
    
    def error_rate(
            self,
            X: torch.Tensor,
            y: torch.Tensor
    ) -> torch.Tensor:
        return self.errors(X=X, y=y).sum() / X.size(0)
