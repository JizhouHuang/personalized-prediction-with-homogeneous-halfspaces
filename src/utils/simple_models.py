from typing import Union, List, Tuple, Any, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

MIN_BOOL_ERROR = 0.1

class LinearModel(nn.Module):
    def __init__(
            self,
            weights: Union[torch.Tensor, torch.sparse.FloatTensor]  # Float [N1, N2, ..., N(k - 1), d]
    ):
        super(LinearModel, self).__init__()
        self.weights: torch.Tensor = weights
        self.device: torch.device = weights.device

    def forward(
            self,
            X: torch.Tensor     # Float     [m, d]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1), m]
        if self.weights.is_sparse:
            return torch.sparse.mm(self.weights, X.t())
        else:
            return torch.matmul(self.weights, X.t())
    
    def size(
            self,
            dim: int = 0
    ) -> torch.Tensor:
        return self.weights.size(dim)
    
    def __getitem__(
            self,
            idx: int
    ):
        return self.weights[idx]
    
    def predict(
            self,
            X: torch.Tensor     # Float     [m, d]
    ) -> torch.Tensor:          # Boolean   [N1, N2, ..., N(k - 1), m]
        return self.forward(X=X) >= 0
    
    def prediction_rate(
            self,
            X: torch.Tensor     # Float     [m, d]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1)]
        return self.predict(X=X).sum(dim=-1) / X.size(0)
    
    def agreements(
            self,
            X: torch.Tensor,                    # Float     [m, d]
            y: torch.Tensor                     # Boolean   [N1, N2, ..., N(k - 1), m]
    ) -> torch.Tensor:                          # Boolean   [N1, N2, ..., N(k - 1), m]
        return self.predict(X=X) != y.bool()    # Boolean   [N1, N2, ..., N(k - 1), m]
    
    def accuracy(
            self,
            X: torch.Tensor,    # Float     [m, d]
            y: torch.Tensor     # Boolean   [N1, N2, ..., N(k - 1), m]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1)]
        return self.agreements(
            X=X,
            y=y
        ).sum(dim=-1) / X.size(0)
    
    def errors(
            self,
            X: torch.Tensor,                    # Float     [m, d]
            y: torch.Tensor                     # Boolean   [N1, N2, ..., N(k - 1), m]
    ) -> torch.Tensor:                          # Boolean   [N1, N2, ..., N(k - 1), m]
        return self.predict(X=X) != y.bool()    # Boolean   [N1, N2, ..., N(k - 1), m]
    
    def error_rate(
            self,
            X: torch.Tensor,    # Float     [m, d]
            y: torch.Tensor     # Boolean   [N1, N2, ..., N(k - 1), m]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1)]
        return self.errors(
            X=X,
            y=y
        ).sum(dim=-1) / X.size(0)
    
    def update(
            self,
            weights: torch.Tensor       # Float [N1, N2, ..., N(k - 1), d]
    ) -> None:
        self.weights += weights         # [N1, N2, ..., N(k - 1), d]
        self.weights /= torch.norm(
            self.weights, 
            p=2,
            dim=-1
        ).unsqueeze(-1)
    
    def proj_grad(
            self,
            X: torch.Tensor,    # Float     [m, d]
            y: torch.Tensor     # Boolean   [N1, N2, ..., N(k - 1), m]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1), ]
        orthogonal_projections = X - torch.matmul(self.forward(X).unsqueeze(-1), self.weights.unsqueeze(-2))      # [N1, N2, ..., N(k - 1), m, d]
                                                                            # X - <X, w>w^T
        return torch.mean(
            self.agreements(
                X=X,
                y=y
            ).unsqueeze(-1) * orthogonal_projections,       # [N1, N2, ..., N(k - 1), m, d]
            dim=-2
        )
    
class ConditionalLinearModel(nn.Module):
    
    def __init__(
            self,
            seletor_weights: torch.Tensor = None,
            predictor: Any = None,
            device: torch.device = torch.device('cpu')
    ):
        super(ConditionalLinearModel, self).__init__()

        self.selector: LinearModel = None
        self.predictor: Any = predictor
        self.device = device

        self.set_selector(weights=seletor_weights)
        self.set_predictor(predictor=predictor)

    def set_selector(
            self,
            weights: torch.Tensor       # [N1, N2, ..., N(k - 1), d]
    ) -> None:
        if weights is not None and isinstance(weights, torch.Tensor):
            self.selector = LinearModel(weights=weights)

    def set_predictor(
            self,
            predictor: Any
    ) -> None:
        if predictor and hasattr(predictor, "errors"):
            self.predictor = predictor

    def select_data(
            self,
            X: torch.Tensor,
            y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        selections = self.selector.predict(X=X)
        return y[selections], X[selections]

    def conditional_error_rate(
            self,
            X: torch.Tensor,                                # [m, d]
            y: torch.Tensor                                 # [N1, N2, ..., N(k - 1), m]
    ) -> torch.Tensor:                                      # [N1, N2, ..., N(k - 1)]
        selections = X
        if self.selector:
            selections = self.selector.predict(X=X)         # [N1, N2, ..., N(k - 1), m]
        errors = y                                          # [N1, N2, ..., N(k - 1), m]
        if self.predictor:
            errors = self.predictor.errors(X=X, y=y)        # [N1, N2, ..., N(k - 1), m]
        sel_errors = selections * errors                    # [N1, N2, ..., N(k - 1), m]
        cond_err_rate = sel_errors.sum(dim=-1) / selections.sum(dim=-1)
        cond_err_rate[torch.isnan(cond_err_rate)] = 1
        return cond_err_rate
    
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
