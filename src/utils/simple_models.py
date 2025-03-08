from typing import Union, List, Tuple, Any, Self
import torch
import torch.nn as nn
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
            dim: int = -1
    ) -> torch.Tensor:
        if dim >= 0:
            return self.weights.size(dim)
        return self.weights.size()
    
    def reduce(
            self,
            ids: torch.Tensor,  # [N]
            dim: int = 0
    ) -> Self:
        if isinstance(self.weights, torch.sparse.Tensor):
            self.weights = self.weights.to_dense()
        if dim == 1:
            self.weights = self.weights[
                torch.arange(self.weights.size(0)),
                ids
            ]
        elif dim == 0:
            self.weights = self.weights[ids]
        else:
            print(f"Reducing LinearModel dimensions can only be applied on the first two dimensions, but the input dimension is: {dim}")

        return self
    
    def __getitem__(
            self,
            idx: int
    ) -> Self:
        return LinearModel(weights=self.weights[idx])
    
    def predict(
            self,
            X: torch.Tensor     # Float     [m, d]
    ) -> torch.Tensor:          # Boolean   [N1, N2, ..., N(k - 1), m]
        return self.forward(X=X) >= 0
    
    # def select_data(
    #         self,
    #         X: torch.Tensor,     # Float     [m, d]
    #         y: torch.Tensor = None      # Boolean   [Ni, ..., N(k - 1), m]
    # ) -> torch.Tensor:          # Boolean   [N1, N2, ..., N(k - 1), m]
    #     selections = self.predict(X=X)
    #     X_sel = X[selections]
    #     if y is not None:
    #         y_sel = y[selections]
    #     else:
    #         y_sel = None
    #     return y_sel, X_sel
    
    def prediction_rate(
            self,
            X: torch.Tensor     # Float     [m, d]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1)]
        return self.predict(X=X).sum(dim=-1) / X.size(0)
    
    def agreements(
            self,
            X: torch.Tensor,                    # Float     [m, d]
            y: torch.Tensor                     # Boolean   [Ni, ..., N(k - 1), m]
    ) -> torch.Tensor:                          # Boolean   [N1, ..., Ni, ..., N(k - 1), m]
        return self.predict(X=X) == y.bool()    # [N1, ..., Ni, ..., N(k - 1), m] * [Ni, ..., N(k - 1), m]
    
    def accuracy(
            self,
            X: torch.Tensor,    # Float     [m, d]
            y: torch.Tensor     # Boolean   [Ni, ..., N(k - 1), m]
    ) -> torch.Tensor:          # Boolean   [N1, ..., Ni, ..., N(k - 1)]
        return self.agreements(
            X=X,
            y=y
        ).sum(dim=-1) / X.size(0)
    
    def errors(
            self,
            X: torch.Tensor,                    # Float     [m, d]
            y: torch.Tensor                     # Boolean   [Ni, ..., N(k - 1), m]
    ) -> torch.Tensor:                          # Boolean   [N1, ..., Ni, ..., N(k - 1), m]
        return self.predict(X=X) != y.bool()    # [N1, ..., Ni, ..., N(k - 1), m] * [Ni, ..., N(k - 1), m]
    
    def error_rate(
            self,
            X: torch.Tensor,    # Float     [m, d]
            y: torch.Tensor     # Boolean   [Ni, ..., N(k - 1), m]
    ) -> torch.Tensor:          # Float     [N1, ..., Ni, ..., N(k - 1)]
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
        ).unsqueeze(-1)                 # [N1, N2, ..., N(k - 1), 1]

    def proj_weights_to(
            self,
            ids: torch.Tensor,  # Boolean [N1, N2, ..., N(k - 1)]
            X: torch.Tensor     # [N2, ..., N(k - 1), d]
    ) -> None:
        if ids.any():
            X_rep = X.unsqueeze(0).expand(self.weights.shape[0], *X.shape[:]) # [N1, N2, ..., N(k - 1), d]

            # <w, X>
            w_proj = (self.weights[ids] * X_rep[ids]).sum(-1).unsqueeze(-1)  # [..., 1]

            # w - <w, X>X^T
            self.weights[ids] -= w_proj * X_rep[ids]

    def proj_data(
            self,
            X: torch.Tensor     # Float     [m, d]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1), m, d]
        # X - <X, w>w^T
        return X - torch.matmul(
            self.forward(X).unsqueeze(-1),  # [N1, N2, ..., N(k - 1), m, 1]
            self.weights.unsqueeze(-2)      # [N1, N2, ..., N(k - 1), 1, d]
        )      # [N1, N2, ..., N(k - 1), m, d]
    
    def proj_grad(
            self,
            X: torch.Tensor,    # Float     [m, d]
            y: torch.Tensor     # Boolean   [Ni, ..., N(k - 1), m]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1), d]                                                                                   
        return torch.mean(
            self.agreements(
                X=X,
                y=y
            ).unsqueeze(-1) * self.proj_data(X=X),       # [N1, N2, ..., N(k - 1), m, d]
            dim=-2
        )
    
    def conditional_one_rate(
            self,
            X: torch.Tensor,    # Float     [m, d]
            y: torch.Tensor     # Boolean   [Ni, ..., N(k - 1), m]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1)]
        y_sel = y * self.predict(X=X)     # [N1, N2, ..., N(k - 1), m]
        return y_sel.sum(dim=-1) / self.predict(X=X).sum(dim=-1)
    
    def conditional_zero_rate(
            self,
            X: torch.Tensor,    # Float     [m, d]
            y: torch.Tensor     # Boolean   [Ni, ..., N(k - 1), m]
    ) -> torch.Tensor:          # Float     [N1, N2, ..., N(k - 1)]
        return 1 - self.conditional_accuracy(
            X=X,
            y=y
        )
    
class ConditionalLinearModel(nn.Module):
    
    def __init__(
            self,
            selector: LinearModel = None,
            predictor: Any = None,
            device: torch.device = torch.device('cpu')
    ):
        super(ConditionalLinearModel, self).__init__()

        self.selector: LinearModel = None
        self.set_selector(selector=selector)

        self.predictor: Any = None
        self.set_predictor(predictor=predictor)

        self.device = device

    def set_selector(
            self,
            selector: LinearModel       # weights: [N1, N2, ..., N(k - 1), d]
    ) -> None:
        if isinstance(selector, LinearModel):
            self.selector = selector

    def set_predictor(
            self,
            predictor: Any
    ) -> None:
        if hasattr(predictor, "errors"):
            self.predictor = predictor

    def reduce(
            self,
            ids: torch.Tensor,
            predictor_dim: int = 0,
            selector_dim: int = 0
    ) -> Self:
        if isinstance(self.predictor, LinearModel):
            self.predictor.reduce(ids=ids, dims=predictor_dim)
        if isinstance(self.selector, LinearModel):
            self.selector.reduce(ids=ids, dims=selector_dim)
        return self

    # def select_data(
    #         self,
    #         X: torch.Tensor,
    #         y: torch.Tensor
    # ) -> Any:
    #     if isinstance(self.selector, LinearModel):
    #         selections = self.selector.predict(X=X)
    #         return y[selections], X[selections]
    #     return None, None

    def conditional_error_rate(
            self,
            X: torch.Tensor,                                # [m, d]
            y: torch.Tensor                                 # [Ni, ..., N(k - 1), m]
    ) -> torch.Tensor:                                      # [N1, N2, ..., N(k - 1)]
        selections = X
        if isinstance(self.selector, LinearModel):
            selections = self.selector.predict(X=X)         # [N1, N2, ..., N(k - 1), m]

        errors = y                                          # [N1, N2, ..., N(k - 1), m]
        if hasattr(self.predictor, "errors"):
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
