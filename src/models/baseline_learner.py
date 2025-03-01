import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from ..utils.simple_models import PredictiveModel

class LogisticRegLearner(PredictiveModel):
    def __init__(
            self,
            max_data_train: int = 50000,
            device: torch.device = torch.device('cpu')
        ):

        # Initialize the logistic regression model
        super().__init__(LogisticRegression(max_iter=2000), max_data_train, device)

class SVMLearner(PredictiveModel):
    def __init__(
            self,
            max_data_train: int = 50000, 
            device: torch.device = torch.device('cpu')
    ):

        super().__init__(
            model=SVC(C=100000, kernel='linear'), 
            max_data_train=max_data_train, 
            device=device
        )

class RandomForestLearner(PredictiveModel):
    def __init__(
            self,
            max_data_train: int = 50000, 
            device: torch.device = torch.device('cpu')
    ):
        
        super().__init__(RandomForestClassifier(n_estimators=100), max_data_train, device)

class XGBoostLearner(PredictiveModel):
    def __init__(
            self,
            max_data_train: int = 50000, 
            device: torch.device = torch.device('cpu')
    ):

        super().__init__(xgb.XGBClassifier(), max_data_train, device)
