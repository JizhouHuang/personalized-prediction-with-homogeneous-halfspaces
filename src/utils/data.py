from typing import Any, Self, Tuple
import torch
from torch.utils.data import Dataset

class MultiLabeledDataset(Dataset):
    def __init__(
            self, 
            data: torch.Tensor, 
            predictor: Any = None
        ):
        """
        Initialize the dataset with a label mapping.
        Note that the label map will generate cluster_size {0, 1} labels for each example.
        A label of "1" indicates the corresponding sparse classifier disagrees with the 
        true original data label.

        Parameters:
        data (torch.Tensor):    The input data.
        predictor:              A tuple of classifier(s) and its corresponding prediction method.
                                Classifier(s) can be any class, while the prediction method must
                                take the classifier(s) and the feature in the form of torch.Tensor as
                                inputs, then outputs a label in the form of torch.Tensor.
        """
        self.data: torch.Tensor = data
        self.label_to_errors(predictor)
        self.device = data.device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
            self, 
            idx: int
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.trans_labels[idx].t(), self.data[idx, 1:]
    
    def decouple(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.trans_labels, self.data[:, 1:]
    
    def features(self) -> torch.Tensor:
        return self.data[:, 1:]
    
    def labels(self) -> torch.Tensor:
        return self.trans_labels
    
    def num_features(self) -> int:
        return self.data.size(1) - 1
    
    def size_feature(
            self,
            dim: int = -1
    ) -> Tuple[int, torch.Size]:
        if dim < 0:
            return torch.Size([self.data.size(0), self.num_features()])
        elif dim == 0:
            return self.data.size(0)
        else:
            return self.num_features()
    
    def num_labels(self) -> int:
        if len(self.trans_labels.size()) == 1:
            return 1
        else:
            return self.trans_labels.size(1)
        
    def size_label(
            self,
            dim: int = -1
    ) -> Tuple[int, torch.Size]:
        if dim < 0:
            return self.trans_labels.size()
        return self.trans_labels.size(dim)
    
    def label_with(
            self,
            predictor: Any
    ) -> Self:
        return MultiLabeledDataset(
            data=self.data,
            predictor=predictor
        )
    
    def random_subset(
            self,
            subset_size: int,
            random_state: int = None
    ) -> Self:
        if random_state is not None:
            data_perm = self.data[
                torch.randperm(
                    self.data.size(0), 
                    generator=torch.Generator().manual_seed(random_state)
                )
            ]
        else:
            data_perm = self.data[torch.randperm(data.size(0))]
        return MultiLabeledDataset(
            data=data_perm[:min(subset_size, data_perm.size(0))],
            predictor=self.predictor
        )
    
    def label_to_errors(
            self,
            predictor: Any
    ) -> None:
        """
        Map the labels according to the given predictor.
        """
        self.predictor = predictor
        if predictor and hasattr(predictor, 'errors'):
            self.trans_labels = predictor.errors(
                X=self.data[:, 1:],     # [data_batch_size, dim_sample]
                y=self.data[:, 0]       # [data_batch_size]
            ).t()                       # [data_batch_size, num predictors]
        else:
            self.trans_labels = self.data[:, 0].bool()

class FixedIterationLoader:
    def __init__(self, dataloader, max_iterations):
        self.dataloader = dataloader
        self.max_iterations = max_iterations

    def __iter__(self):
        self.data_iter = iter(self.dataloader)
        self.iter_count = 0
        return self

    def __next__(self):
        if self.iter_count >= self.max_iterations:
            raise StopIteration
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        self.iter_count += 1
        return batch
    