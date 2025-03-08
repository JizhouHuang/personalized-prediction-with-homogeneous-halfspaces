from typing import List, Tuple, Any
import torch
from torch.utils.data import Dataset

class MultiLabelledDataset(Dataset):
    def __init__(
            self, 
            data: torch.Tensor, 
            predictor: Any = None,
            shuffle: bool = True,
            random_state: int = None
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

        if shuffle:
            if random_state:
                self.data: torch.Tensor = data[
                    torch.randperm(
                        data.size(0), 
                        generator=torch.Generator().manual_seed(random_state)
                    )
                ]
            else:
                self.data: torch.Tensor = data[torch.randperm(data.size(0))]
            
        self.label_to_errors(predictor)
        self.device = data.device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
            self, 
            idx: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.trans_labels[idx], self.data[idx, 1:]
    
    def dim_feature(self) -> torch.Tensor:
        return self.data.size(1) - 1
    
    def dim_label(self) -> torch.Tensor:
        if len(self.trans_labels.size()) == 1:
            return 1
        else:
            return self.trans_labels.size(1)
    
    def label_to_errors(
            self,
            predictor: Any
    ) -> None:
        """
        Map the labels according to the given predictor.
        """
        if predictor and hasattr(predictor, 'errors'):
            self.trans_labels = predictor.errors(
                X=self.data[:, 1:],     # [data_batch_size, dim_sample]
                y=self.data[:, 0]       # [data_batch_size]
            ).t()                       # [data_batch_size, cluster_size]
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
    