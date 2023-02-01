import torch

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """
    Dataset class
    ===
    This class is used to load the dataset since pytorch needs a custom dataset class to load the data.
    """
    def __init__(self, corpus, labels) -> None:
        super().__init__()

        if len(corpus) != len(labels):
            raise ValueError("Corpus and labels must have the same length")
        
        self.corpus = corpus
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return (torch.Tensor(self.corpus[index]).long(), torch.Tensor([self.labels[index]]).reshape(1,1).float())
    