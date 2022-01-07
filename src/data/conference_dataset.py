import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):  
    def __init__(self, path):
        self.texts = torch.load(path)
     
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]