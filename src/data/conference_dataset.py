import os
from torch.utils.data import Dataset

class TextDataset(Dataset):  
    def __init__(self, path, tokenizer=None, max_length=512):
        self.texts = []
        self.tokenizer = tokenizer
        files = os.listdir(path)
        for f in files: 
            with open(path+f) as file:
                text = file.read().replace('\n',' ')
                tokenized_text = self.tokenizer.encode(text, return_tensors='pt')[0][:max_length]
                self.texts.append(tokenized_text)
     
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        text = self.texts[i]
        return text