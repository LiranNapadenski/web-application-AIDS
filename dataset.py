import torch
from torch.utils.data import Dataset

class RequestsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=1024):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        text = row["content"]
        label = int(row["classification"])

        encoded = self.tokenizer.encode(text, max_length=self.max_len)
        mask = self.tokenizer.create_mask(encoded)

        return encoded, mask, None, torch.tensor(label, dtype=torch.long)
