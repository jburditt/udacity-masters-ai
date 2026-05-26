import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class NumberProductDataset(Dataset):
    def __init__(self, data_range):
        self.data = [(i, i + 1, i * (i + 1)) for i in range(data_range)]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

class NumberSumDataset(Dataset):
    def __init__(self, data_range):
        self.data = []
        for i in range(data_range):
            for j in range(data_range):
                self.data.append(([i, j], i + j))

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0], dtype=torch.float32), torch.tensor(self.data[idx][1], dtype=torch.float32)

    def __len__(self):
        return len(self.data)
      
class SumMLP(nn.Module):
    def __init__(self):
        super(SumMLP, self).__init__()
        self.hidden_layer = nn.Linear(2, 128)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x