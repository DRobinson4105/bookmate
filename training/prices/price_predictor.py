import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

SEQUENCE_LAYER_SIZE=16
FINAL_LAYER_SIZE=32
LISTINGS_SIZE=20

def pre_process(list, device):
    result = []
    for x in list:
        while len(x) < LISTINGS_SIZE:
            x += x

        x = x[:LISTINGS_SIZE]
        result.append(x)
    
    return torch.tensor(result, dtype=torch.float, device=device)

class BookDataset(Dataset):
    def __init__(self, prices, conditions, targets=None, device="cpu", start=0, end=1):
        self.prices = pre_process(prices, device)
        self.conditions = pre_process(conditions, device)
        
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float, device=device)
        else:
            self.targets = torch.tensor(self.prices[:, 0:1].tolist(), dtype=torch.float, device=device)

        start_idx = int(start * len(prices))
        end_idx = int(end * len(prices))

        self.prices = self.prices[start_idx:end_idx]
        self.conditions = self.conditions[start_idx:end_idx]
        self.targets = self.targets[start_idx:end_idx]

    def train_test_split(prices, conditions, targets=None, device="cpu", split=0.8):
        return BookDataset(prices, conditions, targets, device, 0, split), BookDataset(prices, conditions, targets, device, split, 1)

    def __len__(self):
        return len(self.prices)
    
    def __getitem__(self, idx):
        return self.prices[idx], self.conditions[idx], self.targets[idx]
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.prices_processor = nn.Sequential(
            nn.Linear(LISTINGS_SIZE, SEQUENCE_LAYER_SIZE),
            nn.BatchNorm1d(SEQUENCE_LAYER_SIZE),
            nn.Dropout()
        )
        self.conditions_processor = nn.Sequential(
            nn.Linear(LISTINGS_SIZE, SEQUENCE_LAYER_SIZE),
            nn.BatchNorm1d(SEQUENCE_LAYER_SIZE),
            nn.Dropout()
        )
        self.middle_layer = nn.Sequential(
            nn.Linear(SEQUENCE_LAYER_SIZE * 2, FINAL_LAYER_SIZE),
            nn.BatchNorm1d(FINAL_LAYER_SIZE),
        )
        self.output_layer = nn.Linear(FINAL_LAYER_SIZE, 1)

    def forward(self, prices, conditions):
        prices_output = F.relu(self.prices_processor(prices))
        conditions_output = F.relu(self.conditions_processor(conditions))
        combined = torch.cat((prices_output, conditions_output), dim=1)
        x = F.relu(self.middle_layer(combined))
        return self.output_layer(x).squeeze()