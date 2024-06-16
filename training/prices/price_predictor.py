import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

DETAILS_LAYER_SIZE=10
SEQUENCE_LAYER_SIZE=10
FINAL_LAYER_SIZE=10
LISTINGS_SIZE=50

def pre_process(list, device):
    result = []
    for x in list:
        while len(x) < LISTINGS_SIZE:
            x += x

        x = x[:LISTINGS_SIZE]
        result.append(x)
    
    return torch.tensor(result, dtype=torch.float, device=device)

class BookDataset(Dataset):
    def __init__(self, details, prices, conditions, targets=None, device="cpu", start=0, end=1):
        self.details = torch.tensor(details, dtype=torch.float, device=device)
        self.prices = pre_process(prices, device)
        self.conditions = pre_process(conditions, device)
        
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float, device=device)
        else:
            self.targets = torch.tensor(self.details[:, 0:1].tolist(), dtype=torch.float, device=device)

        start_idx = int(start * len(details))
        end_idx = int(end * len(details))

        self.details = self.details[start_idx:end_idx]
        self.prices = self.prices[start_idx:end_idx]
        self.conditions = self.conditions[start_idx:end_idx]
        self.targets = self.targets[start_idx:end_idx]

    def train_test_split(details, prices, conditions, targets=None, device="cpu", split=0.8):
        return BookDataset(details, prices, conditions, targets, device, 0, split), BookDataset(details, prices, conditions, targets, device, split, 1)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.details[idx], self.prices[idx], self.conditions[idx], self.targets[idx]
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.details_processor = nn.Linear(2, DETAILS_LAYER_SIZE)
        self.prices_processor = nn.Linear(LISTINGS_SIZE, SEQUENCE_LAYER_SIZE)
        self.conditions_processor = nn.Linear(LISTINGS_SIZE, SEQUENCE_LAYER_SIZE)
        self.middle_layer = nn.Linear(DETAILS_LAYER_SIZE + SEQUENCE_LAYER_SIZE * 2, FINAL_LAYER_SIZE)
        self.output_layer = nn.Linear(FINAL_LAYER_SIZE, 1)

    def forward(self, details, prices, conditions):
        details_output = self.details_processor(details)
        prices_output = self.prices_processor(prices)
        conditions_output = self.conditions_processor(conditions)
        combined = torch.cat((details_output, prices_output, conditions_output), dim=1)
        x = F.relu(self.middle_layer(combined))
        return self.output_layer(x).squeeze()