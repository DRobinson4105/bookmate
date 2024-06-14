import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCALAR_LAYER_SIZE=10
SEQUENCE_LAYER_SIZE=10
FINAL_LAYER_SIZE=10
LISTINGS_SIZE=50

def pre_process(list):
    print(list)
    list = [torch.tensor(x[:LISTINGS_SIZE], dtype=torch.float, device=device) for x in list]
    list = [F.pad(x, (0, max(0, LISTINGS_SIZE - x.size(0))), 'constant', 0) for x in list]
    list = pad_sequence(list).transpose(0, 1)
    return list

class BookDataset(Dataset):
    def __init__(self, scalars, prices, conditions, targets=None):
        self.scalars = torch.tensor(scalars, dtype=torch.float, device=device)
        self.prices = pre_process(prices)
        self.conditions = pre_process(conditions)
        
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float, device=device)
        else:
            self.targets = torch.tensor(self.scalars[:, 0:1].tolist(), dtype=torch.float, device=device)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.scalars[idx], self.prices[idx], self.conditions[idx], self.targets[idx]
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar_processor = nn.Linear(2, SCALAR_LAYER_SIZE)
        self.prices_processor = nn.Linear(LISTINGS_SIZE, SEQUENCE_LAYER_SIZE)
        self.conditions_processor = nn.Linear(LISTINGS_SIZE, SEQUENCE_LAYER_SIZE)
        self.middle_layer = nn.Linear(SCALAR_LAYER_SIZE + SEQUENCE_LAYER_SIZE * 2, FINAL_LAYER_SIZE)
        self.output_layer = nn.Linear(FINAL_LAYER_SIZE, 1)

    def forward(self, scalar_input, prices, conditions):
        scalar_output = self.scalar_processor(scalar_input)
        prices_output = self.prices_processor(prices)
        conditions_output = self.conditions_processor(conditions)
        combined = torch.cat((scalar_output, prices_output, conditions_output), dim=1)
        x = F.relu(self.middle_layer(combined))
        return self.output_layer(x).squeeze(0)