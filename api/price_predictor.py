import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCALAR_LAYER_SIZE=10
SEQUENCE_LAYER_SIZE=20
FINAL_LAYER_SIZE=10
SEQUENCE_SIZE=100

class BookDataset(Dataset):
    def __init__(self, scalars, sequences, targets=None):
        self.scalars = torch.tensor(scalars, dtype=torch.float, device=device)
        self.sequences = sequences
        self.sequences = [torch.tensor(x, dtype=torch.float, device=device) for x in self.sequences]
        self.sequences = [nn.Flatten()(x.unsqueeze(dim=0))[:100].squeeze() for x in self.sequences]
        self.sequences[0] = F.pad(self.sequences[0], (0, max(0, 100 - self.sequences[0].size(0))), 'constant', 0)
        self.sequences = pad_sequence(self.sequences).transpose(0, 1)
        
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float, device=device)
        else:
            self.targets = torch.tensor(self.scalars[:, 0:1].tolist(), dtype=torch.float, device=device)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.scalars[idx], self.sequences[idx], self.targets[idx]
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar_processor = nn.Linear(2, SCALAR_LAYER_SIZE)
        self.sequence_processor = nn.Linear(SEQUENCE_SIZE, SEQUENCE_LAYER_SIZE)
        self.middle_layer = nn.Linear(SCALAR_LAYER_SIZE + SEQUENCE_LAYER_SIZE, FINAL_LAYER_SIZE)
        self.output_layer = nn.Linear(FINAL_LAYER_SIZE, 1)

    def forward(self, scalar_input, sequence_input):
        scalar_output = self.scalar_processor(scalar_input)
        sequence_output = self.sequence_processor(sequence_input)
        combined = torch.cat((scalar_output, sequence_output), dim=1)
        x = F.relu(self.middle_layer(combined))
        return self.output_layer(x).squeeze(0)