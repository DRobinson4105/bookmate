import torch
from torch.utils.data import DataLoader
from price_predictor import BookDataset, Model

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequences = [torch.randn((3, 2)).tolist(), torch.randn((5, 2)).tolist(), torch.randn((4, 2)).tolist()]
scalars = torch.randn((3, 2)).tolist()
targets = torch.randn((3, 1)).tolist()

model = Model().to(device)
model.load_state_dict(torch.load(f="model.pt"))

dataset = BookDataset(scalars=scalars, sequences=sequences, targets=targets)
dataloader = DataLoader(dataset)
scalar, sequence, target = next(iter(dataloader))
scalar, sequence, target = scalar.to(device), sequence.to(device), target.to(device)
with torch.inference_mode():
    print(model(scalar, sequence), target)