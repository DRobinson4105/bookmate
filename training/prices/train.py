import torch
from price_predictor import Model

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
model = Model().to(device)
torch.save(obj=model.state_dict(), f="model.pt")