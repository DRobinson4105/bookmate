import torch
import torch.nn as nn
import pandas as pd
from utils import Model, BookDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# Prepare Data

BATCH_SIZE=32
df = pd.read_csv('dataset.csv')

convert_to_list = lambda column: [[float(x) for x in list.strip(')(][').split(", ")] for list in df[column].tolist()]
train_dataset, test_dataset = BookDataset.train_test_split(convert_to_list('prices'), convert_to_list('conditions'), df['targets'].tolist(), device=device)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Training

model = Model().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-4)

epochs = 50000
train_loss_values = []
test_loss_values = []
avg_train_loss, avg_test_loss = 0, 0

for epoch in tqdm(range(epochs)):
    if (epoch % 100 == 0):
        tqdm.write(f"Train Loss: {round(avg_train_loss, 2)} | Test Loss: {round(avg_test_loss, 2)}")

    # Training
    model.train()
    avg_train_loss = 0

    for prices, conditions, targets in train_dataloader:
        pred = model(prices, conditions)

        loss = loss_fn(pred, targets)
        
        avg_train_loss += loss.detach().cpu().numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss /= len(train_dataloader)
    train_loss_values.append(avg_train_loss)

    # Testing
    model.eval()
    avg_test_loss = 0

    with torch.inference_mode():
        for prices, conditions, targets in test_dataloader:
            pred = model(prices, conditions)

            loss = loss_fn(pred, targets)

            avg_test_loss += loss.detach().cpu().numpy()

        avg_test_loss /= len(test_dataloader)
        test_loss_values.append(avg_test_loss)

# Save Results

torch.save(obj=model.state_dict(), f="model.pt")

loss_df = pd.DataFrame({
    'Epoch': range(epochs),
    'Train_Loss': train_loss_values,
    'Test_Loss': test_loss_values 
})

loss_df.to_csv('loss.csv', index=False)

# Final Test

prices, conditions, targets = next(iter(test_dataloader))
model.eval()
preds = model(prices, conditions)
print("Expected:", targets[:5].tolist())
print("Actual:", preds[:5].tolist())