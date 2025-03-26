import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv("netflix.csv")  
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)

# Extract Close prices and scale data
scaler = MinMaxScaler()
df["Close_Scaled"] = scaler.fit_transform(df[["Close"]])

# Prepare data for supervised learning
seq_len = 15  # Number of past days to use for prediction
X, y = [], []
for i in range(len(df) - seq_len):
    X.append(df["Close_Scaled"].iloc[i:i+seq_len].values)
    y.append(df["Close_Scaled"].iloc[i+seq_len])

X, y = np.array(X), np.array(y)

# Split into train and test sets
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=False)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(train_x.reshape(train_x.shape[0], -1), train_y)
lr_pred = lr_model.predict(test_x.reshape(test_x.shape[0], -1))

# Convert predictions back to original scale
lr_pred = scaler.inverse_transform(lr_pred.reshape(-1, 1))
test_actual = scaler.inverse_transform(test_y.reshape(-1, 1))

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

# Convert to tensors
torch_train_x = torch.tensor(train_x[:, :, None], dtype=torch.float32)
torch_train_y = torch.tensor(train_y[:, None], dtype=torch.float32)
torch_test_x = torch.tensor(test_x[:, :, None], dtype=torch.float32)
torch_test_y = torch.tensor(test_y[:, None], dtype=torch.float32)

# DataLoader
batch_size = 32
dataset = TensorDataset(torch_train_x, torch_train_y)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize LSTM
model = LSTMModel(input_size=1, hidden_size=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        output = model(batch_x)
        loss = loss_fn(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

# Evaluate LSTM model
model.eval()
with torch.no_grad():
    lstm_pred = model(torch_test_x)
lstm_pred = scaler.inverse_transform(lstm_pred.numpy())

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(test_actual, label="Actual Prices", color="green")
plt.plot(lr_pred, label="Linear Regression Predictions", color="blue")
plt.plot(lstm_pred, label="LSTM Predictions", color="red")
plt.legend()
plt.show()

