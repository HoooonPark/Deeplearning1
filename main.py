import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Define the ticker symbol and time period
symbol = 'AAPL'
start_date = '2015-01-01'
end_date = '2021-12-31'

# Download the stock data
data = yf.download(symbol, start=start_date, end=end_date)

# Extract the closing prices
close_data = data['Close'].values.reshape(-1,1)

# Normalize the data
scaler = MinMaxScaler()
close_data = scaler.fit_transform(close_data)

# Split the data into training and testing sets
train_data = close_data[:int(0.8*len(close_data))]
test_data = close_data[int(0.8*len(close_data)):]

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
model = LSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    inputs = torch.from_numpy(train_data[:-1]).float()
    targets = torch.from_numpy(train_data[1:]).float()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
