import torch
import torch.nn as nn

def function():
  mse_loss = nn.MSELoss()
  predicted_price = torch.tensor([320000.0])
  actual_price = torch.tensor([300000.0])
  loss = mse_loss(predicted_price, actual_price)
  print(f"MSE Loss: {loss.item()}")
  return loss