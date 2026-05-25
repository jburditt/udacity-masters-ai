import torch
import torch.nn as nn

def function():
  cross_entropy_loss = nn.CrossEntropyLoss()
  target = torch.tensor([1]) # 0 for cats, 1 for dogs. Label is for dog.

  # Case 1: Prediction is more likely a dog (correct)
  predicted1 = torch.tensor([[2.0, 5.0]])
  loss1 = cross_entropy_loss(predicted1, target)
  print(f"Loss when prediction is more likely a dog: {loss1.item():.4f}")

  # Case 2: Prediction is more likely a cat (incorrect)
  predicted2 = torch.tensor([[1.5, 1.1]])
  loss2 = cross_entropy_loss(predicted2, target)
  print(f"Loss when prediction is more likely a cat: {loss2.item():.4f}")
  
  return loss1, loss2
