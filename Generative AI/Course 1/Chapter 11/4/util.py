import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# Set a seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define a simple Multi-Layer Perceptron (MLP)
class SimpleMLP(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Flatten the image from 28x28 to a 784-element vector
        x = torch.flatten(x, 1)
        # Apply ReLU activation after the first layer
        x = F.relu(self.fc1(x))
        # The output of the second layer are the logits
        x = self.fc2(x)
        return x
    
# Function to compute accuracy from model outputs (logits)
@torch.no_grad()
def accuracy_from_logits(logits, y):
    # Get predicted class by finding the index with the highest logit value
    preds = logits.argmax(dim=1)
    # Check if predictions match the true labels
    correct = preds == y
    # Return the mean accuracy for the batch
    return correct.float().mean().item()

# Function for a single training epoch
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss, running_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # 1. Zero the gradients
        optimizer.zero_grad()
        # 2. Get model predictions (forward pass)
        logits = model(x)
        # 3. Calculate the loss
        loss = criterion(logits, y)
        # 4. Compute gradients (backward pass)
        loss.backward()
        # 5. Update model weights
        optimizer.step()
        
        # ... update stats ...
        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy_from_logits(logits, y) * batch_size
        n += batch_size
        
    return running_loss / n, running_acc / n