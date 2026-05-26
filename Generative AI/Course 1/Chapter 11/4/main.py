from matplotlib import transforms
import matplotlib.pyplot as plt
import util
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

util.set_seed(1234)

# Select the appropriate device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the data transforms: convert images to tensors and normalize them
data_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST-specific mean and std
    ]
)

# Download and load the MNIST training and test datasets
train_ds_full = datasets.MNIST(
    root="./data", train=True, download=True, transform=data_transforms
)
test_ds = datasets.MNIST(
    root="./data", train=False, download=True, transform=data_transforms
)

# Create a validation split from the training set
val_ratio = 0.1
val_size = int(len(train_ds_full) * val_ratio)
train_size = len(train_ds_full) - val_size
train_ds, val_ds = torch.utils.data.random_split(train_ds_full, [train_size, val_size])

print(
    f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}"
)

# Create DataLoaders to handle batching
batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)