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

# Loss function for classification
criterion = nn.CrossEntropyLoss()

# Function to create a new model instance for each experiment
def make_fresh_model():
    return util.SimpleMLP().to(device)

# Define the optimizer configurations we want to compare
optim_configs = {"sgd": {"lr": 1e-2, "momentum": 0.9}, "adam": {"lr": 1e-3}}

# Helper function to create an optimizer for a given model
def make_optimizer(name, params):
    if name == "sgd":
        return optim.SGD(params, **optim_configs["sgd"])
    if name == "adam":
        return optim.Adam(params, **optim_configs["adam"])
    raise ValueError("Unknown optimizer")
  
# Main training loop to compare optimizers
EPOCHS = 3
histories = {}

for opt_name in ["sgd", "adam"]:
    print(f"\n=== Training with {opt_name.upper()} ===")
    util.set_seed(1234)  # Reset seed for a fair comparison
    model = make_fresh_model()
    optimizer = make_optimizer(opt_name, model.parameters())

    # Store metrics for this run
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        # Train for one epoch
        tl, ta = util.train_one_epoch(model, train_loader, criterion, optimizer, device)
        # Evaluate on the validation set
        vl, va = util.evaluate(model, val_loader, criterion, device)

        # Save metrics
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

        print(
            f"Epoch {epoch:02d} | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}"
        )

    histories[opt_name] = history

print("\nDone training all optimizers.")