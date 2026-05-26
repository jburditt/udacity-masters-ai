from matplotlib import transforms
import util
import torch
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