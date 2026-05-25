import torch
import torch.nn as nn
import neural_network
from torch.optim import SGD, Adam
import util
from torch.utils.data import Dataset, DataLoader

A = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)
print("A^1:\\n", torch.matrix_power(A, 1))
print("A^2:\\n", torch.matrix_power(A, 2))
print("A^3:\\n", torch.matrix_power(A, 3))
print("A^4:\\n", torch.matrix_power(A, 4))

model = neural_network.MLP(input_size=10)
print(model)

input_vector = torch.randn(10)
output = model(input_vector)
print(output)

# Assuming 'model' is our MLP defined earlier
optimizer_sgd = SGD(model.parameters(), lr=0.01, momentum=0.9)
print("SGD Optimizer instantiated.")

optimizer_adam = Adam(model.parameters(), lr=0.01)
print("Adam Optimizer instantiated.")

dataset = util.NumberProductDataset(data_range=11)
sample = dataset[3] # pick the 4th element (index 3)
print(f"Input: {sample[0]}, {sample[1]}, Output: {sample[2]}")

dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
for i, (input1, input2, output) in enumerate(dataloader):
    print(f"Batch {i+1}:")
    print(f"  Inputs 1: {input1}")
    print(f"  Inputs 2: {input2}")
    print(f"  Outputs: {output}")
    
sum_dataset = util.NumberSumDataset(data_range=10)
sum_dataloader = DataLoader(sum_dataset, batch_size=10, shuffle=True)
sum_model = util.SumMLP()
loss_function = nn.MSELoss()
optimizer = Adam(sum_model.parameters(), lr=0.01)

num_epochs = 20
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, targets in sum_dataloader:
        optimizer.zero_grad()
        predictions = sum_model(inputs)
        loss = loss_function(predictions.squeeze(), targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(sum_dataloader):.4f}")
    
test_input = torch.tensor([3.0, 7.0])
prediction = sum_model(test_input)
print(f"Prediction for 3 + 7: {prediction.item():.1f}")