import torch
import neural_network

A = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)
print("A^1:\\n", torch.matrix_power(A, 1))
print("A^2:\\n", torch.matrix_power(A, 2))
print("A^3:\\n", torch.matrix_power(A, 3))
print("A^4:\\n", torch.matrix_power(A, 4))

model = neural_network.MLP(input_size=10)
print(model)