from torch.optim import SGD, Adam

# 'model' is our MLP
def function(model):
  optimizer_sgd = SGD(model.parameters(), lr=0.01, momentum=0.9)
  print("SGD Optimizer instantiated.")

  optimizer_adam = Adam(model.parameters(), lr=0.01)
  print("Adam Optimizer instantiated.")