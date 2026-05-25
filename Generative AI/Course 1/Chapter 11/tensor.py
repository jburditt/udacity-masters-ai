import torch
import matplotlib.pyplot as plt

images = torch.rand(4, 28, 28)
# To access the second image
second_image = images[1]
print(second_image)

plt.imshow(second_image, cmap='gray')
plt.show()