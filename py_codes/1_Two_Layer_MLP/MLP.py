import torch
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# how many samples per batch to load
batch_size = 20

# convert data to torch.FloatTensor
normalize = transforms.Normalize((0.1307,), (0.3081,))  # MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# choose the training and test datasets
train_data = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

# prepare data loaders
num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(0.2 * num_train))

train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size, sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size, sampler=valid_sampler)

import matplotlib.pyplot as plt

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    # print out the correct label for each image
    # .item() gets the value contained in a Tensor
    ax.set_title(str(labels[idx].item()))
plt.show()
