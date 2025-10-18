import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# 50x50 pixels
img_size = 50

class Net(nn.Module):
    # constructor
    def __init__(self):
        super().__init__()

        # Neural Network
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)

        # Linear Layers
        self.fc1 = nn.Linear(128*2*2, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # print(f"Shape after conv1: {x.shape}")
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # print(f"Shape after conv2: {x.shape}")
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        # print(f"Shape after conv3: {x.shape}")
        # sys.exit("Trying to get shape for linear layer")
        
        x = x.view(-1, 128*2*2)

        x = self.softmax(self.fc2(F.relu(self.fc1(x))))

        return x

## Checking the input Linear size
# net = Net()

# test_img = torch.randn(img_size, img_size).view(-1, 1, img_size, img_size)
# output = net(test_img)