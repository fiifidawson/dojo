import numpy as np
import torch
from net_class import Net

# 50x50 pixels
img_size = 50

net = Net()
net.load_state_dict(torch.load("model/saved_model.pth"))

net.eval()


testing_data = np.load("melanoma_testing_data.npy", allow_pickle=True)

# Putting image arrays into a tensor
test_X = torch.Tensor([item[0] for item in testing_data])
test_X = test_X / 255 # Normalise

# for row in train_X:
#     print(row)
#     print()
#     input()

# One-hot vector labels tensor
test_y = torch.Tensor([item[1] for item in testing_data])