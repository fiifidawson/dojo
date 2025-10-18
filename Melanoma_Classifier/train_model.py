import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from net_class import Net

# 50x50 pixels
img_size = 50

training_data = np.load("melanoma_training_data.npy", allow_pickle=True)

# Putting image arrays into a tensor
train_X = torch.Tensor([item[0] for item in training_data])
train_X = train_X / 255 # Normalise

# for row in train_X:
#     print(row)
#     print()
#     input()

# One-hot vector labels tensor
train_y = torch.Tensor([item[1] for item in training_data])

# Instantiating class
net = Net()

# Setting optimizer
optimizer = optim.Adam(net.parameters(),
                       lr=0.001)

# Setting loss function
loss_function = nn.MSELoss()

# Setting batch size
batch_size = 100 # no. of images being processed at once

# Setting epoch
epochs = 2
for epoch in range(epochs):
    for i in range(0, len(train_X), batch_size):
        print(f"EPOCH {epoch+1} | fraction compplete: {i/len(train_X)}")

        batch_X = train_X[i: i+batch_size].view(-1, 1, img_size, img_size)
        batch_y = train_y[i: i+batch_size]

        # Reset gradients of model parameters to zero before this pass
        optimizer.zero_grad()

        outputs = net(batch_X)

        

        # Setting loss function: calc loss between predicted outputs and actual image one-hot vector labels
        loss = loss_function(outputs, batch_y)
        ## real label(eg): [0, 1]
        ## model guess(eg): [0.34, 0.66]

        # Setting backpropagation: calc gradients of the loss wrt model params
        loss.backward()

        #  Update the model params based on the recent calculated gradient
        optimizer.step()


torch.save(net.state_dict(), "save_model.pth")