import numpy as np
import torch as pt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import sys, os


class Covid19SicknessInfoNN(nn.Module):
    def __init__(self):
        super(Covid19SicknessInfoNN, self).__init__()
        self.l0_input = nn.Linear(7, 16)
        self.l1_hidden = nn.Linear(16, 16)
        self.l2_output = nn.Linear(16, 1)

    def forward(self, x):

        x = pt.relu(self.l0_input(x))

        x = pt.relu(self.l1_hidden(x))

        x = self.l2_output(x)

        return x

CovidNet = Covid19SicknessInfoNN()

optimizer = optim.Adam(CovidNet.parameters(), lr=0.01)
loss_func = nn.BCEWithLogitsLoss()

train_data = [
    [0, 0, 0, 0., 38, 1, 1],
    [1, 0, 1, 12.5, 38, 1, 1],
    [1, 1, 0, 0., 38, 1, 0],
    [1, 1, 0, 0., 38, 1, 1],
    [0, 0, 1, 1., 38, 1, 0],
    [1, 0, 1, 1.2, 36, 0, 0],
    [0, 0, 0, 0., 36, 0, 0],
    [1, 1, 1, 1., 36, 0, 0]
]

train_data = pt.FloatTensor(train_data)

priority_data = pt.FloatTensor([0.5, 0.9, 1, 0.5, 1, 1, 0.7])

print(train_data)

train_data = priority_data * train_data

print(train_data)

target_data = [
    1, 1, 1, 1, 0, 0, 0, 0
]

target_data = pt.FloatTensor(target_data)

input_data = TensorDataset(train_data, target_data)
trainD = DataLoader(input_data, batch_size=1, shuffle=True)

epochs = 100

for e in range(epochs):
    for (inputs, target) in trainD:
        inputs = Variable(inputs)

        target = Variable(target)

        optimizer.zero_grad()

        net_out = CovidNet(inputs)

        target = target.view(-1,1)

        loss = loss_func(net_out, target)

        loss.backward()

        optimizer.step()

        sys.stdout.write(f"\r E: {e}, L: {loss.item():.5f}")

print()
while(True):
    print("Enter test data")
    print()

    test_data = Variable(pt.FloatTensor(np.array(input().split(' '), dtype=float)))

    print(pt.sigmoid(CovidNet(test_data)))