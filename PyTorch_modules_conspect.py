import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

N = 100  # Количество точек в каждом классе
D = 2  # Размерность
K = 3  # Количество классов
X = np.zeros((N * K, D))  # Матрица данных, где каждая строчка - один экземпляр
y = np.zeros(N * K, dtype='uint8')  # Матрица "легенды"

for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # Радиус по котором распределяются точки
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # Коэффциент тэта
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

plt.figure(figsize=(10, 8))

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.rainbow)
plt.title("Спираль точек", fontsize=15)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$y$', fontsize=14)
plt.ioff()
plt.show()

X = pt.FloatTensor(X)
y = pt.LongTensor(y)

D_in, H, D_out = D, N, K

class PointsClassificationModule(nn.Module):
    def __init__(self):
        super(PointsClassificationModule, self).__init__()
        self.f_lin = nn.Linear(D_in, H)
        self.s_lin = nn.Linear(H, D_out)

    def forward(self, X):
        X = F.relu(self.f_lin(X))
        return F.softmax(self.s_lin(X))

network = PointsClassificationModule()

loss_fn = nn.CrossEntropyLoss()

optim = optim.Adam(network.parameters(), 1e-1)

losses=[]

for t in range(100):
    y_pred = network(X)

    loss = loss_fn(y_pred, y)
    losses.append(loss.item())

    optim.zero_grad()
    loss.backward()
    optim.step()

plt.plot(losses)
plt.show()

class PointsClassificationModule_without_params(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layers = [nn.Linear(D_in, H), nn.Linear(H, D_out)]
        self.my_useless_bias = pt.ones(1, H, requires_grad=True)
        self.more_of_my_useless_bias = [
            pt.ones(1, H, requires_grad=True),
            pt.ones(1, H, requires_grad=True),
            pt.ones(1, H, requires_grad=True)
        ]

    def forward(self, X):
        X = F.relu(self.linear_layers[0](X))
        X += self.my_useless_bias
        return F.softmax(self.linear_layers[1](X))

network_2 = PointsClassificationModule_without_params()

print(list(network_2.parameters()))

class PointsClassificationModule_with_params(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(D_in, H), nn.Linear(H, D_out)])
        self.my_useless_bias = nn.Parameter(pt.ones(1, H, requires_grad=True))
        self.more_of_my_useless_bias = nn.ParameterList([
            nn.Parameter(pt.ones(1, H, requires_grad=True)),
            nn.Parameter(pt.ones(1, H, requires_grad=True)),
            nn.Parameter(pt.ones(1, H, requires_grad=True))
        ])

    def forward(self, X):
        X = F.relu(self.linear_layers[0](X))
        X += self.my_useless_bias
        for b in self.more_of_my_useless_bias:
            X += b
        return F.softmax(self.linear_layers[1](X))

network_3 = PointsClassificationModule_with_params()

print(list(network_3.parameters()))