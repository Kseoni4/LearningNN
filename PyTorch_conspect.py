"""



"""

import torch

# Тензор - основная единица работы с данными в PyTorch
# Тензор - многомерный массив
# ПРЯМ КАК В НАМПАЙ
# ОДНАКО В торче тензоры могут вычислять на GPU.

a = torch.FloatTensor([1, 2]) # 1D тензор
print(a)

print(a.shape) # Размер 1D тензора - ([кол-во элементов])

b = torch.FloatTensor([[1,2,3],[4,5,6]]) # 2D тензор
print(b)

print(b.shape) # Размер 2D тензора: ([количество строк, количество стобцов])

x = torch.FloatTensor(2,3,4)

print(x)

# Тензоры из нулей
x1 = torch.FloatTensor(3,2,4).zero_() # Переопределение на месте при создании

x2 = torch.zeros(3, 2, 4)

x3 = torch.zeros_like(x1)

x = torch.randn((2,3)) # Нормальное распределение среднего 0 и дисперсией 1.
print(x)

print(x.random_(0, 10))
print(x.uniform_(1))
print(x.normal_(mean=0, std=1))
print(x.bernoulli_(p=0.5))

# np.reshape() == torch.view() - аналогичная функция
print(b)

b = b.view(3,2) # изменение формы тензора без изменения данных
#(количество строк, кол-во столбцов)

print(b)

print(b.view(-1)) # Автоматически подсчитывается размер тензора

b.reshape(-1) # Копия тензора со смежными данными (?)

import numpy as np
a = np.random.rand(3,3)

b = torch.from_numpy(a)

def forward_pass(X, w):
    return torch.sigmoid(X @ w)

X = torch.FloatTensor([[-5, 5], [2, 3], [1, -1]])
w = torch.FloatTensor([[-0.5],[2.5]])
result = forward_pass(X, w)
print(f'result {result}')

# CUDA

'''
print(torch.cuda.is_available())

x = torch.FloatTensor(1024, 10024).uniform_()
print(x)

print(x.is_cuda)

x = x.cuda()

print(x)'''

dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 64, 3, 3, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

y_pred = (x @ w1).clamp(min=0).matmul(w2)
loss = (y_pred - y).pow(2).sum()

loss.backward()

print((y_pred - y).pow(2).sum())

print(w1.grad,"\n", w2.grad)

print(loss.grad)


""" ========================================"""

# Ручной подход к созданию НС

"""print()
print()
print()
"""
import torchvision

from torchvision.datasets import MNIST
import torchvision.transforms as tfs

data_tfs = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize((0.5), (0.5))
])

root = "./data/mnist/"

train = MNIST(root, train=True, transform=data_tfs, download=True)
test  = MNIST(root, train=False, transform=data_tfs, download=True)

print(f'Data size:\n\t train {len(train)}, \n\t test {len(test)}')
print(f'Data shape:\n\t features {train[0][0].shape}, \n\t target {type(test[0][1])}')

from torch.utils.data import DataLoader

batch_size = 128

train_loader = DataLoader(train, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, drop_last=True)
"""

x_batch, y_batch = next(iter(train_loader))
print(x_batch.shape, '\n', y_batch.shape)

features = 784
classes = 10

W = torch.FloatTensor(features, classes).uniform_(-1) / features ** 0.5

W.requires_grad_()

epochs = 3
lr = 1e-2
history=[]

import numpy as np
from torch.nn.functional import cross_entropy

for i in range(epochs):
    for x_batch, y_batch in train_loader:

        # Изменяем форму тензора из признаков объектов
        #print(x_batch.shape)
        x_batch = x_batch.reshape(x_batch.shape[0], -1)
        #print(x_batch.shape)
        y_batch = y_batch

        # Вычисляем функцию ошибки (log loss - кроссэнтропия)
        logits = x_batch @ W
        probabilities = torch.exp(logits) / torch.exp(logits).sum(dim=1, keepdim=True)

        loss = -torch.log(probabilities[range(batch_size), y_batch]).mean()

        history.append(loss.item())

        # Вычисление градиента
        loss.backward()

        # Шаг градиентного спуска
        grad = W.grad
        with torch.no_grad():
            W -= lr * grad
        W.grad.zero_()

    print(f'{i+1},\t loss: {history[-1]}')

import  matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))

plt.plot(history)

plt.show()

from sklearn.metrics import accuracy_score

acc = 0
batches = 0

for x_batch, y_batch in test_loader:
    batches += 1

    x_batch = x_batch.view(x_batch.shape[0], -1)
    y_batch = y_batch

    preds = torch.argmax(x_batch @ W, dim=1)
    acc += (preds==y_batch).cpu().numpy().mean()

    print(f'Test accuracy {acc / batches:.3}')
"""""

"""=============================="""

# Sequential подход

import torch.nn as nn
from torchsummary import summary

features = 784
classes = 10

# nn.Sequential - базовый класс на вход которого принимаются слои НС и функции активации друг за другом

model = nn.Sequential(
    nn.Linear(features, 64),
    nn.ReLU(),
    nn.Linear(64, classes)
)

summary(model, (features,), batch_size=228)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

epochs = 3
history = []

for i in range(epochs):
    for x_batch, y_batch in train_loader:

        # Загружаем данные
        x_batch = x_batch.view(x_batch.shape[0], -1)
        y_batch = y_batch

        # Проводим прямой проход данных через нейронку и получаем итоговые выходные данные
        logits = model(x_batch)

        # Считаем ошибку
        loss = criterion(logits, y_batch)
        history.append(loss.item())

        # Считаем градиент
        optimizer.zero_grad()
        loss.backward()

        # Делаем шаг градиентного спуска
        optimizer.step()

    print(f'{i+1},\t loss: {history[-1]}')

import  matplotlib.pyplot as plt

plt.figure()
plt.plot(history)
plt.show()

