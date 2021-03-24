import matplotlib.pyplot as plt
import numpy as np
import torch
import sys as s

N = 350  # Количество точек в каждом классе
D = 2  # Размерность
K = 3  # Количество классов
X = np.zeros((N * K, D))  # Матрица данных, где каждая строчка - один экземпляр
y = np.zeros(N * K, dtype='uint8')  # Матрица "легенды"

""" Генерация данных """

for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # Радиус по котором распределяются точки
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # Коэффциент тэта
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

# Визуализация данных:

plt.figure(figsize=(10, 8))

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.title("Спираль точек", fontsize=15)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.ioff()
plt.show()

from torch import nn
import torch.nn.functional as F  # Мат.Функции работы с НС

# print(nn.Module.__doc__)

# torch.nn.Module - основной абстратный класc для построения НС

# torch.nn.Sequential - класс для работы с НС. Шаблон для построения простой НС

# N - размер батча (или набора примеров) batch_size
# D_in - размерность входа (количество входных нейронов)
# H - размерность скрытых слоёв (количество скрытых нейронов)
# D_out - размерность выходного слоя (количество классов)

N, D_in, H, D_out = 64, 2, 500, 3

t_lnetwork = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),  # Линейный скрытый слой
    torch.nn.ReLU(),  # Активация
    torch.nn.Linear(H, D_out),  # Линейный выходной слой
    torch.nn.Softmax()  # Активация
)

# print("Weight shapes:", [w.shape for w in t_lnetwork.parameters()])

# Создаём блок входных данных
x_batch = torch.tensor((X[:3]), dtype=torch.float32)
y_batch = torch.tensor((y[:3]), dtype=torch.float32)

y_predicted = t_lnetwork(x_batch)[0]

# print("y_predicted = ", y_predicted)

from torch.autograd import Variable


# Генерация случайной выборки входных данных
def batch_gen(X, y, batch_size=128):
    idx = np.random.randint(X.shape[0], size=batch_size)
    X_batch = X[idx]
    y_batch = y[idx]

    return Variable(torch.FloatTensor(X_batch)), Variable(torch.LongTensor(y_batch))


# print(batch_gen(X, y)[1].shape)

# print(t_lnetwork.forward(batch_gen(X, y)[0]))

""" Обучение НС """

loss_fn = torch.nn.CrossEntropyLoss(size_average=False)  # Кроссэнтропия (сумма -log правильного класса)

learning_rate = 0.1

optimizer = torch.optim.SGD(t_lnetwork.parameters(), lr=learning_rate)

epoch = 20000

for t in range(epoch):
    # Выбираем данные из общего пула
    x_batch, y_batch = batch_gen(X, y)

    # Делаем предсказание
    y_predicted = t_lnetwork(x_batch)

    # Подсчёт ошибки
    loss = loss_fn(y_predicted, y_batch)
    s.stdout.write(f"\r{t} {loss.data}")

    # Зануляем градиент
    optimizer.zero_grad()

    # Считаем градиент
    loss.backward()

    # Считаем веса снова
    optimizer.step()

h = 0.02

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
grid_tenzor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

Z = t_lnetwork(torch.autograd.Variable(grid_tenzor))
Z = Z.data.numpy()
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))

plt.contourf(xx, yy, Z, cmap=plt.cm.get_cmap("rainbow"), alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.get_cmap("rainbow"))

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title("Спираль точек", fontsize=15)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.show()
