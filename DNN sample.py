# Импорт требуемых библиотек
import torch as tch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix

# Создание 500 результатов с помощью randn | Присвоен тег 0
X1 = tch.randn(3000, 32)
# Создание ещё 500 немного отличающихся от X1 результатов с помощью randn | Присвоен тег 0
X2 = tch.randn(3000, 32) + 0.5
# Комбинирование X1 и X2
X = tch.cat([X1, X2], dim=0)

# Создание 1000 Y путём комбинирования 50% 0 и 50% 1
Y1 = tch.zeros(3000, 1)
Y2 = tch.ones(3000, 1)
Y = tch.cat([Y1, Y2], dim=0)

# Создание индексов данных для обучения и расщепления для подтверждения:
batch_size = 16
validation_split = 0.2  # 20%
random_seed = 2019

# Перемешивание индексов
dataset_size = X.shape[0]
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)

# Создание индексов для обучения и подтверждения
train_indices, val_indices = indices[split:], indices[:split]
# Создание набора данных для обучения и подтверждения
X_train, x_test = X[train_indices], X[val_indices]
Y_train, y_test = Y[train_indices], Y[val_indices]

# Отрисовка формы каждого набора данных
print("X_train.shape:", X_train.shape)
print("x_test.shape:", x_test.shape)
print("Y_train.shape:", Y_train.shape)
print("y_test.shape:", y_test.shape)


# Создание нейронной сети с 2 скрытыми слоями и 1 выходным слоем
# Скрытые слои имеют 64 и 256 нейронов
# Выходные слои имеют 1 нейрон

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu1 = nn.ReLU()  # A(x) = max(0,x)
        self.fc2 = nn.Linear(64, 256)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(256, 1)
        self.final = nn.Sigmoid()

    def forward(self, x):
        op = self.fc1(x)
        op = self.relu1(op)
        op = self.fc2(op)
        op = self.relu2(op)
        op = self.out(op)
        y = self.final(op)
        return y


model = NeuralNetwork()
loss_function = nn.BCELoss()  # Бинарные потери по перекрёстной энтропии
optimizer = tch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 16

for epoch in range(num_epochs):
    train_loss = 0.0

    # Начало явного обучения модели
    model.train()

    for i in range(0, X_train.shape[0], batch_size):
        # Извлечение пакета обучения из X и Y
        input_data = X_train[i:min(X_train.shape[0], i + batch_size)]
        labels = Y_train[i:min(X_train.shape[0], i + batch_size)]

        # Установление градиентов на 0 перед применением алгоритма  обратного распространения ошибок
        optimizer.zero_grad()

        # Дальнейшая передача данных
        output_data = model(input_data)

        # Подсчёт потерь
        loss = loss_function(output_data, labels)

        # Применение алгоритма  обратного распространения ошибок
        loss.backward()

        # Обновление весов
        optimizer.step()

        train_loss += loss.item() * batch_size

    print("Epoch: {} - Loss:{:.4f}".format(epoch + 1, train_loss / X_train.shape[0]))

# Прогнозирование
y_test_pred = model(x_test)
a = np.where(y_test_pred > 0.5, 1, 0)
print(confusion_matrix(y_test, a))
