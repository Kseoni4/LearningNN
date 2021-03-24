import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import datasets, transforms

import sys

class Neuronka(nn.Module):
    def __init__(self):
        super(Neuronka, self).__init__()
        self.l0_input = nn.Linear((28 * 28), 200)
        self.l1_hidden = nn.Linear(200, 200)
        self.l2_out = nn.Linear(200, 10)

    def forward(self, x):
        # print(f" Входные значения \n {x}")

        x = torch.sigmoid(self.l0_input(x))
        # print(f" Выходные значения l0 \n {x}")

        x = torch.sigmoid(self.l1_hidden(x))
        # print(f" Выходные значения l1 \n {x}")

        x = self.l2_out(x)
        # print(f" Выходные значения взвешенной суммы l2\n {x}")

        return torch.sigmoid(x)

net = Neuronka()

net

learning_rate = 0.9
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
loss_func = nn.NLLLoss()
# TODO: Изучить и рассказать про NLLLoss

def create_nn(batch_size=200, learning_rate=0.01, epochs=10,
              log_interval=10):
    """ ЭТО НЕ ВАЖНО """

    print("Загружаем тренировчные данные")
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    print("Загружаем проверочные данные")
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    """ А ЭТО ВАЖНО """

    """ Описываем структуру нейросети """

    class Neuronka(nn.Module):
        def __init__(self):
            super(Neuronka, self).__init__()
            self.l0_input = nn.Linear((28 * 28), 200)  # Входной слой из 784 нейронов
            self.l1_hidden = nn.Linear(200, 200)  # Скрытый слой из 200 нейронов
            self.l2_out = nn.Linear(200, 10)  # Выходной слой из 10 нейронов на каждую цифру от 0 до 9

        """ Переопределяем функцию прямого распространения """

        def forward(self, x):

            print(f" Входные значения\n {x}")

            x = F.relu(self.l0_input(x))
            print(f" Выходные значения l0\n {x}")

            x = F.relu(self.l1_hidden(x))
            print(f" Выходные значения l1\n {x}")

            x = self.l2_out(x)
            print(f" Выходные значения взвешенной суммы l2\n {x}")

            activation = nn.LogSoftmax()

            print(f" Активация нейронов: {activation(x).data}")
            return activation(x)

    """ Создаём экземпляр нейронки """

    net = Neuronka()
    print(net)

    """ Определяем функцию обучения - SGD - Стохастический градиентный спуск
        А так же функцию стоимости ошибки """

    learning_rate = 0.5
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.3)
    loss_func = nn.NLLLoss()  # Функция стоимости ошибки

    """ Цикл обучения нейросети на тренировочных данных"""

    print(enumerate(train_loader))

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)

            data = data.view(batch_size, 28 * 28)
            optimizer.zero_grad()
            net_out = net(data)
            input()
            """
            # C = (a_out - a_target)
            loss = loss_func(net_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                sys.stdout.write('\rЭпоха обучения: {} [{}/{} ({:.0f}%)] \t Ошибка: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss))
"""
    # Тестовый прогон на проверочных данных после обучения НС
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        data = data.view(-1, 28 * 28)
        net_out = net(data)
        # sum up batch loss
        test_loss += loss_func(net_out, target)
        pred = net_out.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print()
    sys.stdout.write('\rTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


create_nn()