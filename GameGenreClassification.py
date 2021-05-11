import torch as pt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import sys

class GenreClassificationNN(nn.Module):
    def __init__(self):
        super(GenreClassificationNN, self).__init__()

        self.l0_input = nn.Linear(15, 28)
        self.l1_hidden = nn.Linear(28, 28)
        self.l2_output = nn.Linear(28, 7)

    def forward(self, x):

        x = pt.sigmoid(self.l0_input(x))

        x = pt.sigmoid(self.l1_hidden(x))

        x = pt.sigmoid(self.l2_output(x))

        return pt.softmax(x, 1)

    def trainNN(self, train_data, epochs, lr=0.1):

        optimizer = optim.Adam(self.parameters(), lr)

        loss_func = nn.CrossEntropyLoss()

        loss = loss_func

        for e in range(epochs):
            for features, target in train_data:

                features = Variable(features)

                target = Variable(target)

                target = target.view(-1, 1)

                optimizer.zero_grad()

                NOut = self.forward(features)

                loss = loss_func(NOut, target)

                loss.backward()

                optimizer.step()

            sys.stdout.write(f"\r epoch {e+1}/{epochs}, loss = {loss.item():.05f}")
        print()

    def FindGenre(self, inputData):
        out = self.forward(inputData)

        print(out)

        outdict = (["RTS", out[0]],
                   ["FPS", out[1]],
                   ["RPG", out[2]],
                   ["TBS", out[3]],
                   ["Action", out[4]],
                   ["RogueLike", out[5]],
                   ["JRPG",out[6]])

        outg = list()

        print(outdict)

        for g in outdict:
            #print(g[:][1])
            if(g[:][1] >= 0.5):
                outg += list((g[:][0], g[:][1]))

        if(len(outg) > 0):
            print(outg)
            print(outg[0])
        else:
            print("Undefined genre")
"""
0. Постройка базы
1. Наём юнитов
2. Добыча ресурсов
3. Вид от первого лица
4. Вид от третьего лица
5. Вид сверху
6. Прокачка персонажа
7. Линейный сюжет
8. Нелинейный сюжет
9. Открытый мир
10. Пошаговые битвы
11. С противников падает лут
12. Крафтинг
13. Перманентная смерть
14. Процедурная генерация уровней

RTS, FPS, RPG, TBS, Action, Roguelike, JRPG
Action/RPG, FPS/RTS, FPS/RPG, Action/Strategy, Strategy/RPG

"""

features = [
    "Постройка базы",
    "Наём юнитов",
    "Добыча ресурсов",
    "Вид от первого лица",
    "Вид от третьего лица",
    "Вид сверху",
    "Прокачка персонажа",
    "Линейный сюжет",
    "Нелинейный сюжет",
    "Открытый мир",
    "Пошаговые битвы",
    "С противников падает лут",
    "Крафтинг",
    "Перманентная смерть",
    "Процедурная генерация уровней"
]

train_data_tuple = ([
    #0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
    [[1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0,  0,  0,  0,  0],[1, 0, 0, 0, 0, 0, 0]], #RTS
    [[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0],[0, 1, 0, 0, 0, 0, 0]], #FPS
    [[0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0,  1,  1,  0,  0],[0, 0, 1, 0, 0, 0, 0]], #RPG
    [[1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1,  0,  0,  0,  0],[0, 0, 0, 1, 0, 0, 0]], #TBS
    [[0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,  1,  0,  1,  1],[0, 0, 0, 0, 0, 1, 0]], #Roguelike
    [[0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1,  0,  0,  0,  0],[0, 0, 0, 0, 0, 0, 1]], #JRPG
    [[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,  1,  0,  0,  0],[0, 0, 0, 0, 1, 0, 0]]  #Action
])

tr_data = list()
targ_data = list()

for data in train_data_tuple:
    trd, td = data
    tr_data.append(trd)
    targ_data.append(td)


input_data = TensorDataset(pt.FloatTensor(tr_data), pt.FloatTensor(targ_data))
trainD = DataLoader(input_data, batch_size=1, shuffle=False)

def getTensorFromInput():
    outTensor = list()

    i = 0

    for feat in features:
        print(f"Есть {feat}? [1/0]")
        outTensor.append(float(input()))
        print(outTensor)
        i += 1
    outTensor = np.array(outTensor)

    outTensor = pt.FloatTensor(outTensor)

    return outTensor


GCNN = GenreClassificationNN()

GCNN.trainNN(trainD, 1000)

GCNN.FindGenre(getTensorFromInput())

