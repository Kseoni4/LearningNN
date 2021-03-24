import numpy as np
import sys as s

class GoToTheUnickNetwork(object):
    """

    Смещение b = 0

    Вектор нейронов на нулевом слое - а0

    Вектор нейронов на скрытом слое - а1

    Вектор нейронов на выходном слое - а2

    а1 = sig(a0 * W0 + b)

    W0 - значения весов между слоями 0 и 1

    W1 - значения весов между слоями 1 и 2

    LR = 0.1

    sig = 1 / (1 + exp^(-x))

    """

    """ Для более детального просмотра этапов работы НС, просто раскоментируйте print и input """

    def __init__(self, lr=0.1):
        self.W0 = np.random.normal(0.0, 2 ** -1, (2, 3))  # W0 - веса между а0 и а1
        # normal - это случайное нормальное распределение от -0.5 до 0.5
        # (2, 3) - размерность: 2 строки, 3 столбца, потому что 2 нейрона на скрытом слое по три веса на каждый нейрон

        #print("Вектор весов W0 = ", self.W0)
        #input()

        self.W1 = np.random.normal(0.0, 1, (1, 2)) # W1 - веса между а1 и а2

        #print("Вектор весов W1 = ", self.W1)
        #input()

        self.sig = np.vectorize(self.sigmoid) # Функция, которая применяет к каждому элементу вектора сигмоиду

        self.learning_rate = np.array([lr]) # Шаг обучения

        # Объявление "пустых" векторов из нейронов
        self.a1 = np.vectorize
        self.a2 = np.vectorize

    # Функция сигмоиды
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Идём слева направо, предсказываем результат
    def predict(self, a0):
        #print("Входные данные: ", a0)
        #input()

        #Применяем формулу взвешенной суммы в сигмоиду для всех нейронов скрытого слоя
        self.a1 = self.sig(np.dot(self.W0, a0))

        #print("Значения нейронов скрытого слоя sig(W0 * a0) = ", self.a1)
        #input()

        #Применяем формулу взвешенной суммы в сигмоиду для всех нейронов выходного слоя
        self.a2 = self.sig(np.dot(self.W1, self.a1))

        #print("Значение выходного слоя sig(W1 * a1) = ", self.a2)
        #input()

        #Выводим значение выходного слоя
        return self.a2

    # Учим нейронку справа налево
    def train(self, a0, atarget):

        #Записываем текущий вывод нейронки
        a2_current = self.predict(a0)

        #Считаем производную ошибки от текущего вывода
        C2 = np.array([a2_current - atarget])  # dC/da2

        #print("Ошибка обучения C2 = {} - {} = {}".format(str(a2_current), str(atarget), str(a2_current - atarget)))
        #input()

        d_sig2 = a2_current * (1 - a2_current)  # da2/dz2

        #print("Градиент стоимости dC = ", d_sig2)
        #input()

        dWeights2 = C2 * d_sig2

        W1_old = self.W1

        self.W1 = W1_old - (np.dot(dWeights2, self.a1.reshape(1, len(self.a1))) * self.learning_rate)

        #print("Новые веса W1 = ", self.W1)
        #input()

        """ Веса между а0 и а1 """

        C1 = dWeights2 * W1_old  # dC/da2

        #print("Ошибка обучения C1 = {} * {} = {}".format(str(dWeights2), str(W1_old), str(C1)))
        #input()

        d_sig1 = self.a1 * (1 - self.a1)  # da2/dz2

        #print("Градиент стоимости dC = ", d_sig2)
        #input()

        dWeights1 = C1 * d_sig1

        W0_old = self.W0

        self.W0 = W0_old - np.dot(a0.reshape(len(a0), 1), dWeights1).T * self.learning_rate

        #print("Новые веса W0 = ", self.W0)
        #input()


# Входные значения а0 и ожидаемый вывод
input_data = [
    ([0, 0, 0], 0),
    ([0, 0, 1], 1),
    ([0, 1, 0], 0),
#    ([0, 1, 1], 0),
    ([1, 0, 0], 0),
#    ([1, 0, 1], 1),
#    ([1, 1, 0], 0),
#    ([1, 1, 1], 1),
]


# Среднеквадратчное отклонение
def MSE(y, Y):
    return np.mean((y - Y) ** 2)


# Количество итераций предсказаний и обучения (эпох)
epochs = 5000

# Создаём объект нейросети
network = GoToTheUnickNetwork()

""" 
  Каждую эпоху нейросеть предсказывает результат и обучается по всем входным данным, уменьшая ошибку обучения
"""
for e in range(epochs):

    a0_ = []
    correct_predictions = []
    for inputs, a_targets in input_data:
        network.train(np.array(inputs), a_targets)
        a0_.append(np.array(inputs))
        correct_predictions.append(np.array(a_targets))

    train_loss = MSE(network.predict(np.array(a0_).T), np.array(correct_predictions))

    s.stdout.write(f"\rЭпоха: {e} ошибка обучения: {train_loss}")

    #print("\rОшибка обучения: ", train_loss)


""" Можно проверить работу обученной НС. Рекомендую сначала раскоментировать print-ы в predict"""
print("\nВведите проверочные данные: ")
in_ = input().split(' ')

in_ = np.array(in_, dtype=np.float32)

print("Вывод нейросети: ", network.predict(np.array(in_)))