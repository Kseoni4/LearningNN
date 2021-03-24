import numpy as np
import sys

class GoToTheUnickNetwork(object):
    """ Вводные данные

        Смещение b = 0

        Вектор значений входных данных слоя а0

        Вектор значений скрытого слоя а1
        Значения вектора а1 = sig(а0 * W0 + b)

        Вектор значений выходного слоя а2
        Значения a2 = sig(a1 * W1 +b)

        Вектор весов W0 и W1 для связи нейронов между а0 и а1, а так же между а1 и а2 соответственно
        Значения весов устанавливаются путём создания
        Learning rate по умолчанию 0.1

        Формула сигмоиды в программном виде: 1 / (1 + exp(-x))

    """

    # Конструктор класса
    def __init__(self, learning_rate=0.1):
        sys.stdout.write("Нейросеть - идём на пары?")

        self.weights_0_1 = np.random.normal(0.0, 2 ** -1,
                                            (2,
                                             3))  # Определение 2D вектора весов от 0 к 1 уровню (2 нейрона, 3 веса или 2 строки, 3 столбца)

        print("Вектор весов W0 = ", self.weights_0_1)
        input()

        self.weights_1_2 = np.random.normal(0.0, 1,
                                            (1,
                                             2))  # Определение 1D вектора весов от 1 к 2 уровню (1 нейрон, 2 веса или 1 строка 2 столбка)

        print("Вектор весов W1 = ", self.weights_1_2)
        input()

        self.sigmoid_mapper = np.vectorize(
            self.sigmoid)  # Применение ко всем значениям 1D вектора нейронов функции сигмоиды
        self.learning_rate = np.array([learning_rate])  # Создаём массив из значения LR

        """ Эксперименты """
        # Векторы выходных значений на слое 1 и выходном слое 2
        self.outputs_1 = np.vectorize
        self.outputs_2 = np.vectorize

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Метод прямого распространения слева направо
    def predict(self, inputs):
        print("Входные данные: {}".format(str(inputs)))
        input()

        """ Слой 0 и 1 """
        inputs_1 = np.dot(self.weights_0_1,
                          inputs)  # Скалярное произведение 1D вектора весов на вектор входных значений слоя 0

        print("z0 = ", inputs_1, "\n")
        input()

        self.outputs_1 = self.sigmoid_mapper(inputs_1)  # Применение к нейронам слоя 1 сигмоиды

        print("a1 = ", self.outputs_1, "\n")
        input()

        """ Слой 1 и 2 """
        inputs_2 = np.dot(self.weights_1_2,
                          self.outputs_1)  # Скалярное произведение 1D вектора весов на вектор значений нейронов слоя 1

        print("z1 = ", inputs_2, "\n")
        input()

        self.outputs_2 = self.sigmoid_mapper(inputs_2)  # Применение к нейронам слоя 2 функции сигмодиы

        print("a2 = ", self.outputs_2, "Предсказание: ", (self.outputs_2 > 0.5), "\n")
        input()

        return self.outputs_2  # Значение выходного нейрона

    # Обратное распространение
    def train(self, inputs, expected_predict):
        """inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)"""
        actual_predict = self.predict(inputs)

        """ gradC = dC/dw = dC/da_out * da_out/z1 * dz1/w1 """

        error_layer_2 = np.array([actual_predict - expected_predict])  # Подсчёт ошибки (без возведения в квадрат)

        print("Ошибка обучения {} - {} = {}".format(str(actual_predict), str(expected_predict), str(error_layer_2)))

        gradient_layer_2 = actual_predict * (1 - actual_predict)  # Поиск градиента - производная сигмоиды

        print("Градиент ошибки: {} * (1 - {}) = {}".format(str(actual_predict),str(actual_predict),str(gradient_layer_2)))

        weights_delta_layer_2 = error_layer_2 * gradient_layer_2  # Поиск дельты веса
        # numpy.dot - Скалярное произведение (если вектора) и произведение матриц (если это матрицы (array))
        # numpy.reshape - изменение формы массива без изменения данных (можно считать за транспонирование),
        # в данном случае 1 строка с количеством выходных значений нейронов слоя 1
        weights_1_2_old = self.weights_1_2
        self.weights_1_2 -= (np.dot(weights_delta_layer_2,
                                    self.outputs_1.reshape(1, len(
                                        self.outputs_1)))) * self.learning_rate  # Рассчёт нового веса

        error_layer_1 = weights_delta_layer_2 * self.weights_1_2
        gradient_layer_1 = self.outputs_1 * (1 - self.outputs_1)
        weights_delta_layer_1 = error_layer_1 * gradient_layer_1
        self.weights_0_1 -= np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T * self.learning_rate


# Среднеквадратичное отклонение
def MSE(y, Y):
    return np.mean((y - Y) ** 2)


# Слева направо - те же значения, которые сверху вниз
train = [
    ([0, 0, 0], 0),
    ([0, 0, 1], 1),
    ([0, 1, 0], 0),
    ([0, 1, 1], 0),
    ([1, 0, 0], 1),
    ([1, 0, 1], 1),
    ([1, 1, 0], 0),  # Важный предмет, плохая погода, друг не идёт -> не идём на пару
    ([1, 1, 1], 0),
]

# learning_rate = 0.05
epochs = 3000
learning_rate = 0.1

network = GoToTheUnickNetwork(learning_rate=learning_rate)

for e in range(epochs):
    print("\rЭпоха: ", e)
    inputs_ = []
    correct_predictions = []
    for input_stat, correct_predict in train:
        network.train(np.array(input_stat), correct_predict)
        inputs_.append(np.array(input_stat))
        correct_predictions.append(np.array(correct_predict))

    train_loss = MSE(network.predict(np.array(inputs_).T), np.array(correct_predictions))
    sys.stdout.write(
        "\rПрогресс обучения: {}, Ошибка обучения: {}".format(str(100 * e / float(epochs))[:4], str(train_loss)[:5]))

for input_stat, correct_predict in train:
    print("\nДля вводных данных: {} вывод нейросети: {}, ожидалось: {}".format(
        str(input_stat),
        str(network.predict(np.array(input_stat)) > .5),
        str(correct_predict == 1)))

for input_stat, correct_predict in train:
    print("\nДля вводных данных: {} вывод нейросети: {}, ожидалось: {}".format(
        str(input_stat),
        str(network.predict(np.array(input_stat))),
        str(correct_predict == 1)))
