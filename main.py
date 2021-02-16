import numpy as np
import sys


class PartyNN(object):

    # Конструктор класса
    def __init__(self, learning_rate=0.1):
        self.weights_0_1 = np.random.normal(0.0, 2 ** -0.5,
                                            (2, 3))  # Определение 1D вектора весов от 0 к 1 уровню (2 нейрона, 3 веса)
        self.weights_1_2 = np.random.normal(0.0, 1,
                                            (1, 2))  # Определение 1D вектора весов от 1 к 2 уровню (1 нейрон, 2 веса)
        self.sigmoid_mapper = np.vectorize(
            self.sigmoid)  # Применение ко всем значениям 1D вектора нейронов функции сигмоиды
        self.learning_rate = np.array([learning_rate])  # Создаём массив из значения LR

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Метод прямого распространения слева направо
    def predict(self, inputs):
        """ Слой 0 и 1 """
        inputs_1 = np.dot(self.weights_0_1,
                          inputs)  # Скалярное произведение 1D вектора весов на вектор входных значений слоя 0
        outputs_1 = self.sigmoid_mapper(inputs_1)  # Применение к нейронам слоя 1 сигмоиды

        """ Слой 1 и 2 """
        inputs_2 = np.dot(self.weights_1_2,
                          outputs_1)  # Скалярное произведение 1D вектора весов на вектор значений нейронов слоя 1
        outputs_2 = self.sigmoid_mapper(inputs_2)  # Применение к нейронам слоя 2 функции сигмодиы
        return outputs_2  # Значение выходного нейрона

    # Обратное распространение
    def train(self, inputs, expected_predict):
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)
        actual_predict = outputs_2[0]

        error_layer_2 = np.array([actual_predict - expected_predict])  # Подсчёт ошибки (без возведения в квадрат)
        gradient_layer_2 = actual_predict * (1 - actual_predict)  # Поиск градиента - производная сигмоиды
        weights_delta_layer_2 = error_layer_2 * gradient_layer_2  # Поиск дельты веса
        # numpy.dot - Скалярное произведение (если вектора) и произведение матриц (если это матрицы (array))
        # numpy.reshape - изменение формы массива без изменения данных (можно считать за транспонирование),
        # в данном случае 1 строка с количеством выходных значений нейронов слоя 1
        weights_1_2_old = self.weights_1_2
        self.weights_1_2 -= (np.dot(weights_delta_layer_2,
                                    outputs_1.reshape(1, len(outputs_1)))) * self.learning_rate  # Рассчёт нового веса

        error_layer_1 = weights_delta_layer_2 * self.weights_1_2_old
        gradient_layer_1 = outputs_1 * (1 - outputs_1)
        weights_delta_layer_1 = error_layer_1 * gradient_layer_1
        self.weights_0_1 -= np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T * self.learning_rate


# Среднеквадратичное отклонение
def MSE(y, Y):
    return np.mean((y - Y) ** 2)


train = [
    ([0, 0, 0], 0),
    ([0, 0, 1], 1),
    ([0, 1, 0], 0),
    ([0, 1, 1], 0),
    ([1, 0, 0], 1),
    ([1, 0, 1], 1),
    ([1, 1, 0], 0),
    ([1, 1, 1], 0),
]

# learning_rate = 0.05
epochs = 5000
learning_rate = 0.08

network = PartyNN(learning_rate=learning_rate)

for e in range(epochs):
    inputs_ = []
    correct_predictions = []
    for input_stat, correct_predict in train:
        network.train(np.array(input_stat), correct_predict)
        inputs_.append(np.array(input_stat))
        correct_predictions.append(np.array(correct_predict))

    train_loss = MSE(network.predict(np.array(inputs_).T), np.array(correct_predictions))
    sys.stdout.write("\rProgress: {}, Training loss: {}".format(str(100 * e / float(epochs))[:4], str(train_loss)[:5]))

for input_stat, correct_predict in train:
    print("\nFor input: {} the prediction is: {}, expected: {}".format(
        str(input_stat),
        str(network.predict(np.array(input_stat)) > .5),
        str(correct_predict == 1)))

for input_stat, correct_predict in train:
    print("\nFor input: {} the prediction is: {}, expected: {}".format(
        str(input_stat),
        str(network.predict(np.array(input_stat))),
        str(correct_predict == 1)))
