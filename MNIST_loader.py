"""
mnist_loader
~~~~~~~~~~~~

Библиотека загрузки изображений из базы MNIST. Детали структур описаны в комментариях к ``load_data`` и
``load_data_wrapper``.  На практике, ``load_data_wrapper`` - это функция, которую обычно вызывает код НС. """

#### Библиотеки
# Стандартные
import pickle
import gzip

# Сторонние
import numpy as np

def load_data():
    """Вернуть данные MNIST в виде кортежа, содержащего обучающие, подтверждающие и проверочные данные.
    ``training_data`` возвращается как кортеж с двумя вхождениями. Первое содержит сами картинки. Это numpy ndarray с
    50 000 элементами. Каждый элемент – это в свою очередь numpy ndarray с 784 значениями, представляющими 28 * 28 =
    784 пикселя одного изображения MNIST. Второе – это numpy ndarray с 50 000 элементами. Эти элементы – цифры от 0
    до 9 для соответствующих изображений, содержащихся в первом вхождении. ``validation_data`` и ``test_data``
    похожи, только содержат по 10 000 изображений. Это удобный формат данных, но для использования в НС полезно будет
    немного изменить формат ``training_data``. Это делается в функции-обёртке ``load_data_wrapper()``.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Вернуть кортеж, содержащий ``(training_data, validation_data, test_data)``. На основе ``load_data``,
    но формат удобнее для использования в нашей реализации НС. В частности, ``training_data`` - это список из  50 000
    кортежей из двух переменных, ``(x, y)``.  ``x`` - это 784-размерный numpy.ndarray, содержащий входящее
    изображение. ``y`` - это 10-мерный numpy.ndarray, представляющий единичный вектор, соответствующий правильной
    цифре для ``x``. ``validation_data`` и ``test_data`` - это списки, содержащие по 10 000 кортежей из двух
    переменных, ``(x, y)``.  ``x`` - это 784-размерный numpy.ndarray, содержащий входящее изображение, а ``y`` - это
    соответствующая классификация, то есть, цифровые значения (целые числа), соответствующие ``x``. Очевидно,
    это означает, что для тренировочных и подтверждающих данных мы используем немного разные форматы. Они оказываются
    наиболее удобными для использования в коде НС. """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Вернуть 10-мерный единичный вектор с 1.0 в позиции j и нулями на остальных позициях. Это используется для
    преобразования цифры (0..9) в соответствующие выходные данные НС. """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e