#ndarray - N-мерный массив
# https://www.dataquest.io/blog/numpy-cheat-sheet/ - шпаргалка по функциям Numpy
#атрибуты: shape - размер по каждому измерению
#dtype - тип данных
#ndim - размерность массивов

"""
1D - массив, список, вектор
2D - двумерный массив, матрица
3D - куб
4D и больше - тензоры
"""

import numpy as np
data = [1,2,3,4,5] # list

arr = np.array(data) # создание из обычного list

print(arr)
print(arr.shape)
print(arr.dtype)
print(arr.ndim)

arr2 = np.array([1,2,3,4,5]) # Создание через np.array

arr3 = np.array([1,2,3,4,5], dtype=np.float) # Создание через конструктор с явным указанием типа

print(arr3)
print(arr3.shape)
print(arr3.dtype)
print(arr3.ndim)
print(len(arr3)) # Длина массива
print(arr3.size) # Размерность массива

arr3 = arr3.astype(np.int32) # приведение к типу int32

print(arr3)
print(arr3.dtype)

arr4 = np.arange(0, 20, 1.5) #Создаёт массив с набором значений от 0 до 20 с шагом 1.5

arr5 = np.linspace(0,2,5) #Заполняет массив набором значений от 0 до 2 и количеством 5

arr6 = np.linspace(0,2,1000)

print(arr6)

random_arr = np.random.random((5,)) # Генерация списка из 5 случайных ДРОБНЫХ элементов от 0 до 1
random_arr2 = np.random.random_sample((5,)) # Генерация списка из 5 случайных ДРОБНЫХ элементов от 0 до 1 (?)
random_arr3 = (10 - -5) * np.random.random_sample((5,)) - 5 # Генерация списка из 5 случайных элементов по диапазону [-5; 10]
random_arr4 = np.random.randint(-5, 10, 10) #Создание массива из 10 рандомных чисел от -5 до 10
"""
    Формула (b-a) * np.random() + a, где b > a
"""

""" Операции """

arr = np.sqrt(arr) # Применяет к КАЖДОМУ элементу функцию квадратного корня

arr7 = arr+arr3 #Сложение массивов
print(arr7)

arr8 = arr ** 2 #Возведение всех элементов массива в квадрат

""" Агрегаты (max,min,mean,sum,std,median) """

print(arr8.max())
print(arr8.min())
print(arr8.mean())
print(arr8.sum())
print(arr8.std()) #Стандартное отклонение
print(np.median(arr8))

arr8 < 2 # Проверка элементов массива на условие

""" Манипуляции с массивами """

np.insert(arr8, 2, -20) #Вставка числа -20 в позицию 2, массива arr8

np.delete(arr8, 2) #Удаление элемента в позиции 2

np.sort(arr8) #Сортировка массива

arr9 = np.concatenate(arr7,arr8) #Склейка массивов

np.array_split(arr9, 3) #Деление одного списка на 3 отдельных списка

""" Индексы 1D """

index_arr = np.random.randint(0, 20, 10)

index_arr[0] = 12

print(index_arr[0])
print(index_arr[0:2]) #Вывод элементов от 0 до 1 (два элемента)
print(index_arr[::-1]) #Вывод массива в обратном порядке :: означает от 0 до последнего элемента с шагом в обратном порядке
print(index_arr[index_arr < 2]) #Выбрать все элементы меньше 2. Можно выводить любой сложности
index_arr[1:4] = 0 #Присвоить элементам с 1 до 4 (не включая 4) значение 0

""" Матрицы и ND массивы"""

matrix = np.array([(1,2,3),(4,5,6)], dtype=np.float64) #Создание матрицы. Через запятую указываются СТРОКИ, значения строк будут являться СТОЛБЦАМИ
matrix2 = np.array([(1,2,3),(4,5,6),(7,8,9)], dtype=np.float64) #Создание матрицы. Через запятую указываются СТРОКИ, значения строк будут являться СТОЛБЦАМИ

three_d_array = np.array([[1,2], [3,4],[5,6],[7,8]]) #3D массив

print(matrix.shape) #Форма матрицы (кол-во строк, кол-во столбцов)
print(matrix.ndim) #Размерность матрицы
print(matrix.size) #Размер матрицы - кол-во элементов

matrix.reshape(1,9) #Изменение формы - одна строка и 9 элементов - без изменения данных

matrix2.reshape(2,6) # Две строки по 6 элементов

matrix2 = np.resize(matrix2, (2,2)) #Изменение размера матрицы (вместо 3 на 3, 2 на 2), данные могут быть удалены

matrix3 = np.arange(16).reshape(2,8) #Сгенерировать 16 элементов и изменить форму на 2 строки по 8 элементов

""" Специальные матрицы """

np.zeros((2,3)) #Матрица нулей
np.ones((3,3)) #Матрица единиц
np.eye((5)) #Единичная матрица 5 порядка.
np.full((3,3),9) #Заполнение матрицы 3 на 3 элементами - 9

matrix.dot(matrix2) #Скалярное произведение.

""" Axis """
# axis 0 - ось строк (направление)
# axis 1 - ось столбцов (направление)

np.hstack((arr3,arr7)) #Соединение массивов как строки

np.vstack((arr3,arr9)) #Соединение массивов как столбцы

matrix4 = np.array([(1,2,3),(4,5,6),(7,8,9)])

matrix4 = matrix4.T # Транспонирование - стобцы и строки меняются местами

matrix4.flatten() # "Сжатие" в 1D массив

np.linalg.inv(matrix4) # Обратная матрица

np.linalg.det(matrix4) # Определитель

np.linalg.matrix_rank(matrix4) # Ранк матрицы

np.linalg.eig(matrix4) # Находит собственные числа и векторы матрицы

""" Доп.функции """

np.info(np.eye) # Справка. Как работает та или иная функция

np.loadtxt('')
np.genfromtxt()
np.savetxt('.txt')
np.savetxt('.csv')