import MNIST_loader
import NumberClassNN

training_data, validation_data, test_data = MNIST_loader.load_data_wrapper()

net = NumberClassNN.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)