from ML_packages import *

#input -> feature_extractor: (Conv_5x5 -> Pool_2x2 -> Conv_5x5 -> Pool_2x2) -> feature_classification -> Output

device = 'cuda' if pt.cuda.is_available() else 'cpu'
print(device)

"""transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=False, transform=transform)
trainloader = pt.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=False, transform=transform)
testloader = pt.utils.data.DataLoader(trainset, batch_size=4, shuffle=False)

classes = tuple(str(i) for i in range(10))
"""
#print(trainloader.dataset.train_data.shape)

#print(testloader.dataset.test_data.shape)

#print(trainloader.dataset.train_data[0])

#numpy_img = trainloader.dataset.train_data[0].numpy()

#plt.imshow(numpy_img, cmap='gray')
#plt.show()

"""class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(4 * 4 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 4 * 4 * 16)
        #print(x.shape)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)

        return x
"""

#from tqdm import tqdm_notebook

# network = CNNet().to(device)

"""
fig = plt.figure(figsize=(10,7))

ax = fig.add_subplot(1,1,1)

for epoch in range(2):

    running_loss = 0.0
    for i, batch in enumerate(trainloader):

        X_batch, y_batch = batch
        optimizer.zero_grad()

        y_pred = network(X_batch.to(device))
        loss = loss_fn(y_pred, y_batch.to(device))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            losses.append(running_loss)
            running_loss = 0.0
    ax.clear()
    ax.plot(np.arange(len(losses)), losses)
    plt.show()

print('Обучение завершено')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with pt.no_grad():
    for data in testloader:
        images, labels = data
        y_pred = network(images.to(device))
        _, predicted = pt.max(y_pred, 1)

        c = predicted.cpu().detach() == labels
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]
    ))"""


transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = pt.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)
testloader = pt.utils.data.DataLoader(trainset, batch_size=128, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CNNet_CIFAR(nn.Module):
    def __init__(self):
        super(CNNet_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(6 * 6 * 128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):

        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))

        #print(x.shape)
        x = x.view(-1, 6 * 6 * 128)


        #print(x.shape)
        x = f.relu(self.fc1(x))

        x = f.relu(self.fc2(x))

        x = self.fc3(x)
        return x


network = CNNet_CIFAR()

loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-4

optimizer = optim.Adam(network.parameters(), lr=learning_rate)

losses = []

for epoch in range(2):

    running_loss = 0.0
    for i, batch in enumerate(trainloader):

        X_batch, y_batch = batch

        optimizer.zero_grad()

        y_pred = network(X_batch.to(device))

        loss = loss_fn(y_pred, y_batch.to(device))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f'
                  % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Обучение завершено')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with pt.no_grad():
    for data in testloader:
        images, labels = data
        y_pred = network(images.to(device))
        _, predicted = pt.max(y_pred, 1)

        c = predicted.cpu().detach() == labels
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]
    ))