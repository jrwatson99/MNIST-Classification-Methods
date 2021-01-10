import numpy as np
import torch
import torch.nn.functional as F
from mnist import MNIST

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def calc_accuracy(model, x_test, y_test):
    model.eval()
    batch_size = 200
    num_batches = int(x_test.size(0) / batch_size)
    num_correct = 0
    for batch in range(num_batches):
        test_predictions = model(x_test[batch * batch_size:(batch + 1) * batch_size])
        num_correct += torch.sum(test_predictions.argmax(dim=1) == y_test[batch * batch_size:(batch + 1) * batch_size])
    accuracy = (float(num_correct) / len(y_test)) * 100
    print("The model accuracy over the test set is " + str(accuracy) + "%")
    model.train()

class DigitClassifierCNN(torch.nn.Module):
    def __init__(self):
        super(DigitClassifierCNN, self).__init__()

        self.conv1_1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3)
        self.conv1_3 = torch.nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = torch.nn.Dropout(0.4)
        self.batch1 = torch.nn.BatchNorm2d(32)
        self.conv2_1 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
        self.conv2_3 = torch.nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = torch.nn.Dropout(0.4)
        self.batch2 = torch.nn.BatchNorm2d(64)
        self.fc1 = torch.nn.Linear(7*7*64, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = torch.nn.ReLU()(x)
        x = self.batch1(x)
        x = self.drop1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = torch.nn.ReLU()(x)
        x = self.batch2(x)
        x = self.drop2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def open_menu():
    #load the train and test data
    mnist = MNIST('.\\MNIST dataset\\')
    x_train, y_train = mnist.load_training()
    x_test, y_test = mnist.load_testing()
    x_train = torch.tensor(np.asarray(x_train).astype(np.float32)).to(device)
    y_train = torch.tensor(np.asarray(y_train).astype(np.int64)).to(device)
    x_test = torch.tensor(np.asarray(x_test).astype(np.float32)).to(device)
    y_test = torch.tensor(np.asarray(y_test).astype(np.int64)).to(device)

    x_train = x_train.reshape(x_train.size(0), 1, 28, 28)
    x_test = x_test.reshape(x_test.size(0), 1, 28, 28)

    model = DigitClassifierCNN().to(device)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    learn = None

    batch_size = 200
    num_batches = int(60000 / batch_size)

    model.train()

    torch.cuda.empty_cache()
    for epoch in range(20):
        for batch in range(num_batches):
            y_pred = model(x_train[batch * batch_size:(batch + 1) * batch_size])
            loss = loss_fn(y_pred, y_train[batch * batch_size:(batch + 1) * batch_size])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 4:
            print(epoch + 1, loss.item())
    calc_accuracy(model, x_train, y_train)
    calc_accuracy(model, x_test, y_test)