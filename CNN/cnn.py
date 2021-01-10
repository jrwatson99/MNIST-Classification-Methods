import numpy as np
import torch
import torch.nn.functional as F
from mnist import MNIST

#input 1x784 (28x28) image
INPUT_SIZE = 784
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

class DigitClassifierOneHidden(torch.nn.Module):
    def __init__(self):
        super(DigitClassifierOneHidden, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(5, 25, kernel_size=5, stride=1, padding=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = torch.nn.Linear(7*7*25, 7*7*10)
        self.linear2 = torch.nn.Linear(7*7*10, 10)

    def forward(self, x):
        #x = x.reshape(x.size(0), 1, 28, 28)
        x = torch.nn.Dropout(0.1)(x)
        x = self.conv1(x)
        x = torch.nn.ReLU()(x)
        x = self.pool(x)
        x = torch.nn.Dropout(0.1)(x)
        x = self.conv2(x)
        x = torch.nn.ReLU()(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        #x = torch.nn.LogSoftmax(dim=1)(x)
        return x

class InceptionLayer(torch.nn.Module):
    def __init__(self, length):
        super(InceptionLayer, self).__init__()

        self.length = length

        #self.conv_pre = torch.nn.Conv2d(1, 5, kernel_size=3, stride=1, padding=1)

        #self.batch_1 = torch.nn.BatchNorm2d(5)
        self.conv1_1_1 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1_1_2 = torch.nn.Conv2d(1, 5, kernel_size=1, stride=1, padding=0)
        self.conv1_2_1 = torch.nn.Conv2d(1, 5, kernel_size=1, stride=1, padding=0)
        self.conv1_2_2 = torch.nn.Conv2d(5, 10, kernel_size=3, stride=1, padding=1)
        self.conv1_3_1 = torch.nn.Conv2d(1, 5, kernel_size=1, stride=1, padding=0)
        self.conv1_3_2 = torch.nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=2)
        self.conv1_4 = torch.nn.Conv2d(1, 5, kernel_size=1, stride=1, padding=0)
        #ReLU
        self.batch_2 = torch.nn.BatchNorm2d(30)
        self.conv_post = torch.nn.Conv2d(30, 60, kernel_size=3, stride=1, padding=1)
        #ReLU
        self.batch_3 = torch.nn.BatchNorm2d(60)
        #ReLU
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = torch.nn.Linear(int(length*length*60/4), int(length*length*30/4))
        self.linear2 = torch.nn.Linear(int(length*length*30/4), int(length * length / 4))

    def forward(self, x):
        #x = x.reshape(x.size(0), 1, self.length, self.length)
        #x = self.conv_pre(x)
        #x = torch.nn.ReLU()(x)
        #x = self.batch_1(x)
        x1 = self.conv1_1_1(x)
        x2 = self.conv1_2_1(x)
        x3 = self.conv1_3_1(x)
        x1 = self.conv1_1_2(x1)
        x2 = self.conv1_2_2(x2)
        x3 = self.conv1_3_2(x3)
        x4 = self.conv1_4(x)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = torch.nn.ReLU()(x)
        x = self.batch_2(x)
        x = torch.nn.Dropout2d(0.1)(x)
        x = self.conv_post(x)
        x = torch.nn.ReLU()(x)
        x = self.batch_3(x)
        x = torch.nn.Dropout2d(0.1)(x)
        x = self.pool(x)

        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class DigitClassifierInception(torch.nn.Module):
    def __init__(self):
        super(DigitClassifierInception, self).__init__()

        self.inception1 = InceptionLayer(28)
        self.inception2 = InceptionLayer(14)
        self.linear1 = torch.nn.Linear(7*7, 10)
        #self.linear2 = torch.nn.Linear(10*10, 10)

    def forward(self, x):
        x = self.inception1(x)
        x = x.reshape(x.size(0), 1, 14, 14)
        x = self.inception2(x)
        x = self.linear1(x)
        #x = self.linear2(x)

        return x

    '''
        x = x.reshape(x.size(0), 1, 28, 28)
        #x = torch.nn.Dropout(0.1)(x)
        x = self.conv_pre(x)
        x = torch.nn.ReLU()(x)
        x = self.batch_1(x)
        x = self.pool(x)
        x1 = self.conv1_1_1(x)
        x2 = self.conv1_2_1(x)
        x3 = self.conv1_3_1(x)
        x1 = self.conv1_1_2(x1)
        x2 = self.conv1_2_2(x2)
        x3 = self.conv1_3_2(x3)
        x4 = self.conv1_4(x)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = torch.nn.ReLU()(x)
        x = self.batch_2(x)
        x = self.conv_post(x)
        x = torch.nn.ReLU()(x)
        x = self.batch_3(x)
        x = self.pool(x)

        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    '''


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
        #x = self.pool1(x)
        x = torch.nn.ReLU()(x)
        x = self.batch1(x)
        x = self.drop1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        #x = self.pool2(x)
        x = torch.nn.ReLU()(x)
        x = self.batch2(x)
        x = self.drop2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
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

    print(x_train.size())
    print(y_train.size())
    print(x_test.size())
    print(y_test.size())

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

'''
first model:
    1: 98.22
    2: 98.34
    3: 98.11
    
second model:
    98.45
    98.23
    98.45
third model:
    98.34
    98.26
    98.1
    
inception model:
    98.99
    99

32x3>32x5>P>BN>D2>64x3>64x5>P>BN>D2>FC>FC
    99.11
32x3>32x3>P>BN>D2>64x3>64x3>P>BN>D2>FC>FC
    99.11
32x3>32x3>32x5>P>BN>D2>64x3>64x3>64x5>P>BN>D2>FC>FC
    99.28
32x3>32x3>32x5>P>BN>D4>64x3>64x3>64x5>P>BN>D4>FC>FC
    99.03
'''