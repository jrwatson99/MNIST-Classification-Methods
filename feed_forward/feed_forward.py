import numpy as np
import torch
import torch.nn.functional as F
from mnist import MNIST

#input 1x784 (28x28) image
INPUT_SIZE = 784
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def calc_accuracy(model, x_test, y_test):
    test_predictions = model(x_test)
    num_correct = torch.sum(test_predictions.argmax(dim=1) == y_test)
    accuracy = (float(num_correct) / len(y_test)) * 100
    print("The model accuracy over the test set is " + str(accuracy) + "%")

class DigitClassifierOneHidden(torch.nn.Module):
    def __init__(self):
        super(DigitClassifierOneHidden, self).__init__()
        self.linear1 = torch.nn.Linear(784, 392)
        self.linear2 = torch.nn.Linear(392, 10)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        #x = torch.nn.LogSoftmax(dim=1)(x)
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

    print(x_train.size())
    print(y_train.size())
    print(x_test.size())
    print(y_test.size())

    model = DigitClassifierOneHidden().to(device)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(5000):
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        if epoch % 100 == 99:
            print(epoch + 1, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    calc_accuracy(model, x_test, y_test)