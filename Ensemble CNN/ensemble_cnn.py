import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from mnist import MNIST

# Highest accuracy achieved: 99.52 at 45 epochs%
# input 1x784 (28x28) image
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def calc_accuracy(model, loader):
    model.eval()
    with torch.no_grad():
        total = 0
        num_correct = 0
        for images, labels in loader:
            x_batch = images.to(device)
            y_batch = labels.to(device)

            y_pred = model(x_batch)
            total += y_batch.size(0)
            num_correct += (y_pred.argmax(dim=1) == y_batch).sum().item()
        accuracy = (float(num_correct) / total) * 100
        print("Accuracy " + str(accuracy) + "%")
    model.train()

def calc_ensemble_accuracy(model_a, model_b, model_c, loader):
    model_a.eval()
    model_b.eval()
    model_c.eval()

    with torch.no_grad():
        total = 0
        num_correct = 0
        for images, labels in loader:
            x_batch = images.to(device)
            y_batch = labels.to(device)

            y_pred_a = model_a(x_batch)
            y_pred_b = model_b(x_batch)
            y_pred_c = model_c(x_batch)

            y_pred_ensemble = (y_pred_a + y_pred_b + y_pred_c) / 3

            total += y_batch.size(0)
            num_correct += (y_pred_ensemble.argmax(dim=1) == y_batch).sum().item()
        accuracy = (float(num_correct) / total) * 100
        print("Accuracy " + str(accuracy) + "%")
    model_a.train()
    model_b.train()
    model_c.train()


class DigitClassifierOneHidden(torch.nn.Module):
    def __init__(self):
        super(DigitClassifierOneHidden, self).__init__()
        self.linear1 = torch.nn.Linear(784, 392)
        self.batch1 = torch.nn.BatchNorm1d(392)
        self.drop1 = torch.nn.Dropout(0.1)
        self.linear2 = torch.nn.Linear(392, 10)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = torch.nn.ReLU()(x)
        x = self.batch1(x)
        x = self.drop1(x)
        x = self.linear2(x)
        return x

class DigitClassifierSimpleCNN(torch.nn.Module):
    def __init__(self):
        super(DigitClassifierSimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch1 = torch.nn.BatchNorm2d(32)
        self.drop1 = torch.nn.Dropout(0.2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch2 = torch.nn.BatchNorm2d(64)
        self.drop2 = torch.nn.Dropout(0.2)
        self.linear1 = torch.nn.Linear(7*7*64, 128)
        self.linear2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = torch.nn.ReLU()(x)
        x = self.batch1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.nn.ReLU()(x)
        x = self.batch2(x)
        x = self.drop1(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class InceptionLayer(torch.nn.Module):
    def __init__(self, length):
        super(InceptionLayer, self).__init__()

        self.length = length

        # self.conv_pre = torch.nn.Conv2d(1, 5, kernel_size=3, stride=1, padding=1)

        # self.batch_1 = torch.nn.BatchNorm2d(5)
        self.conv1_1_1 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1_1_2 = torch.nn.Conv2d(1, 5, kernel_size=1, stride=1, padding=0)
        self.conv1_2_1 = torch.nn.Conv2d(1, 5, kernel_size=1, stride=1, padding=0)
        self.conv1_2_2 = torch.nn.Conv2d(5, 10, kernel_size=3, stride=1, padding=1)
        self.conv1_3_1 = torch.nn.Conv2d(1, 5, kernel_size=1, stride=1, padding=0)
        self.conv1_3_2 = torch.nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=2)
        self.conv1_4 = torch.nn.Conv2d(1, 5, kernel_size=1, stride=1, padding=0)
        # ReLU
        self.batch_2 = torch.nn.BatchNorm2d(30)
        self.conv_post = torch.nn.Conv2d(30, 60, kernel_size=3, stride=1, padding=1)
        # ReLU
        self.batch_3 = torch.nn.BatchNorm2d(60)
        # ReLU
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = torch.nn.Linear(int(length * length * 60 / 4), int(length * length * 30 / 4))
        self.linear2 = torch.nn.Linear(int(length * length * 30 / 4), int(length * length / 4))

    def forward(self, x):
        # x = x.reshape(x.size(0), 1, self.length, self.length)
        # x = self.conv_pre(x)
        # x = torch.nn.ReLU()(x)
        # x = self.batch_1(x)
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
        self.linear1 = torch.nn.Linear(7 * 7, 10)
        # self.linear2 = torch.nn.Linear(10*10, 10)

    def forward(self, x):
        x = self.inception1(x)
        x = x.reshape(x.size(0), 1, 14, 14)
        x = self.inception2(x)
        x = self.linear1(x)
        # x = self.linear2(x)

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
        self.conv1_2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_3 = torch.nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = torch.nn.Dropout(0.4)
        self.batch1 = torch.nn.BatchNorm2d(32)
        self.conv2_1 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = torch.nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = torch.nn.Dropout(0.4)
        self.batch2 = torch.nn.BatchNorm2d(64)
        self.fc1 = torch.nn.Linear(7 * 7 * 64, 128)
        self.fc2 = torch.nn.Linear(128, 100)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        # x = self.pool1(x)
        x = torch.nn.ReLU()(x)
        x = self.batch1(x)
        x = self.drop1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        # x = self.pool2(x)
        x = torch.nn.ReLU()(x)
        x = self.batch2(x)
        x = self.drop2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # load the train and test data
    batch_size = 200

    train_transform = transforms.Compose([
        transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                                 train=True,
                                                 transform=train_transform,
                                                 download=True)

    test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                                train=False,
                                                transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    model_cnn = DigitClassifierCNN().to(device)
    model_simple = DigitClassifierCNN().to(device)
    model_ff = DigitClassifierCNN().to(device)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=1e-4)
    optimizer_simple = torch.optim.Adam(model_simple.parameters(), lr=1e-4)
    optimizer_ff = torch.optim.Adam(model_ff.parameters(), lr=1e-4)

    batch_size = 200
    num_batches = int(60000 / batch_size)

    model_cnn.train()
    model_simple.train()
    model_ff.train()

    total_batches = len(train_loader)
    torch.cuda.empty_cache()
    for epoch in range(50):
        for i, (images, labels) in enumerate(train_loader):
            x = images.to(device)
            y = labels.to(device)

            y_pred_cnn = model_cnn(x)
            loss_cnn = loss_fn(y_pred_cnn, y)
            optimizer_cnn.zero_grad()
            loss_cnn.backward()
            optimizer_cnn.step()

            y_pred_simple = model_simple(x)
            loss_simple = loss_fn(y_pred_simple, y)
            optimizer_simple.zero_grad()
            loss_simple.backward()
            optimizer_simple.step()

            y_pred_ff = model_ff(x)
            loss_ff = loss_fn(y_pred_ff, y)
            optimizer_ff.zero_grad()
            loss_ff.backward()
            optimizer_ff.step()
        if epoch % 5 == 4:
            print(epoch + 1, "CNN: ", loss_cnn.item(), "Simple: ", loss_simple.item(), "FF: ", loss_ff.item())
            print("CNN train:")
            calc_accuracy(model_cnn, train_loader)
            print("CNN test:")
            calc_accuracy(model_cnn, test_loader)
            print("Simple train:")
            calc_accuracy(model_simple, train_loader)
            print("Simple test:")
            calc_accuracy(model_simple, test_loader)
            print("FF train:")
            calc_accuracy(model_ff, train_loader)
            print("FF test:")
            calc_accuracy(model_ff, test_loader)
            print("Ensemble train:")
            calc_ensemble_accuracy(model_cnn, model_simple, model_ff, train_loader)
            print("Ensemble test:")
            calc_ensemble_accuracy(model_cnn, model_simple, model_ff, test_loader)
