import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from mnist import MNIST

# Highest accuracy achieved: 99.52 at 45 epochs%
# input 1x784 (28x28) image
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CombinerNN(torch.nn.Module):
    def __init__(self):
        super(CombinerNN, self).__init__()
        self.linear1 = torch.nn.Linear(30, 10)

    def forward(self, x):
        x = self.linear1(x)
        return x


def train_combiner(model_1, model_2, model_3, train_loader, test_loader):
    combiner_model = CombinerNN().to(device)
    combiner_model.train()

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(combiner_model.parameters(), lr=1e-4)

    for epoch in range(50):
        for i, (images, labels) in enumerate(train_loader):
            x = images.to(device)
            y = labels.to(device)

            pred_model_1 = model_1(x)
            pred_model_2 = model_2(x)
            pred_model_3 = model_3(x)

            pred_models = torch.cat([pred_model_1, pred_model_2, pred_model_3], dim=1)

            y_pred_combiner = combiner_model(pred_models)
            loss = loss_fn(y_pred_combiner, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 4:
            print(epoch + 1, "Combiner Ensemble: ", loss.item())

        print("Ensemble train:")
        calc_ensemble_accuracy(model_1, model_2, model_3, combiner_model, train_loader)
        print("Ensemble test:")
        calc_ensemble_accuracy(model_1, model_2, model_3, combiner_model, test_loader)

    return combiner_model


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


def calc_ensemble_accuracy(model_a, model_b, model_c, model_combiner, loader):
    model_a.eval()
    model_b.eval()
    model_c.eval()
    model_combiner.eval()

    with torch.no_grad():
        total = 0
        num_correct = 0
        for images, labels in loader:
            x_batch = images.to(device)
            y_batch = labels.to(device)

            y_pred_a = model_a(x_batch)
            y_pred_b = model_b(x_batch)
            y_pred_c = model_c(x_batch)

            pred_models = torch.cat([y_pred_a, y_pred_b, y_pred_c], dim=1)

            y_pred_ensemble = model_combiner(pred_models)

            total += y_batch.size(0)
            num_correct += (y_pred_ensemble.argmax(dim=1) == y_batch).sum().item()
        accuracy = (float(num_correct) / total) * 100
        print("Accuracy " + str(accuracy) + "%")
    model_a.train()
    model_b.train()
    model_c.train()
    model_combiner.train()


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

    model_1 = DigitClassifierCNN().to(device)
    model_2 = DigitClassifierCNN().to(device)
    model_3 = DigitClassifierCNN().to(device)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=1e-4)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=1e-4)
    optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=1e-4)

    model_1.train()
    model_2.train()
    model_3.train()

    torch.cuda.empty_cache()
    for epoch in range(50):
        for i, (images, labels) in enumerate(train_loader):
            x = images.to(device)
            y = labels.to(device)

            y_pred_1 = model_1(x)
            loss_1 = loss_fn(y_pred_1, y)
            optimizer_1.zero_grad()
            loss_1.backward()
            optimizer_1.step()

            y_pred_2 = model_2(x)
            loss_2 = loss_fn(y_pred_2, y)
            optimizer_2.zero_grad()
            loss_2.backward()
            optimizer_2.step()

            y_pred_3 = model_3(x)
            loss_3 = loss_fn(y_pred_3, y)
            optimizer_3.zero_grad()
            loss_3.backward()
            optimizer_3.step()

        if epoch % 5 == 4:
            print(epoch + 1, "Model 1: ", loss_1.item(), "Model 2: ", loss_2.item(), "Model 3: ", loss_3.item())

    combiner_model = train_combiner(model_1, model_2, model_3, train_loader, test_loader)

    print("Model 1 train:")
    calc_accuracy(model_1, train_loader)
    print("Model 1 test:")
    calc_accuracy(model_1, test_loader)
    print("Model 2 train:")
    calc_accuracy(model_2, train_loader)
    print("Model 2 test:")
    calc_accuracy(model_2, test_loader)
    print("Model 3 train:")
    calc_accuracy(model_3, train_loader)
    print("Model 3 test:")
    calc_accuracy(model_3, test_loader)
    print("Ensemble train:")
    calc_ensemble_accuracy(model_1, model_2, model_3, combiner_model, train_loader)
    print("Ensemble test:")
    calc_ensemble_accuracy(model_1, model_2, model_3, combiner_model, test_loader)
