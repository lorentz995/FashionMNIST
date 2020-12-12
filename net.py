import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, input):
        return input * torch.sigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SimpleFashionNet(nn.Module):
    def __init__(self):
        super(SimpleFashionNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)  # 10 class --> out_features = 10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten Layer
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out


class DeepFashionNet(nn.Module):
    def __init__(self):
        super(DeepFashionNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1)  # 28*28
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)  # 28*28
        nn.init.xavier_uniform_(self.conv1.weight)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)  # 26*26
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 13*13
        self.drop1 = nn.Dropout2d(0.25)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)  # 13*13
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(128)  # 13*13
        nn.init.xavier_uniform_(self.conv3.weight)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)  # 11*11
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(128)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 5*5
        self.drop2 = nn.Dropout2d(0.25)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)  # 5*5
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(256)  # 5*5
        nn.init.xavier_uniform_(self.conv5.weight)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)  # 3*3
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(256)
        nn.init.xavier_uniform_(self.conv6.weight)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1*1
        self.drop3 = nn.Dropout2d(0.25)

        self.fc1 = nn.Linear(in_features=256, out_features=10)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.mp1(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.mp2(x)
        x = self.drop2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.bn6(x)
        x = self.mp3(x)
        x = self.drop3(x)

        x = x.view(x.size(0), -1)  # Flatten Layer
        x = self.fc1(x)

        out = self.softmax(x)
        return out


class SwishFashionNet(nn.Module):
    def __init__(self):
        super(SwishFashionNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)  # 28*28
        self.bn1 = nn.BatchNorm2d(16)
        self.swish1 = Swish()
        nn.init.xavier_normal_(self.conv1.weight)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=1)  # 24*24

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # 24*24
        self.bn2 = nn.BatchNorm2d(32)
        self.swish2 = Swish()
        nn.init.xavier_normal_(self.conv2.weight)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 12*12

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # 12*12
        self.bn3 = nn.BatchNorm2d(64)
        self.swish3 = Swish()
        nn.init.xavier_normal_(self.conv3.weight)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 6*6
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.swish1(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.swish2(x)
        x = self.mp2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.swish3(x)
        x = self.mp3(x)

        x = x.view(x.size(0), -1)  # Flatten Layer
        x = self.fc1(x)
        out = self.softmax(x)
        return out
