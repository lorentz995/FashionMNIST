from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FashionNet1(nn.Module):
    def __init__(self):
        super(FashionNet1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 10)  # 10 classi --> out_features = 10

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        x = conv2.view(conv2.size(0), -1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out


class FashionNet2(nn.Module):
    def __init__(self):
        super(FashionNet2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(in_features=50 * 4 * 4, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=10)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        x = conv2.view(conv2.size(0), -1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out


class Swish(nn.Module):
    def forward(selfself, input):
        return input * torch.sigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class DeeperFashionNet(nn.Module):
    def __init__(self):
        super(DeeperFashionNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.swish1 = Swish()
        nn.init.xavier_normal_(self.conv1.weight)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.swish2 = Swish()
        nn.init.xavier_normal_(self.conv2.weight)
        self.mp2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.swish3 = Swish()
        nn.init.xavier_normal_(self.conv3.weight)
        self.mp3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 6 * 6, 10)

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

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        out = self.softmax(x)

        return out
