import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class ClassifierMNIST(torch.nn.Module):
    def __init__(self):
        super(ClassifierMNIST, self).__init__()
        self._conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self._conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self._dropout_1 = torch.nn.Dropout2d()
        self._linear_1 = torch.nn.Linear(320, 50)
        self._linear_2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        print("1--->", x.shape)

        x = self._conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self._conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 320)
        x = self._dropout_1(x)
        x = self._linear_1(x)
        x = F.relu(x)
        x = self._linear_2(x)
        
        x = F.log_softmax(x, dim=1)

        print("2--->", x.shape)

        return x
