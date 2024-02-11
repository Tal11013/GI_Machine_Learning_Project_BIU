from torch import nn
import torch.nn.functional as F
import torch


class DiffuserLayer(nn.Module):
    def __init__(self, input_size, diffuser_weights):
        super(DiffuserLayer, self).__init__()
        try:
            self.input_size = input_size
        except Exception as e1:
            print("Exception e1:", e1)
        try:
            self.diffuser_weights = nn.Parameter(diffuser_weights)
        except Exception as e2:
            print("Exception e2:", e2)

    def forward(self, x):
        print("size of x is:", x.size())
        print("size of diffuser weights:", self.diffuser_weights.size())
        try:
            diffused_input = x * self.diffuser_weights
        except Exception as e3:
            print("Exception e3:", e3)
        return diffused_input


class ConvolutionalNetDiff(nn.Module):
    def __init__(self, num_of_measurements, batch_size):
        super(ConvolutionalNetDiff, self).__init__()
        try:
            self.batch_size = batch_size
        except Exception as e4:
            print("Exception e4:", e4)
        try:
            input_size = num_of_measurements
        except Exception as e5:
            print("Exception e5:", e5)
        try:
            diffuser_weights = torch.ones((input_size,), requires_grad=True)
        except Exception as e6:
            print("Exception e6:", e6)
        try:
            self.diffuser = DiffuserLayer(input_size, diffuser_weights)
        except Exception as e7:
            print("Exception e7:", e7)

        self.mp = nn.MaxPool2d(2)
        try:
            self.fc1 = nn.Linear(num_of_measurements, 2048)
        except Exception as e8:
            print("Exception e8:", e8)
        try:
            self.fc2 = nn.Linear(2048, 1024)
        except Exception as e9:
            print("Exception e9:", e9)
        try:
            self.fc3 = nn.Linear(1024, 2048)
        except Exception as e10:
            print("Exception e10:", e10)
        try:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        except Exception as e11:
            print("Exception e11:", e11)
        try:
            self.bn1 = nn.BatchNorm2d(16)
        except Exception as e12:
            print("Exception e12:", e12)
        try:
            self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        except Exception as e13:
            print("Exception e13:", e13)
        try:
            self.bn2 = nn.BatchNorm2d(32)
        except Exception as e14:
            print("Exception e14:", e14)
        try:
            self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        except Exception as e15:
            print("Exception e15:", e15)
        try:
            self.bn3 = nn.BatchNorm2d(64)
        except Exception as e16:
            print("Exception e16:", e16)
        try:
            self.do = nn.Dropout()
        except Exception as e17:
            print("Exception e17:", e17)
        try:
            self.fc4 = nn.Linear(2048, 1024)
        except Exception as e18:
            print("Exception e18:", e18)
        try:
            self.fc5 = nn.Linear(1024, 2)
        except Exception as e19:
            print("Exception e19:", e19)

    def forward(self, x):
        try:
            x = torch.flatten(x, 1)
        except Exception as e20:
            print("Exception e20:", e20)
        try:
            x = self.diffuser(x)
        except Exception as e21:
            print("Exception e21:", e21)
        try:
            x = self.fc1(x)
        except Exception as e22:
            print("Exception e22:", e22)
        try:
            x = F.relu(x)
        except Exception as e23:
            print("Exception e23:", e23)
        try:
            x = self.fc2(x)
        except Exception as e24:
            print("Exception e24:", e24)
        try:
            x = F.relu(x)
        except Exception as e25:
            print("Exception e25:", e25)
        try:
            x = self.fc3(x)
        except Exception as e26:
            print("Exception e26:", e26)
        try:
            x = F.relu(x)
        except Exception as e27:
            print("Exception e27:", e27)
        try:
            x = torch.reshape(x, (self.batch_size, 1, 32, 64))
        except Exception as e28:
            print("Exception e28:", e28)
        try:
            x = self.conv1(x)
        except Exception as e29:
            print("Exception e29:", e29)
        try:
            x = self.bn1(x)
        except Exception as e30:
            print("Exception e30:", e30)
        try:
            x = self.mp(x)
        except Exception as e31:
            print("Exception e31:", e31)
        try:
            x = F.relu(x)
        except Exception as e32:
            print("Exception e32:", e32)
        try:
            x = self.conv2(x)
        except Exception as e33:
            print("Exception e33:", e33)
        try:
            x = self.bn2(x)
        except Exception as e34:
            print("Exception e34:", e34)
        try:
            x = self.mp(x)
        except Exception as e35:
            print("Exception e35:", e35)
        try:
            x = F.relu(x)
        except Exception as e36:
            print("Exception e36:", e36)
        try:
            x = self.conv3(x)
        except Exception as e37:
            print("Exception e37:", e37)
        try:
            x = self.bn3(x)
        except Exception as e38:
            print("Exception e38:", e38)
        try:
            x = self.mp(x)
        except Exception as e39:
            print("Exception e39:", e39)
        try:
            x = F.relu(x)
        except Exception as e40:
            print("Exception e40:", e40)
        try:
            x = self.do(x)
        except Exception as e41:
            print("Exception e41:", e41)
        try:
            x = torch.reshape(x, (self.batch_size, 2048))
        except Exception as e42:
            print("Exception e42:", e42)
        try:
            x = self.fc4(x)
        except Exception as e43:
            print("Exception e43:", e43)
        try:
            x = F.relu(x)
        except Exception as e44:
            print("Exception e44:", e44)
        try:
            x = self.fc5(x)
        except Exception as e45:
            print("Exception e45:", e45)
        return torch.sigmoid(x)