from torch import nn
import torch.nn.functional as F
import torch


class DiffuserLayer(nn.Module):
    def __init__(self, input_size, diffuser_weights):
        super(DiffuserLayer, self).__init__()
        self.input_size = input_size
        self.diffuser_weights = nn.Parameter(diffuser_weights)

    def forward(self, x):
        diffused_input = x * self.diffuser_weights
        return diffused_input


class ConvolutionalNetDiff(nn.Module):
    def __init__(self, num_of_measurements, batch_size, shape=(128, 128)):
        super(ConvolutionalNetDiff, self).__init__()
        self.num_of_measurements = num_of_measurements
        self.batch_size = batch_size
        self.shape = shape
        num_of_features = shape[0] * shape[1]

        diffuser_weights = torch.ones((num_of_features,), requires_grad=True)
        self.diffuser = DiffuserLayer(num_of_features, diffuser_weights)

        self.mp = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.do = nn.Dropout()

        # Placeholder for the first fully connected layer, will be initialized in forward pass
        self.fc1 = None
        self.fc2 = nn.Linear(2048, 1024)  # Adjusted dynamically if needed
        self.fc3 = nn.Linear(1024, 2048)  # Adjusted dynamically if needed
        self.fc4 = nn.Linear(2048, 1024)  # Adjusted dynamically if needed
        self.fc5 = nn.Linear(1024, 2)

    def forward(self, x):
        batch_size = x.size(0)  # Dynamically determine batch size

        # Pass through the diffuser layer
        x = torch.flatten(x, 1)
        x = self.diffuser(x)

        # Reshape for convolutional layers
        x = x.view(batch_size, 1, *self.shape)

        # Convolutional and pooling layers
        x = F.relu(self.mp(self.bn1(self.conv1(x))))
        x = F.relu(self.mp(self.bn2(self.conv2(x))))
        x = F.relu(self.mp(self.bn3(self.conv3(x))))
        x = self.do(x)

        # Flatten for fully connected layers
        x = torch.flatten(x, 1)

        # Dynamically create or adjust fc1
        if self.fc1 is None or self.fc1.in_features != x.size(1):
            self.fc1 = nn.Linear(x.size(1), 2048).to(x.device)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return torch.sigmoid(x)