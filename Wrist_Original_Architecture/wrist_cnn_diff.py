from torch import nn
import torch.nn.functional as F
import torch


class DiffuserLayer(nn.Module):
    def __init__(self, input_size, num_of_masks):
        super(DiffuserLayer, self).__init__()
        self.input_size = input_size
        self.num_of_masks = num_of_masks

        # Create diffuser weights with the shape (input_size, num_of_masks)
        self.diffuser_weights = nn.Parameter(torch.ones((input_size, num_of_masks), requires_grad=True))

    def forward(self, x):
        # Element-wise multiplication with the masks
        x_after_gi = x @ self.diffuser_weights  # Matrix multiplication between input and diffuser weights
        return x_after_gi


class ConvolutionalNetDiff(nn.Module):
    def __init__(self, batch_size, shape=(128, 128), sampling_rate=1.0):
        super(ConvolutionalNetDiff, self).__init__()
        self.batch_size = batch_size
        self.shape = shape
        self.sampling_rate = sampling_rate
        num_of_features = shape[0] * shape[1]

        # Calculate the number of masks based on the sampling rate
        num_of_masks = round(num_of_features * sampling_rate)

        self.diffuser = DiffuserLayer(num_of_features, num_of_masks)

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

        self.fc_fix = nn.Linear(round(shape[0] * shape[1] * sampling_rate), 16384)

    def forward(self, x):
        batch_size = x.size(0)  # Dynamically determine batch size

        # Pass through the diffuser layer
        x = torch.flatten(x, 1)
        x = self.diffuser(x)

        x = self.fc_fix(x)

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