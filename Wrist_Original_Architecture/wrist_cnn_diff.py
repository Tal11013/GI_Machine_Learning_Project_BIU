from torch import nn
import torch.nn.functional as F
import torch


class DiffuserLayer(nn.Module):
    def __init__(self, input_size, num_of_masks):
        super(DiffuserLayer, self).__init__()
        self.input_size = input_size
        self.num_of_masks = num_of_masks

        # Create diffuser weights for compression
        self.diffuser_weights = nn.Parameter(torch.randn((input_size, num_of_masks), requires_grad=True) * 0.01)
        # Create reconstruction weights to restore original dimension
        self.reconstruction_weights = nn.Parameter(torch.randn((num_of_masks, input_size), requires_grad=True) * 0.01)

    def forward(self, x):
        # Compress
        x_compressed = x @ self.diffuser_weights
        # Reconstruct
        x_reconstructed = x_compressed @ self.reconstruction_weights
        return x_reconstructed


class ConvolutionalNetDiff(nn.Module):
    def __init__(self, batch_size, shape=(128, 128), sampling_rate=1.0):
        super(ConvolutionalNetDiff, self).__init__()
        self.batch_size = batch_size
        self.shape = shape
        self.sampling_rate = sampling_rate
        num_of_features = shape[0] * shape[1]

        # Calculate the number of masks based on the sampling rate
        num_of_masks = round(num_of_features * sampling_rate)

        # Diffuser layer for dimensionality reduction and reconstruction
        self.diffuser = DiffuserLayer(num_of_features, num_of_masks)

        # Use the original CNN architecture after the diffuser
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.mp = nn.MaxPool2d(2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 16, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 2)

        self.do = nn.Dropout()

    def forward(self, x):
        # Store batch size for reshaping
        batch_size = x.shape[0]

        # Flatten the input
        x_flat = x.view(batch_size, -1)

        # Apply diffuser and reconstruction
        x_processed = self.diffuser(x_flat)

        # Reshape back to image format
        x = x_processed.view(batch_size, 1, self.shape[0], self.shape[1])

        # Now proceed with the original CNN architecture
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.mp(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.mp(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.mp(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.do(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)

        return torch.sigmoid(x)