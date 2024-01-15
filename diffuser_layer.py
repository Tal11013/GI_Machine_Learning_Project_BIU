import torch
import torch.nn as nn

class DiffuserLayer(nn.Module):
    def __init__(self, input_size, diffuser_weights):
        super(DiffuserLayer, self).__init__()
        self.input_size = input_size
        self.diffuser_weights = nn.Parameter(diffuser_weights)

    def forward(self, x):
        diffused_input = x * self.diffuser_weights
        return diffused_input


class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # Define the layers of your original model
        self.fc1 = nn.Linear(in_features=..., out_features=...)
        # ... add more layers as needed
        self.fcN = nn.Linear(in_features=..., out_features=...)

        # Initialize diffuser layer with appropriate input size and initial weights
        input_size = 32 * 32  # Adjust the input size accordingly
        diffuser_weights = torch.ones((input_size,), requires_grad=True)
        self.diffuser = DiffuserLayer(input_size, diffuser_weights)

    def forward(self, x):
        # Pass input through the diffuser layer
        diffused_input = self.diffuser(x)

        # Continue with the original layers of your model
        output = self.fc1(diffused_input)
        # ... pass through more layers as needed
        output = self.fcN(output)

        return output


# Example usage:
input_size = 32 * 32  # Adjust the input size accordingly
model = YourModel(input_size)
