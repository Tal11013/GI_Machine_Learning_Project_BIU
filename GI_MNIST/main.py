import torch
from torchvision import transforms
from torch import nn
from train_and_test import train, test
from GI_Mnist import GI_Mnist
from cnn import ConvolutionalNet


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)), ])

# Define the number of measurements
num_of_measurements = 16

# Create the dataset
# shape - the shape to resize the input into
mnist_gi_dataset = GI_Mnist(num_of_measurement=num_of_measurements, shape=32, resize=True, transform=transform)


# split the data to train and test
train_set, test_set = torch.utils.data.random_split(mnist_gi_dataset, [0.8, 0.2])


# define batch size
batch_size = 16


# create a data loader for the train set
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
# create data loader for the test set
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
dataiter = iter(train_loader)
batch_images, batch_labels = next(dataiter)

# create the network
model = ConvolutionalNet(num_of_measurements, batch_size)
# choose a loss function
criterion = nn.CrossEntropyLoss()
# choose an optimizer and learning rate for the training
optimizer = torch.optim.Adam(model.parameters(), lr=0.00002, weight_decay=0.0001)
# choose the number of epochs
number_of_epochs = 5


# Train the model
model = train(model, train_loader, criterion, optimizer, number_of_epochs, batch_size)
# Evaluate the trained model
test(model, test_loader, criterion, batch_size)
