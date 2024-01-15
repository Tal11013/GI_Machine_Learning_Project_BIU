import torch
from torchvision import datasets, transforms


def get_data(num_of_measurement=1, shape=28, resize=False):
    # This function is in charge of the pre-processing section. the function receives the number of measurements
    # chosen by the main program - M. the function then loads the unprocessed dataset of images and multiplies every
    # image with M random vectors of shapexshape size. the function returns the processed data ready for training in
    # the shape of (samples, labels).

    # this part of the function resize MNIST images if required. (in the article they resize MNIST to 32x32)
    if resize:
        transform = transforms.Compose([transforms.Resize(shape), transforms.PILToTensor()])
    else:
        transform = transforms.Compose([transforms.PILToTensor()])

    # load mnist dataset
    mnist_data = datasets.MNIST("./MNIST_data", train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=len(mnist_data))
    ### can switch to fashion_mnist by replacing the next 2 lines with the last 2.
    # fashion_mnist_data = datasets.FashionMNIST("./FashionMNIST_data", train=True, download=True, transform=transform)
    # data_loader = torch.utils.data.DataLoader(fashion_mnist_data, batch_size=len(fashion_mnist_data))

    images, labels = next(iter(data_loader))

    # create reference_matrix
    reference_matrix = torch.randn(num_of_measurement, shape * shape)
    # multiply the reference matrix with the data
    measurements = torch.matmul(reference_matrix, images.view(60000, shape * shape).float().t())
    # return the measurements and the labels
    return measurements.t(), labels
