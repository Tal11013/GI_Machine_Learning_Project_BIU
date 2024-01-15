from data_preprocessing import get_data
from torch.utils.data import Dataset


# This is our custom dataset class
class GI_Mnist(Dataset):
    # The constructor of the dataset calls the get_data function from "data_preprocessing.py" to get the
    # processed dataset simulating GI output
    def __init__(self, num_of_measurement=1, shape=28, resize=False, transform=None):
        self.data, self.labels = get_data(num_of_measurement, shape, resize)
        self.transform = transform

    def __len__(self):
        # returns the length of the dataset.
        return len(self.data)

    def __getitem__(self, index):
        # return the element at a given index in the dataset.
        return self.data[index], self.labels[index]
