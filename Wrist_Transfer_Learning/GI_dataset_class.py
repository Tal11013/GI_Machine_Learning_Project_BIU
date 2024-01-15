import torch
from torch.utils.data import Dataset
import pandas as pd


# This is our custom dataset class
class GI_Wrist(Dataset):
    def __init__(self, csv_file, transform=None):
        # read the csv file
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        # returns the length of the dataset.
        return len(self.data)

    def __getitem__(self, index):
        # read the csv file at the given index
        measurements = pd.read_csv(self.data['path'][index]).values
        # transform the measurements if needed
        if self.transform:
            measurements = self.transform(measurements)
        # if cuda is available, move the measurements to cuda
        if torch.cuda.is_available():
            measurements = measurements.cuda()
        # read the label at the given index and convert it to a tensor
        if self.data['label'][index] == 1:
            y_label = 1
        else:
            y_label = 0

        # return the measurements and the label
        return measurements, y_label