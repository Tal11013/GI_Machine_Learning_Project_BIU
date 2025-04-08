from torch.utils.data import Dataset
import pandas as pd
import torch
import cv2
import numpy as np

def debug_visualize_opencv(image_tensor):
    """
    Uses OpenCV to visualize a single image. Accepts either PyTorch tensor or NumPy array.
    """
    if isinstance(image_tensor, torch.Tensor):
        # Convert from C x H x W (Tensor) to H x W x C (NumPy)
        image_tensor = image_tensor.permute(1, 2, 0).cpu().detach().numpy()

    # Convert float to 8-bit image (0-255 range)
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min()) * 255
    image_tensor = np.uint8(image_tensor)

    cv2.imshow('Debug Visualization', image_tensor)
    cv2.waitKey(0)  # Wait for a key press to close
    cv2.destroyAllWindows()


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
        # convert the measurements to a tensor of type float32
        measurements = torch.tensor(measurements, dtype=torch.float32)
        # transform the measurements if needed
        if self.transform:
            measurements = self.transform(measurements)
        debug_visualize_opencv(measurements)
        # measurements = measurements.repeat(3, 1, 1)
        # read the label at the given index and convert it to a tensor
        if self.data['label'][index] == 1:
            y_label = 1
        else:
            y_label = 0

        # return the measurements and the label
        return measurements, y_label
