# import the required libraries
import torch
import torch.nn.functional as F
from torch import nn

from consts import *

from GI_Wrist import GI_Wrist
import preprocessing_wrist
import preposcessing_wrist2
from consts import ALON_PROCESSED_DATASETS_PATH, IMAGE_SIZE
from wrist_cnn_diff import ConvolutionalNetDiff
from wrist_cnn import ConvolutionalNet
from train_and_test import train, test
import torchvision.transforms as transforms
import gc


# Clear all allocated memory
torch.cuda.empty_cache()

# Optionally reset the CUDA memory allocator
torch.cuda.reset_accumulated_memory_stats()
torch.cuda.reset_peak_memory_stats()
torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_max_memory_cached()

gc.collect()




def main_wrist(config):

    try:
        # use cuda if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        # Clear all allocated memory
        torch.cuda.empty_cache()

        # Optionally reset the CUDA memory allocator
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()

        shape = config.shape
        width, height = map(int, shape.split('_'))
        shape_tuple = (width, height)

        # Define the transform

        transform = transforms.Compose([
            # Resize any 1:2 image tensor to 128x128
            transforms.Lambda(lambda x: F.interpolate(x.unsqueeze(0).unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear',
                                                      align_corners=False).squeeze(0)),
            # Normalize the resized tensor
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        sampling_rate = config.sampling_rate

        # generate the data ### use this line only if you want to generate the data and transform the photos
        # to GI images BEFORE putting it into the net.
        # preposcessing_wrist2.generate_data2(ALON_PROCESSED_DATASETS_PATH, shape_tuple)
        preprocessing_wrist.generate_data(ALON_PROCESSED_DATASETS_PATH, sampling_rate, shape_tuple)

        # preposcessing_wrist2.generate_data2("/content/drive/MyDrive/Processed_Datasets/", shape_tuple)
        # preprocessing_wrist.generate_data("/content/drive/MyDrive/Processed_Datasets/", shape_tuple)

        # create the dataset
        path_ending = str(shape) + CSV_ENDING
        print(path_ending)
        csv_path = ALON_PROCESSED_DATASETS_PATH  + "new_dataset_" + path_ending
        # csv_path = "/content/drive/MyDrive/Processed_Datasets/new_dataset_" + path_ending
        wrist_gi_dataset = GI_Wrist(csv_path, transform = transform)
        # split the data to train and test
        number_of_samples = len(wrist_gi_dataset)
        # define batch size
        batch_size = config.batch_size


        # define the lengths of the train and test datasets to numbers divisible by the batch size
        train_len = (int(number_of_samples * 0.8) // batch_size) * batch_size  # 80% of the data for training
        test_len = (int(number_of_samples * 0.2) // batch_size) * batch_size  # 20% of the data for testing

        # create a new dataset with the desired number of samples (train_len + test_len)
        subset_dataset = torch.utils.data.Subset(wrist_gi_dataset, range(train_len + test_len))

        # split the dataset into train and test sets
        train_set, test_set = torch.utils.data.random_split(subset_dataset, [train_len, test_len])

        ### from here on it is the same as in main.py in GI_MNIST ###

        # create a data loader for the train set
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        # create data loader for the test set
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        # create the network and choose if you want the GI imaging to happen before the entering to the net
        # or during the net - choose only one.

        model = ConvolutionalNet(batch_size, (IMAGE_SIZE, IMAGE_SIZE)).to(device)
        # model = ConvolutionalNetDiff(batch_size, (IMAGE_SIZE, IMAGE_SIZE), sampling_rate).to(device)


        # choose a loss function
        criterion = nn.CrossEntropyLoss()
        # choose an optimizer and learning rate for the training
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0003)
        # choose the number of epochs
        number_of_epochs = config.epoch

        # Train the model
        model = train(model, train_loader, criterion, optimizer, number_of_epochs, batch_size)

        # Evaluate the trained model
        test_acc, test_loss = test(model, test_loader, criterion, batch_size)

        return test_acc, test_loss

    finally:
        for name in list(globals()):
            obj = globals()[name]
            if torch.is_tensor(obj) or isinstance(obj, torch.nn.Module):
                del globals()[name]

        gc.collect()

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# very outdated !!
def main_wrist_dict(params):
    # use cuda if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    num_of_measurements = params["num_of_measurements"]
    # generate the data ### use this line only if you want to generate the data
    # preprocessing_wrist.generate_data("Processed_Dataset/", num_of_measurements, params["shape"])

    # create the dataset
    path_ending = str(num_of_measurements) + "_" + str(params["shape"]) + ".csv"
    csv_path = "../Processed_Dataset/new_dataset_" + path_ending
    wrist_gi_dataset = GI_Wrist(csv_path)

    # split the data to train and test
    number_of_samples = len(wrist_gi_dataset)
    # define batch size
    batch_size = params["batch_size"]

    # define the lengths of the train and test datasets to numbers divisible by the batch size
    train_len = (int(number_of_samples * 0.8) // batch_size) * batch_size  # 80% of the data for training
    test_len = (int(number_of_samples * 0.2) // batch_size) * batch_size  # 20% of the data for testing

    # create a new dataset with the desired number of samples (train_len + test_len)
    subset_dataset = torch.utils.data.Subset(wrist_gi_dataset, range(train_len + test_len))

    # split the dataset into train and test sets
    train_set, test_set = torch.utils.data.random_split(subset_dataset, [train_len, test_len])

    ### from here on it is the same as in main.py in GI_MNIST ###

    # create a data loader for the train set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # create data loader for the test set
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # create the network and choose if you want the GI imaging to happen before the entering to the net
    # or during the net - choose only one.
    #model = ConvolutionalNetDiff(num_of_measurements, batch_size, sampling_rate).to(device)
    model = ConvolutionalNet(num_of_measurements, batch_size).to(device)
    # choose a loss function
    criterion = nn.CrossEntropyLoss()
    # choose an optimizer and learning rate for the training
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=0.0003)
    # choose the number of epochs
    number_of_epochs = params["epoch"]

    # Train the model
    model = train(model, train_loader, criterion, optimizer, number_of_epochs, batch_size)
    # Evaluate the trained model
    test_acc, test_loss = test(model, test_loader, criterion, batch_size)
    return test_acc, test_loss

