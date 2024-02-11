# import the required libraries
import torch
from torch.utils.data import Dataset
from torch import nn
import preprocessing_wrist
from preprocessing_wrist import generate_data
import preposcessing_wrist2
from preposcessing_wrist2 import generate_data2
from GI_Wrist import GI_Wrist
from wrist_cnn_diff import ConvolutionalNetDiff
from wrist_cnn import ConvolutionalNet
from train_and_test import train, test


def main_wrist(config):
    # use cuda if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    num_of_measurements = config.num_of_measurements
    shape = config.shape
    # generate the data ### use this line only if you want to generate the data and transform the photos
    # to GI images BEFORE putting it into the net.
    #try:
        #preposcessing_wrist2.generate_data2("C:\\Users\\iker1\\OneDrive\\מסמכים\\GitHub\\GI_Machine_Learning_Project_BIU\\Processed_Dataset\\", num_of_measurements, shape)
        #preprocessing_wrist.generate_data("C:\\Users\\iker1\\OneDrive\\מסמכים\\GitHub\\GI_Machine_Learning_Project_BIU\\Processed_Dataset\\", num_of_measurements, shape)
    #except Exception as d:
        #print("errord1:", d)
    # create the dataset
    path_ending = str(config.num_of_measurements) + "_" + str(shape) + ".csv"
    print(path_ending)
    csv_path = "C:\\Users\\iker1\\OneDrive\\מסמכים\\GitHub\\GI_Machine_Learning_Project_BIU\\Processed_Dataset\\new_dataset_" + path_ending
    try:
        wrist_gi_dataset = GI_Wrist(csv_path)
    except Exception as e:
        print("error2:", e)
    try:
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
    except Exception as c:
        print("error3", c)
    #model = ConvolutionalNet(num_of_measurements, batch_size).to(device)
    model = ConvolutionalNetDiff(num_of_measurements, batch_size).to(device)
    try:
        # choose a loss function
        criterion = nn.CrossEntropyLoss()
    except Exception as e1:
        print("error4:", e1)
    try:
        # choose an optimizer and learning rate for the training
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0003)
    except Exception as e2:
        print("error5:", e2)
        # choose the number of epochs
    number_of_epochs = config.epoch
    try:
        # Train the model
        model = train(model, train_loader, criterion, optimizer, number_of_epochs, batch_size)
    except Exception as e3:
        print("error6:", e3)
        # Evaluate the trained model
    try:
        test_acc, test_loss = test(model, test_loader, criterion, batch_size)
        return test_acc, test_loss
    except Exception as e4:
        print("error7:", e4)
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
    #model = ConvolutionalNetDiff(num_of_measurements, batch_size).to(device)
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

