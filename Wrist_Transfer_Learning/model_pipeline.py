from train_and_test import train, test
from GI_dataset_class import GI_Wrist
import torch
import torch.nn as nn
from torchvision import transforms
from our_nn import MyNeuralNet
import pandas as pd

accuracy_data = pd.DataFrame(columns=['num_of_layers_unfreeze', 'model_name', 'lr', 'epochs', 'accuracy',
                                      'reshape_size', 'num_of_measurements'])


def model_pipeline(config):
    existing_data = accuracy_data[
        (accuracy_data['num_of_layers_unfreeze'] == config.num_of_layers_unfreeze) &
        (accuracy_data['model_name'] == config.model_name) &
        (accuracy_data['lr'] == config.lr) &
        (accuracy_data['epochs'] == config.epochs) &
        (accuracy_data['reshape_size'] == config.reshape_size) &
        (accuracy_data['num_of_measurements'] == config.num_of_measurements)
        ]
    # if config not exists in the accuracy_data dataframe then train the model
    if existing_data.empty:
        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer = make(config)

        # check if cuda is available
        if torch.cuda.is_available():
            # if it is then move the model to the GPU
            print("Using CUDA")
            model.cuda()

        sum_of_epochs = 0
        for num_of_epoch in [5, 5, 5]:
            model = train(model, train_loader, criterion, optimizer, num_of_epoch, 16)
            acc = test(model, test_loader, criterion, 16)
            sum_of_epochs += num_of_epoch
            # add the accuracy to the accuracy_data dataframe
            accuracy_data.loc[len(accuracy_data)] = [config.num_of_layers_unfreeze, config.model_name, config.lr,
                                                     sum_of_epochs, acc, config.reshape_size,
                                                     config.num_of_measurements]

        return accuracy_data[(accuracy_data['num_of_layers_unfreeze'] == config.num_of_layers_unfreeze) &
                             (accuracy_data['model_name'] == config.model_name) &
                             (accuracy_data['lr'] == config.lr) &
                             (accuracy_data['epochs'] == config.epochs) &
                             (accuracy_data['reshape_size'] == config.reshape_size) &
                             (accuracy_data['num_of_measurements'] == config.num_of_measurements)].iloc[0]['accuracy']
    else:
        # if config exists in the accuracy_data dataframe then return the accuracy
        return existing_data.iloc[0]['accuracy']


def make(config):
    batch_size = 16
    # Make the data
    train_loader, test_loader = make_loaders(batch_size, config)

    # Make the model
    model = make_model(config, batch_size)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    return model, train_loader, test_loader, criterion, optimizer


def make_loaders(given_batch_size, config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        #        transforms.Resize((32, 32), antialias=True),
        transforms.Normalize(mean=[0.485], std=[0.229]),  # normalize for a single channel
    ])
    # create the dataset
    path_ending = str(config.num_of_measurements) + "_" + config.reshape_size + ".csv"
    # /home/sharonlabgpu/noa_and_hadar/wrist_project/wrist_project/wrist_dataset
    csv_path = "../Processed_Dataset/new_dataset_" + path_ending
    wrist_gi_dataset = GI_Wrist(csv_path, transform=transform)

    # split the data to train and test
    number_of_samples = len(wrist_gi_dataset)
    # define batch size
    batch_size = given_batch_size
    # define the lengths of the train and test datasets to numbers divisible by the batch size
    train_len = (int(number_of_samples * 0.8) // batch_size) * batch_size  # 80% of the data for training
    test_len = (int(number_of_samples * 0.2) // batch_size) * batch_size  # 20% of the data for testing

    # create a new dataset with the desired number of samples (train_len + test_len)
    subset_dataset = torch.utils.data.Subset(wrist_gi_dataset, range(train_len + test_len))

    # split the dataset into train and test sets
    train_set, test_set = torch.utils.data.random_split(subset_dataset, [train_len, test_len])

    # create a data loader for the train set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # create data loader for the test set
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# def make_model(config):
#     if config.model_name == "ResNet":
#         model = models.resnet18(pretrained=True)
#         model.fc = nn.Linear(512, 2)
#     elif config.model_name == "VGG":
#         model = models.vgg16(pretrained=True)
#         model.classifier[6] = nn.Linear(4096, 2)
#     elif config.model_name == "efficientnet":
#         model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
#         model.classifier.fc = nn.Linear(1280, 2)
#     elif config.model_name == "inception":
#         model = models.inception_v3(pretrained=True)
#         model.fc = nn.Linear(2048, 2)
#
#     # crate a dictionary of the last 6 layers of all networks
#     last_layers = {"ResNet": ["layer1", "layer2", "layer3", "layer4", "avgpool", "fc"],
#                    "VGG": ["features.26", "features.28", "avgpool", "classifier.0", "classifier.3", "classifier.6"],
#                    "efficientnet": ["blocks.6", "blocks.7", "blocks.8", "blocks.9", "blocks.10", "features",
#                                     "classifier.fc"],
#                    "inception": ["Mixed_6e", "AuxLogits", "Mixed_7a", "Mixed_7b", "Mixed_7c", "avgpool", "fc"]}
#     # freeze the layers before the number of layers specified in the config file
#     for name, param in model.named_parameters():
#         if name.split(".")[0] in last_layers[config.model_name][config.num_of_layers_unfreeze:]:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False
#
#     return model

def make_model(config, batch_size):
    model = MyNeuralNet(config.model_name, config.num_of_layers_unfreeze, batch_size, config.num_of_measurements)
    return model
