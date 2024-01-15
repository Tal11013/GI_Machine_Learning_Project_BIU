import torch
import torch.nn as nn
import torchvision.models as models


class MyNeuralNet(nn.Module):
    def __init__(self, model_name, num_of_layers_unfreeze, batch_size=16, input_size=1024):
        super(MyNeuralNet, self).__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_of_layers_unfreeze = num_of_layers_unfreeze
        self.model = self.make_model()
        self.addition = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.addition(x)
        # Reshape the input tensor to (16, 1, 64, 128)
        reshaped_input = x.reshape(16, 1, 64, 128)
        if self.model_name == "inception":
            resized_tensor = torch.nn.functional.interpolate(reshaped_input, size=(299, 299), mode='bilinear',
                                                             align_corners=False)
        else:
            # Resize the reshaped input tensor to (16, 3, 224, 224)
            resized_tensor = torch.nn.functional.interpolate(reshaped_input, size=(224, 224), mode='bilinear',
                                                             align_corners=False)
        # Repeat the tensor 3 times to get (16, 3, 224, 224)
        x = torch.cat([resized_tensor] * 3, dim=1)
        x = self.model(x)
        return x

    def make_model(self):
        if self.model_name == "ResNet":
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(512, 2)
        elif self.model_name == "VGG":
            model = models.vgg16(pretrained=True)
            model.classifier[6] = nn.Linear(4096, 2)
        elif self.model_name == "efficientnet":
            model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
            model.classifier.fc = nn.Linear(1280, 2)
        elif self.model_name == "inception":
            model = models.inception_v3(pretrained=True)
            model.fc = nn.Linear(2048, 2)
        elif self.model_name == "ResNet152":
            model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
            model.fc = nn.Linear(2048, 2)
            return model
        elif self.model_name == "densenet169":
            model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet169', pretrained=True)
            model.classifier = nn.Linear(1664, 2)
            return model

        # crate a dictionary of the last 6 layers of all networks
        last_layers = {"ResNet": ["layer1", "layer2", "layer3", "layer4", "avgpool", "fc"],
                       "VGG": ["features.26", "features.28", "avgpool", "classifier.0", "classifier.3", "classifier.6"],
                       "efficientnet": ["blocks.6", "blocks.7", "blocks.8", "blocks.9", "blocks.10", "features",
                                        "classifier.fc"],
                       "inception": ["Mixed_6e", "AuxLogits", "Mixed_7a", "Mixed_7b", "Mixed_7c", "avgpool", "fc"]}

        # freeze the layers before the number of layers specified in the config file
        for name, param in model.named_parameters():
            if name.split(".")[0] in last_layers[self.model_name][self.num_of_layers_unfreeze:]:
                param.requires_grad = True
            else:
                param.requires_grad = False

        return model
