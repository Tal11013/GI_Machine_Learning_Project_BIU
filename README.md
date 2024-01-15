# Fracture Detection using Ghost Imaging and Neural Networks
#### Authors: Noa Tal & Hadar Leiman  
Welcome to our project focusing on Fracture Detection through the innovative combination of Ghost Imaging (GI) techniques and advanced machine learning.
In this project, we explore the potential of using machine learning algorithms to identify bone fractures from X-rays captured using the Ghost imaging method.
Our goal is to investigate whether this approach can contribute to accurate medical diagnoses while minimizing radiation exposure, offering a safer alternative
for patients undergoing X-ray evaluations.

## Installation
### Dataset
We utilized a publicly available dataset of wrist X-ray images which we later preprocessed to generate GI measurements for training and evaluating our neural network model. You can download the dataset from [here](https://www.nature.com/articles/s41597-022-01328-z#Sec9).

Please refer to the dataset's documentation for usage terms, licensing, and any specific instructions provided by the dataset creators.

- Alternatively, you can use the measurements found in the "Processed Data" folder as explained below.

### Packages required
Install the required packages using the command:  
`pip install torch torchvision pandas numpy tqdm Pillow wandb`
### Code Parts
Our project consists of three distinct code parts, each contained in its respective folder:  
1. **GI_MINST**: This folder contains the code for reproducing the results of [this article](https://pubmed.ncbi.nlm.nih.gov/34624000/).
2. **Wrist_original_architecture**: In this folder, you'll find the code where we adapted the architecture from the article to work with wrist fracture images.
3. **Wrist_transfer_learning**: This folder holds the code for the third part, where we applied transfer learning to the wrist fracture dataset.
   
Each folder contains a main file, a Dataset class file and a Neural Network class file.  
Please download/clone the folder you are interested in. In addition, download/clone the file `train_and_test.py`.  
For the second and third folders also download/clone `preprocessing_wrist.py` or use the Processed Data folder instead (created using reshape = 64*128, measurements = 1024).

## How to use
### First Part - GI MNIST

To run the first part of the project, follow these steps:

1. Open the `main.py` file located in the `GI_MNIST` folder.
2. Within the `main.py` file, you can experiment with various parameters directly (learning rate, number of epochs, number of GI measurements and more)

Additionally, in the `data_preprocessing.py` file, you have the option to switch between using the MNIST dataset and the Fashion MNIST dataset.

### Second and Third Part - Wrist Original Architecture & Transfer Learning

#### Preprocessing
To generate processed data in the form of GI measurements from the X-ray image dataset for model usage, follow these steps:
1. Use the `preprocessing_wrist.py` file located in the main folder.
2. Pass the following parameters:
   - Number of Ghost Imaging measurements
   - Resize size for preprocessing
   - Path to the original dataset

#### Running with WandB
If you prefer to utilize WandB for experiment tracking, you have two options:

- In the `Wrist_original_architecture` folder, execute the `sweep_conf.py` script.
- In the `Wrist_transfer_learning` folder, execute the `main_sweep.py` script.

Both options initiate training runs with various configurations, and the results will be logged to your WandB account.

#### Running with Custom Configuration
For a more tailored approach to your experiments, follow these steps:

1. Open the `main_for_wrist.py` file in the `Wrist_original_architecture` folder or the `model_pipeline.py` file in the `Wrist_transfer_learning` folder.
2. Locate the `main` function in `main_for_wrist.py` or the `model_pipeline` function in `model_pipeline.py`.
3. Specify your desired configuration, including:
   - Learning rate
   - Number of epochs
   - Resize size for preprocessing
   - Number of Ghost Imaging measurements
   - Batch size (in `main_for_wrist.py`)
   - Model name (in `model_pipeline.py`)
   - Number of layers to unfreeze (in `model_pipeline.py`)

   For example: `main((0.001, 10, (64, 128), 10, 32))` or `model_pipeline((5, 2, ResNet, 0.0001, 512, (64, 128)))`  
   **Note:** Ensure the combination of Resize size for preprocessing and Number of Ghost Imaging measurements exists (create it during the preprocessing step).
