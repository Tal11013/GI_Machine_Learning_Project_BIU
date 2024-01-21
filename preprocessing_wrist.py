import numpy as np
import pandas as pd
import torch
from PIL import Image
import os
from tqdm import tqdm
import traceback


def generate_data(folder_path, num_of_measurement, shape=(64, 128)):
    # create reference_matrix - a random matrix in the shape of number of measurments (photos) and the pixels each photo
    # contain, so shape[0]*shape[1]
    reference_matrix = np.random.randn(num_of_measurement, shape[0] * shape[1])
    # read the csv file given as an input to the function
    data = pd.read_csv(folder_path + "dataset.csv")
    # create a new csv file for later use
    new_data = pd.DataFrame(columns=['path', 'label'])
    # iterate on the rows of the  data (given CSV file in the function)
    for i in tqdm(range(len(data))):            # clear cache and unused memory from the GPU
        torch.cuda.empty_cache()
        # get the image path from the csv file - .iloc[i,0] is used to access the value of row i and column 0 in the
        # data file (column 0 in the data gives the image path)
        image_path = "C:\\Users\\iker1\\Downloads\\Xray_images\\"
        image_path += data.iloc[i, 0]
        # read the png image from the path
        image_path += ".png"
        img = Image.open(image_path)
        # resize the image to shape (input) and resamples the pixels in a Bilinear way (the color weight given to the
        # pixel is calculated using the 4 nearest pixels
        img = img.resize(size=shape, resample=Image.BILINEAR)
        # convert the black and white image to numpy array of shape (shape[0], shape[1]). every pixel is assigned a
        # value between 0 and 255 to determine how gray they are.
        img = np.array(img.convert('L'))
        # if cuda is available, move the img to cuda
        if torch.cuda.is_available():
            # changes the img and refernce_matrix to Pytorch tensors. type float32 means the values in the tensors
            # will be floats in size 32 bit
            img = torch.tensor(img, dtype=torch.float32).cuda()
            reference_matrix_cuda = torch.tensor(reference_matrix, dtype=torch.float32).cuda()
            # multiply the reference matrix with the img. .view() reshapes the tensor from shape[0]*shape[1] to
            # (shape[0]*shape[1], 1 - flattening the tensor to prepare for matrix multiplication.
            # .cpu() moves the tensor to the CPU to perform further operation that are not supported by the GPU like
            # converting it to a numpy array.
            measurements = torch.matmul(reference_matrix_cuda, img.view(shape[0] * shape[1], 1)).cpu().numpy()
        else:
            # multiply the reference matrix with the img
            measurements = np.matmul(reference_matrix, img.reshape(shape[0] * shape[1]))
        # save the measurements to csv file
        pd.DataFrame(measurements).to_csv(folder_path + "measurements/"+str(num_of_measurement) + "_"+str(shape[0]) +
                                              "_" + str(shape[1]) + "/" + str(i) + ".csv", index=False)
        # save the  path and the label to the new general csv file ("new_data")
        try:
            new_data = pd.concat([new_data, pd.DataFrame([[folder_path + "measurements/"+str(num_of_measurement) + "_"
                                                           + str(shape[0]) + "_" + str(shape[1]) + "/" + str(i) + ".csv",
                                                           data['label'].iloc[i]]], columns=['path', 'label'])])
        except Exception as a:
            print(f"error: {type(a).__name__} - {a}")
            traceback.print_exc()
        # save the new general csv file
        try:
            new_data.to_csv(folder_path + "new_dataset_" + str(num_of_measurement) + "_" + str(shape[0]) + "_"+str(shape[1])
                        + ".csv", index=False)
        except Exception as b:
            print(f"error1f: {type(b).__name__} - {b}")