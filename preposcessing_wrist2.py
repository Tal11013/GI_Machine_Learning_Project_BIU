import numpy as np
import pandas as pd
import torch
from PIL import Image
import os
from tqdm import tqdm
import cv2


def generate_data2(folder_path, num_of_measurement, shape=(128, 256)):
    # Convert the string shape to a tuple of integers
    width, height = map(int, shape.split('_'))
    shape_tuple = (width, height)
    # read the csv file given as an input to the function
    data = pd.read_csv(folder_path + "dataset.csv")
    # create a new csv file for later use
    new_data = pd.DataFrame(columns=['path', 'label'])
    # iterate on the rows of the  data (given CSV file in the function)
    for i in tqdm(range(len(data))):
        # clear cache and unused memory from the GPU
        torch.cuda.empty_cache()
        # get the image path from the csv file - .iloc[i,0] is used to access the value of row i and column 0 in the
        # data file (column 0 in the data gives the image path)
        image_path = "C:\\Users\\iker1\\Downloads\\Xray_images\\"
        image_path += data.iloc[i, 0]
        # read the png image from the path
        image_path += ".png"
        img = cv2.imread(image_path)
        # Resize the image
        img_resized = cv2.resize(img,shape_tuple)

        # Convert to grayscale (black and white)
        img_bw = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Display the resized image for debugging
        # resize the image to shape (input) and resamples the pixels in a Bilinear way (the color weight given to the
        # pixel is calculated using the 4 nearest pixels
        # convert the black and white image to numpy array of shape (shape[0], shape[1]). every pixel is assigned a
        # value between 0 and 255 to determine how gray they are.
        img = np.array(img_bw)
        # if cuda is available, move the img to cuda
        # Create the parent folder if it doesn't exist
        parent_folder = folder_path + "measurements/" + str(num_of_measurement) + "_" + str(shape_tuple[0]) + "_" + str(
            shape_tuple[1])
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)
        # save the measurements to csv file
        pd.DataFrame(img).to_csv(folder_path + "measurements/"+str(num_of_measurement) + "_"+str(shape_tuple[0]) +
                                          "_" + str(shape_tuple[1]) + "/" + str(i) + ".csv", index=False)
        # save the path and the label to the new general csv file ("new_data")
        new_data = pd.concat([new_data, pd.DataFrame([[folder_path + "measurements/"+str(num_of_measurement) + "_"
                                                       + str(shape_tuple[0]) + "_" + str(shape_tuple[1]) + "/" + str(i) + ".csv",
                                                       data['fracture_visible'].iloc[i]]], columns=['path', 'label'])])

    # save the new general csv file
    new_data.to_csv(folder_path + "new_dataset_" + str(num_of_measurement) + "_" + str(shape_tuple[0]) + "_"+str(shape_tuple[1])
                    + ".csv", index=False)