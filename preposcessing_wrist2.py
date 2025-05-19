import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm
import cv2


def generate_data2(folder_path, shape=(128, 256)):
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
        image_path = "C:\\Users\\alonl\\Downloads\\Xray_images\\"
        # image_path = "/content/drive/MyDrive/Processed_Datasets_Xray_Images_Unzipped/Xray_Images/"
        image_path += data.iloc[i, 0]
        # read the png image from the path
        image_path += ".png"

        img = cv2.imread(image_path)
        # Resize the image
        img_resized = cv2.resize(img, shape)
        # Convert to grayscale (black and white)
        img_bw = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        # convert the black and white image to numpy array of shape (shape[0], shape[1]). every pixel is assigned a
        # value between 0 and 255 to determine how gray they are.
        img = np.array(img_bw)
        # if cuda is available, move the img to cuda
        # Create the parent folder if it doesn't exist
        parent_folder = folder_path + "measurements/" + str(shape[0]) + "_" + str(
            shape[1])
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)
        # save the measurements to csv file
        csv_save_path = folder_path + "measurements/" + str(shape[0]) + "_" + str(
            shape[1]) + "/" + str(i) + ".csv"
        pd.DataFrame(img).to_csv(csv_save_path, index=False)
        loaded_img = pd.read_csv(csv_save_path)

        # save the path and the label to the new general csv file ("new_data")
        new_data = pd.concat([new_data, pd.DataFrame([[folder_path + "measurements/"
                                                       + str(shape[0]) + "_" + str(shape[1]) + "/" + str(i) + ".csv",
                                                       data['fracture_visible'].iloc[i]]], columns=['path', 'label'])])

    # save the new general csv file
    new_data.to_csv(folder_path + "new_dataset_" + str(shape[0]) + "_"+str(shape[1])
                    + ".csv", index=False)