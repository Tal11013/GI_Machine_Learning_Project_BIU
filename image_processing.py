import cv2
import numpy as np
import os

from consts import *


def straighten_and_crop_image(image_path, output_dir, output_filename):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Thresholding to get a binary image
    _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour assuming it's the bone
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    # Get the rotation matrix
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")
    m = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Perform the perspective transformation
    straightened_image = cv2.warpPerspective(image, m, (width, height))

    # Check if the height is smaller than the width
    if straightened_image.shape[0] < straightened_image.shape[1]:
        # Rotate 90 degrees clockwise
        straightened_image = cv2.rotate(straightened_image, cv2.ROTATE_90_CLOCKWISE)

    # Save the processed image with the same filename as the original
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, straightened_image)

    print(f"Processed image saved to: {output_path}")


def process_directory():

    # Create output directory if it doesn't exist
    if not os.path.exists(XRAY_IMAGES_OUTPUT_DIR):
        os.makedirs(XRAY_IMAGES_OUTPUT_DIR)

    # Process each image in the input directory
    for filename in os.listdir(XRAY_IMAGES_INPUT_DIR):
        if filename.endswith(PNG_ENDING) or filename.endswith(JPG_ENDING):
            input_path = os.path.join(XRAY_IMAGES_INPUT_DIR, filename)
            # Use the same filename as the original image
            output_filename = filename
            straighten_and_crop_image(input_path, XRAY_IMAGES_OUTPUT_DIR, output_filename)


if __name__ == "__main__":
    process_directory()