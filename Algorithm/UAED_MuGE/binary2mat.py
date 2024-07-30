import cv2
import scipy.io as sio
import os


def process_image(image_path):
    target_size = (481, 321)
    # Read the binary image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    # Ensure the image is binary
    _, binary_image = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)

    return binary_image


def save_as_mat(binary_image, output_path):
    # Save the binary image as a .mat file
    sio.savemat(output_path, {'binary_image': binary_image})

    return True


def process_images_from_list(file_list_path):
    with open(file_list_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        columns = line.strip().split()
        jpg_path = columns[0]
        image_path = columns[1]
        print(image_path)
        if os.path.isfile(image_path):
            binary_image = process_image(image_path)
            output_path = os.path.splitext(image_path)[0] + '.mat'
            save_as_mat(binary_image, output_path)
            # if save_as_mat(binary_image, output_path):
            #     with open('data/train.lst', 'a', newline='') as lst_file:
            #         lst_file.write(jpg_path + ' ' + output_path + '\n')
            print(f'Saved {output_path}')


# Path to your .lst file
file_list_path = 'datasets/train_edges_aug1.lst'

# Process the images listed in the .lst file
process_images_from_list(file_list_path)
