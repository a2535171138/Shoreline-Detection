import numpy as np
import cv2

def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel

def remove_noise(image, min_size):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    img2 = np.zeros((output.shape), np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2

def replace_with_red(input_img, predicted_img):
    output_img = input_img.copy()
    mask = predicted_img > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_channel = np.zeros_like(output_img[:, :, 0])
    red_channel[mask] = 255
    red_channel = cv2.dilate(red_channel, kernel)

    output_img[red_channel > 0] = [0, 0, 255]

    return output_img
