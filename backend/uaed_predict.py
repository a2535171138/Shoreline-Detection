import torch
import cv2
import numpy as np
import argparse
import os
from uaed_structure import Mymodel
from postprocess import skeletonize, remove_noise, replace_with_red
import torchvision.transforms as transforms
from confidence import calculate_confidence
import logging

def predict(checkpoint_path, input_image):
    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')
    img_shape = input_image.shape[:2]

    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution', default="gs", type=str, help='the output distribution')
    args, unknown = parser.parse_known_args()
    model = Mymodel(args).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['state_dict'])
    model.eval()

    with torch.no_grad():
        target_size = (481, 321)
        input_image = cv2.resize(input_image, target_size, interpolation=cv2.INTER_LINEAR)
        input_image = transforms.ToTensor()(input_image)
        input_image = input_image[:, 1:input_image.size(1), 1:input_image.size(2)]
        input_image = input_image.float()
        input_image = input_image.to(device).unsqueeze(0)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        mean, std = model(input_image)

        outputs_dist = torch.distributions.Independent(torch.distributions.Normal(loc=mean, scale=std + 0.001), 1)
        outputs = torch.sigmoid(outputs_dist.rsample())
        png = torch.squeeze(outputs.detach()).cpu().numpy()

        output_img = (png * 255).astype(np.uint8)
        output_img = cv2.resize(output_img, (img_shape[1], img_shape[0]))
        torch.cuda.empty_cache()

    return output_img


def UAED_predict(input_img, checkpoint_path, threshold=200):
    predicted_img = predict(checkpoint_path, input_img)
    confidence = calculate_confidence(predicted_img)
    if len(predicted_img.shape) == 3:
        predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(predicted_img, threshold, 255, cv2.THRESH_BINARY)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    binary_result = skeletonize(binary_image)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_result = cv2.morphologyEx(binary_result, cv2.MORPH_CLOSE, kernel)
    binary_result = remove_noise(binary_result, 5)

    color_result = replace_with_red(input_img, binary_result)
    color_result = cv2.cvtColor(color_result, cv2.COLOR_BGR2RGB)

    coordinates = np.column_stack(np.where(binary_result > 0))
    pixels_result = "MULTILINESTRING (("
    for i, (y, x) in enumerate(coordinates):
        pixels_result += f"{x:.4f} {y:.4f}"
        if i < len(coordinates) - 1:
            pixels_result += ", "
        else:
            pixels_result += "))"

    logging.info(f"Final confidence value: {confidence}")
    return binary_result, color_result, pixels_result, confidence
