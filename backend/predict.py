# 批量读取图像并预测，输出二值图、彩图、点集
import pandas as pd
import os
import re
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from skimage import morphology

def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)

        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class CoFusion(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=3,
                               stride=1, padding=1)
        self.relu = nn.ReLU()

        self.norm_layer1 = nn.GroupNorm(4, 64)
        self.norm_layer2 = nn.GroupNorm(4, 64)

    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = self.relu(self.norm_layer2(self.conv2(attn)))
        attn = F.softmax(self.conv3(attn), dim=1)

        # return ((fusecat * attn).sum(1)).unsqueeze(1)
        return ((x * attn).sum(1)).unsqueeze(1)

class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        # self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=3, stride=1, padding=2, bias=True)),
        self.add_module('norm1', nn.BatchNorm2d(out_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, bias=True)),
        self.add_module('norm2', nn.BatchNorm2d(out_features))

    def forward(self, x):
        x1, x2 = x

        new_features = super(_DenseLayer, self).forward(F.relu(x1))  # F.relu()
        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features


class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads=[0,0,1,3,7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]  # kernel_size-1
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride,
                 use_bs=True
                 ):
        super(SingleConvBlock, self).__init__()
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features,
                 out_features=None,
                 stride=1,
                 use_act=True):
        super(DoubleConvBlock, self).__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,
                               3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_features)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_act:
            x = self.relu(x)
        return x

class DexiNed(nn.Module):
    """ Definition of the DXtrem network. """

    def __init__(self):
        super(DexiNed, self).__init__()
        self.block_1 = DoubleConvBlock(3, 32, 64, stride=2,)
        self.block_2 = DoubleConvBlock(64, 128, use_act=False)
        self.dblock_3 = _DenseBlock(2, 128, 256) # [128,256,100,100]
        self.dblock_4 = _DenseBlock(3, 256, 512)
        self.dblock_5 = _DenseBlock(3, 512, 512)
        self.dblock_6 = _DenseBlock(3, 512, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(64, 128, 2)
        self.side_2 = SingleConvBlock(128, 256, 2)
        self.side_3 = SingleConvBlock(256, 512, 2)
        self.side_4 = SingleConvBlock(512, 512, 1)
        self.side_5 = SingleConvBlock(512, 256, 1)
        # Sory I forget to comment this line :(

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(128, 256, 2)
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)

        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)
        self.block_cat = SingleConvBlock(6, 1, stride=1, use_bs=False)
         # hed fusion method
        # self.block_cat = CoFusion(6,6)# cats fusion method

        self.apply(weight_init)

    def slice(self, tensor, slice_shape):
        t_shape = tensor.shape
        height, width = slice_shape
        if t_shape[-1]!=slice_shape[-1]:
            new_tensor = F.interpolate(
                tensor, size=(height, width), mode='bicubic',align_corners=False)
        else:
            new_tensor=tensor
        return new_tensor

    def forward(self, x):
        assert x.ndim == 4, x.shape

        # Block 1
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_down = self.maxpool(block_3) # [128,256,50,50]
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)

        # Block 4
        block_2_resize_half = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(block_3_down+block_2_resize_half)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)

        # Block 5
        block_5_pre_dense = self.pre_dense_5(
            block_4_down) #block_5_pre_dense_512 +block_4_down
        block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])
        block_5_add = block_5 + block_4_side

        # Block 6
        block_6_pre_dense = self.pre_dense_6(block_5)
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        out_5 = self.up_block_5(block_5)
        out_6 = self.up_block_6(block_6)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        # return results
        results.append(block_cat)
        return results


def image_normalization(img, img_min=0, img_max=255, epsilon=1e-12):
    img = np.float32(img)
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img

def transform(img):
    img = np.array(img, dtype=np.float32)
    mean_pixel_values = [103.939,116.779,123.68, 137.86]
    mean_bgr=mean_pixel_values[0:3] if len(mean_pixel_values) == 4 else mean_pixel_values,
    img -= mean_bgr
    i_h, i_w,_ = img.shape
    crop_size = 512

    img = cv2.resize(img, dsize=(crop_size, crop_size))
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img.copy()).float()
    return img

def replace_with_red(input_img, predicted_img):
    output_img = input_img.copy()

    for i in range(predicted_img.shape[0]):
        for j in range(predicted_img.shape[1]):
            if predicted_img[i, j] > 0:
                cv2.circle(output_img, (j, i), 1, (255, 0, 0), -1)
    return output_img

def postprocessImg(tensor):
  edge_maps = []
  for i in tensor:
      tmp = torch.sigmoid(i).cpu().detach().numpy()
      edge_maps.append(tmp)
  tensor = np.array(edge_maps)

  tmp = tensor[:, 0, ...]
  tmp = np.squeeze(tmp)
  tmp_img = tmp[6]
  tmp_img = np.uint8(image_normalization(tmp_img)).astype(np.uint8)
  return tmp_img

def predict(checkpoint_path, input_image):
  device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')

  img_shape = input_image.shape[:2]
  input_image = transform(input_image)

  model = DexiNed().to(device)
  model.load_state_dict(torch.load(checkpoint_path, map_location=device))
  model.eval()

  with torch.no_grad():
    images = input_image.to(device).unsqueeze(0)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    preds = model(images)

    output_img = postprocessImg(preds)
    output_img = cv2.resize(output_img, (img_shape[1], img_shape[0]))
    torch.cuda.empty_cache()
  return output_img


def fill_holes(binary_image):
    inverted_binary_image = cv2.bitwise_not(binary_image)

    kernel = np.ones((3, 3), np.uint8)
    filled_image = cv2.morphologyEx(inverted_binary_image, cv2.MORPH_CLOSE, kernel)
    filled_image = cv2.bitwise_not(filled_image)

    return filled_image

def skeletonize(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    ret, img = cv2.threshold(image, 127, 255, 0)
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

# def Dexined_predict(input_img, checkpoint_path, threshold = 200):
#     predicted_img = predict(checkpoint_path, input_img)
#
#     _, binary_image = cv2.threshold(predicted_img, threshold, 255, cv2.THRESH_BINARY)
#     filled_image = fill_holes(binary_image)
#     binary_result = skeletonize(filled_image)
#
#     input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
#     color_result = replace_with_red(input_img, binary_result)
#
#     coordinates = np.column_stack(np.where(binary_result > 0))
# #     pixels_result = [tuple(coord) for coord in coordinates]
# #       pixels_result = [[int(coord[1]), int(coord[0])] for coord in coordinates]
#     pixels_result = "MULTILINESTRING (("
#     for i, (y, x) in enumerate(coordinates):
#         pixels_result += f"{x:.4f} {y:.4f}"
#         if i < len(coordinates) - 1:
#             pixels_result += ", "
#         else:
#             pixels_result += "))"
#
#
#     return binary_result, color_result, pixels_result

def remove_noise(image, min_size):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    img2 = np.zeros((output.shape), np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2

def Dexined_predict(input_img, checkpoint_path, threshold = 200):
    predicted_img = predict(checkpoint_path, input_img)
    # plt.imshow(predicted_img)
    # plt.axis('off')
    # plt.show()

    # print(predicted_img)
    # print("Predicted Image Range:", predicted_img.min(), predicted_img.max())
    if len(predicted_img.shape) == 3:
      predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2GRAY)


    # binary_image = predicted_img
    _, binary_image = cv2.threshold(predicted_img, 200, 255, cv2.THRESH_BINARY)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    binary_result = skeletonize(binary_image)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_result = cv2.morphologyEx(binary_result, cv2.MORPH_CLOSE, kernel)
    # print('5')
    # plt.imshow(binary_result)
    # plt.axis('off')
    # plt.show()

    binary_result = remove_noise(binary_result, 7)
    # print('noise')
    # plt.imshow(binary_result)
    # plt.axis('off')
    # plt.show()

    color_result = replace_with_red(input_img, binary_result)
    # print('color-5')
    # plt.imshow(color_result)
    # plt.axis('off')
    # plt.show()

    # filled_image = fill_holes(binary_image)

    # binary_result = skeletonize(filled_image)
    # filter_binary_result = remove_single_pixel_noise(binary_result)

    # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    # color_result = replace_with_red(input_img, binary_result)

    coordinates = np.column_stack(np.where(binary_result > 0))
    pixels_result = "MULTILINESTRING (("
    for i, (y, x) in enumerate(coordinates):  # 缩进修正
        pixels_result += f"{x:.4f} {y:.4f}"
        if i < len(coordinates) - 1:
            pixels_result += ", "
        else:
            pixels_result += "))"

    # return filled_image, binary_result, filter_binary_result, color_result

    return binary_result,color_result,pixels_result

# def main(checkpoint_path, input_csv, nums, threshold):
#   df = pd.read_csv(input_csv)
#   df_shuffled = df.sample(frac=1)
#
#   for index in range(nums):
#     row = df_shuffled.iloc[index]
#     input_img = cv2.imread(row['path'], cv2.IMREAD_COLOR)
#
#     binary_result, color_result, pixels_result = Dexined_predict(input_img, checkpoint_path, threshold)
#
#     input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
#
#     plt.imshow(input_img)
#     plt.axis('off')
#     plt.show()
#
#     plt.imshow(binary_result)
#     plt.axis('off')
#     plt.show()
#
#     plt.imshow(color_result)
#     plt.axis('off')
#     plt.show()
#
#     print(pixels_result)
#
# checkpoint_path = 'checkpoints/BIPED/29/29_model.pth'
# input_csv = 'test_set_200.csv'
# nums = 20
# threshold = 200
# main(checkpoint_path, input_csv, nums, threshold)