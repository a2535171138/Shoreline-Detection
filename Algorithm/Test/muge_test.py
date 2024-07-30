import argparse
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import url_map, url_map_advprop, get_model_params
import numpy as np
import pandas as pd
import os
import re
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
import random
from skimage import morphology
import argparse
import torchvision.transforms as transforms
import time
import subprocess
import sys
from scipy.spatial.distance import cdist
from shapely import LineString
from shapely import frechet_distance
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from collections import defaultdict
dim=1
class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
    def forward(self, x):
        return self.attention(x)


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x



class UnetPlusPlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )
        encoder_channels=list(encoder_channels)
        encoder_channels=[(i) for i in encoder_channels]
        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels

        self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
                        dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features
                    )
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth-1}"])
        return dense_x[f"x_{0}_{self.depth}"]



def patch_first_conv(model, new_in_channels, default_in_channels=3, pretrained=True):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)
        )
        module.reset_parameters()

    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)

    else:
        new_weight = torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)



class EncoderMixin:
    """Add encoder functionality such as:
    - output channels specification of feature tensors (produced by encoder)
    - patching first convolution for arbitrary input channels
    """

    _output_stride = 32

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels, pretrained=True):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        patch_first_conv(model=self, new_in_channels=in_channels, pretrained=pretrained)



class EfficientNetEncoder(EfficientNet, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, model_name, depth=5):

        blocks_args, global_params = get_model_params(model_name, override_params=None)
        super().__init__(blocks_args, global_params)

        self._stage_idxs = stage_idxs
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        del self._fc

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self._conv_stem, self._bn0, self._swish),
            self._blocks[: self._stage_idxs[0]],
            self._blocks[self._stage_idxs[0] : self._stage_idxs[1]],
            self._blocks[self._stage_idxs[1] : self._stage_idxs[2]],
            self._blocks[self._stage_idxs[2] :],
        ]

    def forward(self, x):
        stages = self.get_stages()

        block_number = 0.0
        drop_connect_rate = self._global_params.drop_connect_rate

        features = []
        for i in range(self._depth + 1):

            # Identity and Sequential stages
            if i < 2:
                x = stages[i](x)

            # Block stages need drop_connect rate
            else:
                for module in stages[i]:
                    drop_connect = drop_connect_rate * block_number / len(self._blocks)
                    block_number += 1.0
                    x = module(x, drop_connect)

            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("_fc.bias", None)
        state_dict.pop("_fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)



def _get_pretrained_settings(encoder):
    pretrained_settings = {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": url_map[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        },
        "advprop": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "url": url_map_advprop[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    }
    return pretrained_settings





def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):
    encoders = {
    "efficientnet-b7": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b7"),
        "params": {
            "out_channels": (3, 64, 48, 80, 224, 640),
            "stage_idxs": (11, 18, 38, 55),
            "model_name": "efficientnet-b7",
        },
    },
}
    Encoder = encoders[name]["encoder"]

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        settings = encoders[name]["pretrained_settings"][weights]
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels, pretrained=weights is not None)

    return encoder


from math import sqrt
def spectrum_noise( img_fft, alpha):
    ratio=0.5
    batch_size, h, w, c = img_fft.shape
    img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)
    img_abs = torch.fft.fftshift(img_abs, dim=(1))
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = 0
    img_abs_ = img_abs.clone()
    masks=torch.ones_like(img_abs)
    for i_m in range(alpha.shape[0]):
        masks[i_m]=masks[i_m]*(alpha[i_m])
    masks[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :]=1
    # img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] = \
    #                 img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :]*(1-torch.exp(alpha)).view(-1,1,1,1)
    img_abs=img_abs_*masks
    img_abs = torch.fft.ifftshift(img_abs, dim=(1))  # recover
    img_mix = img_abs * (np.e ** (1j * img_pha))
    return img_mix


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)
class Mymodel(nn.Module):
    def __init__(self,args,encoder_name="efficientnet-b7",encoder_weights="imagenet",in_channels=3,classes=1):
        super(Mymodel, self).__init__()

        self.encoder_depth=5

        self.decoder_use_batchnorm= True,
        self.decoder_channels= (256, 128, 64, 32, 16)
        self.decoder_attention_type= None

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=self.encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=self.decoder_channels,
            n_blocks=self.encoder_depth,
            use_batchnorm=self.decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=self.decoder_attention_type,
        )


        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=classes,
            kernel_size=3,
        )

        self.decoder_1 = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=self.decoder_channels,
            n_blocks=self.encoder_depth,
            use_batchnorm=self.decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=self.decoder_attention_type,
        )


        self.segmentation_head_1 = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=classes,
            kernel_size=3,
        )

        self.args=args

    def forward(self, x,label_style):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]

        features = self.encoder(x)

        for i in range(1,len(features)):
            B,C,a,b=features[i].shape
            C=int(C/2)
            img_fft=features[i][:,C:,:,:].permute(0, 2, 3, 1)
            img_fft = torch.fft.rfft2(img_fft, dim=(1, 2), norm='ortho')
            img_fft=spectrum_noise(img_fft, label_style)
            img_fft = torch.fft.irfft2(img_fft, s=(a, b), dim=(1, 2), norm='ortho')
            img_fft=img_fft.permute(0, 3,1,2)
            features[i]=torch.cat([features[i][:,:C,:,:]*(label_style).view(-1,1,1,1),img_fft],1)
        decoder_output = self.decoder(*features)
        results = self.segmentation_head(decoder_output)
        ### center crop
        results = crop(results, img_H, img_W, 0 , 0)
        if self.args.distribution=="beta":
            results = nn.Softplus()(results)

        decoder_output_1 = self.decoder_1(*features)
        results_1 = self.segmentation_head_1(decoder_output_1)
        ### center crop
        std = crop(results_1, img_H, img_W, 0 , 0)
        if self.args.distribution!="residual":
            std = nn.Softplus()(std)

        return results,std
# Based on BDCN Implementation @ https://github.com/pkuCactus/BDCN
def crop(data1, h, w , crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    assert(h <= h1 and w <= w1)
    data = data1[:, :, crop_h:crop_h+h, crop_w:crop_w+w]
    return data

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

def draw_shoreline(image, multiline_string):
    height, width = image.shape[:2]

    black_image = np.zeros((height, width), dtype=np.uint8)

    pattern = r"\(\((.*?)\)\)"
    matches = re.findall(pattern, multiline_string)

    for match in matches:
        points = match.split(", ")
        coordinates = []
        for point in points:
            x, y = point.split()
            x = x.strip("()")
            y = y.strip("()")
            coordinates.append((int(float(x)), int(float(y))))

        for i in range(len(coordinates) - 1):
            cv2.line(black_image, coordinates[i], coordinates[i + 1], (255,), 1)

    return black_image

def combine_binary_images(pred_mask, gt_mask):
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    combined_img = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    combined_img[pred_mask == 1] = [255, 255, 0]
    combined_img[gt_mask == 1] = [0, 255, 255]

    return combined_img

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
  # device = torch.device('cpu')
  # torch.load with map_location=torch.device('cpu')
  img_shape = input_image.shape[:2]

  parser = argparse.ArgumentParser()
  parser.add_argument('--distribution', default="gs", type=str, help='the output distribution')
  args, unknown = parser.parse_known_args()
  model = Mymodel(args).to(device)
  # model.load_state_dict(torch.load(checkpoint_path,map_location='cpu')['state_dict'])
  model.load_state_dict(torch.load(checkpoint_path, map_location=device)['state_dict'])
  model.eval()

  with torch.no_grad():
    target_size = (481, 321)
    input_image = cv2.resize(input_image, target_size, interpolation=cv2.INTER_LINEAR)
    input_image = transforms.ToTensor()(input_image)
    input_image = input_image[:, 1:input_image.size(1), 1:input_image.size(2)]
    input_image = input_image.float()
    input_image = input_image.to(device).unsqueeze(0)
    # input_image = input_image.cuda()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    label_bias = torch.ones(1).to(device).unsqueeze(0)
    label_bias = label_bias * 1
    mean, std = model(input_image, label_bias)
    # print(mean)
    # print(std)
    outputs_dist=torch.distributions.Independent(torch.distributions.Normal(loc=mean, scale=std+0.001), 1)
    outputs = torch.sigmoid(outputs_dist.rsample())
    png=torch.squeeze(outputs.detach()).cpu().numpy()

    # output_img = postprocessImg(preds)
    _, _, H, W = input_image.shape
    result = np.zeros((H + 1, W + 1))
    result[1:, 1:] = png
    # result_png = Image.fromarray((result * 255).astype(np.uint8))
    result_png = (png * 255).astype(np.uint8)


    # plt.imshow(result_png)
    # plt.axis('off')
    # plt.show()

    output_img = cv2.resize(result_png, (img_shape[1], img_shape[0]))

    # outputs_dist=torch.distributions.Independent(torch.distributions.Normal(loc=mean, scale=std+0.001), 1)
    # outputs = torch.sigmoid(outputs_dist.rsample())
    # png=torch.squeeze(outputs.detach()).cpu().numpy()
    # _, binary_image = cv2.threshold(png, threshold, 255, cv2.THRESH_BINARY)
    # result=np.zeros((H+1,W+1))
    # result[1:,1:]=png
    # output_img = Image.fromarray((png * 255).astype(np.uint8))
    # output_img = (png * 255).astype(np.uint8)
    # output_img = cv2.resize(output_img, (img_shape[1], img_shape[0]))
    torch.cuda.empty_cache()

  return output_img


def fill_holes(binary_image):
    inverted_binary_image = cv2.bitwise_not(binary_image)

    kernel = np.ones((3, 3), np.uint8)
    filled_image = cv2.morphologyEx(inverted_binary_image, cv2.MORPH_CLOSE, kernel)
    filled_image = cv2.bitwise_not(filled_image)

    return filled_image


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


def remove_single_pixel_noise(binary_image):
    # Create a copy of the image to modify
    filtered_image = binary_image.copy()

    # Find all connected components (white blobs) in the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Iterate through all connected components
    for label in range(1, num_labels):  # Skip the background label (0)
        # If the component is a single pixel (area = 1), remove it
        if stats[label, cv2.CC_STAT_AREA] == 1:
            x, y, w, h, area = stats[label]
            filtered_image[y:y+h, x:x+w] = 0

    return filtered_image

def remove_noise(image, min_size):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    img2 = np.zeros((output.shape), np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2

def MUGE_predict(input_img, checkpoint_path, threshold = 1):
    predicted_img = predict(checkpoint_path, input_img)
    # plt.imshow(predicted_img)
    # plt.axis('off')
    # plt.show()

    # print(predicted_img)
    # print("Predicted Image Range:", predicted_img.min(), predicted_img.max())
    if len(predicted_img.shape) == 3:
      predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2GRAY)


    # binary_image = predicted_img
    _, binary_image = cv2.threshold(predicted_img, 60, 255, cv2.THRESH_BINARY)
    # print(binary_image)
    # plt.imshow(binary_image)
    # plt.axis('off')
    # plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    binary_result = skeletonize(binary_image)
    # print('21')
    # plt.imshow(binary_result)
    # plt.axis('off')
    # plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_result = cv2.morphologyEx(binary_result, cv2.MORPH_CLOSE, kernel)
    # print('5')
    # plt.imshow(binary_result)
    # plt.axis('off')
    # plt.show()

    binary_result = remove_noise(binary_result, 5)
    # print('noise')
    # plt.imshow(binary_result)
    # plt.axis('off')
    # plt.show()

    # color_result = replace_with_red(input_img, binary_result)
    # print('color-5')
    # plt.imshow(color_result)
    # plt.axis('off')
    # plt.show()

    # filled_image = fill_holes(binary_image)

    # binary_result = skeletonize(filled_image)
    # filter_binary_result = remove_single_pixel_noise(binary_result)

    # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    # color_result = replace_with_red(input_img, binary_result)

    # coordinates = np.column_stack(np.where(binary_result > 0))
    # pixels_result = [tuple(coord) for coord in coordinates]

    # return filled_image, binary_result, filter_binary_result, color_result

    return binary_result

def calculate_ODS_metrics(pred_binary, gt_binary, distance_threshold=2.0):
    pred_binary = np.array(pred_binary, dtype=bool)
    gt_binary = np.array(gt_binary, dtype=bool)
    pred_points = np.argwhere(pred_binary)
    gt_points = np.argwhere(gt_binary)

    successful_points = 0

    if len(pred_points) > 0 and len(gt_points) > 0:
        distances = cdist(pred_points, gt_points)
        min_distances = np.min(distances, axis=1)
        min_indices = np.argmin(distances, axis=1)

        matched_gt_points = np.zeros(len(gt_points), dtype=bool)

        for i in range(len(pred_points)):
            while min_distances[i] < distance_threshold:
                matched_gt_index = min_indices[i]

                if not matched_gt_points[matched_gt_index]:
                    successful_points += 1
                    matched_gt_points[matched_gt_index] = True
                    break
                else:
                    distances[i, matched_gt_index] = distance_threshold + 1
                    min_distances[i] = np.min(distances[i])
                    min_indices[i] = np.argmin(distances[i])
            else:
                continue

    if len(gt_points) > 0:
        recall = successful_points / len(gt_points)
    else:
        recall = 0.0

    if len(pred_points) > 0:
        precision = successful_points / len(pred_points)
    else:
        precision = 0.0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return precision, recall, f1_score

def filter_points_by_distance(pred_points, gt_points, threshold):
    tree = cKDTree(gt_points)
    distances, _ = tree.query(pred_points)
    filtered_points = pred_points[distances <= threshold]
    return filtered_points

def calculate_frechet_similarity(pred_mask, gt_mask, max_distance=1000, threshold=200):
    pred_points = np.argwhere(pred_mask > 0)
    gt_points = np.argwhere(gt_mask > 0)

    if len(pred_points) > 1 and len(gt_points) > 1:
        # Filter pred_points based on distance threshold
        pred_points_filtered = filter_points_by_distance(pred_points, gt_points, threshold)

        if len(pred_points_filtered) > 1:
            pred_line = LineString(pred_points_filtered)
            gt_line = LineString(gt_points)

            frechet_dist = frechet_distance(pred_line, gt_line)
        else:
            frechet_dist = float('inf')
    else:
        frechet_dist = float('inf')

    similarity = max(0, 1 - frechet_dist / max_distance)

    return similarity

def eval(checkpoint_path, input_csv, metric_method, binary_threshold, distance_threshold, save_path):
    df = pd.read_csv(input_csv)
    stats = defaultdict(lambda: defaultdict(lambda: {'num': 0, 'sum': 0, 'mean_metric': 0}))
    overall_stats = {'num': 0, 'sum': 0, 'mean_metric': 0}
    total_time = 0

    for index, row in df.iterrows():
        start_time = time.time()

        input_img = cv2.imread(row['path'], cv2.IMREAD_COLOR)
        pred_mask = MUGE_predict(input_img, checkpoint_path, threshold = binary_threshold)
        gt_mask = draw_shoreline(input_img, row['label'])
        
        if metric_method == 'ODS':
            precision, recall, metric = calculate_ODS_metrics(pred_mask, gt_mask, distance_threshold)
        elif metric_method == 'frechet_similarity':
            metric = calculate_frechet_similarity(pred_mask, gt_mask)

        overall_stats['num'] += 1
        overall_stats['sum'] += metric
        overall_stats['mean_metric'] = overall_stats['sum'] / overall_stats['num']

        for col in df.columns:
            if col not in ['path', 'label']:
                feature_value = row[col]
                stats[col][feature_value]['num'] += 1
                stats[col][feature_value]['sum'] += metric
                stats[col][feature_value]['mean_metric'] = stats[col][feature_value]['sum'] / stats[col][feature_value]['num']

        iteration_time = time.time() - start_time  # End timer
        total_time += iteration_time

        print(f'{index + 1} -------- current_metric {metric:.3f} -------- mean_metric {overall_stats["mean_metric"]:.3f} -------- Time: {iteration_time:.3f} s')

    print("Final mean metric for each feature and type:")
    for feature in stats:
        for feature_value in stats[feature]:
            print(f'Feature: {feature}, Type: {feature_value}, Mean metric: {stats[feature][feature_value]["mean_metric"]}')
    print(f'Overall Mean metric: {overall_stats["mean_metric"]}')
    average_time = total_time / overall_stats['num']
    print(f'Total average time per iteration: {average_time:.2f} seconds')

    with open(save_path, 'a') as file:
      file.write("Final mean metric for each feature and type:\n")
      for feature in stats:
          for feature_value in stats[feature]:
              file.write(f'Feature: {feature}, Type: {feature_value}, Mean metric: {stats[feature][feature_value]["mean_metric"]}\n')
      file.write(f'Overall Mean metric: {overall_stats["mean_metric"]}\n')
      average_time = total_time / overall_stats['num']
      file.write(f'Total average time per iteration: {average_time:.2f} seconds\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coastline detection and evaluation script")
    parser.add_argument("--binary_threshold", type=int, default=200, help="Binary threshold")
    parser.add_argument("--model_path", type=str, help="Path to the trained model file")
    parser.add_argument("--metric_method", type=str, default='ODS', help="Evaluation metric method")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the results")
    parser.add_argument("--distance_threshold", type=int, default=50, help="Distance threshold for evaluation")

    args = parser.parse_args()

    with open(args.save_path, 'a') as file:
      file.write(f'------ Evaluating with checkpoint: {args.model_path} ------\n')
    print(f'------ Evaluating with checkpoint: {args.model_path} ------')
    eval(args.model_path, args.input_csv, args.metric_method, args.binary_threshold, args.distance_threshold, args.save_path)