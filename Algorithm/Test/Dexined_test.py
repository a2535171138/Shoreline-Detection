import argparse
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
from collections import defaultdict
import time
from scipy.spatial.distance import cdist
from shapely import LineString
from shapely import frechet_distance
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree

def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        # torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        # torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
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
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.norm_layer1 = nn.GroupNorm(4, 64)
        self.norm_layer2 = nn.GroupNorm(4, 64)

    def forward(self, x):
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = self.relu(self.norm_layer2(self.conv2(attn)))
        attn = F.softmax(self.conv3(attn), dim=1)

        return ((x * attn).sum(1)).unsqueeze(1)

class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        # self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(input_features, out_features, kernel_size=3, stride=1, padding=2, bias=True)),
        self.add_module('norm1', nn.BatchNorm2d(out_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, bias=True)),
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
    def __init__(self, in_features, out_features, stride, use_bs=True):
        super(SingleConvBlock, self).__init__()
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=True)
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
    def __init__(self):
        super(DexiNed, self).__init__()
        self.block_1 = DoubleConvBlock(3, 32, 64, stride=2,)
        self.block_2 = DoubleConvBlock(64, 128, use_act=False)
        self.dblock_3 = _DenseBlock(2, 128, 256) # [128,256,100,100]
        self.dblock_4 = _DenseBlock(3, 256, 512)
        self.dblock_5 = _DenseBlock(3, 512, 512)
        self.dblock_6 = _DenseBlock(3, 512, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.side_1 = SingleConvBlock(64, 128, 2)
        self.side_2 = SingleConvBlock(128, 256, 2)
        self.side_3 = SingleConvBlock(256, 512, 2)
        self.side_4 = SingleConvBlock(512, 512, 1)
        self.side_5 = SingleConvBlock(512, 256, 1) # Sory I forget to comment this line :(

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
        self.block_cat = SingleConvBlock(6, 1, stride=1, use_bs=False) # hed fusion method
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
    # whenever an inconsistent image
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

def Dexined_predict(input_img, checkpoint_path, threshold = 200):
    predicted_img = predict(checkpoint_path, input_img)
    _, binary_image = cv2.threshold(predicted_img, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    binary_result = skeletonize(binary_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_result = cv2.morphologyEx(binary_result, cv2.MORPH_CLOSE, kernel)
    binary_result = remove_noise(binary_result, 5)

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
        pred_mask = Dexined_predict(input_img, checkpoint_path, threshold = binary_threshold)
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