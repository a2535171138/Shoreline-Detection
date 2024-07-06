import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import re

def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
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
        self.add_module('conv1', nn.Conv2d(input_features, out_features, kernel_size=3, stride=1, padding=2, bias=True))
        self.add_module('norm1', nn.BatchNorm2d(out_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, bias=True))
        self.add_module('norm2', nn.BatchNorm2d(out_features))

    def forward(self, x):
        x1, x2 = x
        new_features = super(_DenseLayer, self).forward(F.relu(x1))
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
        all_pads = [0, 0, 1, 3, 7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(out_features, out_features, kernel_size, stride=2, padding=pad))
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
    def __init__(self, in_features, mid_features, out_features=None, stride=1, use_act=True):
        super(DoubleConvBlock, self).__init__()
        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features, 3, padding=1, stride=stride)
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
        self.block_1 = DoubleConvBlock(3, 32, 64, stride=2)
        self.block_2 = DoubleConvBlock(64, 128, use_act=False)
        self.dblock_3 = _DenseBlock(2, 128, 256)
        self.dblock_4 = _DenseBlock(3, 256, 512)
        self.dblock_5 = _DenseBlock(3, 512, 512)
        self.dblock_6 = _DenseBlock(3, 512, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.side_1 = SingleConvBlock(64, 128, 2)
        self.side_2 = SingleConvBlock(128, 256, 2)
        self.side_3 = SingleConvBlock(256, 512, 2)
        self.side_4 = SingleConvBlock(512, 512, 1)
        self.side_5 = SingleConvBlock(512, 256, 1)
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
        self.apply(weight_init)

    def slice(self, tensor, slice_shape):
        t_shape = tensor.shape
        height, width = slice_shape
        if t_shape[-1] != slice_shape[-1]:
            new_tensor = F.interpolate(tensor, size=(height, width), mode='bicubic', align_corners=False)
        else:
            new_tensor = tensor
        return new_tensor

    def forward(self, x):
        assert x.ndim == 4, x.shape
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_down = self.maxpool(block_3)
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)
        block_2_resize_half = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(block_3_down + block_2_resize_half)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)
        block_5_pre_dense = self.pre_dense_5(block_4_down)
        block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])
        block_5_add = block_5 + block_4_side
        block_6_pre_dense = self.pre_dense_6(block_5)
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        out_5 = self.up_block_5(block_5)
        out_6 = self.up_block_6(block_6)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]
        block_cat = torch.cat(results, dim=1)
        block_cat = self.block_cat(block_cat)
        results.append(block_cat)
        return results

def image_normalization(img, img_min=0, img_max=255, epsilon=1e-12):
    img = np.float32(img)
    img = (img - np.min(img)) * (img_max - img_min) / ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img

def transform(img):
    img = np.array(img, dtype=np.float32)
    mean_pixel_values = [103.939, 116.779, 123.68, 137.86]
    mean_bgr = mean_pixel_values[0:3] if len(mean_pixel_values) == 4 else mean_pixel_values,
    img -= mean_bgr
    crop_size = 512
    img = cv2.resize(img, dsize=(crop_size, crop_size))
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img.copy()).float()
    return img

def non_max_suppression_for_edges(edge_image):
    sobelx = cv2.Sobel(edge_image, cv2.CV_64F, 1, 0, ksize=21)
    sobely = cv2.Sobel(edge_image, cv2.CV_64F, 0, 1, ksize=21)
    gradient_direction = np.arctan2(sobely, sobelx)
    H, W = edge_image.shape
    suppressed_image = np.zeros((H, W), dtype=np.float32)
    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            q = 255
            r = 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = edge_image[i, j + 1]
                r = edge_image[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):
                q = edge_image[i + 1, j - 1]
                r = edge_image[i - 1, j + 1]
            elif (67.5 <= angle[i, j] < 112.5):
                q = edge_image[i + 1, j]
                r = edge_image[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q = edge_image[i - 1, j - 1]
                r = edge_image[i + 1, j + 1]
            if (edge_image[i, j] >= q) and (edge_image[i, j] >= r):
                suppressed_image[i, j] = edge_image[i, j]
            else:
                suppressed_image[i, j] = 0
    suppressed_image = (suppressed_image / suppressed_image.max() * 255).astype(np.uint8)
    return suppressed_image

def replace_with_red(input_img, predicted_img, threshold=200):
    output_img = input_img.copy()
    for i in range(predicted_img.shape[0]):
        for j in range(predicted_img.shape[1]):
            if predicted_img[i, j] > threshold:
                red_intensity = int((predicted_img[i, j] / 255) * 255)
                output_img[i, j] = [0, 0, red_intensity]
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
#     print(device)
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

def combine_binary_images(binary1, binary2):
    combined_image = np.zeros((binary1.shape[0], binary1.shape[1], 3), dtype=np.uint8)
    combined_image[binary1 > 0] = [255, 255, 0]
    combined_image[binary2 > 0] = [0, 255, 255]
    return combined_image

def process_images(checkpoint_path, image_paths, threshold):
    for image_path in image_paths:
        input_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        predicted_img = predict(checkpoint_path, input_img)
        _, binary_image = cv2.threshold(predicted_img, threshold, 255, cv2.THRESH_BINARY)
        filled_image = fill_holes(binary_image)
        skeleton = skeletonize(filled_image)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        output_img = replace_with_red(input_img, skeleton, threshold)
        # plt.imshow(input_img)
        # plt.axis('off')
        # plt.show()
        plt.imshow(predicted_img)
        plt.axis('off')
        plt.show()
        # plt.imshow(skeleton)
        # plt.axis('off')
        # plt.show()
        # plt.imshow(output_img)
        # plt.axis('off')
        # plt.show()


# def process_images(checkpoint_path, image_paths, threshold):
#     for image_path in image_paths:
#         print(f"Reading image from path: {image_path}")
#         input_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
#         if input_img is None:
#             print(f"Error: Unable to read image at path: {image_path}")
#             continue
        
#         predicted_img = predict(checkpoint_path, input_img)
        
#         if predicted_img is None:
#             print(f"Error: Prediction failed for image at path: {image_path}")
#             continue
        
#         plt.figure(figsize=(8, 6))
#         plt.imshow(predicted_img, cmap='gray')
#         plt.axis('off')
#         plt.title("Predicted Image")
#         output_image_path = image_path.replace('.jpg', '_predicted.jpg')
#         plt.savefig(output_image_path)
#         plt.show()

# Example usage:
# checkpoint_path = "C:\\Users\\padra\\comp9900_coaste-detect\\backend\\29_model.pth"
# image_paths = [
#     "C:\\Users\\padra\\Downloads\\dataset\\train\\images\\1662771387.Sat.Sep.10_10_56_27.AEST.2022.manly.snap.DocHarleyMD_registered.jpg", 
#     "C:\\Users\\padra\\Downloads\\dataset\\train\\images\\1663107120.Wed.Sep.14_08_12_00.AEST.2022.manly.snap.CatCoghlan_registered.jpg"
# ]
# threshold = 200

# process_images(checkpoint_path, image_paths, threshold)