import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import re
import matplotlib.pyplot as plt
import uuid
from PIL import Image, ImageDraw, ImageOps
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

shadow_open = False
augment_open = False
write_img = False
train_csv = '/content/drive/My Drive/train_set_1000_clean_unbalanced_lessShadow.csv'
test_csv = '/content/drive/My Drive/test_set_200.csv'

def bezier_curve(p1, p2, control, num_points=100):
    points = []
    for t in np.linspace(0, 1, num_points):
        x = (1-t)**2 * p1[0] + 2*(1-t)*t * control[0] + t**2 * p2[0]
        y = (1-t)**2 * p1[1] + 2*(1-t)*t * control[1] + t**2 * p2[1]
        points.append((x, y))
    return points

def generate_random_polygon(width, height, num_vertices, max_offset, size_factor=0.6):
    cx = random.uniform(width * 0.1, width * 0.9)
    cy = random.uniform(height * 0.1, height * 0.9)
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
    points = []
    for angle in angles:
        r = random.uniform(size_factor * 0.3, size_factor * 0.5) * min(width, height)
        x = cx + r * np.cos(angle) + random.uniform(-max_offset, max_offset)
        y = cy + r * np.sin(angle) + random.uniform(-max_offset, max_offset)
        points.append((x, y))
    return points

def add_random_shadow(draw, width, height, shadow_count, num_vertices, max_shadow_size, max_offset, size_factor):
    shadows = []
    for _ in range(shadow_count):
        points = generate_random_polygon(width, height, num_vertices, max_offset, size_factor)

        alpha = random.randint(50, 150)
        shadow_color = (0, 0, 0, alpha)

        bezier_points = []
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            control = ((p1[0] + p2[0]) / 2 + random.uniform(-max_offset, max_offset),
                       (p1[1] + p2[1]) / 2 + random.uniform(-max_offset, max_offset))
            bezier_points.extend(bezier_curve(p1, p2, control))

        shadow_polygon = Polygon(bezier_points)
        if not shadow_polygon.is_valid:
            shadow_polygon = shadow_polygon.buffer(0)

        shadows.append(shadow_polygon)

    merged_shadows = unary_union(shadows)

    if merged_shadows.geom_type == 'Polygon':
        shadow_polygons = [merged_shadows]
    elif merged_shadows.geom_type == 'MultiPolygon':
        shadow_polygons = merged_shadows.geoms
    else:
        shadow_polygons = []

    for shadow in shadow_polygons:
        shadow_points = [(x, y) for x, y in shadow.exterior.coords]
        draw.polygon(shadow_points, fill=shadow_color)
        
def adjust_contrast(image, factor):
    """Adjusts the contrast of an image."""
    mean = np.mean(image)
    return np.uint8(np.clip((image - mean) * factor + mean, 0, 255))

def adjust_brightness(image, factor):
  adjusted_image = np.uint8(image * factor)
  adjusted_image = np.clip(adjusted_image, 0, 255)
  return adjusted_image

def adjust_saturation(image, factor):
    """Adjusts the saturation of an image."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
def augment_image(image):
    """Performs data augmentation on the image."""
    
    # Randomly adjust brightness (delta range: -50 to 50)
    brightness_delta = np.random.uniform(0.5, 1.5)
    image = adjust_brightness(image, brightness_delta)
    
    # Randomly adjust contrast (factor range: 0.5 to 1.5)
    contrast_factor = np.random.uniform(0.3, 1.7)
    image = adjust_contrast(image, contrast_factor)
    
    # Randomly adjust saturation (factor range: 0.5 to 1.5)
    saturation_factor = np.random.uniform(0.5, 1.5)
    image = adjust_saturation(image, saturation_factor)

    return image
    
    
DATASET_NAMES = [
    'BIPED',
    'BSDS',
    'BRIND',
    'BSDS300',
    'CID',
    'DCD',
    'MDBD', #5
    'PASCAL',
    'NYUD',
    'CLASSIC'
]  # 8

def dataset_info(dataset_name, is_linux=True):
    if is_linux:

        config = {
            'BSDS': {
                'img_height': 512, #321
                'img_width': 512, #481
                'train_list': 'train_pair.lst',
                'test_list': 'test_pair.lst',
                'data_dir': '/opt/dataset/BSDS',  # mean_rgb
                'yita': 0.5
            },
            'BRIND': {
                'img_height': 512,  # 321
                'img_width': 512,  # 481
                'train_list': 'train_pair2.lst',
                'test_list': 'test_pair.lst',
                'data_dir': '/opt/dataset/BRIND',  # mean_rgb
                'yita': 0.5
            },
            'BSDS300': {
                'img_height': 512, #321
                'img_width': 512, #481
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/BSDS300',  # NIR
                'yita': 0.5
            },
            'PASCAL': {
                'img_height': 416, # 375
                'img_width': 512, #500
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/PASCAL',  # mean_rgb
                'yita': 0.3
            },
            'CID': {
                'img_height': 512,
                'img_width': 512,
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/CID',  # mean_rgb
                'yita': 0.3
            },
            'NYUD': {
                'img_height': 448,#425
                'img_width': 560,#560
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/NYUD',  # mean_rgb
                'yita': 0.5
            },
            'MDBD': {
                'img_height': 720,
                'img_width': 1280,
                'test_list': 'test_pair.lst',
                'train_list': 'train_pair.lst',
                'data_dir': '/opt/dataset/MDBD',  # mean_rgb
                'yita': 0.3
            },
            'BIPED': {
                'img_height': 720, #720 # 1088
                'img_width': 1280, # 1280 5 1920
                'test_list': 'test_pair.lst',
                'train_list': 'train_rgb.lst',
                'data_dir': '/opt/dataset/BIPED',  # mean_rgb
                'yita': 0.5
            },
            'CLASSIC': {
                'img_height': 512,
                'img_width': 512,
                'test_list': None,
                'train_list': None,
                'data_dir': 'data',  # mean_rgb
                'yita': 0.5
            },
            'DCD': {
                'img_height': 352, #240
                'img_width': 480,# 360
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/DCD',  # mean_rgb
                'yita': 0.2
            }
        }
    else:
        config = {
            'BSDS': {'img_height': 512,  # 321
                     'img_width': 512,  # 481
                     'test_list': 'test_pair.lst',
                     'train_list': 'train_pair.lst',
                     'data_dir': 'C:/Users/xavysp/dataset/BSDS',  # mean_rgb
                     'yita': 0.5},
            'BSDS300': {'img_height': 512,  # 321
                        'img_width': 512,  # 481
                        'test_list': 'test_pair.lst',
                        'data_dir': 'C:/Users/xavysp/dataset/BSDS300',  # NIR
                        'yita': 0.5},
            'PASCAL': {'img_height': 375,
                       'img_width': 500,
                       'test_list': 'test_pair.lst',
                       'data_dir': 'C:/Users/xavysp/dataset/PASCAL',  # mean_rgb
                       'yita': 0.3},
            'CID': {'img_height': 512,
                    'img_width': 512,
                    'test_list': 'test_pair.lst',
                    'data_dir': 'C:/Users/xavysp/dataset/CID',  # mean_rgb
                    'yita': 0.3},
            'NYUD': {'img_height': 425,
                     'img_width': 560,
                     'test_list': 'test_pair.lst',
                     'data_dir': 'C:/Users/xavysp/dataset/NYUD',  # mean_rgb
                     'yita': 0.5},
            'MDBD': {'img_height': 720,
                         'img_width': 1280,
                         'test_list': 'test_pair.lst',
                         'train_list': 'train_pair.lst',
                         'data_dir': 'C:/Users/xavysp/dataset/MDBD',  # mean_rgb
                         'yita': 0.3},
            'BIPED': {'img_height': 720,  # 720
                      'img_width': 1280,  # 1280
                      'test_list': 'test_pair.lst',
                      'train_list': 'train_rgb.lst',
                      'data_dir': 'C:/Users/xavysp/dataset/BIPED',  # WIN: '../.../dataset/BIPED/edges'
                      'yita': 0.5},
            'CLASSIC': {'img_height': 512,
                        'img_width': 512,
                        'test_list': None,
                        'train_list': None,
                        'data_dir': 'data',  # mean_rgb
                        'yita': 0.5},
            'DCD': {'img_height': 240,
                    'img_width': 360,
                    'test_list': 'test_pair.lst',
                    'data_dir': 'C:/Users/xavysp/dataset/DCD',  # mean_rgb
                    'yita': 0.2}
        }
    return config[dataset_name]


class TestDataset(Dataset):
    def __init__(self,
                 data_root,
                 test_data,
                 mean_bgr,
                 img_height,
                 img_width,
                 test_list=None,
                 arg=None
                 ):
        if test_data not in DATASET_NAMES:
            raise ValueError(f"Unsupported dataset: {test_data}")

        self.data_root = data_root
        self.test_data = test_data
        self.test_list = test_list
        self.args=arg
        # self.arg = arg
        # self.mean_bgr = arg.mean_pixel_values[0:3] if len(arg.mean_pixel_values) == 4 \
        #     else arg.mean_pixel_values
        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.data_index = self._build_index()

        print(f"mean_bgr: {self.mean_bgr}")

    def _build_index(self):
        sample_indices = []
        # if self.test_data == "CLASSIC":
        #     # for single image testing
        #     images_path = os.listdir(self.data_root)
        #     labels_path = None
        #     sample_indices = [images_path, labels_path]
        # else:
        #     # image and label paths are located in a list file

        #     if not self.test_list:
        #         raise ValueError(
        #             f"Test list not provided for dataset: {self.test_data}")

        #     list_name = os.path.join(self.data_root, self.test_list)
        #     if self.test_data.upper()=='BIPED':

        #         with open(list_name) as f:
        #             files = json.load(f)
        #         for pair in files:
        #             tmp_img = pair[0]
        #             tmp_gt = pair[1]
        #             sample_indices.append(
        #                 (os.path.join(self.data_root, tmp_img),
        #                  os.path.join(self.data_root, tmp_gt),))
        #     else:
        #         with open(list_name, 'r') as f:
        #             files = f.readlines()
        #         files = [line.strip() for line in files]
        #         pairs = [line.split() for line in files]

        #         for pair in pairs:
        #             tmp_img = pair[0]
        #             tmp_gt = pair[1]
        #             sample_indices.append(
        #                 (os.path.join(self.data_root, tmp_img),
        #                  os.path.join(self.data_root, tmp_gt),))
        
        
        df = pd.read_csv(test_csv)
        
        for _, row in df.iterrows():
          sample_indices.append((row['path'], row['label']))
          
        return sample_indices

    def __len__(self):
        # return len(self.data_index[0]) if self.test_data.upper()=='CLASSIC' else len(self.data_index)
        return len(self.data_index)

    def __getitem__(self, idx):
        # # get data sample
        # # image_path, label_path = self.data_index[idx]
        # if self.data_index[1] is None:
        #     image_path = self.data_index[0][idx]
        # else:
        #     image_path = self.data_index[idx][0]
        # label_path = None if self.test_data == "CLASSIC" else self.data_index[idx][1]
        # img_name = os.path.basename(image_path)
        # file_name = os.path.splitext(img_name)[0] + ".png"

        # # base dir
        # if self.test_data.upper() == 'BIPED':
        #     img_dir = os.path.join(self.data_root, 'imgs', 'test')
        #     gt_dir = os.path.join(self.data_root, 'edge_maps', 'test')
        # elif self.test_data.upper() == 'CLASSIC':
        #     img_dir = self.data_root
        #     gt_dir = None
        # else:
        #     img_dir = self.data_root
        #     gt_dir = self.data_root

        # # load data
        # image = cv2.imread(os.path.join(img_dir, image_path), cv2.IMREAD_COLOR)
        # if not self.test_data == "CLASSIC":
        #     label = cv2.imread(os.path.join(
        #         gt_dir, label_path), cv2.IMREAD_COLOR)
        # else:
        #     label = None

        
        image_path, label_ = self.data_index[idx]

        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = draw_shoreline(image, label_)
        
        im_shape = [image.shape[0], image.shape[1]]
        file_name = os.path.basename(image_path)
        
        image, label = self.transform(img=image, gt=label)

        return dict(images=image, labels=label, file_names=file_name, image_shape=im_shape)

    def transform(self, img, gt):
        # gt[gt< 51] = 0 # test without gt discrimination
        if self.test_data == "CLASSIC":
            img_height = self.img_height
            img_width = self.img_width
            print(
                f"actual size: {img.shape}, target size: {( img_height,img_width,)}")
            # img = cv2.resize(img, (self.img_width, self.img_height))
            img = cv2.resize(img, (img_width,img_height))
            gt = None

        # Make images and labels at least 512 by 512
        elif img.shape[0] < 512 or img.shape[1] < 512:
            img = cv2.resize(img, (self.args.test_img_width, self.args.test_img_height)) # 512
            gt = cv2.resize(gt, (self.args.test_img_width, self.args.test_img_height)) # 512

        # Make sure images and labels are divisible by 2^4=16
        elif img.shape[0] % 16 != 0 or img.shape[1] % 16 != 0:
            img_width = ((img.shape[1] // 16) + 1) * 16
            img_height = ((img.shape[0] // 16) + 1) * 16
            img = cv2.resize(img, (img_width, img_height))
            gt = cv2.resize(gt, (img_width, img_height))
        else:
            img_width =self.args.test_img_width
            img_height =self.args.test_img_height
            img = cv2.resize(img, (img_width, img_height))
            gt = cv2.resize(gt, (img_width, img_height))

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        _, gt = cv2.threshold(gt, 50, 255, cv2.THRESH_BINARY)
        
        if self.test_data == "CLASSIC":
            gt = np.zeros((img.shape[:2]))
            gt = torch.from_numpy(np.array([gt])).float()
        else:
            gt = np.array(gt, dtype=np.float32)
            if len(gt.shape) == 3:
                gt = gt[:, :, 0]
            gt /= 255.
            gt = torch.from_numpy(np.array([gt])).float()

        return img, gt


class BipedDataset(Dataset):
    train_modes = ['train', 'test', ]
    dataset_types = ['rgbr', ]
    data_types = ['aug', ]

    def __init__(self,
                 data_root,
                 img_height,
                 img_width,
                 mean_bgr,
                 train_mode='train',
                 dataset_type='rgbr',
                 #  is_scaling=None,
                 # Whether to crop image or otherwise resize image to match image height and width.
                 crop_img=False,
                 arg=None
                 ):
        self.data_root = data_root
        self.train_mode = train_mode
        self.dataset_type = dataset_type
        self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.arg = arg

        self.data_index = self._build_index()

    def _build_index(self):
        assert self.train_mode in self.train_modes, self.train_mode
        assert self.dataset_type in self.dataset_types, self.dataset_type
        assert self.data_type in self.data_types, self.data_type

        data_root = os.path.abspath(self.data_root)
        sample_indices = []

        print(train_csv)
        df = pd.read_csv(train_csv)
        
        for _, row in df.iterrows():
          sample_indices.append((row['path'], row['label']))
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_ = self.data_index[idx]

        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = draw_shoreline(image, label_)
        
        if augment_open:
            if write_img:
                cv2.imwrite('/content/drive/My Drive/augment_images/' + str(idx) + "_original.jpg", image)
                
            image = augment_image(image)
            
            if write_img:
                cv2.imwrite('/content/drive/My Drive/augment_images/' + str(idx) + ".jpg", image)
                print('Write to '+'/content/drive/My Drive/augment_images/' + str(idx) + ".jpg")

        if shadow_open:
            if random.random() < 0.5:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image)
                draw = ImageDraw.Draw(pil_image, 'RGBA')
            
                shadow_count = random.randint(5, 15)
                num_vertices = random.randint(8, 15) 
                max_shadow_size = random.randint(100, 300)
                max_offset = random.randint(30, 50)
                size_factor = 0.4 
                width, height = pil_image.size
        
                add_random_shadow(draw, width, height, shadow_count, num_vertices, max_shadow_size, max_offset, size_factor)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)
                
                if write_img:
                    cv2.imwrite('/content/drive/My Drive/shadow_images/' + str(idx) + ".jpg", image)
                    print('Write to '+'/content/drive/My Drive/shadow_images/' + str(idx) + ".jpg")
                
        image, label = self.transform(img=image, gt=label)
        return dict(images=image, labels=label)

    def transform(self, img, gt):
        crop_size = self.img_height if self.img_height == self.img_width else None#448# MDBD=480 BIPED=480/400 BSDS=352

        gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        _, gt = cv2.threshold(gt, 50, 255, cv2.THRESH_BINARY)
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        gt /= 255. # for DexiNed input and BDCN
        gt[gt > 0.1] +=0.2#0.4
        gt = np.clip(gt, 0., 1.)
        
        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        i_h, i_w,_ = img.shape
        img = cv2.resize(img, dsize=(crop_size, crop_size))
        
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()
        return img, gt


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