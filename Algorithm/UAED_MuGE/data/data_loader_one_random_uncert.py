from torch.utils import data
from os.path import join, abspath, splitext, split, isdir, isfile
import numpy as np
import torch
import imageio
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random
import cv2
import re
import pandas as pd
import os


test_csv = '/content/drive/My Drive/test_set_200.csv'

def draw_shoreline(image, multiline_string):
    height, width = image.shape[:2]
    blank_image = np.zeros((height, width), dtype=np.uint8)
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
            cv2.line(blank_image, coordinates[i], coordinates[i + 1], (255,), 1)
    
    return blank_image

def replace_with_red(input_img, predicted_img):
    output_img = input_img.copy()
    mask = predicted_img > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_channel = np.zeros_like(output_img[:, :, 0])
    red_channel[mask] = 255
    red_channel = cv2.dilate(red_channel, kernel)
    output_img[red_channel > 0] = [255, 0, 0]

    return output_img
    
class BSDS_RCFLoader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='datasets', split='train', transform=False, csv_path='/content/drive/My Drive/train_set_1000_clean_balanced_weighted2.csv', epoch=0, aug_open=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.epoch = epoch
        self.aug_open = aug_open
        if self.split == 'train':
            print('------Train set CSV path:', csv_path)
            print('------aug_open:', aug_open)
            self.filelist = []
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.filelist.append((row['path'], row['label']))
        elif self.split == 'test':
            self.filelist = []
            df = pd.read_csv(test_csv)
            for _, row in df.iterrows():
                self.filelist.append((row['path'], row['label']))
        else:
            raise ValueError("Invalid split type!")

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        target_size = (481, 321)
        # target_size = (481, 361)
        image_path, label_ = self.filelist[index]
        img_ = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img_, target_size, interpolation=cv2.INTER_LINEAR)
    
        if self.split == "train":
            label = draw_shoreline(img_, label_)
            
            # Thicker
            kernel = np.ones((5, 5), np.uint8)
            dilated_image = cv2.dilate(label, kernel, iterations=1)
            dilated_only = dilated_image - label
            colored_image = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
            colored_image[dilated_only > 0] = (128, 128, 128)
            # label = colored_image.astype(np.float32)
            if colored_image.ndim == 3:
              colored_image = np.squeeze(colored_image[:, :, 0])
            assert colored_image.ndim == 2
            label = colored_image + label
            
            label = cv2.resize(label, target_size, interpolation=cv2.INTER_LINEAR)
    
            if self.epoch < 5 and self.aug_open:
                # 将图像转换为 PIL 图像
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                label = label.astype(np.uint8)  # 确保标签为 uint8 类型
                label_pil = Image.fromarray(label, mode='L')
    
                # 图像增强
                proportion = 0.7 # 0.5
                if random.random() > proportion:
                    img_pil = TF.adjust_contrast(img_pil, random.uniform(0.5, 1.5))
                if random.random() > proportion:
                    img_pil = TF.adjust_saturation(img_pil, random.uniform(0.5, 1.5))
                if random.random() > proportion:
                    img_pil = TF.adjust_brightness(img_pil, random.uniform(0.5, 1.5))
                if random.random() > proportion:
                    angle = random.uniform(-15, 15)
                    img_pil = TF.rotate(img_pil, angle)
                    label_pil = TF.rotate(label_pil, angle)
                if random.random() > proportion:
                    translate = (random.uniform(-0.1, 0.1) * target_size[1], random.uniform(-0.1, 0.1) * target_size[0])
                    img_pil = TF.affine(img_pil, angle=0, translate=translate, scale=1, shear=0)
                    label_pil = TF.affine(label_pil, angle=0, translate=translate, scale=1, shear=0)
                if random.random() > proportion:
                    scale = random.uniform(0.8, 1.4)
                    img_pil = TF.affine(img_pil, angle=0, translate=(0, 0), scale=scale, shear=0)
                    label_pil = TF.affine(label_pil, angle=0, translate=(0, 0), scale=scale, shear=0)
    
                # 将增强后的图像转换回 numpy 数组
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                label = np.array(label_pil)
    
            # # 保存增强后的图片
            # color_result = replace_with_red(img, label * 255)
            # folder_path = 'Augment Images'
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)
            # save_path = f'{folder_path}/epoch_{self.epoch}_rand_{random.randint(0, 10000)}.jpg'
            # cv2.imwrite(save_path, color_result) 
    
            
            
            _, label = cv2.threshold(label, 50, 1, cv2.THRESH_BINARY)
            label = torch.from_numpy(label)
            label = label[1:label.size(0), 1:label.size(1)]
            label = label.float()
            lb_mean = label.mean(dim=0).unsqueeze(0)
            lb_std = label.std(dim=0).unsqueeze(0)
    
        img = transforms.ToTensor()(img)
        img = img[:, 1:img.size(1), 1:img.size(2)]
        img = img.float()
    
        if self.split == "train":
            return img, label, lb_mean, lb_std
        else:
            return img
            
    def set_epoch(self, epoch):
        self.epoch = epoch
        
        if self.epoch < 5 and self.aug_open:
            print('------Augment is open')
        else:
            print('------Augment is close')
