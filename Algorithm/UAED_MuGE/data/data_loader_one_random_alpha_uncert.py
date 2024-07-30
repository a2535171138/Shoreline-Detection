from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
import numpy as np
import torch
import imageio
import torchvision.transforms as transforms
import scipy.io
import random
import cv2
import re
import pandas as pd

train_csv = '/content/drive/My Drive/train_set_20.csv'
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

        _, binary_image = cv2.threshold(blank_image, 127, 1, cv2.THRESH_BINARY)

    return binary_image


#
# class BSDS_RCFLoader(data.Dataset):
#     """
#     Dataloader BSDS500
#     """
#     def __init__(self, root='./dataset', split='train', transform=False):
#         self.root = root
#         self.split = split
#         self.transform = transform
#         if self.split == 'train':
#             self.filelist = './dataset/train.lst'
#             # self.filelist = join(self.root, 'train.lst')
#
#         elif self.split == 'test':
#             # self.filelist = join(self.root, 'test.lst')
#             self.filelist = './dataset/test.lst'
#         else:
#             raise ValueError("Invalid split type!")
#         with open(self.filelist, 'r') as f:
#             self.filelist = f.readlines()
#
#     def __len__(self):
#         return len(self.filelist)
#
#     def __getitem__(self, index):
#         target_size = (481,321)
#
#         if self.split == "train":
#             img_lb_file = self.filelist[index].strip("\n").split(" ")
#             img_file=img_lb_file[0]
#
#             label_list=[]
#             # lb = scipy.io.loadmat(join(img_lb_file[1]))
#             # # print(lb)
#             # lb = np.asarray(lb['image'])
#             # label = torch.from_numpy(lb)
#             # label = label[1:label.size(0), 1:label.size(1)]
#             # label = label.float()
#
#
#
#             for i_label in range(1,len(img_lb_file)):
#                 lb = scipy.io.loadmat(join(img_lb_file[i_label]))
#                 lb=np.asarray(lb['image'])
#                 label = torch.from_numpy(lb)
#                 label = label[1:label.size(0), 1:label.size(1)]
#                 label = label.float()
#                 label_list.append(label.unsqueeze(0))
#             # for i_label in range(1, len(img_lb_file)):
#             #     lb = scipy.io.loadmat(img_lb_file[i_label])
#             #     lb = np.asarray(lb['image'])
#             #     label = torch.from_numpy(lb).float()
#             #     label = label[:label.size(0)-1, :label.size(1)-1]
#             #     label_list.append(label.unsqueeze(0))
#
#             labels=torch.cat(label_list,0)
#             label_max,label_min=(torch.sum(labels,dim=(1,2))).max(),(torch.sum(labels,dim=(1,2))).min()
#
#             lb_mean=label.mean(dim=0).unsqueeze(0)
#             lb_std=label.std(dim=0).unsqueeze(0)
#             lb_index=random.randint(2,len(img_lb_file))-1
#             lb_file=img_lb_file[lb_index]
#
#         else:
#             img_lb_file = self.filelist[index].strip("\n").split(" ")
#             img_file = img_lb_file[0]
#             # img_file = self.filelist[index].rstrip()
#
#
#
#         img = imageio.imread(join(img_file))
#         img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
#
#         img = transforms.ToTensor()(img)
#         img = img[:, 1:img.size(1), 1:img.size(2)]
#         img = img.float()
#
#
#         if self.split == "train":
#
#             # lb = scipy.io.loadmat(join(lb_file))
#             # lb=np.asarray(lb['image'])
#             # label = torch.from_numpy(lb)
#             # label = label[1:label.size(0), 1:label.size(1)]
#             # label = label.unsqueeze(0)
#             # label = label.float()
#             lb = scipy.io.loadmat(join(lb_file))
#             lb=np.asarray(lb['image'])
#             label = torch.from_numpy(lb)
#             # print(label.shape)
#             label = label[1:label.size(0), 1:label.size(1)]
#             label = label.unsqueeze(0)
#             label = label.float()
#             label_style=(torch.sum(label)-label_min)/(label_max-label_min+1e-10)
#             # print(img.shape)
#             # print(label.shape)
#
#             # assert label_style>=0 and label_style<=1
#             return img, label,lb_mean,lb_std,label_style
#             # return img, label, lb_mean, lb_std
#
#         else:
#             return img


class BSDS_RCFLoader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='./dataset', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            # self.filelist = join(self.root, 'train.lst')
            self.filelist = []
            df = pd.read_csv(train_csv)
            for _, row in df.iterrows():
                self.filelist.append((row['path'], row['label']))


        elif self.split == 'test':

            # self.filelist = join(self.root, 'test.lst')

            self.filelist = []

            df = pd.read_csv(test_csv)

            for _, row in df.iterrows():
                self.filelist.append((row['path'], row['label']))
        else:
            raise ValueError("Invalid split type!")
        # with open(self.filelist, 'r') as f:
        #     self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        target_size = (481, 321)

        image_path, label_ = self.filelist[index]
        # img = imageio.imread(join(self.root,img_file))
        # print(image_path)
        img_ = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # print(img_)
        # resize !!!!!!!!!!
        img = cv2.resize(img_, target_size, interpolation=cv2.INTER_LINEAR)
        # print(f"img: {img.shape}")


        if self.split == "train":
            # img_lb_file = self.filelist[index].strip("\n").split(" ")
            img_lb_file = self.filelist[index]
            img_file = img_lb_file[0]

            label_list = []
            # lb = scipy.io.loadmat(join(img_lb_file[1]))
            # # print(lb)
            # lb = np.asarray(lb['image'])
            # label = torch.from_numpy(lb)
            # label = label[1:label.size(0), 1:label.size(1)]
            # label = label.float()

            for i_label in range(1, len(img_lb_file[0])):
                # lb = scipy.io.loadmat(join(img_lb_file[i_label]))
                # lb = np.asarray(lb['image'])
                # label = torch.from_numpy(lb)
                # label = label[1:label.size(0), 1:label.size(1)]
                # label = label.float()
                # label_list.append(label.unsqueeze(0))

                label = draw_shoreline(img_, label_)
                label = cv2.resize(label, target_size, interpolation=cv2.INTER_LINEAR)
                _, label = cv2.threshold(label, 127, 1, cv2.THRESH_BINARY)
                label = torch.from_numpy(label)
                label = label[1:label.size(0), 1:label.size(1)]
                label = label.float()
                label_list.append(label.unsqueeze(0))



            labels = torch.cat(label_list, 0)
            label_max, label_min = (torch.sum(labels, dim=(1, 2))).max(), (torch.sum(labels, dim=(1, 2))).min()

            lb_mean = label.mean(dim=0).unsqueeze(0)
            lb_std = label.std(dim=0).unsqueeze(0)
            lb_index = random.randint(2, len(img_lb_file)) - 1
            lb_file = img_lb_file[lb_index]

        # else:
        #     # img_lb_file = self.filelist[index].strip("\n").split(" ")
        #     img_lb_file = self.filelist[index]
        #     img_file = img_lb_file[0]
            # img_file = self.filelist[index].rstrip()

        # img = imageio.imread(join(img_file))
        # img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        img = transforms.ToTensor()(img)
        img = img[:, 1:img.size(1), 1:img.size(2)]
        img = img.float()


        if self.split == "train":

            # lb = scipy.io.loadmat(join(lb_file))
            # lb=np.asarray(lb['image'])
            # label = torch.from_numpy(lb)
            # label = label[1:label.size(0), 1:label.size(1)]
            # label = label.unsqueeze(0)
            # label = label.float()
            # lb = scipy.io.loadmat(join(lb_file))
            # lb = np.asarray(lb['image'])
            # label = torch.from_numpy(lb)
            # # print(label.shape)
            # label = label[1:label.size(0), 1:label.size(1)]
            # label = label.unsqueeze(0)
            # label = label.float()
            label_style = (torch.sum(label) - label_min) / (label_max - label_min + 1e-10)
            # print(img.shape)
            # print(label.shape)

            # assert label_style>=0 and label_style<=1
            return img, label, lb_mean, lb_std, label_style
            # return img, label, lb_mean, lb_std

        else:
            return img