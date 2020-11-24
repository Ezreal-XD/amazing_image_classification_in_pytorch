# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset
import cv2
import numpy as np
import random


class SeedDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs_list = []
        for line in fh:
            img, img_mask = line.split('\t')
            imgs_list.append((img, img_mask))

        self.imgs = imgs_list  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.imgs[index]
        pre_path = 'Z:/CreatingCodeHere/seed/data/'
        img = cv2.imread(pre_path + img, cv2.IMREAD_COLOR)
        label = cv2.imread(pre_path + label, cv2.IMREAD_GRAYSCALE)

        img = img[:, :, ::-1]  # change to RGB

        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等

        return img.copy(), label.copy()

    def __len__(self):
        return len(self.imgs)
