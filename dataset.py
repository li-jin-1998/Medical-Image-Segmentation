import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
from skimage.io import imread
import cv2
from glob import glob
import imageio


class Dataset(data.Dataset):
    def __init__(self, state,path, transform=None, target_transform=None):
        self.state = state
        # self.root = 'E:/lijin/pancreas_tumor_seg/pancreas_tumor_2d/'
        self.root=path
        # self.shuffle = shuffle_data
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform
    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test' or self.state =='test1'
        root=None
        if self.state == 'train':
            self.root = self.root+'train'
        if self.state == 'val':
            self.root = self.root+'test'
        if self.state == 'test':
            self.root = self.root+'test'
        if self.state == 'test1':
            self.root = self.root
        pics = []
        masks = []
        paths=os.listdir(self.root+'/image')
        from sklearn.utils import shuffle
        # if self.shuffle:
        #     paths = shuffle(paths, random_state=2022)
        # print(n)
        for path in paths:
            img = os.path.join(self.root,'image', path) # liver is %03d
            mask = os.path.join(self.root,'seg', path)
            pics.append(img)
            masks.append(mask)
        return pics,masks
    def __getitem__(self, index):
        #x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        origin_x = cv2.imread(x_path)
        origin_y = cv2.imread(y_path,cv2.COLOR_BGR2GRAY)
        image_size=512
        origin_x = cv2.resize(origin_x,(image_size,image_size), interpolation=cv2.INTER_CUBIC)
        origin_y = cv2.resize(origin_y, (image_size,image_size), interpolation=cv2.INTER_CUBIC)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y,x_path,y_path

    def __len__(self):
        return len(self.pics)
