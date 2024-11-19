import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Atari_Dataset(Dataset):

    def __init__(self, root='./data', train=True,
                 transform=None,
                 index_path=None, index=None, base_sess=None, img_dim=240):
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        

        self.IMAGE_PATH = ['/home/student/atari155']
        self.SPLIT_PATH = './data'

        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        print(csv_path)
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1

        self.wnids = []
        self.base_s = base_sess

        i = 0
        
        # iterate over csv
        for l in lines:
            # get name of image and class
            path, wnid = l.split(',')
            # record the classes 
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            # record all the data from the path
            self.data.append(path)
            # record all the labels
            self.targets.append(lb)
            # record the relationship of path and label
            self.data2label[path] = lb
        
        print("Length of Targets: ",len(self.targets))
        print("Number classes: ",len(self.wnids))
        print("Base Session: ", base_sess)
        print("Train: ",train)
        
        if base_sess:
            print("Num base classes: ", len(index))
        if train:
            image_size = img_dim
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                # transforms.Grayscale(num_output_channels=1), # added this
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            
            # base session is set for get_base_dataloader
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            image_size = img_dim
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                # transforms.Grayscale(num_output_channels=1), # added this
                transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            
            
    def SelectfromTxt(self, data2label, index_path):
        print("Select from text")
        index=[]
        
        # read seassion text lines
        print(index_path)
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        data_tmp = []
        targets_tmp = []
        
        for line in lines:
            # image names
            data_tmp.append(line)
            targets_tmp.append(data2label[line])
        
        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        print("Select from classes")
        data_tmp = []
        targets_tmp = []
        for i in index:
            # get 500 images corresponding to the class
            ind_cl = np.where(i == targets)[0]
            # append each data path and target
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])
        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]

        try:    
            image = Image.open(path)
        
            if image.mode == 'L':
                print(f"Converting grayscale image to RGB: {path}")
                image = image.convert("RGB")
            image = self.transform(image)
        except OSError:
            print(f"Error reading file: {path}")
            return None, None
        # image = self.transform(Image.open(path))#.convert('RGB'))
        return image, targets
