from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
# from data_processing import data_process_voc_parse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np

def data_process_voc_parse(path):
    # Get imgs
    files = sorted([f"{path}/{f}" for f in listdir(path)])
    # imgs_dict = {name: imgs}
    obj_name = []
    bndbox_values = []
    imgs = []
    class_names = ["apple", "banana", "orange","Hand"]
    for f in files:

        # if file is xml, check xml for bndbox; if no bndbox, skip
        if '.xml' in f:
            tree = ET.parse(f)
            root = tree.getroot()
            objects = root.findall('object')
            if len(objects) != 0:
                tmp = []
                tmp2 = []
                for obj in objects:
                    
                    name = obj.find('name').text
                    if name not in class_names:
                        continue
                    tmp.append(name)
                    bndbox = obj.findall('bndbox')
                    for box in bndbox:
                        xmin = int(float(box.find('xmin').text))
                        ymin = int(float(box.find('ymin').text))
                        xmax = int(float(box.find('xmax').text))
                        ymax = int(float(box.find('ymax').text))
                    tmp2.append([xmin, ymin, xmax, ymax])
                bndbox_values.append(tmp2)
                obj_name.append(tmp)
            if len(objects) == 0:
                imgs.pop(-1)
        if '.jpg' in f:
            imgs.append(f)
    return bndbox_values, obj_name, imgs
   


class fruit_dataset(Dataset):

    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transforms = transform
        # load all image files, sorting them to
        # ensure that they are aligned
        self.target_transform = target_transform
        self.bndbox, self.label, self.imgs = data_process_voc_parse(root)
        self.class_names = ["background", "apple", "banana", "orange", "Hand"]

    def __getitem__(self, idx):
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        label = self.label[idx]
        if len(label) == 0:
            print(1)
        bndbox = self.bndbox[idx]
        img = self.imgs[idx]
        img = Image.open(img).convert("RGB")
        img = np.array(img)
        labels = []
        for _label in label:
            labels.append(self.class_names.index(_label))

        boxes = np.array(bndbox,dtype=np.float32)
        labels = np.array(labels)
        if self.transforms is not None:
            img, boxes, labels = self.transforms(img, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
            
        return img, boxes, labels

    def __len__(self):
        return len(self.imgs)


