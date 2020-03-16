# import some common libraries
import os
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm.auto import tqdm

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode


class get_kitti_dicts():

    def __init__(self, img_dir, annot_dir):

        self.img_dir=img_dir
        self.annot_dir=annot_dir

        self.dataset_dicts = []
        self.dataset_train = []
        self.dataset_val = []

        self.thing_classes=['Car', 'Pedestrian']
        

        for idx, img_name in tqdm(enumerate(os.listdir(img_dir)),desc='Getting kitti dicts'):
            if 'det' not in img_name:
                record = {}
                filename = os.path.join(img_dir, img_name)
                height, width = Image.open(filename).size

                record["file_name"] = filename
                record["image_id"] = idx
                record["height"] = height
                record["width"] = width

                f = open(os.path.join(annot_dir,img_name.split('.')[0]+".txt"), "r")
                annos=f.readline().strip()

                img = np.asarray(Image.open(filename))

                objs = []
                while annos!="":
                    if annos.split(' ')[0] not in self.thing_classes:
                        annos=f.readline().strip()
                        continue
                    bbox = [float(coord) for coord in annos.split(' ')[4:8]]
                    obj = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": self.thing_classes.index(annos.split(' ')[0]),
                        "iscrowd": 0
                    }
                    objs.append(obj)
                    annos=f.readline().strip()

                record["annotations"] = objs
                self.dataset_dicts.append(record)

        self.dataset_train = self.dataset_dicts

    def __len__(self):
        return len(self.dataset_dicts)
    
    def train_val_split(self, split=0):
        val_samples = random.choices(np.arange(0,len(self),1),k=int(len(self)*split))
        self.dataset_train = [dic for dic in self.dataset_dicts if not dic['image_id'] in val_samples]
        self.dataset_val = [dic for dic in self.dataset_dicts if dic['image_id'] in val_samples]

    def get_dicts(self, data_split):
        if data_split is 'train':
            return self.dataset_train
        else:
            return self.dataset_val

class get_kitti_mots_dicts():

    def __init__(self, img_dir, annot_dir, thing_classes, map_classes=None):

        self.img_dir=img_dir
        self.annot_dir=annot_dir

        self.dataset_dicts = []
        self.dataset_train = []
        self.dataset_val = []

        self.thing_classes = thing_classes
        

        for i, folder_name in tqdm(enumerate(os.listdir(img_dir)),desc='Getting kitti dicts'):
            for j, img_name in tqdm(enumerate(os.listdir(os.path.join(img_dir,folder_name)))):
                
                if 'png' not in img_name and 'jpg' not in img_name:
                    continue

                record = {}
                filename = os.path.join(img_dir, folder_name, img_name)
                annot_filename = os.path.join(annot_dir,folder_name, img_name.split('.')[0]+'.png')

                annot = np.asarray(Image.open(annot_filename))

                height, width = annot.shape[:]

                record["file_name"] = filename
                record["image_id"] = i+j
                record["height"] = height
                record["width"] = width

                # Identify different patterns indexes
                patterns = list(np.unique(annot))[1:-1]

                objs = []
                for pattern in patterns:

                    # Coordinates of pattern pixels
                    coords = np.argwhere(annot==pattern)

                    # Bounding box of pattern
                    x0, y0 = coords.min(axis=0)     
                    x1, y1 = coords.max(axis=0)

                    bbox = [y0, x0, y1, x1]
            
                    obj = {
                            "bbox": bbox,
                            "bbox_mode":BoxMode.XYXY_ABS,
                            "category_id": map_classes[int(np.floor(annot[coords[0][0]][coords[0][1]]/1e3))],
                            "iscrowd": 0
                            }
                    
                    objs.append(obj)
                
                record["annotations"] = objs
                self.dataset_dicts.append(record)

        self.dataset_train = self.dataset_dicts

    def __len__(self):
        return len(self.dataset_dicts)
    
    def train_val_split(self, split=0):
        random.seed(len(self))
        val_samples = random.choices(np.arange(0,len(self),1),k=int(len(self)*split))
        self.dataset_train = [dic for dic in self.dataset_dicts if not dic['image_id'] in val_samples]
        self.dataset_val = [dic for dic in self.dataset_dicts if dic['image_id'] in val_samples]

    def get_dicts(self, data_split):
        if data_split is 'train':
            return self.dataset_train
        else:
            return self.dataset_val


'''
dataset = 'KITTI-MOTS'

img_dir = '/media/gemma/My Passport/Master/datasets/MOTSChallenge/train/images'
annot_dir = '/media/gemma/My Passport/Master/datasets/MOTSChallenge/train/instances'

kitti_mots_dataset = get_kitti_mots_dicts(img_dir,annot_dir,['Pedestrian'])
kitti_mots_dataset.train_val_split(.2)

for d in ['train', 'val']:
    DatasetCatalog.register(dataset + '_' + d, lambda d=d: kitti_mots_dataset.get_dicts(d))
    MetadataCatalog.get(dataset + '_' + d).set(thing_classes=['Pedestrian'])

metadata = MetadataCatalog.get(dataset + '_train')

dataset_dicts = kitti_mots_dataset.dataset_dicts
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    plt.imshow(vis.get_image()[:, :, ::-1])
    plt.pause(5)
'''