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

## Custom KITTI dataset ##

class get_kitti_dicts():

    def __init__(self, img_dir, annot_dir):

        self.img_dir=img_dir
        self.annot_dir=annot_dir

        self.dataset_dicts = []
        self.dataset_train = []
        self.dataset_val = []

        self.thing_classes=['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Misc', 'Van', 'Tram', 'Person_sitting']#['Car','Pedestrian','Cyclist'] 'DontCare'
        

        for idx, img_name in tqdm(enumerate(os.listdir(img_dir)),desc='Getting kitti dicts'):
            if 'det' not in img_name:
                record = {}
                filename = os.path.join(img_dir, img_name)
                width, height = Image.open(filename).size

                record["file_name"] = filename
                record["image_id"] = idx
                record["height"] = height
                record["width"] = width

                f = open(os.path.join(annot_dir,img_name.split('.')[0]+".txt"), "r")
                annos=f.readline().strip()

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
        return len(os.listdir(self.img_dir))
    
    def train_val_split(self, split=0):
        val_samples = random.choices(np.arange(0,len(self),1),k=int(len(self)*split))
        val_img = [str(img).zfill(6)+'.png' for img in val_samples]
        self.dataset_train = [dic for dic in self.dataset_dicts if not dic['file_name'].split('/')[-1] in val_img]
        self.dataset_val = [dic for dic in self.dataset_dicts if dic['file_name'].split('/')[-1] in val_img]

    def get_dicts(self, data_split):
        if data_split is 'train':
            return self.dataset_train
        else:
            return self.dataset_val



