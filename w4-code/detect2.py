import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from cv2 import imshow as cv2_imshow
import os
from os.path import join
from tqdm.auto import tqdm
import torch, torchvision
import pickle

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog,build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.catalog import DatasetCatalog
from datetime import datetime
now = datetime.now()
date_time = now.strftime("%m-%d-%Y_%H:%M:%S")

import argparse

parser = argparse.ArgumentParser(description='Define the model')
parser.add_argument('--data', default='KITTI-MOTS', type=str, 
                    choices=['KITTI-MOTS','MOTSChallenge'], help='define dataset')
parser.add_argument('--model', default='ResNeXt101-FPN', type=str,
                    choices=['ResNet50-FPN','ResNet101-FPN','ResNeXt101-FPN','R50-C4','R101-C4','R50-DC5','R101-DC5','Cityscapes'], help='define model')
parser.add_argument('--iter', default=15000, type=int, help='define num of iters')
parser.add_argument('--thr', default=0.5,type=float,help='define the threshold')
parser.add_argument('--lr', default=1e-3, type=float, help='define learning rate')
parser.add_argument('--lr_sch', default=None, type=str,
                    choices=['WarmupMultiStepLR','WarmupCosineLR', None], help='define learning rate scheduler')
parser.add_argument('--train', default=False, help='train model')

args = parser.parse_args()

# import dataset file
from datasets import*

## Defining Dataset ##
dataset = args.data

## Defining model ##
model = args.model
train = args.train
val = True
test = False
thr = args.thr

print('[INFO] Using '+model+' model, thr = '+str(thr))

if model in 'ResNet50-FPN':
    config_file="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
elif model in 'ResNet101-FPN':
    config_file="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
elif model in 'ResNeXt101-FPN':
    config_file="COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
elif model in 'R50-C4':
    config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"
elif model in 'R101-C4':
    config_file = "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"
elif model in 'R50-DC5':
    config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"
elif model in 'R101-DC5':
    config_file = "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"
elif model in 'Cityscapes':
    config_file = "Cityscapes/mask_rcnn_R_50_FPN.yaml"

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thr  # set threshold for this model
# Model from detectron2's model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

## Dataloaders ##

if dataset in 'KITTI-MOTS':

    # Loading images

    img_dir = '/home/mcv/datasets/KITTI-MOTS/training/image_02'
    annot_dir = '/home/mcv/datasets/KITTI-MOTS/instances'
    img_test_dir = '/home/mcv/datasets/KITTI-MOTS/testing/image_02'
    
    if train:
        thing_classes = ['Car', 'Pedestrian']
        map_classes = {1:0,2:1}
    else:
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        thing_classes = metadata.thing_classes
        map_classes = {1:2,2:0}

    # Dataloaders

    kitti_mots_dataset = get_kitti_mots_dicts(img_dir,annot_dir,thing_classes,map_classes)
    kitti_mots_dataset.train_val_split(.2)

    for d in ['train', 'val']:
        DatasetCatalog.register(dataset + '_' + d, lambda d=d: kitti_mots_dataset.get_dicts(d))
        MetadataCatalog.get(dataset + '_' + d).set(thing_classes=thing_classes)


if dataset in 'MOTSChallenge':

    # Loading images

    img_dir = '/home/mcv/datasets/MOTSChallenge/train/images'
    annot_dir = '/home/mcv/datasets/MOTSChallenge/train/instances'
    thing_classes = ['Pedestrian']
    map_classes = {2:0}

    # Dataloaders

    kitti_mots_dataset = get_kitti_mots_dicts(img_dir,annot_dir,thing_classes,map_classes)
    kitti_mots_dataset.train_val_split(.2)

    for d in ['train', 'val']:
        DatasetCatalog.register(dataset + '_' + d, lambda d=d: kitti_mots_dataset.get_dicts(d))
        MetadataCatalog.get(dataset + '_' + d).set(thing_classes=thing_classes)

metadata = MetadataCatalog.get(dataset + '_train')

cfg.DATASETS.TEST = (dataset + '_val',)
#new output dir
if args.lr_sch:
    cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, dataset, model, 'thr='+str(thr), str(args.iter), str(args.lr), str(args.lr_sch))
else:
    cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, dataset, model, 'thr='+str(thr), str(args.iter), str(args.lr), 'none')
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


## Fine-tune coco-pretrained Faster R-CNN ##

if train:

    # Setting training hyperparameters

    cfg.DATASETS.TRAIN = (dataset + '_train',)
    cfg.DATASETS.TEST = (dataset + '_val',)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = args.lr
    if args.lr_sch:
        cfg.SOLVER.LR_SCHEDULER = args.lr_sch
        cfg.SOLVER.WARMUP_FACTOR = 1e-4
    cfg.SOLVER.MAX_ITER = args.iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(kitti_mots_dataset.thing_classes)

    # Training and saving model

    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')

## Evaluate performance on validation set ##

if val:
    predictor = DefaultPredictor(cfg)
    if train:
        evaluator = COCOEvaluator(dataset + '_val', cfg, False, output_dir=join(cfg.OUTPUT_DIR,'eval_trained'))
    else:
        evaluator = COCOEvaluator(dataset + '_val', cfg, False, output_dir=join(cfg.OUTPUT_DIR,'eval'))
    val_loader = build_detection_test_loader(cfg, dataset + '_val')
    inference_on_dataset(predictor.model, val_loader, evaluator)

## Predictions on Test ##

if test:

    predictor = DefaultPredictor(cfg)

    for folder_name in tqdm(os.listdir(img_test_dir),desc='Getting test results'):

        if '0001' not in folder_name and '0016' not in folder_name:
            continue

        for img_name in tqdm(os.listdir(os.path.join(img_test_dir,folder_name))):

            if 'png' not in img_name:
                continue

            im = cv2.imread(join(img_test_dir,folder_name,img_name))
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            if train:
                save_path = join(model,'Trained',str(thr),folder_name)
            else:
                save_path = join(model,'Prerained',str(thr),folder_name)

            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(join(save_path,img_name),v.get_image()[:, :, ::-1])
