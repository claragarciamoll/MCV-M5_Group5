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
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.catalog import DatasetCatalog
from datetime import datetime

import argparse

parser = argparse.ArgumentParser(description='Define the model')
parser.add_argument('--data_train', nargs='+', default=['vKITTI'], help='define train dataset')
parser.add_argument('--data_val', nargs='+', default='', help='define val dataset')
parser.add_argument('--data_test', nargs='+', default='', help='define val dataset')
parser.add_argument('--model', default='Cityscapes', type=str,
                    choices=['ResNet50-FPN', 'ResNeXt101-FPN', 'Cityscapes'], help='define model')
parser.add_argument('--iter', default=5000, type=int, help='define num of iters')
parser.add_argument('--thr', default=0.5, type=float, help='define the threshold')
parser.add_argument('--lr', default=3e-4, type=float, help='define learning rate')
parser.add_argument('--train', default=False, type=bool, help='train model')
parser.add_argument('--real_size', default=0.1, type=float, help='train percent split')
parser.add_argument('--val', default=True, type=bool, help='validate model')
parser.add_argument('--test', default=False, type=bool, help='test model')

parser.add_argument('--data_augm', default=False, type=bool, help='apply data augmentation')
parser.add_argument('--scale', default=1, type=float, help='resize input image')
parser.add_argument('--crop', default=[1, 1], type=float, nargs='+', help='apply random crop')
parser.add_argument('--flip', default=False, type=bool, help='apply a flip')
parser.add_argument('--rot', default=False, type=bool, help='apply a rotation')
parser.add_argument('--angle', default=0, type=int, help='rotation angle')

parser.add_argument('--tracker', default="overlap", type=str, choices=['overlap','centroid','sort','OF'])

parser.add_argument('--dirout', default=None, type=str)
parser.add_argument('--task', default='task_a', choices=['task_a', 'task_b', 'task_c', 'task_f'], type=str)


args = parser.parse_args()

# import customed libraries
from utils import *
from tracking.multi_tracker import MultiTracker
from tracking.sortTracker import SortTracker

## Defining Dataset ##
dataset_train = args.data_train
dataset_val = args.data_val
dataset_test = args.data_test

## Defining model ##
model = args.model
train = args.train
val = args.val
test = args.test
thr = args.thr
tracker_sel = args.tracker


print('[INFO] Using ' + model + ' model, thr = ' + str(thr))

if model in 'ResNet50-FPN':
    config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
elif model in 'ResNeXt101-FPN':
    config_file = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
elif model in 'Cityscapes':
    config_file = "Cityscapes/mask_rcnn_R_50_FPN.yaml"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thr  # set threshold for this model
# Model from detectron2's model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

## Dataloaders ##

metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
thing_classes = metadata.thing_classes

# Loading images

data_dir_kitti = '/home/mcv/datasets/KITTI-MOTS'
data_dir_vkitti = '/home/mcv/datasets/vKITTI'
map_classes = {1: 2, 2: 0}

# Dataloaders

dataset_kitti_mots = get_kitti_mots_dicts(data_dir_kitti, thing_classes, map_classes)
dataset = get_vKitti_dicts(data_dir_vkitti, thing_classes, map_classes)

if 'KITTI-MOTS' in dataset_train and 'vKITTI' not in dataset_train:
    print('[INFO] Training on real')

    DatasetCatalog.register('dataset_train', lambda d='train': dataset_kitti_mots.get_dicts('train'))
    MetadataCatalog.get('dataset_train').set(thing_classes=thing_classes)

elif 'KITTI-MOTS' in dataset_train and 'vKITTI' in dataset_train:
    print('[INFO] Training a "{:.2%} on real'.format(args.real_size))

    dataset_kitti_mots.train_split(args.real_size)
    dataset.add_train_dicts(dataset_kitti_mots.dataset_train,len(dataset_kitti_mots))
    
if 'vKITTI' in dataset_train:
    print('[INFO] Training on virtual clones')

    DatasetCatalog.register('dataset_train', lambda d='train': dataset.get_dicts('train'))
    MetadataCatalog.get('dataset_train').set(thing_classes=thing_classes)


print('[INFO] Validating and testing on real')
for d in ['val','test']:
    DatasetCatalog.register('dataset_'+d, lambda d=d: dataset_kitti_mots.get_dicts(d))
    MetadataCatalog.get('dataset_'+d).set(thing_classes=thing_classes)

## Defining path to train dir ##

if args.train:

    # path to specified task
    cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, args.task)

    # new output dir for data augmentation
    if args.data_augm:
        if args.scale != 1:
            cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, 'thr=' + str(thr), 'resize', str(args.scale))
        if args.crop != [1, 1]:
            cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, 'thr=' + str(thr), 'crop', str(args.crop[0]) + str(args.crop[1]))
        if args.flip:
            cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, 'thr=' + str(thr), 'flip')
        if args.rot:
            cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, 'thr=' + str(thr), 'rot', str(args.angle))
    
    # new output dir depending on training set
    else:
        if 'KITTI-MOTS' in dataset_train and 'vKITTI' in dataset_train:
            cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, '_'.join(dataset_train), model, 'thr=' + str(thr), str(args.iter),
                                    str(args.lr), str(args.real_size))
        else:
            cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, '_'.join(dataset_train), model, 'thr=' + str(thr), str(args.iter),
                                    str(args.lr))

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

## Fine-tune coco-pretrained Faster R-CNN ##

# Setting training hyperparameters

cfg.DATASETS.TEST = ('dataset_test',)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = args.lr
cfg.SOLVER.MAX_ITER = args.iter
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
cfg.SOLVER.FG_THR = 0.7
cfg.SOLVER.BG_THR_HI = 0.7

# Training and saving model

if train:

    cfg.DATASETS.TRAIN = ('dataset_train',)
    
    # Applying data augmentation modified trainer
    if args.data_augm:
        data_augment = DataAugmentation(resize_factor=args.scale, crop_size=args.crop, hflip=args.flip, rot=args.rot, angle=args.angle)
        trainer = MapperTrainer(cfg, mapper=data_augment.mapper)
    else:
        trainer = DefaultTrainer(cfg)
    
    # Saving validation loss 
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    
    #Training and saving model
    trainer.resume_or_load(resume=False)
    trainer.train()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')


## Evaluate performance on validation set and test set ##

if val:
    predictor = DefaultPredictor(cfg)

    # Quantitative results on validation set

    evaluator = COCOEvaluator('dataset_val', cfg, False, output_dir=join(cfg.OUTPUT_DIR, 'eval'))
    val_loader = build_detection_test_loader(cfg, 'dataset_val')
    inference_on_dataset(predictor.model, val_loader, evaluator)
    
    #  Quantitative results on test set

    evaluator = COCOEvaluator('dataset_test', cfg, False, output_dir=join(cfg.OUTPUT_DIR, 'test'))
    val_loader = build_detection_test_loader(cfg, 'dataset_test')
    inference_on_dataset(predictor.model, val_loader, evaluator)

## Qualitative results on Test Set ##

if test:

    ## Tracker selection ##
    if 'task_f' in args.task:
        if tracker_sel == 'sort':
            # from .sort import sort
            tracker = SortTracker(key=get_centroid, max_age=6, min_age = 0, min_hit_streak=0, life_window=None, iou_threshold=0.3)
        else:
            tracker = MultiTracker(ttype=tracker_sel, key=get_centroid, maxDisappeared = 0, pix_tol = 100, iou_threshold=0.3)

    folders=['0004','0005','0007','0008','0009','0011','0015']

    predictor = DefaultPredictor(cfg)

    for j, folder_name in tqdm(enumerate(os.listdir(join(data_dir_kitti, 'training/image_02'))), desc='Getting test results'):

        if folders[0] not in folder_name and folders[1] not in folder_name:
            continue
        
        i=0
        for img_name in tqdm(os.listdir(join(data_dir_kitti, 'training/image_02', folder_name))):

            if 'png' not in img_name and 'jpg' not in img_name:
                continue
            
            if not 'task_f' in args.task:

                im = cv2.imread(join(data_dir_kitti, 'training/image_02', folder_name, img_name))
                outputs = predictor(im)
                v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

                save_path = join(cfg.OUTPUT_DIR, 'Qualitative')
                os.makedirs(join(save_path, folder_name), exist_ok=True)
                cv2.imwrite(join(save_path, folder_name, img_name), v.get_image()[:, :, ::-1])
            
