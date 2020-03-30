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
parser.add_argument('--data_train', nargs='+', default='', help='define train dataset')
parser.add_argument('--data_val', default='MOTSChallenge', type=str, 
                    choices=['KITTI-MOTS','MOTSChallenge'], help='define val dataset')
parser.add_argument('--model', default='ResNet50-FPN', type=str,
                    choices=['ResNet50-FPN','ResNet101-FPN','ResNeXt101-FPN','R50-C4','R101-C4','R50-DC5','R101-DC5','Cityscapes'], help='define model')
parser.add_argument('--iter', default=5000, type=int, help='define num of iters')
parser.add_argument('--thr', default=0.5,type=float,help='define the threshold')
parser.add_argument('--lr', default=3e-4, type=float, help='define learning rate')
parser.add_argument('--lr_sch', default=None, type=str,
                    choices=['WarmupMultiStepLR','WarmupCosineLR', None], help='define learning rate scheduler')
parser.add_argument('--train', default=False, type=bool, help='train model')
parser.add_argument('--val', default=False, type=bool, help='validate model')
parser.add_argument('--test', default=False, type=bool, help='test model')

parser.add_argument('--data_augm', default=False, type=bool, help='apply data augmentation')
parser.add_argument('--scale', default=1, type=float, help='resize input image')
parser.add_argument('--crop', default=[1,1], type=float, nargs='+', help='apply random crop')
parser.add_argument('--dirout', default = None, type=str)


args = parser.parse_args()

# import dataset file
from utils import*

## Defining Dataset ##
dataset_train = args.data_train
dataset_val = args.data_val

## Defining model ##
model = args.model
train = args.train
val = args.val
test = args.test
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
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thr  # set threshold for this model
# Model from detectron2's model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

## Dataloaders ##

metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
thing_classes = metadata.thing_classes

if 'KITTI-MOTS' in dataset_train:

    print('[INFO] Using KITTI-MOTS as training set')

    # Loading images

    data_dir = '/home/mcv/datasets/KITTI-MOTS'
    img_test_dir = '/home/mcv/datasets/KITTI-MOTS/testing/image_02'
    map_classes = {1:2,2:0}

    # Dataloaders

    dataset = get_kitti_mots_dicts(data_dir,thing_classes,map_classes)
    dataset.train_val_split(0.2)

if 'MOTSChallenge' in dataset_train:

    print('[INFO] Using MOTSChallenge as training set')

    # Loading images

    data_dir = '/home/mcv/datasets/MOTSChallenge/train'
    map_classes = {2:0}

    if 'KITTI-MOTS' in dataset_train:
        mots_dataset = get_kitti_mots_dicts(data_dir,thing_classes,map_classes)
        dataset.join_dicts(mots_dataset.dataset_dicts)
    else: 
        dataset = get_kitti_mots_dicts(data_dir,thing_classes,map_classes)

## Dataloaders

if 'KITTI-MOTS' in dataset_val:

    print('[INFO] Using KITTI-MOTS as validation set')

    if 'KITTI-MOTS' not in dataset_train: 
        # Loading images

        data_dir = '/home/mcv/datasets/KITTI-MOTS'
        img_test_dir = '/home/mcv/datasets/KITTI-MOTS/testing/image_02'
        map_classes = {1:2,2:0}

        # Dataloaders

        dataset_kitti_mots = get_kitti_mots_dicts(data_dir,thing_classes,map_classes)
        dataset_kitti_mots.train_val_split(0.2)

        DatasetCatalog.register('dataset_val', lambda d='val':dataset_kitti_mots.get_dicts('val'))
        MetadataCatalog.get('dataset_val').set(thing_classes=thing_classes)

    else:
        DatasetCatalog.register('dataset_val', lambda d='val':dataset.get_dicts('val'))
        MetadataCatalog.get('dataset_val').set(thing_classes=thing_classes)

if 'MOTSChallenge' in dataset_val:

    print('[INFO] Using MOTSChallenge as validation set')

    # Loading images

    data_dir = '/home/mcv/datasets/MOTSChallenge/train'
    img_test_dir = '/home/mcv/datasets/MOTSChallenge/train/images'

    map_classes = {2:0}

    mots_dataset = get_kitti_mots_dicts(data_dir,thing_classes,map_classes)
    mots_dataset.train_val_split(0.2)

    DatasetCatalog.register('dataset_val', lambda d='val': mots_dataset.get_dicts('val'))
    MetadataCatalog.get('dataset_val').set(thing_classes=thing_classes)

if args.train:

    DatasetCatalog.register('dataset_train', lambda d='train': dataset.get_dicts('train'))
    MetadataCatalog.get('dataset_train').set(thing_classes=thing_classes)

    if args.data_augm:
        cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, 'task_c')
    else:
        if 'MOTSChallenge' in dataset_val:
            cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, 'task_a')
        elif 'KITTI-MOTS' in dataset_val:
            cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, 'task_b')
    

    #new output dir
    if args.lr_sch:
        cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, '_'.join(dataset_train), model, 'thr='+str(thr), str(args.iter), str(args.lr), str(args.lr_sch))
    elif args.data_augm:
        if args.scale != 1:
            cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, 'thr='+str(thr), 'resize', str(args.scale))
        if args.crop != [1,1]:
            cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, 'thr='+str(thr), 'crop', str(args.crop[0])+str(args.crop[1]))
    else:
        cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, '_'.join(dataset_train), model, 'thr='+str(thr), str(args.iter), str(args.lr), 'none')

else:
    cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, model, 'thr='+str(thr))

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


## Fine-tune coco-pretrained Faster R-CNN ##

# Setting training hyperparameters

cfg.DATASETS.TEST = ('dataset_val',)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = args.lr
if args.lr_sch:
    cfg.SOLVER.LR_SCHEDULER_NAME = args.lr_sch
cfg.SOLVER.MAX_ITER = args.iter
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
cfg.SOLVER.FG_THR = 0.7
cfg.SOLVER.BG_THR_HI = 0.7


# Training and saving model

if train:
    
    cfg.DATASETS.TRAIN = ('dataset_train',)
    if args.data_augm:
        data_augment=DataAugmentation(resize_factor=args.scale, crop_size=args.crop)
        trainer = MapperTrainer(cfg,mapper=data_augment.mapper)
    else:
        trainer = DefaultTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')

## Evaluate performance on validation set ##

if val:
    predictor = DefaultPredictor(cfg)

    if train:
        evaluator = COCOEvaluator('dataset_val', cfg, False, output_dir=join(cfg.OUTPUT_DIR,'eval_trained'))
    else:
        evaluator = COCOEvaluator('dataset_val', cfg, False, output_dir=join(cfg.OUTPUT_DIR,'eval'))
    val_loader = build_detection_test_loader(cfg, 'dataset_val')
    inference_on_dataset(predictor.model, val_loader, evaluator)

## Predictions on Test ##

if test:

    if 'MOTSChallenge' in dataset_val:

        folders=['0002','0011']
    
    elif 'KITTI-MOTS' in dataset_val:

        folders=['0001','0016']
    
    predictor = DefaultPredictor(cfg)   
    
    for folder_name in tqdm(os.listdir(img_test_dir), desc='Getting test results'):

        if folders[0] not in folder_name and folders[1] not in folder_name:
            continue

        for img_name in tqdm(os.listdir(os.path.join(img_test_dir, folder_name))):

            if 'png' not in img_name and 'jpg' not in img_name:
                continue

            im = cv2.imread(join(img_test_dir, folder_name, img_name))
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            if train:
                save_path = join(model, 'Trained', str(thr), folder_name)
            else:
                save_path = join(model, 'Pretrained', str(thr), folder_name)

            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(join(save_path, img_name), v.get_image()[:, :, ::-1])
