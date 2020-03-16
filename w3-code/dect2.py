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
parser.add_argument('--model', default='FasterRCNN', type=str, 
                    choices=['FasterRCNN','RetinaNet'], help='define model')
parser.add_argument('--iter', default=15000, type=int, help='define num of iters')
parser.add_argument('--lr', default=1e-3, type=float, help='define learning rate')

args = parser.parse_args()


# import dataset file
from datasets import*

## Defining Dataset ##
dataset = args.data

## Defining model ##
model = args.model
train = True
val = False
test = False


print('[INFO] Using '+model+' model')

if model in 'FasterRCNN':
    config_file="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
elif model in 'RetinaNet':
    config_file="COCO-Detection/retinanet_R_101_FPN_3x.yaml"

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Model from detectron2's model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)


if not train:

    ## Loading images ##

    train_data_dir = '/home/mcv/datasets/KITTI-MOTS/training'
    test_data_dir = '/home/mcv/datasets/KITTI-MOTS/testing'

    # Dataloaders
    dataset_train = torchvision.datasets.ImageFolder(train_data_dir)
    dataset_test = torchvision.datasets.ImageFolder(test_data_dir)
    datasets = torch.utils.data.ConcatDataset([dataset_train, dataset_test])

    ## Predicting from pre-trained model ##

    predictor = DefaultPredictor(cfg)

    ## Predicting bboxes ##

    dataset_name = ['train', 'test']

    for x, dataset in enumerate(datasets.datasets):

        for img in tqdm(dataset.imgs,desc='Predicting bboxes at '+ dataset_name[x] + ' dataset'):
            
            im = cv2.imread(img[0])

            outputs = predictor(im)

            # Predictions on the image
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            
            os.makedirs(model+'/'+('/').join(img[0].split('/')[-3:-1]), exist_ok=True)

            cv2.imwrite(model+'/'+('/').join(img[0].split('/')[-3:]),v.get_image()[:, :, ::-1])

else:
    if dataset in 'KITTI-MOTS':

        ## Loading images ##
    
        img_dir = '/home/mcv/datasets/KITTI-MOTS/training/image_02'
        annot_dir = '/home/mcv/datasets/KITTI-MOTS/instances'
        img_test_dir = '/home/mcv/datasets/KITTI-MOTS/testing/image_02'

        if val:
            metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            thing_classes = metadata.thing_classes
            map_classes = {1:2,2:0}
        else:
            thing_classes = ['Car', 'Pedestrian']
            map_classes = {1:0,2:1}

        # Dataloaders

        kitti_mots_dataset = get_kitti_mots_dicts(img_dir,annot_dir,thing_classes,map_classes)
        kitti_mots_dataset.train_val_split(.2)

        for d in ['train', 'val']:
            DatasetCatalog.register(dataset + '_' + d, lambda d=d: kitti_mots_dataset.get_dicts(d))
            MetadataCatalog.get(dataset + '_' + d).set(thing_classes=thing_classes)


    if dataset in 'MOTSChallenge':

        print('hey')

        ## Loading images ##
    
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

    ## Fine-tune coco-pretrained Faster R-CNN ##

    # Setting training hyperparameters

    cfg.DATASETS.TRAIN = (dataset + '_train',)
    cfg.DATASETS.TEST = (dataset + '_val',)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(kitti_mots_dataset.thing_classes)

    # Training and saving model 

    cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, dataset, model, str(cfg.SOLVER.MAX_ITER)) #new output dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)

    if val:

        ## Evaluate performance of pre-trained model##

        evaluator = COCOEvaluator(dataset + '_val', cfg, False, output_dir=join(dataset, model,'eval'))
        val_loader = build_detection_test_loader(cfg, dataset + '_val')
        inference_on_dataset(trainer.model, val_loader, evaluator)

    if False:

        # Training and saving model 

        trainer.train()
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
        
        ## Evaluate performance ##

        evaluator = COCOEvaluator(dataset + '_val', cfg, False, output_dir=join(cfg.OUTPUT_DIR,'eval'))
        val_loader = build_detection_test_loader(cfg, dataset + '_val')
        inference_on_dataset(trainer.model, val_loader, evaluator)

    ## Predictions on Test ##

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model

    if test:

        predictor = DefaultPredictor(cfg)

        for folder_name in tqdm(os.listdir(img_test_dir),desc='Getting test results'):
            for img_name in tqdm(os.listdir(os.path.join(img_test_dir,folder_name))):

                if 'png' not in img_name:
                    continue

                im = cv2.imread(join(img_test_dir,folder_name,img_name))
                outputs = predictor(im)
                v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

                os.makedirs(join('test_pred',folder_name), exist_ok=True)
                cv2.imwrite(join('test_pred',folder_name,img_name),v.get_image()[:, :, ::-1])
