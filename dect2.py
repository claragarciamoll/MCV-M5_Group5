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
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog,build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.catalog import DatasetCatalog


# import dataset file
from kitti_dataset import*


## Defining model ##
model = 'FastRCNN'
train = True

print('[INFO] Using '+model+' model')

if model is 'FastRCNN':
    config_file="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
elif model is 'RetinaNet':
    config_file="COCO-Detection/retinanet_R_101_FPN_3x.yaml"

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Model from detectron2's model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)


if not train:

    ## Loading images ##

    train_data_dir = '/Users/Downloads/bovw/MIT_split/train'
    test_data_dir = '/Users/Downloads/bovw/MIT_split/test'

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

    ## Loading images ##
    
    img_dir = '/home/mcv/datasets/KITTI/data_object_image_2/training/image_2'
    annot_dir = '/home/mcv/datasets/KITTI/training/label_2'

    # Dataloaders

    kitti_dataset = get_kitti_dicts(img_dir,annot_dir)
    kitti_dataset.train_val_split(.2)
    
    
    for d in ['train', 'val']:   
        DatasetCatalog.register('kitti_' + d, lambda d=d: kitti_dataset.get_dicts(d))
        MetadataCatalog.get('kitti_' + d).set(thing_classes=['Car', 'Pedestrian', 'Cyclist', 'Truck', 'Misc', 'Van', 'Tram', 'Person_sitting'])
    
    kitti_metadata = MetadataCatalog.get('kitti_train')

    ## Fine-tune coco-pretrained Faster R-CNN ##

    # Setting training hyperparameters

    cfg.DATASETS.TRAIN = ('kitti_train',)
    cfg.DATASETS.TEST = ('kitti_val',)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 15000 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(kitti_dataset.thing_classes)

    # Training and saving model 

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    cfg.MODEL.WEIGHTS = join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    
    ## Evaluate performance ##

    evaluator = COCOEvaluator('kitti_val', cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, 'kitti_val')
    inference_on_dataset(trainer.model, val_loader, evaluator)


    ## Predictions on Test ##

    predictor = DefaultPredictor(cfg)
    img_test_dir = '/home/mcv/datasets/KITTI/data_object_image_2/testing/image_2'

    for filename in os.listdir(img_test_dir):    
        im = cv2.imread(join(img_test_dir,filename))
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=kitti_metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        os.makedirs('test_pred', exist_ok=True)
        cv2.imwrite(join('test_pred',filename),v.get_image()[:, :, ::-1])