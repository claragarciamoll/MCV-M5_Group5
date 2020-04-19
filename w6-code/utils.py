# import some common libraries
import os
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm.auto import tqdm
import pickle
from os.path import join
import copy
import logging
from collections import OrderedDict
from fvcore.nn.precise_bn import get_bn_modules
from torch.nn.parallel import DistributedDataParallel
from scipy.ndimage import rotate
import math

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, SimpleTrainer, hooks
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.structures import BoxMode
from pycocotools import coco
import torch
from detectron2.engine import HookBase, SimpleTrainer
from detectron2.data import detection_utils as utils
import detectron2.utils.comm as comm
from detectron2.data.transforms import*
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.logger import setup_logger

class get_kitti_mots_dicts():

    def __init__(self, data_dir, thing_classes, map_classes=None):
        """
        Prepares kitti-mots dataset to coco format.

        Args:
            data_dir (str): path to data.
            thing_classes (list): class names.
            map_classes (dict): mapping labels.
        """
        
        train_seq=['0001','0002','0006','0018','0020']
        val_seq=['0000','0003','0010','0012','0014']
        test_seq=['0004','0005','0007','0008','0009','0011','0015']

        self.img_dir=join(data_dir, 'training/image_02')
        
        self.mask_dir=join(data_dir, 'instances')
        self.annot_dir=join(data_dir, 'instances_txt')

        self.dataset_dicts = []
        self.dataset_train = []
        self.dataset_val = []
        self.dataset_test = []

        self.thing_classes = thing_classes
        
        # Load pickle data
        pkl_dir='data'
        if os.path.exists(pkl_dir):
            with open(join(pkl_dir,'train_kitti.pkl'), 'rb') as pkl_file:
                self.dataset_train = pickle.load(pkl_file)
                print('[INFO] Read kitti train pkl, len='+str(len(self.dataset_train)))
            with open(join(pkl_dir,'val_kitti.pkl'), 'rb') as pkl_file:
                self.dataset_val = pickle.load(pkl_file)
                print('[INFO] Read kitti val pkl, len='+str(len(self.dataset_val)))
            with open(join(pkl_dir,'test_kitti.pkl'), 'rb') as pkl_file:
                self.dataset_test = pickle.load(pkl_file)
                print('[INFO] Read kitti test pkl, len='+str(len(self.dataset_test)))
            
            self.dataset_dicts=self.dataset_train+self.dataset_val+self.dataset_test

            return

        os.makedirs(pkl_dir, exist_ok=True)

        image_id=0
        
        for i, folder_name in tqdm(enumerate(os.listdir(self.img_dir)),desc='Getting kitti-mots dicts'):

            for j, img_name in tqdm(enumerate(os.listdir(join(self.img_dir,folder_name)))):
                if 'png' not in img_name and 'jpg' not in img_name:
                    continue

                record = {}
                filename = join(self.img_dir, folder_name, img_name)
                mask_filename = join(self.mask_dir,folder_name, img_name.split('.')[0]+'.png')
                annot_filename = join(self.annot_dir, folder_name+'.txt')

                width, height = Image.open(filename).size

                record["file_name"] = filename
                record["image_id"] = image_id
                record["height"] = height
                record["width"] = width

                with open(annot_filename,'r') as annot:
                    lines = annot.readlines()
                    lines = [l.split(' ') for l in lines]
                
                frame_lines = [l for l in lines if int(l[0]) == int(img_name.split('.')[0])]
                if not frame_lines:
                    continue
                
                objs = []
                for pattern in frame_lines:
                    category_id = int(pattern[2])
                    if category_id not in map_classes.keys():
                        continue

                    rle = {
                        'counts': pattern[-1].strip(),
                        'size': [height, width]
                    }
                    bbox = coco.maskUtils.toBbox(rle).tolist()
                    bbox[2] += bbox[0]
                    bbox[3] += bbox[1]
                    bbox = [int(item) for item in bbox]

                    mask = coco.maskUtils.decode(rle)
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    poly = [[int(i) for i in c.flatten()] for c in contours]
                    poly = [p for p in poly if len(p) >= 6]
                    if not poly:
                        continue

                    obj = {
                            "bbox": bbox,
                            "bbox_mode":BoxMode.XYXY_ABS,
                            "segmentation": poly,
                            "category_id": map_classes[category_id],
                            "iscrowd": 0
                            }
                    
                    objs.append(obj)
                
                record["annotations"] = objs
                self.dataset_dicts.append(record)

                if folder_name in train_seq:
                    self.dataset_train.append(record)
                elif folder_name in val_seq:
                    self.dataset_val.append(record)
                elif folder_name in test_seq:
                    self.dataset_test.append(record)

                image_id+=1
        
        # Save data to pickle files
        with open('data/train_kitti.pkl', 'wb') as autosave:
            pickle.dump(self.dataset_train, autosave, protocol=pickle.HIGHEST_PROTOCOL)
        with open('data/val_kitti.pkl', 'wb') as autosave:
            pickle.dump(self.dataset_val, autosave, protocol=pickle.HIGHEST_PROTOCOL)
        with open('data/test_kitti.pkl', 'wb') as autosave:
            pickle.dump(self.dataset_test, autosave, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.dataset_dicts)
    
    def train_val_split(self, split=0):
        """
        Apply random split to specific propotion of the dataset.

        Args:
            split (float): proportion of the dataset that will be for validation.
        """
        random.seed(len(self))
        val_samples = random.choices(np.arange(0,len(self),1),k=int(len(self)*split))
        self.dataset_train = [dic for dic in self.dataset_dicts if not dic['image_id'] in val_samples]
        self.dataset_val = [dic for dic in self.dataset_dicts if dic['image_id'] in val_samples]
    
    def train_split(self, split=0):
        """
        Apply random split to specific propotion of the train set.

        Args:
            split (float): proportion of the train set that will be used.
        """
        self.dataset_train = random.choices(self.dataset_train,k=int(len(self.dataset_train)*split))

    def get_dicts(self, data_split):

        """
        Retruns train, val or test set dictionary.

        Args:
            data_split (str): name of wanted split ['train', 'val', 'test'].
        """
        if data_split is 'train':
            return self.dataset_train
        elif data_split is 'val':
            return self.dataset_val
        else:
            return self.dataset_test

class get_vKitti_dicts():

    def __init__(self, data_dir, thing_classes, map_classes=None):

        """
        Prepares virtual kitti dataset to coco format.

        Args:
            data_dir (str): path to data.
            thing_classes (list): class names.
            map_classes (dict): mapping labels.
        """

        train_seq=['01','02','06','18','20']
        
        self.data_dir = data_dir
        
        self.dataset_dicts = []
        self.dataset_train = []

        self.thing_classes = thing_classes
        
        # Load pickle data
        pkl_dir = 'data'
        if os.path.exists(pkl_dir):
            with open(join(pkl_dir,'train_vkitti.pkl'), 'rb') as pkl_file:
                self.dataset_train = pickle.load(pkl_file)
                print('[INFO] Read vkitti train pkl, len='+str(len(self.dataset_train)))
            self.dataset_dicts=self.dataset_train
            
            return

        image_id=0
        
        for i, folder_name in tqdm(enumerate(os.listdir(self.data_dir)), desc='Getting vkitti dicts'):            
            for j, class_name in tqdm(enumerate(os.listdir(join(self.data_dir,folder_name)))):
                
                imfolder_name = class_name + '/frames/rgb'
                annot_name = class_name + '/frames/instanceSegmentation'
                dir1 = join(self.data_dir,folder_name)
                img_dir = join(dir1,imfolder_name)
                annot_dir = join(dir1,annot_name)
                
                for k, camera_id in tqdm(enumerate(os.listdir(img_dir))):
                    for l, img_name in tqdm(enumerate(os.listdir(join(img_dir,camera_id)))):             
                        if 'png' not in img_name and 'jpg' not in img_name:
                            continue

                        record = {}
                        filename = join(img_dir, camera_id, img_name)
                        annot_name = 'instancegt_'+img_name.split('_')[1].split('.')[0]+'.png'
                        annot_filename = join(annot_dir,camera_id, annot_name)
                        
                        width, height = Image.open(filename).size

                        record["file_name"] = filename
                        record["image_id"] = image_id
                        record["height"] = height
                        record["width"] = width
                
                        objs = []
                        annot = np.array(Image.open(annot_filename).convert('RGB'))
                        annot = cv2.cvtColor(annot, cv2.COLOR_RGB2GRAY)
                        patterns = list(np.unique(annot))[1:]

                        for pattern in patterns:

                            # Coordinates of pattern pixels
                            coords = np.argwhere(annot==pattern)

                            # Bounding box of pattern
                            x0, y0 = coords.min(axis=0)     
                            x1, y1 = coords.max(axis=0)

                            bbox = [y0, x0, y1, x1]
                            
                            copy = annot.copy()
                            copy[annot==pattern] = 255
                            copy[annot!=pattern] = 0
                            copy = np.asarray(copy,np.uint8)

                            contours, _ = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                            poly = [[int(i) for i in c.flatten()] for c in contours]
                            poly = [p for p in poly if len(p) >= 6]
                            if not poly:
                                continue
                            
                            obj = {
                                "bbox": bbox,
                                "bbox_mode":BoxMode.XYXY_ABS,
                                "segmentation": poly,
                                "category_id": 2,
                                "iscrowd": 0
                                }
                        
                            objs.append(obj)                            
                        
                        record["annotations"] = objs
                        self.dataset_dicts.append(record)
                        self.dataset_train.append(record)
        
                        image_id+=1
        
        # Save data to pickle file
        with open('data/train_vkitti.pkl', 'wb') as autosave:
            pickle.dump(self.dataset_train, autosave, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.dataset_dicts)

    def add_train_dicts(self, dicts, prev_len):
        """
        Add more data to train set.

        Args:
            dicts (dict): train dicts to be added.
            prev_len (int): length of the other dataset.

        """
        new_dicts=[]
        for record in self.dataset_train:
            record["image_id"]+=prev_len
            new_dicts.append(record)
        self.dataset_train=dicts+new_dicts

    def get_dicts(self, data_split):
        """
        Retruns train set dictionary.
        """
        return self.dataset_train


class RotateTransform(Transform):
    """
    Perform rotation.
    """

    def __init__(self, h: int, w: int, angle: int):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Rotate the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the rotated image(s).
        """

        return rotate(img, self.angle, axes=(0,1))

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Rotate the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.
        """
        ox, oy = int(self.h/2), int(self.w/2)

        new_coords = coords.copy()

        new_coords[:, 0] = ox + math.cos(self.angle) * (coords[:, 0] - ox) - math.sin(self.angle) * (coords[:, 1] - oy)
        new_coords[:, 1] = oy + math.sin(self.angle) * (coords[:, 0] - ox) + math.cos(self.angle) * (coords[:, 1] - oy)

        return new_coords

class RandomRotate(TransformGen):
    """
    Rotate the image by a maximum angle with the given probability.
    """

    def __init__(self, angle=0, prob=0.5):
        """
        Args:
            angle (int): maximum angle of rotation.
            prob (float): probability of rotation.
        """
        super().__init__()

        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            return RotateTransform(h,w,self.angle)
        else:
            return NoOpTransform()


class DataAugmentation():

    """
    A mapper to add data augmentation at the train dataloader.

    """

    def __init__(self, resize_factor=1, crop_size=[1,1], hflip=False, p_hflip=0.5, rot=False, angle=0):

        """
        Args:
            resize_factor (float): percentage of the original image size.
            crop_size (list of floats): relative range of the original image size, if different to [1,1], apply random crop.
            hflip (bool): if True apply horizontal flip.
            p_hflip (float): if hflip=True, probability to apply horizontal flip.
            rot (bool): if True apply rotation.
            angle (int): degrees of maximum rotation.
        """

        self.resize_factor=resize_factor
        self.crop_size=crop_size
        self.hflip=hflip
        self.p_hflip=p_hflip
        self.rot=rot
        self.angle=angle


    def mapper(self, dataset_dict):

        """
        Configurates data augmentation mapper.

        Args:
            dataset_dict (dict): dataset dictionary in coco format.
        """

        dataset_dict = copy.deepcopy(dataset_dict) 
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        new_shape = (int(image.shape[0]*self.resize_factor), int(image.shape[1]*self.resize_factor))
        
        if self.rot:
            image, transforms = apply_transform_gens([RandomRotate(angle=5)], image)
        elif self.resize_factor!=1:
            image, transforms = apply_transform_gens([Resize(new_shape),
                                                      RandomFlip(prob=self.p_hflip,horizontal=self.hflip)], image)
        elif self.crop_size != [1,1]:
            image, transforms = apply_transform_gens([RandomCrop('relative_range', self.crop_size),
                                                      RandomFlip(prob=self.p_hflip,horizontal=self.hflip)], image)
        else:
            image, transforms = apply_transform_gens([RandomFlip(prob=self.p_hflip,horizontal=self.hflip)], image)

        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict


class MapperTrainer(SimpleTrainer):
    """
    A trainer with default training logic. Compared to `DefaultTrainer`, it
    contains the data augmentation in addition.

    """

    def __init__(self, cfg, mapper):
        """
        Args:
            cfg (CfgNode):
        """
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg,mapper=mapper)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        super().__init__(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it.

        Otherwise, load a model specified by the config.

        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter = (
            self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume).get(
                "iteration", -1
            )
            + 1
        )

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.

        It is now implemented by:

        .. code-block:: python

            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]

        """
        # Here the default print/log frequency of each writer is used.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(cls, cfg, mapper):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg,mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        raise NotImplementedError(
            "Please either implement `build_evaluator()` in subclasses, or pass "
            "your evaluator as arguments to `DefaultTrainer.test()`."
        )

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        #build dataloader as train
        self.val_loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        data_val = next(self.val_loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data_val)

            loss_dict_val = {"val_" + k: v.item() for k, v in
                            comm.reduce_dict(loss_dict).items()}
            total_val_loss = sum(loss for loss in loss_dict_val.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=total_val_loss,
                                                 **loss_dict_val)


def get_centroid(rect):
    x1, y1, x2, y2 = rect
    x = int((x1+x2)/2)
    y = int((y1+y2)/2)
    return (x, y)
