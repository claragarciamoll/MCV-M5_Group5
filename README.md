# Team 05: Object Detection, Recognition and Segmentation

Members:
- Gemma Alaix: gemma.alaix@e-campus.uab.cat
- Sergi Solà: sergi.solac@e-campus.uab.cat
- Clara Garcia: clara.garciamo@e-campus.uab.cat

Overleaf link: https://www.overleaf.com/read/gjtkhhxbdryp



## WEEK 1: Introduction to Pytorch

On the first week it is implemented a model, done at first with keras, using pytorch. The model created should be able to classify landscapes images into 8 different classes such as, coast, forest, mountain, and so on.

The model built is called Pokenet which is simple, fast and efficient. This model has 4 convolutional layers and at the end an average pooling that helps to reduce the number of parameters.

Slides Week 1: https://docs.google.com/presentation/d/1EObySNg2QzrBMco_XzRF06RO_lm9tRwX5Z2FnkJPpmw/edit?usp=sharing

## WEEK 2: Introduction to Object Detection

On this second week it is implemented an object detector algorithm using Detectron2. In the code there are 3 options, first of all, when the model is not trained that could be run the code using Faster R-CNN or using RetinaNet. On the other hand, when the model is trained using Kitti dataset where the code is run with Faster R-CNN.

Full slides Week 2: https://docs.google.com/presentation/d/1Jqn9qbvMEgt0Hi_OwwWv9SsKKzMiekAKXNUeloeEPew/edit?usp=sharing

## WEEK 3: Introduction to M5 Project: Multiple Object Tracking and Segmentation (MOTS)

On this third week it is implemented the methods to solve two challenges proposed on CVPR 2020 workshop (https://motchallenge.net/workshops/bmtt2020/tracking.html).
The first challenge is based on pedestrians tracking and segmentation, while the second challenge is to track and segmentate not only pedestrians, but also cars. In order to achieve both challenges two datasets are given, the first one is callet MOTSChallenge whereas the second one is called KITTI-MOTS and two models are used, the Faster R-CNN and the RetinaNet using Detectron2 as in previous week.

Full slides Week 3: https://docs.google.com/presentation/d/1_Uoyy5iyPBSmU83a5mO9xzRgbldVN5zLVR-uq6PiIEg/edit?usp=sharing

## WEEK 4: Introduction to Object Segmentation

On this fourth week it is implemented an image instance segmentation algorithm using Detectron2. In the code there are many options that let us to use different models from COCO that uses the Mask R-CNN with different backbone combinations such as:
* ResNet50-FPN: Use a ResNet50+FPN with standard convolutional layers and FC heads for mask and box prediction.
* ResNet101-FPN: Use a ResNet101+FPN with standard conv layers and FC heads for mask and box prediction. 
* ResNeXt101-FPN: Use a ResNeXt101+FPN with standard conv and FC heads for mask and box prediction.
* R50-C4: Use a ResNet50 conv4 backbone with conv5 head.
* R101-C4: Use a ResNet101 conv4 backbone with conv5 head.
* R50-DC5: Use a ResNet50 conv5 backbone with dilations in conv5, and standard conv and FC heads for mask and box prediction.
* R101-DC5: Use a ResNet101 conv5 backbone with dilations in conv5, and standard conv and FC heads for mask and box prediction.
* Cityscapes: Use Mask R-CNN on Cityscapes instance segmentation.

https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md

Moreover there are more options such as, using the pretrained models or train the model using our own dataset, which is KITTI-MOTS dataset. There is the option to choose different thresholds and different learning rates 

On the slides below there are some results obtained usind some different options mentioned before.

Full slides Week 4: https://docs.google.com/presentation/d/1FevNbnYPYh_Ra6Ob_N1no4BKM0hCSgRR8rPPd5mcWEQ/edit?usp=sharing

## WEEK 5: Transfer Learning for Object Detection and Segmentation

On this fifth week what is introduced is trasnfer learning and the goal was to use pre-trained and fine-tuning using different datasets datasets in order to train the model. The different combinations used on this project was:

1) Using MOTSChallenge dataset to evaluate the model:
  * COCO
  * COCO + Cityscapes
  * COCO + KITTI-MOTS
  * COCO + Cityscapes + KITTI-MOTS
  
 2) Using KITTI-MOTS dataset to evaluate the models
  * COCO
  * COCO + MOTSChallenge
  * COCO + Cityscapes
  * COCO + Cityscapes + MOTSChallenge
  * COCO + KITTI-MOTS
  * COCO + KITTI-MOTS + MOTSChallenge
  * COCO + Cityscapes + KITTI-MOTS
  * COCO + Cityscapes + KITTI-MOTS + MOTSChallenge
  
And finally in this week it is also added the hyperparameters in order to obtain some results and analyse them. The hyperparameters used were:

* Input Resolution
* Background threshold and Foreground threshold
* Data augmentation using cropping

On the slides below there are some results obtained usind some different options mentioned before.

Full slides Week 5: https://docs.google.com/presentation/d/1ffRBOUElYEvtR4ZL01RCWqa-9Or2ycHTxAt-cCGvkCA/edit?usp=sharing

## WEEK 6: Data Augmentation, Semantic Segmentation and Video Object Segmentation

On this last week there are included data augmentation parameters like crop, horitzontal flip and rotation. Moreover, it is included the usage of the virtual clone of KITTI-MOTS dataset in order to observe how it could improve the results obtained on previous weeks.

Finally, it is used DeepLabv3+ (https://github.com/tensorflow/models/tree/master/research/deeplab), which is a deep learning model for semantic image segmentation. Trying to reproduce the experiments from Table 7 (a) in Chen et
al [1]. <a href='w6-code/semantic_segmentation.md'>Training semantic segmentation on cityscapes dataset.</a><br>


Full slides Week 6: https://docs.google.com/presentation/d/1roW6wcd_nX_8PDHGwg0frgG0eMX95xO5_bwbnzgl-hg/edit?usp=sharing

Final Project Presentation: https://docs.google.com/presentation/d/11YdVDG5UOdnC7ujvP2Od4EOKS8Tkpb3CdPmt7Tc4sYo/edit?usp=sharing 

## References

1.  **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**<br />
    Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam (+ equal
    contribution). <br />
    [[link]](https://arxiv.org/pdf/1802.02611.pdf). In ECCV, 2018.
