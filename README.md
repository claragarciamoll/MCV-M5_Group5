# Team 05: Object Detection, Recognition and Segmentation

Members:
- Gemma Alaix: gemma.alaix@e-campus.uab.cat
- Sergi Solà: sergi.solac@e-campus.uab.cat
- Clara Garcia: clara.garciamo@e-campus.uab.cat

Overleaf link: https://www.overleaf.com/read/vtqcqjffdbpq



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
