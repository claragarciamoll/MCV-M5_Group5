# Running DeepLab on Cityscapes Semantic Segmentation Dataset

This page walks through the steps required trying to reproduce DeepLab performance on Cityscapes on a
local machine.

## Dependencies

DeepLab depends on the following libraries:

*   Numpy
*   Pillow 1.0
*   tf Slim (which is included in the "tensorflow/models/research/" checkout)
*   Jupyter notebook
*   Matplotlib
*   Tensorflow (<2)

For detailed steps to install Tensorflow 1, follow the [Tensorflow installation
instructions](https://www.tensorflow.org/install/). A typical user can install
Tensorflow using one of the following commands:

```bash
# For CPU
pip install tensorflow (<2)
# For GPU
pip install tensorflow-gpu (<2)
```

## Download DeepLab source code

Clone tensorflow models repository.

``
git clone https://github.com/tensorflow/models.git
``

## Download dataset and convert to TFRecord

You should add cityscapes to datasets folder, from the directory `tensorflow/models/research/deeplab`, but also include `cityscapesscripts`, by ``git clone https://github.com/mcordts/cityscapesScripts.git``. Directory structure should look as follows:

```
+ datasets
  + cityscapes
    + cityscapesscripts
    + leftImg8bit
    + gtFine
```

Run the script (under the folder `datasets`) to convert Cityscapes
dataset to TFRecord.

```bash
# From the tensorflow/models/research/deeplab/datasets directory.
sh convert_cityscapes.sh
```

The converted dataset will be saved at ./deeplab/datasets/cityscapes/tfrecord.

## Running the train/eval/vis jobs

To run DeepLab, you must add ``export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`` to your local `.bashrc`.

A local training job using `xception_65` can be run with the following command:

```bash
# From tensorflow/models/research/
python deeplab/train.py \
    --logtostderr \
    --save_summaries_secs=120 \
    --save_interval_secs=600 \
    --training_number_of_steps=30000 \
    --train_split="train_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --add_image_level_feature=True \
    --decoder_output_stride=4 \
    --train_crop_size="769,769" \
    --train_batch_size=1 \
    --fine_tune_batch_norm=False \
    --dataset="cityscapes" \
    --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
    --train_logdir=${PATH_TO_TRAIN_DIR} \
    --dataset_dir=${PATH_TO_DATASET}
```

where ${PATH_TO_INITIAL_CHECKPOINT} is the path to the initial checkpoint
(usually an ImageNet pretrained checkpoint), ${PATH_TO_TRAIN_DIR} is the
directory in which training checkpoints and events will be written to, and
${PATH_TO_DATASET} is the directory in which the Cityscapes dataset resides.

Checkpoints used in this case will be:

Model name                                                                             | File Size
-------------------------------------------------------------------------------------- | :-------:
[xception_65_imagenet](http://download.tensorflow.org/models/deeplabv3_xception_2018_01_04.tar.gz) | 447MB
[xception_71_imagenet](http://download.tensorflow.org/models/xception_71_2018_05_09.tar.gz  ) | 474MB

In order to skip Image-Level, just remove line `add_image_level_feature`, and also not to use decoder remove `decoder_output_stride` line.

**Note that for {train,eval,vis}.py**:

1.  In order to reproduce our results, one needs to use large batch size (> 8),
    and set fine_tune_batch_norm = True. Here, we simply use small batch size
    during training for the lack of GPU memory and fine_tune_batch_norm = False.

2.  In order to skip Image-Level flag, just remove line `add_image_level_feature`, and also not to use decoder remove `decoder_output_stride` line.

3.  Change and add the following flags in order to use the provided dense
    prediction cell. 

```bash
--model_variant="xception_71"
--dense_prediction_cell_json="deeplab/core/dense_prediction_cell_branch5_top1_cityscapes.json"
```

A local evaluation job using `xception_65` can be run with the following
command:

```bash
# From tensorflow/models/research/
python3 deeplab/eval.py \
    --logtostderr \
    --eval_split="val_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --add_image_level_feature=True \
    --decoder_output_stride=4 \
    --train_crop_size="769,769" \
    --train_batch_size=2 \
    --fine_tune_batch_norm=False \
    --eval_crop_size="1025,2049" \
    --dataset="cityscapes" \
    --checkpoint_dir=${PATH_TO_CHECKPOINT} \
    --eval_logdir=${PATH_TO_EVAL_DIR} \
    --dataset_dir=${PATH_TO_DATASET}
```

where ${PATH_TO_CHECKPOINT} is the path to the trained checkpoint (i.e., the
path to train_logdir), ${PATH_TO_EVAL_DIR} is the directory in which evaluation
events will be written to, and ${PATH_TO_DATASET} is the directory in which the
Cityscapes dataset resides.

A local visualization job using `xception_65` can be run with the following
command:

```bash
# From tensorflow/models/research/
python3 deeplab/vis.py \
    --logtostderr \
    --vis_split="val_fine" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --add_image_level_feature=True \
    --decoder_output_stride=4 \
    --train_crop_size="769,769" \
    --train_batch_size=2 \
    --fine_tune_batch_norm=False \
    --vis_crop_size="1025,2049" \
    --dataset="cityscapes" \
    --colormap_type="cityscapes" \
    --checkpoint_dir=${PATH_TO_CHECKPOINT} \
    --vis_logdir=${PATH_TO_VIS_DIR} \
    --dataset_dir=${PATH_TO_DATASET}
```

where ${PATH_TO_CHECKPOINT} is the path to the trained checkpoint (i.e., the
path to train_logdir), ${PATH_TO_VIS_DIR} is the directory in which evaluation
events will be written to, and ${PATH_TO_DATASET} is the directory in which the
Cityscapes dataset resides.

## Running Tensorboard

Progress for training and evaluation jobs can be inspected using Tensorboard. If
using the recommended directory structure, Tensorboard can be run using the
following command:

```bash
tensorboard --logdir=${PATH_TO_LOG_DIRECTORY}
```

where `${PATH_TO_LOG_DIRECTORY}` points to the directory that contains the
train, eval, and vis directories (e.g., the folder `train_on_train_set` in the
above example). 
