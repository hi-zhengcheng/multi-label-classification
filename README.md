# multi-label-classification

## 1. Data Preparation

To make tensorflow run in high efficiency, first save data in TFRecord files.

1. Create one dir and copy all images into this dir. We call it `image_dir`.

1. Create `image_list` txt file. The format is like:

    ```
    COCO_val2014_000000320715.jpg 8
    COCO_val2014_000000379048.jpg 2
    COCO_val2014_000000014562.jpg 9
    ...
    ```

    Tip: create two files, one for training, one for evaluation.

1. Create `image_label` txt file. The format is like:

    ```
    1 1 1 1 0 0 1 0 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    1 0 1 1 1 1 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
    ...
    ```

    In the above example, assume there are total **35** labels. Each line corresponds to one image in `image_list` txt file. Each label has one fixed index. The value 1 means image has this label, 0 means not. The number in second column of `image_list` means how many labels the image file has.

    Tip: create two files, one for training, one for evaluation.

1. Create `tfrecords` file. The model will read data from TFRecords data format. Just run script:

    ```
    python create_tfrecord.py \
        --image_dir="/path/to/images_dir" \
        --imglist_file="/path/to/image_list_file" \
        --imglabel_file="/path/to/image_label_file" \
        --output_file="/path/to/xx.tfrecords" \
        --gpu="1"
    ```

    Tip: Create `train.tfrecords` and `eval.tfrecords` separately. `read_tfrecord.py` is just a tool script to read data from tfrecords for test purpose.

## 2. Base network definition and pre-trained checkpoints.
1. This library do image feature extraction by pre-trained resnet_50 model. I have downloaded network definition files(`resnet_utils.py`, `resnet_v2.py`) from [ResNet V2 50](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py), you still need to [download the pre-trained checkpoint](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz).

1. You can also change the file `multi_label_classification_model.py` to use rest101 or other models. [Find the networks and pre-trained models from here](https://github.com/tensorflow/models/tree/master/research/slim).

## 3. Multi-label-classification Model

1. Model is defined in file `multi_label_classification_model.py`. I just choose one endpoint in the pre-trained model, then **add three conv2d layers** in the end.

1. Input image processing is very import. The logic is:
    1. In training process, first resize image to a larger size, then random crop to the target size, and do some image augmentations. Finally use this randomly created image for training.

    1. In evaluation process, I just resize image to the target size.

    1. In inference process, first resize image to a larger size, then use `10 crops evaluation` method: for one image, using 10 crops(top-left, top-right, bottom-left, bottom-right, center and the mirrors) go throw the model, and compute the mean or max value of 10 outputs.


## 4. Training

1. Config model. In `train.py`, modify the params of `ModelConfig` creation.

1. Config train. In `train.py`, modify the params of `TrainConfig`.

1. Run script:

    ```
    python train.py
    ```

## 5. Evaluation

After start train script, start the evaluation script, let it run in parallel with train:

1. Config model. In `evaluate.py`, modify the params of `ModelConfig` creation.

1. Config eval. In `evaluate.py`, modify the params of `EvalConfig`.

1. Run script:
    ```
    python evaluate.py
    ```

## 6. Inference

Use `tensorboard` to monitor the training process. When the model is likely to be overfitting, start it, choose one good checkpoint, and use this checkpoint to do inference operation on test dataset:

1. Config model. In `inference.py`, modify the params of `ModelConfig` creation.

1. Config eval. In `inference.py`, modify the params of `InferenceConfig`.

1. Implement the `get_test_image_list` funciton in `inference.py`, let it return a list of image paths, like:

    ```
    [
        '/path/to/img1.jpg',
        '/path/to/img2.jpg',
        ...
    ]
    ```

1. Run script:
    ```
    python inference.py
    ```

## 7. Threshold calibration

After inference, the model produces score (or confidence) values for each label. It's time to choose threshold values to decide whether specific label belongs to an image or not. Method is:

1. Use trained model to do inference on evaluation dataset. It produces the scores for evaluation dataset.

1. Use `threshold_calibration.py` to compute optimal thresholds for each label.

1. Use the computed optimal thresholds on the test dataset's inference result.
