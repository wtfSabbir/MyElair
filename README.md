# ECLAIR: Attention Fully Convolutional Semantic Segmentation Algorithm

## Summary

- [Installation](#installation---procedure-followed-to-recreate-the-environment)
- [PLO Prediction](#plo-prediction-with-eclair)
  - [Predicting Point Clouds](#vertical-object-prediction-with-eclair-on-photogrammetric-clouds-lasz)
  - [Reproducing Models](#reproducibility-of-training)
  - [Comparing Results with Existing Model](#evaluating-results)
- [LiDAR Cloud Prediction](#lidar-cloud-prediction-with-eclair)
- [Launch a Training with ECLAIR](#launch-a-training-with-eclair)
  - [Point Clouds](#point-clouds)
  - [Training Configuration](#training-configuration)
  - [Transformations Applied to Clouds](#transformations-applied-to-clouds)
  - [Train the Model](#train-the-model)
- [Predict Point Clouds](#predict-point-clouds)
- [Integration into PSANP](#integration-into-psanp)
- [Learn More](#learn-more)

---

## Installation - Procedure Followed to Recreate the Environment


Eclair uses the MinkowskiEngine library, which is only available on Linux.

**Prerequisites:**
* Debian-based Linux distribution (e.g., Ubuntu)
* [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
* OpenBLAS (`sudo apt install -y libopenblas-dev`)
* Python 3.10 (virtual environment strongly recommended)

```bash
pip install --index-url="[https://pypi.geo-sat.com/simple](https://pypi.geo-sat.com/simple)" eclair
```


<br>
<br>

## PLO Prediction with Eclair

### Vertical Object Prediction with Eclair on Photogrammetric Clouds (`.la[s/z]`).

Prediction is launched with the command:

```bash
python -m eclair.plos.predict.predict --weight_path1 [path to the basemodel weights] --weight_path2 [path to the PLOs model weights] --config_file [path to the config.yaml file] --pointclouds [path to the pointclouds folder] --save_path [path to the save folder]
```
* The model weights **--weight_path1** and **--weight_path2** are available on MLFlow: https://gsvision.geo-sat.com/mlflow/#/experiments/911728019003067248/runs/70d9b7c1332b47af81d23a5bb8495e28/artifacts
* The **--config_file** is the file `src/eclair/plos/predict/config.yaml`
* The folders for point clouds to predict and for saving predictions must be specified in the arguments 
```--config_file``` and ```--save_path```

Each predicted and saved cloud contains 3 new fields:

* **basemodel_v13** : contains the predictions of the base Eclair model, with the classes:
  ```text
  Unclassified: 0
  Ground: 1
  PLO: 2
  Trunk: 3
  Vegetation: 4
  Building: 5
  ```
* **plo_v13** : contains the predictions of the Eclair model specialized in PLOs, with the classes:
  ```text
  Unclassified: 0
  Pole: 1
  Sign: 2
  Bollard: 3
  ```
* **fuze** : contains the fusion of the previous predictions, including decision rules based on the 
  predicted classes. The class mapping is as follows:
  ```text
  Unclassified: 0
  Ground: 1
  Trunk: 2
  Vegetation: 3
  Building: 4
  Pole: 5
  Sign: 6
  Bollard: 7
  ```

> ⚠️ The 'classification' field is also edited with the same classification as the 'fuze' field, to allow visualization on Potree or Paintcloud

> ⚠️ Clouds must be a maximum size of 40x40x40 meters! Points beyond this will all be classified as 'Unclassified'

---

### Reproducibility of Training

The scripts present in the folder `/src/eclair/plos/train` allow reproducing 
the training of the 2 models used for PLO prediction:
- Configuration files *first_model.yaml* for the base model and *plo.yaml* for the PLO model
- File *train.py* to execute the training.
- The datasets used for these trainings as well as the train/val distribution files
are available here: `R:\9_temporary_files\for_temporary_storage_1_month\malo\Eclair_preprod\v13\train_data`

<br>

### Evaluating Results

The script `src/eclair/plos/predict/evaluate_mlflow.py` allows evaluating prediction results 
with manually annotated "ground truth" clouds. The metrics used are precision and recall scores
for the classes *Pole*, *Sign*, *Bollard*, and *Trunk*. The metrics are calculated by comparing
the *Classification* fields of two folders of `.la[s/z]` point clouds. The clouds must have exactly the same
filename in both folders.

The script is executed with the command:

```bash
python -m eclair.plos.predict.evaluate_mlflow --gt_dir [path to the ground truth pointclouds folder] --pred_dir [path to the predicted pointclouds folder] --config_path [path to the evaluate_mlflow.json file] [--log_mlflow]
```

* The configuration file is available here: `src/eclair/plos/predict/evaluate_mlflow.json`
  * The **log_artifacts_files** parameter of this file allows storing files
  linked to this experiment (model weights or training configuration file, for example).
  It expects a list of paths to the files you wish to save. By default, no file is saved.
* The manually annotated 'ground truth' clouds are stored here: `R:\9_temporary_files\for_temporary_storage_1_month\malo\Eclair_preprod\v13\pointclouds_test_gt`
* The argument ```--log_mlflow``` activates the recording of results on MLflow

The different experiments on PLO segmentation are available here: https://gsvision.geo-sat.com/mlflow/#/experiments/911728019003067248



<br>
<br>


### LiDAR Cloud Prediction with Eclair

An Eclair model is available for LiDAR cloud segmentation. The segmented classes are as follows: 

  ```text
  Unclassified: 0
  Ground: 1
  Vegetation: 2
  Building: 3
  Human: 4
  Urban Furniture: 5
  Vehicle: 6
  Cable: 7
  ```

The weights, training and prediction scripts, as well as the location of the training set are available on MLFlow:
https://gsvision.geo-sat.com/mlflow/#/experiments/433491281747178879/runs/85a12cadffb8468491bf86f71be0bab0

> ⚠️ To use this model, the clouds must be a maximum size of 30x30x30 meters! Points beyond this will all be classified as 'Unclassified'


<br>
<br>

## Launch a Training with ECLAIR

The scripts present in the `/scripts` folder allow launching an Eclair training and predicting point clouds
with a trained model. 

### Point Clouds

Point clouds must be in .la[s/z] format.
* A .json file with the names of your clouds and their distribution in train/val/
  test is required. The file must have the form:

```json
[
  {
    "tile_name": "pointcloud1.laz",
    "split": "train"
  },
  {
    "tile_name": "pointcloud2.laz",
    "split": "test"
  },
  {
    "tile_name": "pointcloud3.laz",
    "split": "val"
  },
  ...
]
```
* The clouds in train and val must be placed in the same folder, which will be specified for training.

<br>

### Training Configuration

All training configuration is done with the file `scripts/config.yaml`


#### Data-related Parameters

- The **features_name** parameter lists the features to use to train the model. Just comment out the unwanted ones in the config file. Four features are supported:
    - Intensity (`intensity`)
    - RGB values (`colors`)
    - Return number (`return_number`)
    - Number of returns (`number_of_returns`)
  
  

- The **num_features** parameter must contain the total number of features for your model. Example: if you want to
  take into account intensity and RGB values as features, the total number of features is 4 (intensity=1,
  RGB=3)

```yaml
feature_names:
  - intensity # size: 1
  #  - return_number # size: 5 (one-hot)
  #  - number_of_returns # size: 5 (one-hot)
  - colors # size: 3

num_features: 4
```

<br>

- The "num_classes" parameter indicates the number of classes for your model. Note, if you only have 2
  classes (one class vs everything else), you must indicate num_classes = 1

```yaml
num_classes: 3  # If two classes, indicate 1. Otherwise, indicate normally the nb of classes.
```

<br>

* If you wish to reorder your classes, merge them, or perform training with only certain ones, you
  can use the **input_mapping** parameter. It allows mapping between the class IDs of your clouds
  and those that will actually be used during training. The IDs at the output of **input_mapping** must start at zero and be consecutive.

```yaml
#Example to go from 11 to 3 classes

input_mapping:
  0: 0
  1: 1
  2: 0
  3: 1
  4: 0
  5: 2
  6: 2
  7: 2
  8: 0
  9: 0
  10: 0
```

* If you do not wish to touch your cloud classes, simply map each ID to itself

```yaml
input_mapping:
  0: 0
  1: 1
  2: 2
  3: 3
  4: 4
  5: 5
  6: 6
  7: 7
  8: 8
  9: 9
  10: 10
```

<br>

#### Model Parameters

* The **model_name** parameter defines the model used. Twelve models are available. All these models are based on the U-Net architecture for 3D data, using the
MinkowskiEngine library.
They share an encoder/decoder structure but differ in their depth (number of layers) and width
(number of channels), which directly influences their learning capacity, training time, and performance.
Variants of the same model (like 18A, 18B, 18D) adjust channels to adapt to different resource constraints
or precision needs. Here are the available models and a brief description of each:

| Model       | Description                                                                                                   |
|-------------|---------------------------------------------------------------------------------------------------------------|
| MinkUNet14  | Very lightweight model with few blocks, ideal for tests or small datasets.                                    |
| MinkUNet14C | Variant of 14 with increased width (custom PLANES) for a better capacity/speed compromise.                    |
| MinkUNet18  | Standard balanced model, commonly used for a good performance/complexity compromise.                          |
| MinkUNet18A | Lightweight variant of 18 with fewer channels in the decoder, for faster execution.                           |
| MinkUNet18B | More homogeneous variant of 18 with constant channels in the decoder.                                         |
| MinkUNet18D | Larger variant of 18, with significantly more output channels (decoder).                                      |
| MinkUNet34  | Deeper version than 18, adapted for more complex tasks.                                                       |
| MinkUNet34A | Variant of 34 with a wider decoder, maintaining architecture symmetry.                                        |
| MinkUNet34B | Variant of 34 with progressive channel reduction in the decoder.                                              |
| MinkUNet34C | Variant of 34 with a moderately reduced decoder, between A and B.                                             |
| MinkUNet50  | Deep model using bottleneck blocks, more memory efficient for its size.                                       |
| MinkUNet101 | Very large model with bottleneck blocks, adapted for tasks requiring strong representation capacity.          |

* **voxel_size**: voxel size, in meters.
* **batch_size**: batch size during training.
* **test_batch_size**: batch size during testing.
* **epoch**: number of epochs for training.
* **patience**: number of consecutive epochs without improvement in validation loss before training stops.
* **num_worker**: number of parallel processes to read data.
* **pin_memory**: enables optimization for faster data transfer to GPU (True/False).
* **learning_rate**: the 'step' used to update model weights at each iteration.
* **weight_decay**: penalty applied to the size of model weights (reduces overfitting).
* **self_attention**: inserts 0, 1, or 2 self-attention modules into the model.
* **multihead_attention**: inserts 0, 1, or 2 multihead attention modules into the model.
* **num_head**: number of heads for MultiHeadAttention modules.
* **loss**: desired loss for training. Three losses are available:
    * For binary classification (id 0: everything not searched for, id 1: the class searched for), it is recommended to use *TverskyFocaleLoss*.
    * For a number of classes > 2, *FocaleLoss* and *FocalTverskyLoss_multiclass* are proposed.
* **loss_weights**: Hyperparameters of the loss.
    * For *TverskyFocaleLoss*, **loss_weights** must be of the form [alpha, beta]. For your class with id 1, alpha
      will penalize false positives and beta false negatives (0 < alpha < 1 & 0 < beta < 1).
    * For *FocalTverskyLoss_multiclass*, **loss_weights** must have the form: [[alpha_0, alpha_1, ..., alpha_n],[beta_0, beta_1, ..., beta_n]], where alpha_i and beta_i penalize false positives and false negatives of class i respectively.
    * For *FocaleLoss*, **loss_weights** must be of the form [w_0, w_1, ..., w_n], where w_i is the weight associated with the
      i-th class.

* **weights_name**: the name of the file containing the model weights that will be saved in the folder indicated in the training command line.

```yaml
model_name: MinkUNet14 #See readme for possibilities
voxel_size: 0.08 #Voxel size in meters.
batch_size: 2
test_batch_size: 1
epochs: 1000
patience: 100
num_worker: 1
pin_memory: False
learning_rate: 1e-4
weight_decay: 1e-5
self_attention: 0 # 0, 1 or 2 layers
multihead_attention: 1 # 0, 1 or 2 layers.
num_head: 4 #Number of heads for MultiHeadAttention
loss: FocalTverskyLoss_multiclass #Choice: FocalTverskyLoss (binary case), FocalLoss (any case), FocalTverskyLoss_multiclass (> 2 classes)
loss_weights: [[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5]] #See Readme for more info depending on desired loss.
weights_name: 'eclair' #name of weights that will be saved
```

<br>

#### Transformations Applied to Clouds

A certain number of transformations are applied to clouds before being placed as input to the model. 

* **cloud_size** crops the points of the cloud into a fixed-size 3D box and filters
  outlier points (noise under the cloud) according to a quantile on the Z-axis. This transformation standardizes the spatial scale of data
  to
  facilitate their processing by the model. If clouds are smaller than the indicated size, a dummy point will be
  added (then removed at the end, don't worry).
  If clouds are larger, points beyond the defined limits will not be taken into account for training
  and will be classified as zero for prediction.

Transformations can be applied or not to data to improve training and are to be indicated in the **train_transforms** parameter:

* Eleven data-augmentation methods are available. Simply comment out unwanted ones to not apply them:
  * *Intensity_RGB_Variation*: Applies a random variation to intensity and RGB values (if available) between ±15%. RGB values are limited to a valid range (default 65535 for 16 bits).
  * *Scale*: Applies a random variation of ±10 % to the XYZ coordinates of the point cloud.
  * *Noise*: Adds uniform random noise between -5 % and +5 % to intensity and RGB fields (if available). RGB values are limited to a valid range (default 65535).
  * *Rotate*: Applies a random rotation around the Z-axis (angle between 0 and 2π) to the point cloud coordinates.
  * *Flip*: Inverses Y coordinates with a probability of 50 % (no inversion on X axis in the current version).
  * *RandomCrop*: Randomly cuts a portion of the point cloud, selecting a bounding box whose dimensions vary between 90 % and 100 % of the cloud size.
  * *RandomDelete*: Randomly deletes between 0.1 % and 10 % of the cloud points, updating associated attributes (intensity, RGB, classification, etc.).
  * *AddRandomPoints*: Adds a random number of points (up to 0.001 % of the total) with coordinates in a bounding box slightly extended around the cloud. New points have random RGB values (0 to 65535) and a classification fixed at 0.
  * *ElasticDistortion*: Applies an elastic distortion by generating a field of random displacements smoothed by a Gaussian filter. Parameters sigma (0.01 to 0.3) and scale (0.01 to 0.1) control the amplitude of the distortion.
  * *ShearTransform*: Applies a random shear to coordinates with an amplitude between 0.001 and 0.01, modifying relations between X, Y, and Z axes.
  * *StretchTransform*: Applies a random independent stretch or compression on each axis (X, Y, Z) with factors varying in a random interval (default between 0.85 and 1.15).

  
* 3 data formatting methods (advised not to touch), used for train, val, and test data:
    - *NormalizeCoordinates* / *NormalizeCoordinates_predict*: normalizes coordinates and crops points in the defined bounding box.
    - *NormalizeFeatures*: normalizes features.
    - *RemapClassification*: reorders IDs from values indicated in "input_mapping".

```yaml
cloud_size: #Fixed cloud size. See readme for more info.
  max_range_xy: 40.0
  max_range_z: 40.0

#Data-augmentation applied. More info in the readme. Simply comment to not apply augmentation.
train_transforms:
  - name: Rotate
    params:
      feature_names: ${feature_names}
  - name: Flip
    params:
      feature_names: ${feature_names}
  - name: Scale
    params:
      feature_names: ${feature_names}
  - name: RandomDelete
    params:
      feature_names: ${feature_names}
#  - name: RandomCrop
#    params:
#      feature_names: ${feature_names}
#  - name: AddRandomPoints
  - name: ElasticDistortion
  - name: ShearTransform
  - name: StretchTransform
  - name: Intensity_RGB_Variation
    params:
      feature_names: ${feature_names}
  - name: Noise
    params:
      feature_names: ${feature_names}
  - name: NormalizeCoordinates
  - name: NormalizeFeatures
    params:
      feature_names: ${feature_names}
  - name: RemapClassification
    params:
      class_mapping: ${input_mapping}


###Basic normalization for test and validation clouds. It is advised not to touch it.
val_transforms:
  - name: NormalizeCoordinates
  - name: NormalizeFeatures
    params:
      feature_names: ${feature_names}
  - name: RemapClassification
    params:
      class_mapping: ${input_mapping}

test_transforms:
  - name: NormalizeCoordinates_predict
  - name: NormalizeFeatures
    params:
      feature_names: ${feature_names}
  - name: RemapClassification
    params:
      class_mapping: ${input_mapping}
```

#

#### Train the Model

Once everything is in place, simply execute the command: 

```bash
python scripts/train.py --data_dir [the directory path where the pointclouds are stored (train & val)] --label_json [path to the.json that specifies which set (train, validation, or test) each cloud belongs to] --config_yaml [path to the config.yaml file] --save_dir [the directory path where the weights will be stored]
```

* The saved weight files all have a name of the form: **weights_name**_[Weight Type].pth
* [Weight Type] can take the form of:
    * "final_weights" for weights from the last epoch.
    * "bestvallossweights" for weights minimizing validation loss
    * "bestvalaccweights" for weights maximizing validation accuracy
    * "bestvalf1scoreweights" for weights maximizing the F1 score of class number 1 (only in the case
      of binary classification)

<br>

## Predict Point Clouds

Point cloud prediction is done with the script ```scripts/predict.py```. The script uses the same configuration file as for training.

This script is launched with the command:

```bash
python predict.py --weight_path [path to the model weights] --config_file [path to the config.yaml file used for training] --pointclouds [either .json file or path to a folder with pointclouds to predict] --savepath [path to the save folder]
```

The clouds indicated in "test" in the .json will then be predicted, or all clouds in the folder, depending on what is
specified in the argument ```--pointclouds```

Predictions are saved in a new **eclair** field.

## Integration into PSANP

This project implements a [plugin for PSANP](http://geosat-docs.s3-website.eu-west-3.amazonaws.com/psanp-abstractions/master/plugins.html),
which allows chaining an Eclair classification with other types of processing
(sub-sampling, ortho-image generation...). For PSANP to be able to load 
the weights and configuration of a model, the following structure must be respected:

```
<classifiers>
├───EclairBaseClassifier
│       model.yaml
│       weights.pth
│
├───EclairPLOClassifier
│       model.yaml
│       weights.pth
│
└───...
```

**Notes**

- In the example above, the folders `EclairBaseClassifier` and `EclairPLOClassifier` are called *bundles*.
- They **must** contain an *Eclair* configuration file `model.yaml` and a weights file `weights.pth`.
- Their name **must** start with `Eclair` and end with `Classifier`, and follow the *PascalCase* convention.
- For PSANP to be able to load bundles, the environment variable 
  `PSANP_BUNDLE_FOLDER` must point to the `<classifiers>` folder

> ℹ️ The script `scripts/check_psanp_bundles.py` allows verifying if a bundle is valid for PSANP 

---

## Learn More

* The git repo of the MinkowskiEngine project used by Eclair is available here: https://gitlab.geo-sat.com/rnd/research/minkowski-engine 
* Research reports on Eclair are available here: `R:\4_archived_projects\4_reports\1_research_reports\Eclair`
