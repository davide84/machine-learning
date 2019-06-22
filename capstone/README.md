# README.md

## Introduction

The project was developed on a personal computer running Ubuntu 18.04 with Python 3.6. All required packages have been installed with pip; the GPU variant of tensorflow was installed, to make use of the local Nvidia GPU.

## Software dependencies

The project uses the following packages, in alphabetical order:

* cv2
* glob
* keras
* matplotlib
* numpy
* os
* PIL
* random
* shutil
* sklearn
* tensowflow


## Dataset and folders

The dataset can be downloaded from
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/downloads/chest-xray-pneumonia.zip/2

The downloaded zip file contains the images in a nested archive; the data is actually contained in three folders at the end of the hierarchy (train/ val/ and test/). The project notebook needs to know where these three folders are located; the path can be configured, the default value is ./dataset/chest_xray - meaning that training images will be in ./dataset/chest_xray/train/ folder.

The notebook will generate preprocessed images in ./dataset/resized/ folder, but only if the resized/ subfolder is not existing. Removing it will cause the images to be reprocessed. This path can also be configured if needed.

Finally, the notebook needs permission to write in the current folder, and also folder ./saved_models/ must be existing and writeable.
