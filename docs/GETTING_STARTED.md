## Dataset Preparation
We suggest to organize the dataset in the following way:
* -TODO structure here-
* -TODO maybe code here to create structure-


```
View-of-Delft-Dataset (root)
│   ├── lidar (kitti dataset where velodyne contains the LiDAR point clouds)
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & image_2 & label_2
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
```


## Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 14.04/16.04/18.04/20.04/21.04)
* Python 3.6+
* Note that Python 3.6+ is required for the devkit and its examples, but the dataset can be accessed with any programming language and Python version.

### Installation of the Devkit

a. Clone this repository.
```shell
git clone https://github.com/tudelft-iv/view-of-delft-dataset
```

b. Create a virtual environtment with the requried dependencies:

```shell
# using pip
-TODO virtual env here-
pip install -r requirements.txt

# or using Conda
conda create --name vod-dataset --file requirements.txt
```

