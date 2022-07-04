## Dataset Preparation
We suggest to organize the dataset in the following way, which is also the format of how the dataset is provided:

```
View-of-Delft-Dataset (root)
    ├── lidar (kitti dataset where velodyne contains the LiDAR point clouds)
    │   │── ImageSets
    │   │── training
    │   │   ├──calib & velodyne & image_2 & label_2
    │   │── testing
    │       ├──calib & velodyne & image_2
    | 
    ├── radar (kitti dataset where velodyne contains the radar point clouds)
    │   │── ImageSets
    │   │── training
    │   │   ├──calib & velodyne & image_2 & label_2
    │   │── testing
    │       ├──calib & velodyne & image_2
    | 
    ├── radar_3_scans (kitti dataset where velodyne contains the accumulated radar point clouds of 3 scans)
    │   │── ImageSets
    │   │── training
    │   │   ├──calib & velodyne & image_2 & label_2
    │   │── testing
    │       ├──calib & velodyne & image_2
    |
    ├── radar_5_scans (kitti dataset where velodyne contains the radar point clouds of 5 scans)
        │── ImageSets
        │── training
        │   ├──calib & velodyne & image_2 & label_2
        │── testing
            ├──calib & velodyne & image_2
```


## Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 14.04/16.04/18.04/20.04/21.04)
* Python 3.6+
* Note that Python 3.6+ is required for the devkit and its examples, but the dataset can be accessed with any programming language and Python version.

### Installation of the Devkit from GitHub

1. Clone the repository: `git@github.com:tudelft-iv/view-of-delft-dataset.git`
2. Inside the root folder, use the `environment.yml` to create a new conda environment using: `conda env create -f environment.yml`
3. Activate the environment using: `conda activate view-of-delft-env`
4. In the same terminal windows typing `jupyter notebook` will start the notebook server.

In case the interactive plots do not show up in the notebooks use: `jupyter nbextension install --py --sys-prefix k3d`

### Installation of the Devkit using package manager
The devkit is also available as a pip package. To install the devkit, run the following command:
```
pip install vod-tudelft
```

### Usage
After fetching both the data and the devkit, please refer to these manuals for several examples of how to use them, including data loading, fetching and applying transformations, and 2D/3D visualization:
[Frame Information](https://github.com/tudelft-iv/view-of-delft-dataset/blob/main/1_frame_information.ipynb)
[Frame Transformations](https://github.com/tudelft-iv/view-of-delft-dataset/blob/main/2_frame_transformations.ipynb)
[2D Visualization](https://github.com/tudelft-iv/view-of-delft-dataset/blob/main/3_2d_visualization.ipynb)

### Evaluation
The devkit provides a set of evaluation scripts that can be used to evaluate the performance of your model.
The evaluation script is located under `vod.evaluation`. The fourth notebook provides an example of how to use the evaluation script.

The evaluation script is based on the origin Kitti C++ code, which has been translated to Python by a Github user 
named [Yan Yan](https://github.com/traveller59/kitti-object-eval-python). Our contribution to the evaluation script are:
- Removing CUDA dependency
- Adding further evaluation metrics
- Fixing Numba issues
- PEP8 style guide


