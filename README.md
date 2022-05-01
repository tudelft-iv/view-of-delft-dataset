<img src="docs/figures/LOGO.png" align="right" width="20%">

# The View of Delft dataset

This repository shares the documentation and development kit of the View of Delft automotive dataset.

<div align="center">
<figure>
<img src="docs/figures/example_frame_2.png" alt="Prius sensor setup" width="700"/>
</figure>
<br clear="left"/>
<b>Example frame from our dataset with camera, LiDAR, 3+1D radar, and annotations overlaid.</b>
</div>
<br />
<br />

## Overview
- [Introduction](#introduction)
- [Sensors and Data](docs/SENSORS.md)
- [Annotation](docs/ANNOTATIONS.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Examples and Demo](docs/EXAMPLES.md)
- [Citation](#citation)
- [Original paper](https://ieeexplore.ieee.org/document/9699098)
- [Links](#links)

## Introduction

We present the novel View-of-Delft (VoD) automotive dataset. It contains 8600 frames of synchronized and calibrated 64-layer LiDAR-, (stereo) camera-, and 3+1D radar-data acquired in complex, urban traffic. It consists of more than 76000 3D bounding box annotations, including more than 15000 pedestrian, 7100 cyclist and 17000 car labels.

<div align="center">
<p float="center">
<img src="docs/figures/sensors.gif" alt="Prius sensor setup" width="400"/>
<img src="docs/figures/labels.gif" alt="Prius sensor setup" width="400"/>
</p>
</div>

A short introduction video of the dataset can be found here (click the thumbnail below):

[![Thumbnail of the demo video](https://img.youtube.com/vi/R8r3lrkicJ0/0.jpg)](https://www.youtube.com/watch?v=R8r3lrkicJ0)

## Sensors and data
The LiDAR sensor is a Velodyne 64 operating at 10 Hz. The camera provides colored images of 1936 × 1216 pixels at around 12 Hz. The horizontal field of view is ~64° (± 32°), vertical field of view is ~ 44° (± 22°). Odometry is a filtered combination of several inputs: RTK GPS, IMU, and wheel odometry with a frame rate around 100 Hz. Odometry is given relative to the starting position at the beginning of the current sequence, not in absolute coordinates. Note that while camera and odometry operate at a higher frequency than the LiDAR sensor, timestamps of the LiDAR sensor were chosen as “lead”, and we provide the closest camera frame and odometry information available. 

The provided LiDAR point-clouds are already ego-motion compensated. More specifically, misalignment of LiDAR and camera data originating from ego-motion during the scan (i.e. one full rotation of the LiDAR sensor) and ego-motion between the capture of LiDAR and camera data has been compensated. See an example frame for our data with calibrated and ego-motion compensated LiDAR point-cloud overlaid on the camera image. 

We also provide intrinsic calibration for the camera and extrinsic calibration of all three sensors in a format specified by the annotating party.

<div align="center">
<img src="docs/figures/Prius_sensor_setup_5.png" alt="Prius sensor setup" width="600"/>
</div>


## Annotation

The dataset contains 3D bounding box annotations for 13 road user classes with occlusion, activity, information, along with a track id to follow objects across frames. For more details, please refer to the [Annotation](docs/ANNOTATION.md) documentation.

## Getting Started

Please refer to the [GETTING_STARTED](docs/GETTING_STARTED.md) manual to learn more usage about this project.

## Examples and Demos

Please refer to this [EXAMPLES](docs/EXAMPLES.md) manual for several examples of how to use the dataset and the development kit, including data loading, fething and applying transformations, and 2D/3D visualization.

## License

* TODO the development kit is realeased under the TBD license.
* The dataset can be used by accepting the [Research Use License](https://intelligent-vehicles.org/datasets/view-of-delft/view-of-delft-dataset-research-use-license/).

## Acknowledgement
* Annotation was done by [understand.ai](https://understand.ai), a subsidiary of DSpace.
* This work was supported by the Dutch Science Foundation NWO-TTW, within the SafeVRU project (nr. 14667).
* During our experiments, we used the OpenPCDET library both for training, and for evaluation purposes.

## Citation 
If you find the dataset useful in your research, please consider citing it as:

```
@ARTICLE{apalffy2022,
  author={Palffy, Andras and Pool, Ewoud and Baratam, Srimannarayana and Kooij, Julian F. P. and Gavrila, Dariu M.},
  journal={IEEE Robotics and Automation Letters}, 
  title={Multi-Class Road User Detection With 3+1D Radar in the View-of-Delft Dataset}, 
  year={2022},
  volume={7},
  number={2},
  pages={4961-4968},
  doi={10.1109/LRA.2022.3147324}}
```


## Links
- [Original paper](https://ieeexplore.ieee.org/document/9699098)
- [The paper on Rearch Gate](https://www.researchgate.net/publication/358328092)
- [Demo video of the dataset](https://youtu.be/R8r3lrkicJ0)
- [Visit our website](https://intelligent-vehicles.org/)
- [Reach Use Licence](https://intelligent-vehicles.org/datasets/view-of-delft/view-of-delft-dataset-research-use-license/)


