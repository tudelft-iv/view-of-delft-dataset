# Sensors and Data

<div align="center">
<img src="figures/Prius_sensor_setup_5.png" alt="Prius sensor setup" width="500"/>
</div>

<br>

---

## Camera
The camera provides colored, rectified images of 1936 × 1216 pixels at around 30 Hz.   
The horizontal field of view is ~64° (± 32°), vertical field of view is ~ 44° (± 22°). 

### Data format
Images are stored in jpg files.
<br>
<br>

---
## LiDAR
The LiDAR sensor is a Velodyne 64 sensor mounted on the top of the vehicle, operating at 10 Hz.  
The provided LiDAR point clouds are ego-motion compensated both for ego-motion during the scan (i.e. one full rotation of the LiDAR sensor) and ego-motion between the capture of LiDAR and camera data (i.e. overlaying camera and LiDAR data should give a consistent image).

### Data format
LiDAR point clouds are stored in bin files.  
Each bin file contains a 360° scan in a form of a Nx4 array, where N is the number of points, and 4 is the number of features:
`[x,y,z,reflectance]`
<br>
<br>

---
## Radar
The radar sensor is a ZF FRGen21 3+1D radar (∼13 Hz) mounted behind the front bumper.  
The provided radar point clouds are ego-motion compensated for ego-motion between the capture of radar and camera data (i.e. overlaying camera and radar data should give a consistent image).
We provide radar point clouds in three flavors:
- Single scan
- 3 scans (accumulation of the last 3 radar scans)
- 5 scans (accumulation of the last 5 radar scans)  

Accumulation (i.e. radar_3_scans and radar_5_scans folders) is implemented by transforming point clouds from previous scans to the coordinate system of the last scan.

### Data format
The radar point clouds are stored in bin files.  
Each bin file contains a set of points in the form of a Nx7 array, where N is the number of points, and 7 is the number of features:  

`[x, y, z, RCS, v_r, v_r_compensated, time]`

where `v_r` is the relative radial velocity, and `v_r_compensated` is the absolute (i.e. ego motion compensated) radial velocity of the point.

`time` is the time id of the point, indicating which scan it originates from. E.g., a point from the current scan has a t = 0,
while a point from the third most recent scan has a t = −2. 

<br>
<br>

---
## Odometry
Odometry is a filtered combination of several inputs: RTK GPS, IMU, and wheel odometry with a frame rate around 30 Hz. 

### Data format
We provide odometry information as transformations. For convenience, three transformations is defined for each frame:
- map to camera  (global coordinate system)
- odom to camera (local coordinate system)
- UTM to camera  (Official [UTM](https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system) coordinate system)

<br>
<br>

---
## Calibration files
We provide extrinsic calibration between the point cloud sensors (LiDAR, radar) and the camera in KITTI format.
Further transformations, e.g. LiDAR to radar, or UTM to LiDAR can be derived with our devkit through the transformations described in the calibration files and the odometry data as shown in the examples.
<br>
<br>

---
## Syncronization
Output of the sensors were recorded in an asyncronus way (i.e. no connected triggering) with accurate, synced timestamps.  
For convenience, we provide the dataset in synchronized “frames” similar to the  KITTI dataset, consisting of a: 
- a rectified mono-camera image, 
- a LiDAR point cloud,
- three radar point clouds (single-, 3, and 5 scans), 
- and a pose file describing the position of the egovehicle. 
 
Timestamps of the LiDAR sensor were chosen as lead (~10 Hz), and we chose the closest camera, radar and odometry information available (maximum tolerated time difference is set to **0.04 seconds**).
To get the best possible syncronization, we synced radar and camera data to the moment when the LiDAR sensor **scanned the middle of the camera field of view**.

Corressponding  camera, radar, LiDAR, and pose messages (i.e. content of a frame) are connected via their filenames, see the [GETTING_STARTED](GETTING_STARTED.md) manual and the [EXAMPLES](EXAMPLES.md) manual.

We also share the metadata of the syncronized messages, i.e. the original timestamp of each syncronized message in the frame.



