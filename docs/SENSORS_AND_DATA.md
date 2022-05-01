# Sensors and Data

## Camera
The camera provides colored images of 1936 × 1216 pixels at around 30 Hz. The horizontal field of view is ~64° (± 32°), vertical field of view is ~ 44° (± 22°). 

## LiDAR
The LiDAR sensor is a Velodyne 64 operating at 10 Hz. 
The provided LiDAR point-clouds are already ego-motion compensated. More specifically, misalignment of LiDAR data from ego-motion during the scan (i.e. one full rotation of the LiDAR sensor) and ego-motion between the capture of LiDAR and camera data has been compensated. 

## Radar

## Odometry
Odometry is a filtered combination of several inputs: RTK GPS, IMU, and wheel odometry with a frame rate around 30 Hz. 

## Calibration files
We also provide intrinsic calibration for the camera and extrinsic calibration of all three sensors.

## Syncronization
Output of the sensors were recorded in an asyncronus way (i.e. no connected triggering) with accurate, synced timestamps.
