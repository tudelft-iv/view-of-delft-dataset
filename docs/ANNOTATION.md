## Annotation information

# Type of Annotations
We annotated the data with 3D bounding boxes with 9 degree of freedom (DoF):
* x, y, z: 3D location of the box's center 
* l, w, h: length, width, height of the box
* yaw, pitch, roll: orientation of the box

For each object, we also annotated the level of occlusion for two types of occlusions (“spatial” and “lighting”) and an activity attribute (“stopped,” “moving,” “parked,” “pushed,” “sitting”). 
Furthermore, same physical objects were assigned unique object ids over frames to make the dataset suitable for tracking and prediction tasks. 


# KITTI formatted labels
For now, we will provide the annotation in [KITTI format](https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt), in the camera frame.

Note that there are 3 important differences between VoD and KITTI labels:
- **2D bounding boxes are automatically calculated**: Annotation was done in 3D on the LiDAR point cloud. While we provide the 2D bounding boxes in the KITTI formatted labels, these were calculated automatically by projecting the 3D bounding boxes to the camera plane, and assigning a minimum fit rectangle.
- **Truncation**: We do not provide truncation information (it is used for other meta data) for the same reason (no annotation in image plane). **Important:** please be sure not to use Truncation values in your evaluation, if you do not use the provided eval module, see [this issue](https://github.com/tudelft-iv/view-of-delft-dataset/issues/8).
- **Rotation**: the original KITTI devkit assumes that camera's and LiDAR's vertical axes (Y and Z) are parallel, just pointing to different directions. In our research vehicle however, the camera is slightly tilted. Thus for convenience, we define the rotation of objects around the LiDAR’s negative vertical ( -Z) axis. This is in fact what many open source library assumes anyway: that the LiDAR’s and camera’s vertical axes (Z and Y) are perfectly aligned.

```
#Values    Name      Description
----------------------------------------------------------------------------
   1    Class        Describes the type of object: 'Car', 'Pedestrian', 'Cyclist', etc.
   1    truncated    Not used, only there to be compatible with KITTI format.
   1    occluded     Integer (0,1,2) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded.
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates. This was automatically calculated from the 3D boxes.
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation     Rotation around -Z axis of the LiDAR sensor [-pi..pi]
```

# Tracking IDs

The labels in the original release do not include track ids. 
If you are interested in the track ids, the zip file below has to be downloaded, and its content placed in the relevant location in: 
```<your root of view_of_delft>/lidar/training/label_2```, overwriting the original labels. There is no other difference between the two sets of labels, all boxes are identical.  

Annotations with tracking IDs can be downloaded with [this link](https://surfdrive.surf.nl/files/index.php/s/CjKsQGz69uZKkNL).

Using your password received in email after registration.  

We share the tracking IDs by overriding the standard's KITTI format's truncation value, see above, i.e. the first number after the class string.

For example the following line in the annotations:

```bicycle 1757 1 -0.5150583918601345 1692.8588 873.00977 1935.0 1064.7266 0.9959256326426174 0.4582897348611458 1.737482152677817 5.230204792744421 2.477337074657124 8.676091008791296 0.027439126666472635 1```

means that this annotated bicycle is the 1757th object in the dataset, and this number will be its tracking id. This number is going to be consistent along frames, i.e., if the bicycle is visible later, it will have the same number printed at this location.  

# Annotated area
Any object of interest within 50 meters of the LiDAR sensor and partially or fully within the camera’s field of view (horizontal FoV: ±32°, vertical FoV: ± 22°). 
was annotated.

# Annotated classes
13 object classes were annotated:
- Car
- Pedestrian
- Cyclist (including both the bycicle and the rider)
- Rider (the human on the bycicle, motor, etc. separately)
- Unused bicycle
- Bicycle rack
- Human depiction (e.g. statues)
- Moped or scooter
- Motor
- Truck
- Other ride
- Other vehicle
- Uncertain ride





Annotation instructions will be available here soon (TODO).
