## Annotation information

# Type of Annotations
We provide 3D bounding boxes with 9 degree of freedom (DoF):
* x, y, z: 3D location of the box's center 
* l, w, h: length, width, height of the box
* yaw, pitch, roll: orientation of the box

For each object, we also annotated the level of occlusion for two types of occlusions (“spatial” and “lighting”) and an activity attribute (“stopped,” “moving,” “parked,” “pushed,” “sitting”). 
Furthermore, same physical objects were assigned unique object ids over frames to make the dataset suitable for tracking and prediction tasks. 
Annotation instructions with detailed descriptions of the classes and attributes will be shared along with the dataset.

For now, we will provide the annotation in KITTI format.

# KITTI formatted labels
[Example label definition here](https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt).
'''
'Car' 
'''


# Annotated area
Any object of interest within 50 meters of the LiDAR sensor and partially or fully within the camera’s field of view (horizontal FoV: ±32°, vertical FoV: ± 22°). 
was annotated with a six degree of freedom ( DoF) 3D bounding box. 

13 object classes were annotated:
* TODO
