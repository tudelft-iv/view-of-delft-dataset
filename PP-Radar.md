# PP-Radar on VoD

While 3+1D Radars do output a spatially three-dimensional point cloud, it is not very straightforward to train PointPillars on the radar point cloud using OpenPCDet. A few modifications are needed to adapt this library to our radar point cloud containing seven features per target - *[x, y, z, RCS, doppler, compensated_doppler, time]*.

Note: The steps listed here are only to provide a starting point to the users and **not** a manual for reproducing results.

## OpenPCDet version

All experiments presented in our paper are performed by modifying the following version of OpenPCDet:
[https://github.com/open-mmlab/OpenPCDet/tree/0642cf06d0fd84f50cc4c6c01ea28edbc72ea810](https://github.com/open-mmlab/OpenPCDet/tree/0642cf06d0fd84f50cc4c6c01ea28edbc72ea810)

We do not track their repository and hence may not be able to offer support for later versions. It is the recommended version to implement the modifications listed below.

## Config Files

Through dataset and model configuration files, we bring changes to parameters including but not limited to:

- Point Cloud Range
- Voxel Size
- Point Features
- Augmentations
- Max Point per Voxel
- VFE (Voxel Feature Encoder)

### Dataset Config

OpenPCDet first generates an info .pkl file for training and evaluation of networks. Refer to their document linked below for detailed steps.

[https://github.com/open-mmlab/OpenPCDet/docs/GETTING_STARTED.md](https://github.com/open-mmlab/OpenPCDet/blob/0642cf06d0fd84f50cc4c6c01ea28edbc72ea810/docs/GETTING_STARTED.md)

To facilitate this, create a dataset configuration file named **radar_5frames_as_kitti_dataset.yaml** under

> tools/cfgs/dataset_configs/
> 

Use the following config parameters as a starting point.

```
DATASET: 'KittiDataset'
DATA_PATH: '/view_of_delft/radar_5frames'

POINT_CLOUD_RANGE: [0, -25.6, -3, 51.2, 25.6, 2]

DATA_SPLIT: {
    'train': train,
    'test': val
}

INFO_PATH: {
    'train': [kitti_infos_train.pkl],
    'test': [kitti_infos_val.pkl],
}

FOV_POINTS_ONLY: True

DATA_AUGMENTOR:
    DISABLE_AUG_LIST: ['placeholder']
    AUG_CONFIG_LIST:
        - NAME: random_world_flip
          ALONG_AXIS_LIST: ['x']

        - NAME: random_world_scaling
          WORLD_SCALE_RANGE: [0.95, 1.05]

POINT_FEATURE_ENCODING: {
    encoding_type: absolute_coordinates_encoding,
    used_feature_list: ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'],
    src_feature_list: ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'],
}

DATA_PROCESSOR:
    - NAME: mask_points_and_boxes_outside_range
      REMOVE_OUTSIDE_BOXES: True

    - NAME: shuffle_points
      SHUFFLE_ENABLED: {
        'train': True,
        'test': False
      }

    - NAME: transform_points_to_voxels
      VOXEL_SIZE: [0.16, 0.16, 5]
      MAX_POINTS_PER_VOXEL: 10
      MAX_NUMBER_OF_VOXELS: {
        'train': 16000,
        'test': 40000
      }

```

### PointPillar Config

In order to train/evaluate/infer a PointPillar on our radar point cloud, create a model configuration file under

> tools/cfgs/kitti_models/
> 

Again, use the following setup as a starting point for your models.

```
CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']

DATA_CONFIG:
    _BASE_CONFIG_: cfgs/dataset_configs/radar_5frames_as_kitti_dataset.yaml
    POINT_CLOUD_RANGE: [0, -25.6, -3, 51.2, 25.6, 2]
    DATA_PROCESSOR:
        - NAME: mask_points_and_boxes_outside_range
          REMOVE_OUTSIDE_BOXES: True

        - NAME: shuffle_points
          SHUFFLE_ENABLED: {
            'train': True,
            'test': False
          }

        - NAME: transform_points_to_voxels
          VOXEL_SIZE: [0.16, 0.16, 5]
          MAX_POINTS_PER_VOXEL: 10
          MAX_NUMBER_OF_VOXELS: {
            'train': 16000,
            'test': 40000
          }
    DATA_AUGMENTOR:
        DISABLE_AUG_LIST: ['random_world_rotation', 'gt_sampling']
        AUG_CONFIG_LIST:

            - NAME: random_world_flip
              ALONG_AXIS_LIST: ['x']

            - NAME: random_world_scaling
              WORLD_SCALE_RANGE: [0.95, 1.05]

MODEL:
    NAME: PointPillar

    VFE:
        NAME: Radar7PillarVFE
        USE_XYZ: True
        USE_RCS: True
        USE_VR: True
        USE_VR_COMP: True
        USE_TIME: True
        USE_NORM: True
        USE_ELEVATION: True
        USE_DISTANCE: False
        NUM_FILTERS: [64]

    MAP_TO_BEV:
        NAME: PointPillarScatter
        NUM_BEV_FEATURES: 64

    BACKBONE_2D:
        NAME: BaseBEVBackbone
        LAYER_NUMS: [3, 5, 5]
        LAYER_STRIDES: [2, 2, 2]
        NUM_FILTERS: [64, 128, 256]
        UPSAMPLE_STRIDES: [1, 2, 4]
        NUM_UPSAMPLE_FILTERS: [128, 128, 128]

    DENSE_HEAD:
        NAME: AnchorHeadSingle
        CLASS_AGNOSTIC: False

        USE_DIRECTION_CLASSIFIER: True
        DIR_OFFSET: 0.78539
        DIR_LIMIT_OFFSET: 0.0
        NUM_DIR_BINS: 2

        ANCHOR_GENERATOR_CONFIG: [
            {
                'class_name': 'Car',
                'anchor_sizes': [[3.9, 1.6, 1.56]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.78],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.45
            },
            {
                'class_name': 'Pedestrian',
                'anchor_sizes': [[0.8, 0.6, 1.73]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            },
            {
                'class_name': 'Cyclist',
                'anchor_sizes': [[1.76, 0.6, 1.73]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            }
        ]

        TARGET_ASSIGNER_CONFIG:
            NAME: AxisAlignedTargetAssigner
            POS_FRACTION: -1.0
            SAMPLE_SIZE: 512
            NORM_BY_NUM_EXAMPLES: False
            MATCH_HEIGHT: False
            BOX_CODER: ResidualCoder

        LOSS_CONFIG:
            LOSS_WEIGHTS: {
                'cls_weight': 1.0,
                'loc_weight': 2.0,
                'dir_weight': 0.2,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }

    POST_PROCESSING:
        RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
        SCORE_THRESH: 0.1
        OUTPUT_RAW_SCORE: False

        EVAL_METRIC: kitti

        NMS_CONFIG:
            MULTI_CLASSES_NMS: False
            NMS_TYPE: nms_gpu
            NMS_THRESH: 0.01
            NMS_PRE_MAXSIZE: 4096
            NMS_POST_MAXSIZE: 500

OPTIMIZATION:
    BATCH_SIZE_PER_GPU: 16
    NUM_EPOCHS: 80

    OPTIMIZER: adam_onecycle
    LR: 0.003
    WEIGHT_DECAY: 0.01
    MOMENTUM: 0.9

    MOMS: [0.95, 0.85]
    PCT_START: 0.4
    DIV_FACTOR: 10
    DECAY_STEP_LIST: [35, 45]
    LR_DECAY: 0.1
    LR_CLIP: 0.0000001

    LR_WARMUP: False
    WARMUP_EPOCH: 1

    GRAD_NORM_CLIP: 10

```

## RadarPillarVFE

To utilize the radar features like RCS and Doppler which are not available in LiDAR point cloud, we need to modify the PillarVFE in the following file:

> /pcdet/models/backbones_3d/vfe/pillar_vfe.py
> 

Create a new PillarVFE class like below.

```
class Radar7PillarVFE(VFETemplate):
def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
    super().__init__(model_cfg=model_cfg)

    num_point_features = 0
    self.use_norm = self.model_cfg.USE_NORM  # whether to use batchnorm in the PFNLayer
    self.use_xyz = self.model_cfg.USE_XYZ
    self.with_distance = self.model_cfg.USE_DISTANCE
    self.selected_indexes = []

    ## check if config has the correct params, if not, throw exception
    radar_config_params = ["USE_RCS", "USE_VR", "USE_VR_COMP", "USE_TIME", "USE_ELEVATION"]

    if all(hasattr(self.model_cfg, attr) for attr in radar_config_params):
        self.use_RCS = self.model_cfg.USE_RCS
        self.use_vr = self.model_cfg.USE_VR
        self.use_vr_comp = self.model_cfg.USE_VR_COMP
        self.use_time = self.model_cfg.USE_TIME
        self.use_elevation = self.model_cfg.USE_ELEVATION

    else:
        raise Exception("config does not have the right parameters, please use a radar config")

    self.available_features = ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']

    num_point_features += 6  # center_x, center_y, center_z, mean_x, mean_y, mean_z, time, we need 6 new

    self.x_ind = self.available_features.index('x')
    self.y_ind = self.available_features.index('y')
    self.z_ind = self.available_features.index('z')
    self.rcs_ind = self.available_features.index('rcs')
    self.vr_ind = self.available_features.index('v_r')
    self.vr_comp_ind = self.available_features.index('v_r_comp')
    self.time_ind = self.available_features.index('time')

    if self.use_xyz:  # if x y z coordinates are used, add 3 channels and save the indexes
        num_point_features += 3  # x, y, z
        self.selected_indexes.extend((self.x_ind, self.y_ind, self.z_ind))  # adding x y z channels to the indexes

    if self.use_RCS:  # add 1 if RCS is used and save the indexes
        num_point_features += 1
        self.selected_indexes.append(self.rcs_ind)  # adding  RCS channels to the indexes

    if self.use_vr:  # add 1 if vr is used and save the indexes. Note, we use compensated vr!
        num_point_features += 1
        self.selected_indexes.append(self.vr_ind)  # adding  v_r_comp channels to the indexes

    if self.use_vr_comp:  # add 1 if vr is used (as proxy for sensor cue) and save the indexes
        num_point_features += 1
        self.selected_indexes.append(self.vr_comp_ind)

    if self.use_time:  # add 1 if time is used and save the indexes
        num_point_features += 1
        self.selected_indexes.append(self.time_ind)  # adding  time channel to the indexes

    ### LOGGING USED FEATURES ###
    print("number of point features used: " + str(num_point_features))
    print("6 of these are 2 * (x y z)  coordinates realtive to mean and center of pillars")
    print(str(len(self.selected_indexes)) + " are selected original features: ")

    for k in self.selected_indexes:
        print(str(k) + ": " + self.available_features[k])

    self.selected_indexes = torch.LongTensor(self.selected_indexes)  # turning used indexes into Tensor

    self.num_filters = self.model_cfg.NUM_FILTERS
    assert len(self.num_filters) > 0
    num_filters = [num_point_features] + list(self.num_filters)

    pfn_layers = []
    for i in range(len(num_filters) - 1):
        in_filters = num_filters[i]
        out_filters = num_filters[i + 1]
        pfn_layers.append(
            PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
        )
    self.pfn_layers = nn.ModuleList(pfn_layers)

    ## saving size of the voxel
    self.voxel_x = voxel_size[0]
    self.voxel_y = voxel_size[1]
    self.voxel_z = voxel_size[2]

    ## saving offsets, start of point cloud in x, y, z + half a voxel, e.g. in y it starts around -39 m
    self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
    self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
    self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

def get_output_feature_dim(self):
    return self.num_filters[-1]  # number of outputs in last output channel

def get_paddings_indicator(self, actual_num, max_num, axis=0):
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator

def forward(self, batch_dict, **kwargs):
    ## coordinate system notes
    # x is pointing forward, y is left right, z is up down
    # spconv returns voxel_coords as  [batch_idx, z_idx, y_idx, x_idx], that is why coords is indexed backwards

    voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
        'voxel_coords']

    if not self.use_elevation:  # if we ignore elevation (z) and v_z
        voxel_features[:, :, self.z_ind] = 0  # set z to zero before doing anything

    orig_xyz = voxel_features[:, :, :self.z_ind + 1]  # selecting x y z

    # calculate mean of points in pillars for x y z and save the offset from the mean
    # Note: they do not take the mean directly, as each pillar is filled up with 0-s. Instead, they sum and divide by num of points
    points_mean = orig_xyz.sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
    f_cluster = orig_xyz - points_mean  # offset from cluster mean

    # calculate center for each pillar and save points' offset from the center. voxel_coordinate * voxel size + offset should be the center of pillar (coords are indexed backwards)
    f_center = torch.zeros_like(orig_xyz)
    f_center[:, :, 0] = voxel_features[:, :, self.x_ind] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
    f_center[:, :, 1] = voxel_features[:, :, self.y_ind] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
    f_center[:, :, 2] = voxel_features[:, :, self.z_ind] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

    voxel_features = voxel_features[:, :, self.selected_indexes]  # filtering for used features

    features = [voxel_features, f_cluster, f_center]

    if self.with_distance:  # if with_distance is true, include range to the points as well
        points_dist = torch.norm(orig_xyz, 2, 2, keepdim=True)  # first 2: L2 norm second 2: along 2. dim
        features.append(points_dist)

    ## finishing up the feature extraction with correct shape and masking
    features = torch.cat(features, dim=-1)

    voxel_count = features.shape[1]
    mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
    mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
    features *= mask

    for pfn in self.pfn_layers:
        features = pfn(features)
    features = features.squeeze()
    batch_dict['pillar_features'] = features
    return batch_dict

```

Further, ensure that this class is added in the following file.

> /pcdet/models/backbones_3d/vfe/__init__.py
> 

It should then look something like this.

```
from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE, Radar7PillarVFE
from .vfe_template import VFETemplate
__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'Radar7PillarVFE': Radar7PillarVFE,
}

```

## kitti_dataset.py

While loading the point cloud from .bin files for radar, the get function needs to be modified at:

> pcdet/datasets/kitti/kitti_dataset.py#L62
> 

```
number_of_channels = 7  # ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']
points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, number_of_channels)

# replace the list values with statistical values; for x, y, z and time, use 0 and 1 as means and std to avoid normalization
means = [0, 0, 0, 0, 0, 0, 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
stds =  [1, 1, 1, 1, 1, 1, 1]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'

#we then norm the channels
points = (points - means)/stds

```
