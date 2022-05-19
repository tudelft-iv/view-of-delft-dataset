import json
import os
import logging
from typing import Optional

import numpy as np

from .data_loader import FrameDataLoader


class FrameTransformMatrix:
    """
    This class is responsible for providing the possible homogenous transform matrices between the possible
     coordinate frames.
    """

    def __init__(self, frame_data_loader_object: FrameDataLoader):
        """
Constructor which creates the backing fields for the properties in the class
        :param frame_data_loader_object: FrameDataLoader object, for a specific frame which requires the transformation.
        matrices.
        """
        self.frame_data_loader: FrameDataLoader = frame_data_loader_object

        # Local transformations
        self._camera_projection_matrix: Optional[np.ndarray] = None
        self._T_camera_lidar: Optional[np.ndarray] = None
        self._T_camera_radar: Optional[np.ndarray] = None

        self._T_lidar_camera: Optional[np.ndarray] = None
        self._T_radar_camera: Optional[np.ndarray] = None
        self._T_lidar_radar: Optional[np.ndarray] = None
        self._T_radar_lidar: Optional[np.ndarray] = None

        # World transformations
        self._T_odom_camera: Optional[np.ndarray] = None
        self._T_map_camera: Optional[np.ndarray] = None
        self._T_UTM_camera: Optional[np.ndarray] = None

        self._T_camera_odom: Optional[np.ndarray] = None
        self._T_camera_map: Optional[np.ndarray] = None
        self._T_camera_UTM: Optional[np.ndarray] = None

    @property
    def camera_projection_matrix(self):
        """
Property which gets the camera projection matrix.
        :return: Numpy array of the camera projection matrix.
        """
        if self._camera_projection_matrix is not None:
            # When the data is already loaded.
            return self._camera_projection_matrix
        else:
            # Load data if it is not loaded yet.
            self._camera_projection_matrix, self._T_camera_lidar = self.get_sensor_transforms('lidar')
            return self._camera_projection_matrix

    @property
    def t_camera_lidar(self):
        """
Property which returns the homogeneous transform matrix from the lidar frame, to the camera frame.
        :return: Numpy array of the homogeneous transform matrix from the lidar frame, to the camera frame.
        """
        if self._T_camera_lidar is not None:
            # When the data is already loaded.
            return self._T_camera_lidar
        else:
            # Load data if it is not loaded yet.
            self._camera_projection_matrix, self._T_camera_lidar = self.get_sensor_transforms('lidar')
            return self._T_camera_lidar

    @property
    def t_camera_radar(self):
        """
Property which returns the homogeneous transform matrix from the radar frame, to the camera frame.
        :return: Numpy array of the homogeneous transform matrix from the radar frame, to the camera frame.
        """
        if self._T_camera_radar is not None:
            # When the data is already loaded.
            return self._T_camera_radar
        else:
            # Load data if it is not loaded yet.
            self._camera_projection_matrix, self._T_camera_radar = self.get_sensor_transforms('radar')
            return self._T_camera_radar

    @property
    def t_lidar_camera(self):
        """
Property which returns the homogeneous transform matrix from the camera frame, to the lidar frame.
        :return: Numpy array of the homogeneous transform matrix from the camera frame, to the lidar frame.
        """
        if self._T_lidar_camera is not None:
            # When the data is already loaded.
            return self._T_lidar_camera
        else:
            # Calculate data if it is calculated yet.
            self._T_lidar_camera = np.linalg.inv(self.t_camera_lidar)
            return self._T_lidar_camera

    @property
    def t_radar_camera(self):
        """
Property which returns the homogeneous transform matrix from the camera frame, to the radar frame.
        :return: Numpy array of the homogeneous transform matrix from the camera frame, to the radar frame.
        """
        if self._T_radar_camera is not None:
            # When the data is already loaded.
            return self._T_radar_camera
        else:
            # Calculate data if it is calculated yet.
            self._T_radar_camera = np.linalg.inv(self.t_camera_radar)
            return self._T_radar_camera

    @property
    def t_lidar_radar(self):
        """
Property which returns the homogeneous transform matrix from the radar frame, to the lidar frame.
        :return: Numpy array of the homogeneous transform matrix from the radar frame, to the lidar frame.
        """
        if self._T_lidar_radar is not None:
            # When the data is already loaded.
            return self._T_lidar_radar
        else:
            # Calculate data if it is calculated yet.
            self._T_lidar_radar = np.dot(self.t_lidar_camera, self.t_camera_radar)
            return self._T_lidar_radar

    @property
    def t_radar_lidar(self):
        """
Property which returns the homogeneous transform matrix from the lidar frame, to the radar frame.
        :return: Numpy array of the homogeneous transform matrix from the lidar frame, to the radar frame.
        """
        if self._T_radar_lidar is not None:
            # When the data is already loaded.
            return self._T_radar_lidar
        else:
            # Calculate data if it is calculated yet.
            self._T_radar_lidar = np.dot(self.t_radar_camera, self.t_camera_lidar)
            return self._T_radar_lidar

    @property
    def t_odom_camera(self):
        """
Property which returns the homogeneous transform matrix from the camera frame, to the odom frame.
        :return: Numpy array of the homogeneous transform matrix from the camera frame, to the odom frame.
        """
        if self._T_odom_camera is not None:
            # When the data is already loaded.
            return self._T_odom_camera
        else:
            # Load data if it is not loaded yet.
            self._T_odom_camera, self._T_map_camera, self._T_UTM_camera = self.get_world_transform()
            return self._T_odom_camera

    @property
    def t_map_camera(self):
        """
Property which returns the homogeneous transform matrix from the camera frame, to the map frame.
        :return: Numpy array of the homogeneous transform matrix from the camera frame, to the map frame.
        """
        if self._T_map_camera is not None:
            # When the data is already loaded.
            return self._T_map_camera
        else:
            # Load data if it is not loaded yet.
            self._T_odom_camera, self._T_map_camera, self._T_UTM_camera = self.get_world_transform()
            return self._T_map_camera

    @property
    def t_utm_camera(self):
        """
Property which returns the homogeneous transform matrix from the camera frame, to the UTM frame.
        :return: Numpy array of the homogeneous transform matrix from the camera frame, to the UTM frame.
        """
        if self._T_UTM_camera is not None:
            # When the data is already loaded.
            return self._T_UTM_camera
        else:
            # Load data if it is not loaded yet.
            self._T_odom_camera, self._T_map_camera, self._T_UTM_camera = self.get_world_transform()
            return self._T_UTM_camera

    @property
    def t_camera_odom(self):
        """
Property which returns the homogeneous transform matrix from the odom frame, to the camera frame.
        :return: Numpy array of the homogeneous transform matrix from the odom frame, to the camera frame.
        """
        if self._T_camera_odom is not None:
            # When the data is already loaded.
            return self._T_camera_odom
        else:
            # Calculate data if it is calculated yet.
            self._T_camera_odom = np.linalg.inv(self.t_odom_camera)
            return self._T_camera_odom

    @property
    def t_camera_map(self):
        """
Property which returns the homogeneous transform matrix from the map frame, to the camera frame.
        :return: Numpy array of the homogeneous transform matrix from the map frame, to the camera frame.
        """
        if self._T_camera_map is not None:
            # When the data is already loaded.
            return self._T_camera_map
        else:
            # Calculate data if it is calculated yet.
            self._T_camera_map = np.linalg.inv(self.t_map_camera)
            return self._T_camera_map

    @property
    def t_camera_utm(self):
        """
Property which returns the homogeneous transform matrix from the UTM frame, to the camera frame.
        :return: Numpy array of the homogeneous transform matrix from the UTM frame, to the camera frame.
        """
        if self._T_camera_UTM is not None:
            # When the data is already loaded.
            return self._T_camera_UTM
        else:
            # Calculate data if it is calculated yet.
            self._T_camera_UTM = np.linalg.inv(self.t_utm_camera)
            return self._T_camera_UTM

    def get_sensor_transforms(self, sensor: str):  # -> Optional[(np.ndarray, np.ndarray)]:
        """
This method returns the corresponding intrinsic and extrinsic transformation from the dataset.
        :param sensor: Sensor name in string for which the transforms to be read from the dataset.
        :return: A numpy array tuple of the intrinsic, extrinsic transform matrix.
        """
        if sensor == 'radar':
            try:
                calibration_file = os.path.join(self.frame_data_loader.kitti_locations.radar_calib_dir,
                                                f'{self.frame_data_loader.frame_number}.txt')
            except FileNotFoundError:
                logging.error(f"{self.frame_data_loader.frame_number}.txt does not exist at"
                              f" location: {self.frame_data_loader.kitti_locations.radar_calib_dir}!")
                return None, None

        elif sensor == 'lidar':
            try:
                calibration_file = os.path.join(self.frame_data_loader.kitti_locations.lidar_calib_dir,
                                                f'{self.frame_data_loader.frame_number}.txt')
            except FileNotFoundError:
                logging.error(f"{self.frame_data_loader.frame_number}.txt does not exist at"
                              f" location: {self.frame_data_loader.kitti_locations.lidar_calib_dir}!")
                return None, None
        else:
            raise AttributeError('Not valid sensor')

        with open(calibration_file, "r") as f:
            lines = f.readlines()
            intrinsic = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Intrinsics
            extrinsic = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Extrinsic
            extrinsic = np.concatenate([extrinsic, [[0, 0, 0, 1]]], axis=0)

        return intrinsic, extrinsic

    def get_world_transform(self):  # -> Optional[(np.ndarray, np.ndarray, np.ndarray)]:
        """
This method returns the world transformations matrices from the dataset.
        :return: A numpy array tuple of the t_odom_camera, t_map_camera and t_utm_camera matrices.
        """
        try:
            pose_file = os.path.join(self.frame_data_loader.kitti_locations.pose_dir,
                                     f'{self.frame_data_loader.frame_number}.json')
        except FileNotFoundError:
            logging.error(f"{self.frame_data_loader.frame_number}.json does not exist at"
                          f" location: {self.frame_data_loader.kitti_locations.pose_dir}!")
            return None, None, None

        jsons = []
        for line in open(pose_file, 'r'):
            jsons.append(json.loads(line))

        t_odom_camera = np.array(jsons[0]["odomToCamera"], dtype=np.float32).reshape(4, 4)
        t_map_camera = np.array(jsons[1]["mapToCamera"], dtype=np.float32).reshape(4, 4)
        t_utm_camera = np.array(jsons[2]["UTMToCamera"], dtype=np.float32).reshape(4, 4)

        return t_odom_camera, t_map_camera, t_utm_camera


def homogeneous_transformation(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
This function applies the homogenous transform using the dot product.
    :param points: Points to be transformed in a Nx4 numpy array.
    :param transform: 4x4 transformation matrix in a numpy array.
    :return: Transformed points of shape Nx4 in a numpy array.
    """
    if transform.shape != (4, 4):
        raise ValueError(f"{transform.shape} must be 4x4!")
    if points.shape[1] != 4:
        raise ValueError(f"{points.shape[1]} must be Nx4!")
    return transform.dot(points.T).T


def homogeneous_coordinates(points: np.ndarray) -> np.ndarray:
    """
This function returns the given point array in homogenous coordinates.
    :param points: Input ndarray of shape Nx3.
    :return: Output ndarray of shape Nx4.
    """
    if points.shape[1] != 3:
        raise ValueError(f"{points.shape[1]} must be Nx3!")

    return np.hstack((points,
                      np.ones((points.shape[0], 1),
                              dtype=np.float32)))


def project_3d_to_2d(points: np.ndarray, projection_matrix: np.ndarray):
    """
This function projects the input 3d ndarray to a 2d ndarray, given a projection matrix.
    :param points: Homogenous points to be projected.
    :param projection_matrix: 4x4 projection matrix.
    :return: 2d ndarray of the projected points.
    """
    if points.shape[-1] != 4:
        raise ValueError(f"{points.shape[-1]} must be 4!")

    uvw = projection_matrix.dot(points.T)
    uvw /= uvw[2]
    uvs = uvw[:2].T
    uvs = np.round(uvs).astype(np.int)

    return uvs


def canvas_crop(points, image_size, points_depth=None):
    """
This function filters points that lie outside a given frame size.
    :param points: Input points to be filtered.
    :param image_size: Size of the frame.
    :param points_depth: Filters also depths smaller than 0.
    :return: Filtered points.
    """
    idx = points[:, 0] > 0
    idx = np.logical_and(idx, points[:, 0] < image_size[1])
    idx = np.logical_and(idx, points[:, 1] > 0)
    idx = np.logical_and(idx, points[:, 1] < image_size[0])
    if points_depth is not None:
        idx = np.logical_and(idx, points_depth > 0)

    return idx


def min_max_filter(points, max_value, min_value):
    """
This function can be used to filter points based on a maximum and minimum value.
    :param points: Points to be filtered.
    :param max_value: Maximum value.
    :param min_value: Minimum value.
    :return: Filtered points.
    """
    idx = points < max_value
    idx = np.logical_and(idx, points > min_value)
    return idx


def project_pcl_to_image(point_cloud, t_camera_pcl, camera_projection_matrix, image_shape):
    """
A helper function which projects a point clouds specific to the dataset to the camera image frame.
    :param point_cloud: Point cloud to be projected.
    :param t_camera_pcl: Transformation from the pcl frame to the camera frame.
    :param camera_projection_matrix: The 4x4 camera projection matrix.
    :param image_shape: Size of the camera image.
    :return: Projected points, and the depth of each point.
    """
    point_homo = np.hstack((point_cloud[:, :3],
                            np.ones((point_cloud.shape[0], 1),
                                    dtype=np.float32)))

    radar_points_camera_frame = homogeneous_transformation(point_homo,
                                                           transform=t_camera_pcl)

    point_depth = radar_points_camera_frame[:, 2]

    uvs = project_3d_to_2d(points=radar_points_camera_frame,
                           projection_matrix=camera_projection_matrix)

    filtered_idx = canvas_crop(points=uvs,
                               image_size=image_shape,
                               points_depth=point_depth)
    uvs = uvs[filtered_idx]
    point_depth = point_depth[filtered_idx]

    return uvs, point_depth


def transform_pcl(points: np.ndarray, transform_matrix: np.ndarray):
    """
This function transforms homogenous points using a transformation matrix.
    :param points: Points to be transformed.
    :param transform_matrix: Homogenous transformation matrix.
    :return: Transformed homogenous points.
    """
    point_homo = np.hstack((points[:, :3],
                            np.ones((points.shape[0], 1),
                                    dtype=np.float32)))

    points_new_frame = homogeneous_transformation(point_homo, transform=transform_matrix)

    return points_new_frame
