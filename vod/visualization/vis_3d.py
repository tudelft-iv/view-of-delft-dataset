import k3d
import numpy as np
from .helpers import k3d_get_axes, get_transformed_3d_label_corners, k3d_plot_box, \
    get_radar_velocity_vectors, get_default_camera
from vod.frame import FrameDataLoader, FrameTransformMatrix, FrameLabels, transform_pcl
from .settings import *


class Visualization3D:
    """
    This class is responsible for creating 3D plots inside Jupyter notebooks using the k3d library.
    """

    def __init__(self, frame_data: FrameDataLoader, origin='camera'):
        """
Constructor which prepared the 3D plot in the requested frame.
        :param frame_data:
        """
        self.plot = None
        self.frame_data = frame_data
        self.frame_transforms = FrameTransformMatrix(self.frame_data)

        self.origin = origin

        if self.origin == 'camera':
            self.transform_matrices = {
                'camera': np.eye(4, dtype=float),
                'lidar': self.frame_transforms.t_camera_lidar,
                'radar': self.frame_transforms.t_camera_radar
            }
        elif self.origin == 'lidar':
            self.transform_matrices = {
                'camera': self.frame_transforms.t_lidar_camera,
                'lidar': np.eye(4, dtype=float),
                'radar': self.frame_transforms.t_lidar_radar
            }
        elif self.origin == 'radar':
            self.transform_matrices = {
                'camera': self.frame_transforms.t_radar_camera,
                'lidar': self.frame_transforms.t_radar_lidar,
                'radar': np.eye(4, dtype=float)
            }
        else:
            raise ValueError("Origin must be camera, lidar or radar!")

    def __call__(self, radar_origin_plot: bool = False,
                 lidar_origin_plot: bool = False,
                 camera_origin_plot: bool = False,
                 lidar_points_plot: bool = False,
                 radar_points_plot: bool = False,
                 radar_velocity_plot: bool = False,
                 plot_annotations: bool = False):

        self.draw_plot(radar_origin_plot,
                       lidar_origin_plot,
                       camera_origin_plot,
                       lidar_points_plot,
                       radar_points_plot,
                       radar_velocity_plot,
                       plot_annotations)

    def plot_radar_origin(self,
                          label: bool = True,
                          color: int = radar_plot_color_3d,
                          axis_length: float = axis_length_3d,
                          label_size: float = axis_label_size):
        """
This method plots the radar origin in the requested frame.
        :param axis_length: Vector length of the axis.
        :param label: Bool which sets if the label should be displayed.
        :param color: Color of the label in int.
        :param label_size: Size of the label.
        """
        self.plot += k3d_get_axes(self.transform_matrices['radar'], axis_length)

        if label:
            self.plot += k3d.text("radar",
                                  position=self.transform_matrices['radar'][0:3, 3],
                                  color=color,
                                  size=label_size)

    def plot_lidar_origin(self,
                          label: bool = True,
                          color: int = lidar_plot_color_3d,
                          axis_length: float = axis_length_3d,
                          label_size: float = axis_label_size):
        """
This method plots the lidar origin in the requested frame.
        :param axis_length: Vector length of the axis.
        :param label: Bool which sets if the label should be displayed.
        :param color: Color of the label in int.
        :param label_size: Size of the label.
        """
        self.plot += k3d_get_axes(self.transform_matrices['lidar'], axis_length)

        if label:
            self.plot += k3d.text("lidar",
                                  position=self.transform_matrices['lidar'][0:3, 3],
                                  color=color,
                                  size=label_size)

    def plot_camera_origin(self,
                           label: bool = True,
                           color: int = lidar_plot_color_3d,
                           axis_length: float = axis_length_3d,
                           label_size: float = axis_label_size):
        """
This method plots the camera origin in the requested frame.
        :param axis_length: Vector length of the axis.
        :param label: Bool which sets if the label should be displayed.
        :param color: Color of the label in int.
        :param label_size: Size of the label.
        """
        self.plot += k3d_get_axes(self.transform_matrices['camera'], axis_length)

        if label:
            self.plot += k3d.text("camera",
                                  position=self.transform_matrices['camera'][0:3, 3],
                                  color=color,
                                  size=label_size)

    def plot_lidar_points(self,
                          pcl_size: float = lidar_pcl_size,
                          color: int = lidar_plot_color_3d):
        """
This method plots the lidar pcl on the requested frame.
        :param pcl_size: Size of the pcl particles in the graph.
        :param color: Color of the pcl particles in the graph.
        """
        lidar_points_camera_frame = transform_pcl(points=self.frame_data.lidar_data,
                                                  transform_matrix=self.transform_matrices['lidar'])

        self.plot += k3d.points(positions=np.asarray(lidar_points_camera_frame[:, :3], dtype=float),
                                point_size=pcl_size,
                                color=color)

    def plot_radar_points(self,
                          pcl_size: float = radar_pcl_size,
                          color: int = radar_plot_color_3d
                          ):
        """
This method plots the radar pcl on the requested frame.
        :param pcl_size: Size of the pcl particles in the graph.
        :param color: Color of the pcl particles in the graph.
        """
        radar_points_camera_frame = transform_pcl(points=self.frame_data.radar_data,
                                                  transform_matrix=self.transform_matrices['radar'])

        self.plot += k3d.points(positions=np.asarray(radar_points_camera_frame[:, :3], dtype=float),
                                point_size=pcl_size,
                                color=color)

    def plot_radar_radial_velocity(self, color: int = radar_velocity_color_3d):
        """
This method plots the radar radial velocity vectors for each radar point in the requested frame.
        :param color: Color of the vector.
        """
        compensated_radial_velocity = self.frame_data.radar_data[:, 5]
        radar_points_camera_frame = transform_pcl(points=self.frame_data.radar_data,
                                                  transform_matrix=self.transform_matrices['radar'])

        pc_radar = radar_points_camera_frame[:, 0:3]

        velocity_vectors = get_radar_velocity_vectors(pc_radar, compensated_radial_velocity)

        self.plot += k3d.vectors(origins=pc_radar, vectors=velocity_vectors, color=color)

    def plot_annotations(self, class_colors=label_color_palette_3d, class_width=label_line_width_3d):
        """
This method plots the annotations in the requested frame.
        :param class_colors: Dictionary that contains the colors for the annotations.
        :param class_width: Dictionary that contains the line width for the annotations.
        """
        labels: FrameLabels = FrameLabels(self.frame_data.raw_labels)

        bboxes = get_transformed_3d_label_corners(labels,
                                                  self.transform_matrices['lidar'],
                                                  self.frame_transforms.t_camera_lidar)

        for box in bboxes:
            object_class = box['label_class']

            object_class_color = class_colors[object_class]
            object_class_width = class_width[object_class]

            corners_object = box['corners_3d_transformed']

            k3d_plot_box(self.plot, corners_object, object_class_color, object_class_width)

    def draw_plot(self,
                  radar_origin_plot: bool = False,
                  lidar_origin_plot: bool = False,
                  camera_origin_plot: bool = False,
                  lidar_points_plot: bool = False,
                  radar_points_plot: bool = False,
                  radar_velocity_plot: bool = False,
                  annotations_plot: bool = False,
                  write_to_html: bool = False,
                  html_name: str = "example",
                  grid_visible: bool = False,
                  auto_frame: bool = False,
                  ):
        """
This method displays the plot with the specified arguments.
        :param auto_frame: When set to true, the frame is size automatically.
        :param grid_visible: Plot grid background.
        :param radar_origin_plot: Plots the radar origin axis.
        :param lidar_origin_plot: Plots the lidar origin axis.
        :param camera_origin_plot: Plots the camera origin axis.
        :param lidar_points_plot: Plots the lidar PCL.
        :param radar_points_plot: Plots the radar PCL.
        :param radar_velocity_plot: Plots the radar velocity vectors.
        :param annotations_plot: Plots the annotations.
        :param write_to_html: Allows the plot to be written to html.
        :param html_name: Name of the html file if written to disk.
        """

        self.plot = k3d.plot(camera_auto_fit=auto_frame, axes_helper=0.0, grid_visible=grid_visible)

        if radar_origin_plot:
            self.plot_radar_origin()

        if lidar_origin_plot:
            self.plot_lidar_origin()

        if camera_origin_plot:
            self.plot_camera_origin()

        if lidar_points_plot:
            self.plot_lidar_points()

        if radar_points_plot:
            self.plot_radar_points()

        if radar_velocity_plot:
            self.plot_radar_radial_velocity()

        if annotations_plot:
            self.plot_annotations()

        if not auto_frame:
            self.plot.camera = get_default_camera(self.transform_matrices['lidar'])

        self.plot.display()

        if write_to_html:
            self.plot.snapshot_type = 'inline'

            data = self.plot.get_snapshot()

            with open(f'{html_name}.html', 'w') as f:
                f.write(data)

