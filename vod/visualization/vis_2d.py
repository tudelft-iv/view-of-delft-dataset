from matplotlib import pyplot as plt

from vod.frame import FrameDataLoader, FrameTransformMatrix, FrameLabels, project_pcl_to_image, min_max_filter

from .helpers import plot_boxes, get_2d_label_corners
from .settings import label_color_palette_2d


class Visualization2D:
    """
    This class is responsible for plotting a frame from the set, and visualize
     its image with its point clouds (radar and/or LiDAR), annotations projected and overlaid.
    """
    def __init__(self,
                 frame_data_loader: FrameDataLoader,
                 classes_visualized: list = ['Cyclist', 'Pedestrian', 'Car']
                 ):
        """
Constructor of the class, which loads the required frame properties, and creates a copy of the picture data.
        :param frame_data_loader: FrameDataLoader instance.
        :param classes_visualized: A list of classes to be visualized.
        """
        self.frame_data_loader = frame_data_loader
        self.frame_transformations = FrameTransformMatrix(self.frame_data_loader)

        self.classes_visualized = classes_visualized

        self.image_copy = self.frame_data_loader.image

    def plot_gt_labels(self, max_distance_threshold):
        """
This method plots the ground truth labels on the frame.
        :param max_distance_threshold: The maximum distance where labels are rendered.
        """
        frame_labels_class = FrameLabels(self.frame_data_loader.raw_labels)
        box_points = get_2d_label_corners(frame_labels_class, self.frame_transformations)

        # Class filter
        filtered = list(filter(lambda elem: elem['label_class'] in self.classes_visualized, box_points))

        # Distance filter
        filtered = list(filter(lambda elem: elem['range'] < max_distance_threshold, filtered))

        colors = [label_color_palette_2d[v["label_class"]] for v in filtered]
        labels = [d['corners'] for d in filtered]

        plot_boxes(labels, colors)

    def plot_predictions(self, score_threshold, max_distance_threshold):
        """
This method plots the prediction labels on the frame.
        :param score_threshold: The minimum score to be rendered.
        :param max_distance_threshold: The maximum distance where labels are rendered.
        """
        frame_labels_class = FrameLabels(self.frame_data_loader.predictions)
        box_points = get_2d_label_corners(frame_labels_class, self.frame_transformations)

        # Class filter
        filtered = list(filter(lambda elem: elem['label_class'] in self.classes_visualized, box_points))

        # Score filter
        filtered = list(filter(lambda elem: elem['score'] > score_threshold, filtered))

        # Distance filter
        filtered = list(filter(lambda elem: elem['range'] < max_distance_threshold, filtered))

        colors = [label_color_palette_2d[v["label_class"]] for v in filtered]
        labels = [d['corners'] for d in filtered]

        plot_boxes(labels, colors)

    def plot_radar_pcl(self, max_distance_threshold, min_distance_threshold):
        """
This method plots the radar pcl on the frame. It colors the points based on distance.
        :param max_distance_threshold: The maximum distance where points are rendered.
        :param min_distance_threshold: The minimum distance where points are rendered.
        """
        uvs, points_depth = project_pcl_to_image(point_cloud=self.frame_data_loader.radar_data,
                                                 t_camera_pcl=self.frame_transformations.t_camera_radar,
                                                 camera_projection_matrix=self.frame_transformations.camera_projection_matrix,
                                                 image_shape=self.frame_data_loader.image.shape)

        min_max_idx = min_max_filter(points=points_depth,
                                     max_value=max_distance_threshold,
                                     min_value=min_distance_threshold)
        uvs = uvs[min_max_idx]
        points_depth = points_depth[min_max_idx]

        plt.scatter(uvs[:, 0], uvs[:, 1], c=-points_depth, alpha=0.8, s=(70 / points_depth) ** 2, cmap='jet')

    def plot_lidar_pcl(self,max_distance_threshold, min_distance_threshold):
        """
This method plots the lidar pcl on the frame. It colors the points based on distance.
        :param max_distance_threshold: The maximum distance where points are rendered.
        :param min_distance_threshold: The minimum distance where points are rendered.
        """
        uvs, points_depth = project_pcl_to_image(point_cloud=self.frame_data_loader.lidar_data,
                                                 t_camera_pcl=self.frame_transformations.t_camera_lidar,
                                                 camera_projection_matrix=self.frame_transformations.camera_projection_matrix,
                                                 image_shape=self.frame_data_loader.image.shape)

        min_max_idx = min_max_filter(points=points_depth,
                                     max_value=max_distance_threshold,
                                     min_value=min_distance_threshold)
        uvs = uvs[min_max_idx]
        points_depth = points_depth[min_max_idx]

        plt.scatter(uvs[:, 0], uvs[:, 1], c=-points_depth, alpha=0.4, s=1, cmap='jet')

    def draw_plot(self, plot_figure=True,
                  save_figure=True,
                  show_gt: bool = False,
                  show_pred: bool = False,
                  show_lidar: bool = False,
                  show_radar: bool = False,
                  max_distance_threshold: float = 50.0,
                  min_distance_threshold: float = 0.0,
                  score_threshold: float = 0, ):
        """
This method can be called to draw the frame with the required information.
        :param plot_figure: Should the figure be displayed.
        :param save_figure: Should the figure be saved.
        :param show_gt: Should the ground truth be plotted.
        :param show_pred: Should the predictions be plotted.
        :param show_lidar: Should the lidar pcl be plotted.
        :param show_radar: Should the radar pcl be plotted.
        :param max_distance_threshold: Maximum distance of objects to be plotted.
        :param min_distance_threshold:  Minimum distance of objects to be plotted.
        :param score_threshold: Minimum score for objects to be plotted.
        """
        fig = plt.figure(figsize=(12, 8))
        fig.set_dpi(150)

        plt.clf()

        if show_gt:
            self.plot_gt_labels(max_distance_threshold=max_distance_threshold)

        if show_pred:
            self.plot_predictions(max_distance_threshold=max_distance_threshold,
                                  score_threshold=score_threshold)

        if show_lidar:
            self.plot_lidar_pcl(max_distance_threshold=max_distance_threshold,
                                min_distance_threshold=min_distance_threshold)

        if show_radar:
            self.plot_radar_pcl(max_distance_threshold=max_distance_threshold,
                                min_distance_threshold=min_distance_threshold)

        plt.imshow(self.image_copy, alpha=1)
        plt.axis('off')

        if save_figure:
            plt.savefig(self.frame_data_loader.kitti_locations.output_dir + f'{self.frame_data_loader.file_id}.png',
                        bbox_inches='tight', transparent=True)
        if plot_figure:
            plt.show()

        plt.close(fig)

        return
