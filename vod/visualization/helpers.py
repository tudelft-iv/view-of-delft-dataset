import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import k3d

from vod import FrameLabels, FrameTransformMatrix
from vod.frame import transformations


def get_default_camera(pose_transform=np.eye(4, 4)):
    """
This function returns the list required for creating a camera in k3d.
    :param pose_transform: 4x4 transformation matrix of the used coordinate system.
    :return: List required by k3d to create a camera view.
    """
    # Homogenous camera positions.
    camera_pos = [-10, 0, 20, 1]
    camera_focus_point = [10, 0, 1, 1]
    camera_up = [0, 0, 90, 1]
    default_camera = np.array([camera_pos, camera_focus_point, camera_up])

    pose_camera = pose_transform.dot(default_camera.T).T
    pose_camera_up = pose_camera[2, :3] - pose_camera[0, :3]
    return pose_camera[:2, :3].flatten().tolist() + pose_camera_up.tolist()


def get_3d_label_corners(labels: FrameLabels):
    """
This function returns a list of 3d corners of each label in a frame given a FrameLabels object.
    :param labels: FrameLabels object.
    :return: List of 3d corners.
    """
    label_corners = []

    for label in labels.labels_dict:
        x_corners = [label['l'] / 2,
                     label['l'] / 2,
                     -label['l'] / 2,
                     -label['l'] / 2,
                     label['l'] / 2,
                     label['l'] / 2,
                     -label['l'] / 2,
                     -label['l'] / 2]
        y_corners = [label['w'] / 2,
                     -label['w'] / 2,
                     -label['w'] / 2,
                     label['w'] / 2,
                     label['w'] / 2,
                     -label['w'] / 2,
                     -label['w'] / 2,
                     label['w'] / 2]
        z_corners = [0,
                     0,
                     0,
                     0,
                     label['h'],
                     label['h'],
                     label['h'],
                     label['h']]

        corners_3d = np.vstack([x_corners, y_corners, z_corners])
        label_corners.append({'label_class': label['label_class'],
                              'score': label['score'],
                              'corners_3d': corners_3d})

    return label_corners


def get_transformed_3d_label_corners(labels: FrameLabels, transformation, t_camera_lidar):
    corners_3d = get_3d_label_corners(labels)

    transformed_3d_label_corners = []

    for index, label in enumerate(labels.labels_dict):
        rotation = -(label['rotation'] + np.pi / 2)  # undo changes made to rotation
        rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                               [np.sin(rotation), np.cos(rotation), 0],
                               [0, 0, 1]])

        center = (np.linalg.inv(t_camera_lidar) @ np.array([label['x'],
                                                             label['y'],
                                                             label['z'],
                                                             1]))[:3]

        new_corner_3d = np.dot(rot_matrix, corners_3d[index]['corners_3d']).T + center
        new_corners_3d_hom = np.concatenate((new_corner_3d, np.ones((8, 1))), axis=1)
        new_corners_3d_hom = transformations.homogeneous_transformation(new_corners_3d_hom,
                                                                              transformation)

        transformed_3d_label_corners.append({'label_class': label['label_class'],
                                             'corners_3d_transformed': new_corners_3d_hom,
                                             'score': label['score']})

    return transformed_3d_label_corners


def get_2d_label_corners(labels: FrameLabels, transformations_matrix: FrameTransformMatrix):
    bboxes = []
    corners_3d = get_3d_label_corners(labels)

    for index, label in enumerate(labels.labels_dict):
        rotation = -(label['rotation'] + np.pi / 2)  # undo changes made to rotation
        rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                               [np.sin(rotation), np.cos(rotation), 0],
                               [0, 0, 1]])

        center = (transformations_matrix.t_lidar_camera @ np.array([label['x'],
                                                                    label['y'],
                                                                    label['z'],
                                                                    1]))[:3]

        new_corner_3d = np.dot(rot_matrix, corners_3d[index]['corners_3d']).T + center
        new_corners_3d_hom = np.concatenate((new_corner_3d, np.ones((8, 1))), axis=1)
        new_corners_3d_hom = transformations.homogeneous_transformation(new_corners_3d_hom,
                                                                              transformations_matrix.t_camera_lidar)

        corners_img = np.dot(new_corners_3d_hom, transformations_matrix.camera_projection_matrix.T)
        corners_img = (corners_img[:, :2].T / corners_img[:, 2]).T
        corners_img = corners_img.tolist()
        distance = np.linalg.norm((label['x'], label['y'], label['z']))

        bboxes.append({'label_class': label['label_class'],
                       'corners': corners_img,
                       'score': label['score'],
                       'range': distance})

    bboxes = sorted(bboxes, key=lambda d: d['range'])

    return bboxes


def mask_pcl(scan, scan_C2, inds, scan_c2_depth):
    mask = np.linalg.norm(scan[:, :3], axis=1) < 30
    return scan[mask], scan_C2[mask], inds[mask], scan_c2_depth[mask]


def line(p1, p2, color):
    plt.gca().add_line(
        Line2D((p1[0], p2[0]), (p1[1], p2[1]), color=color, linewidth=1))


def face(corners: np.ndarray, color: tuple, alpha: float = 0.3):
    xs = corners[:, 0]
    ys = corners[:, 1]
    plt.fill(xs, ys, color=color, alpha=alpha)


def plot_boxes(boxes: list, colors=None):
    for j in range(len(boxes)):
        corners_img = np.array(boxes[j])

        if colors is not None:
            color = colors[j]
        else:
            color = (1.0, 1.0, 1.0)

        if color == (1.0, 1.0, 1.0):
            alpha = 0.15
        else:
            alpha = 0.2

        # draw the 6 faces
        face(corners_img[:4], color, alpha)
        face(corners_img[4:], color, alpha)
        face(np.array([corners_img[0], corners_img[1], corners_img[5], corners_img[4]]), color, alpha)
        face(np.array([corners_img[1], corners_img[2], corners_img[6], corners_img[5]]), color, alpha)
        face(np.array([corners_img[2], corners_img[3], corners_img[7], corners_img[6]]), color, alpha)
        face(np.array([corners_img[0], corners_img[3], corners_img[7], corners_img[4]]), color, alpha)
    return


def k3d_get_axes(hom_transform_matrix=np.eye(4, dtype=float), axis_length=1.0):
    """
Function which returns an axis system in the requested frame based on the transform matrix.
    :param hom_transform_matrix: The homogenous transform matrix.
    :param axis_length: Vector length of the axis.
    :return: k3d vector of the axis system.
    """
    hom_vector = np.asarray([[axis_length, 0, 0, 1],
                         [0, axis_length, 0, 1],
                         [0, 0, axis_length, 1]],
                        dtype=float)

    origin = [hom_transform_matrix[:3, 3]] * 3
    axis = hom_transform_matrix.dot(hom_vector.T).T[:, :3]

    pose_axes = k3d.vectors(
        origins=origin,
        vectors=axis[:, :3] - origin,
        colors=[0xFF0000, 0xFF0000, 0x00FF00, 0x00FF00, 0x0000FF, 0x0000FF])

    return pose_axes


def k3d_plot_box(plot, box_corners, color, width):
    lines = [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [0, 4], [1, 5], [3, 7], [2, 6]]

    for plot_line in lines:
        plot += k3d.line(box_corners[plot_line, 0:3], color=color, width=width)


def get_radar_velocity_vectors(pc_radar, compensated_radial_velocity):
    radial_unit_vectors = pc_radar / np.linalg.norm(pc_radar, axis=1, keepdims=True)
    velocity_vectors = compensated_radial_velocity[:, None] * radial_unit_vectors

    return velocity_vectors
