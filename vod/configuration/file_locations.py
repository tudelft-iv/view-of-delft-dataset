import os


class KittiLocations:
    """
    This class contains the information regarding the locations of data for the dataset.
    """
    def __init__(self, root_dir: str, output_dir: str = None, frame_set_path: str = None, pred_dir: str = None):
        """
Constructor which based on a few parameters defines the locations of possible data.
        :param root_dir: The root directory of the dataset.
        :param output_dir: Optional parameter of the location where output such as pictures should be generated.
        :param frame_set_path: Optional parameter of the text file of which output should be generated.
        :param pred_dir: Optional parameter of the locations of the prediction labels.
        """

        # Input parameters
        self.root_dir: str = root_dir
        self.output_dir: str = output_dir
        self.frame_set_path: str = frame_set_path
        self.pred_dir: str = pred_dir

        # Automatically defined variables. The location of sub-folders can be customized here.
        # Current definitions are based on the recommended locations.
        self.camera_dir = os.path.join(self.root_dir, 'lidar', 'training', 'image_2')

        self.lidar_dir = os.path.join(self.root_dir, 'lidar', 'training', 'velodyne')
        self.lidar_calib_dir = os.path.join(self.root_dir, 'lidar', 'training', 'calib')

        self.radar_dir = os.path.join(self.root_dir, 'radar', 'training', 'velodyne')
        self.radar_calib_dir = os.path.join(self.root_dir, 'radar', 'training', 'calib')

        self.pose_dir = os.path.join(self.root_dir, 'lidar', 'training', 'pose')
        self.pose_calib_dir = os.path.join(self.root_dir, 'lidar', 'training', 'calib')

        self.label_dir = os.path.join(self.root_dir, 'lidar', 'training', 'label_2')
