import os
from vod.evaluation import evaluation_common as kitti
from .kitti_official_evaluate import get_official_eval_result
from vod import get_frame_list_from_folder


class Evaluation:
    """
    Evaluation class for KITTI evaluation.
    """
    def __init__(self, test_annotation_file):
        """
        Initialize the evaluation class for KITTI evaluation.
        :param test_annotation_file: Location of the test annotation files.
        """
        self.test_annotation_file = test_annotation_file

    def evaluate(self,
                 result_path,
                 current_class=None,
                 score_thresh=-1,
                 ):
        """
        Evaluate the results.
        :param result_path: Detection labels path.
        :param current_class: Class to evaluate.
        :param score_thresh: Score threshold to use.
        :return: Results of the evaluation.
        """

        if current_class is None:
            current_class = [0, 1, 2]

        val_image_ids = get_frame_list_from_folder(result_path)

        dt_annotations = kitti.get_label_annotations(result_path, val_image_ids)

        if score_thresh > 0:
            dt_annotations = kitti.filter_annotations_low_score(dt_annotations, score_thresh)

        gt_annotations = kitti.get_label_annotations(self.test_annotation_file, val_image_ids)

        evaluation_result = {}
        evaluation_result.update(get_official_eval_result(gt_annotations, dt_annotations, current_class, custom_method=0))
        evaluation_result.update(get_official_eval_result(gt_annotations, dt_annotations, current_class, custom_method=3))

        return evaluation_result


if __name__ == '__main__':

    evaluation = Evaluation(test_annotation_file=os.path.join('..', '..', 'example_set', 'label'))

    results = evaluation.evaluate(
        result_path=os.path.join('..', '..', 'example_set', 'detection'),
        current_class=[0, 1, 2])

    print("Results: \n"
          f"Entire annotated area: \n"
          f"Car: {results['entire_area']['Car_3d_all']} \n"
          f"Pedestrian: {results['entire_area']['Pedestrian_3d_all']} \n"
          f"Cyclist: {results['entire_area']['Cyclist_3d_all']} \n"
          f"mAP: {(results['entire_area']['Car_3d_all'] + results['entire_area']['Pedestrian_3d_all'] + results['entire_area']['Cyclist_3d_all']) / 3} \n"
          f"Driving corridor area: \n"
          f"Car: {results['roi']['Car_3d_all']} \n"
          f"Pedestrian: {results['roi']['Pedestrian_3d_all']} \n"
          f"Cyclist: {results['roi']['Cyclist_3d_all']} \n"
          f"mAP: {(results['roi']['Car_3d_all'] + results['roi']['Pedestrian_3d_all'] + results['roi']['Cyclist_3d_all']) / 3} \n"
          )
