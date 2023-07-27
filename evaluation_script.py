import sys
from vod.evaluation import Evaluation
import os

detection_folder_path = sys.argv[1]
ground_truth_path = sys.argv[2]

# When the instance is created, the label locations are required.
evaluation = Evaluation(test_annotation_file=ground_truth_path)

# Using the evaluate method, the model can be evaluated on the detection labels.
results = evaluation.evaluate(
    result_path=detection_folder_path,
    current_class=[0, 1, 2])

print('Results: \n'
      'Entire annotated area: \n'
      'Car: ', results['entire_area']['Car_3d_all'], '\n'
      'Pedestrian: ', results['entire_area']['Pedestrian_3d_all'], '\n'
      'Cyclist: ', results['entire_area']['Cyclist_3d_all'], '\n'
      'mAP: ', (results['entire_area']['Car_3d_all'] + results['entire_area']['Pedestrian_3d_all'] + results['entire_area']['Cyclist_3d_all']) / 3, '\n'
      'Driving corridor area: \n'
      'Car: ', results['roi']['Car_3d_all'], '\n'
      'Pedestrian: ', results['roi']['Pedestrian_3d_all'], '\n'
      'Cyclist: ', results['roi']['Cyclist_3d_all'], '\n'
      'mAP: ', (results['roi']['Car_3d_all'] + results['roi']['Pedestrian_3d_all'] + results['roi']['Cyclist_3d_all']) / 3, '\n'
)
