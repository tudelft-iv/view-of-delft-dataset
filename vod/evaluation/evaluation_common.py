import pathlib
import re
import numpy as np


def filter_annotations_low_score(image_annotations, thresh):
    new_image_annotations = []
    for anno in image_annotations:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, s in enumerate(anno['score']) if s >= thresh
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = (
                anno[key][relevant_annotation_indices])
        new_image_annotations.append(img_filtered_annotations)
    return new_image_annotations


def get_label_annotation(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    annotations['name'] = np.array([x[0] for x in content])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
        -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros([len(annotations['bbox'])])
    return annotations


def get_label_annotations(label_folder, image_ids=None):
    if image_ids is None:
        filepaths = pathlib.Path(label_folder).glob('*.txt')
        prog = re.compile(r'^\d{5}.txt$')
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        image_ids = [int(p.stem) for p in filepaths]
        image_ids = sorted(image_ids)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    annos = []
    label_folder = pathlib.Path(label_folder)
    for idx in image_ids:
        label_filename = label_folder / (idx + '.txt')
        annos.append(get_label_annotation(label_filename))
    return annos
