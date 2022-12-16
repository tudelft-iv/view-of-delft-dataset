# Based on https://github.com/traveller59/kitti-object-eval-python
# Licensed under The MIT License
import numpy as np
import numba
from .rotate_iou_cpu import rotate_iou_eval


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt,
                   num_sample_pts=41):  # thresholds here are the confidence scores of the model predictions
    scores.sort()
    scores = scores[::-1]  # collect elements along the last axis
    current_recall = 0
    thresholds = []

    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue

        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)

    return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty, roi_clean=False):  # per frame
    valid_class_names = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'truck']
    min_instance_height = [40]
    max_instance_occlusion = [4]

    left = -4
    right = +4
    max_distance = 25

    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = valid_class_names[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0

    # Cleaning ground truth based on difficulty level (one of three levels) for current evaluation class
    for i in range(num_gt):  # per frame
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]  # in pixels [bounding box with min height only evaluated]

        # validate class based on query class
        if gt_name == current_cls_name:
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif current_cls_name == "Car".lower() and "Van".lower() == gt_name:
            valid_class = 0
        else:  # classes not used for evaluation
            valid_class = -1

        ignore = False

        if ((gt_anno["occluded"][i] > max_instance_occlusion[difficulty])
                or (height <= min_instance_height[difficulty])):
            ignore = True

        # ignore gts with centers outside the lane or farther than 25m
        # this is called "Driving Corridor" in the paper, but addressed a roi (region of interest) here
        # not to be confused with roi proposals of detection model
        if roi_clean:
            x, y, z = gt_anno['location'][i]
            if x < left or x > right or z > max_distance:
                ignore = True

        # set ignored vector for gt
        # current class and not ignored (total no. of ground truth is detected for recall denominator)
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        #  neighboring class, or current class but ignored
        elif valid_class == 0 or (ignore and (valid_class == 1)):
            ignored_gt.append(1)
        # all others class
        else:
            ignored_gt.append(-1)

        # extract don't care areas to suppress unfair fp count later
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])

    # extract detections bounding boxes of the current class
    for i in range(num_dt):
        x, y, z = dt_anno['location'][i]

        if dt_anno["name"][i].lower() == current_cls_name:
            valid_class = 1
        else:
            valid_class = -1

        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if height < min_instance_height[difficulty]:
            ignored_dt.append(1)

        elif (x < left or x > right or z > max_distance) and roi_clean:
            ignored_dt.append(1)

        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        # [0,1,2,3] = [left, top, right, bottom] pixel co-ordinates
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            # check if there is an overlap in width
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                # check if there is an overlap in height (then only can there be an intersection)
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                                (boxes[n, 2] - boxes[n, 0]) *
                                (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, q_boxes, criterion=-1):
    r_iou = rotate_iou_eval(boxes, q_boxes, criterion)
    return r_iou


@numba.jit(nopython=True)
def d3_box_overlap_kernel(boxes, q_boxes, r_inc, criterion=-1):
    # ONLY support overlap in CAMERA, not lidar.
    N, K = boxes.shape[0], q_boxes.shape[0]
    for i in range(N):
        for j in range(K):
            if r_inc[i, j] > 0:
                iw = (min(boxes[i, 1], q_boxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], q_boxes[j, 1] - q_boxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = q_boxes[j, 3] * q_boxes[j, 4] * q_boxes[j, 5]
                    inc = iw * r_inc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    r_inc[i, j] = inc / ua
                else:
                    r_inc[i, j] = 0.0


def d3_box_overlap(boxes, q_boxes, criterion=-1):
    r_inc = rotate_iou_eval(boxes[:, [0, 2, 3, 5, 6]],
                           q_boxes[:, [0, 2, 3, 5, 6]], 2)  # r_inc calculates bev overlap
    d3_box_overlap_kernel(boxes, q_boxes, r_inc, criterion)
    return r_inc


@numba.jit(nopython=False)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0.0,
                           compute_fp=False,
                           compute_aos=False):
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]
    # gt_bboxes = gt_datas[:, :4]

    # Will be changed to True when score,overlap cross threshold and when ignore_gt/dt = 0 or 1
    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size  # to ignore a detection if score is below threshold

    # detections with a low score are ignored for computing precision (needs FP)
    if compute_fp:
        for i in range(det_size):
            if dt_scores[i] < thresh:
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    delta = np.zeros((gt_size,))  # diff in local orientation bwn gt and det
    delta_idx = 0

    # iterate overall all gt bboxes
    for i in range(gt_size):
        # this ground truth is not of the current or a neighboring class and therefore ignored
        if ignored_gt[i] == -1:
            continue

        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False  # when you assign an ignored detection to a ground truth object

        for j in range(det_size):

            # detections not of the current class, already assigned or with a low threshold are ignored
            if ignored_det[j] == -1:
                continue
            if assigned_detection[j]:
                continue
            if ignored_threshold[j]:
                continue

            overlap = overlaps[j, i]  # fetch overlap score calculated through iou
            dt_score = dt_scores[j]  # fetch detection confidence

            # for computing recall thresholds, the candidate with the highest score is considered
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j  # get the index of detection label
                valid_detection = dt_score  # store the confidence in the valid_detection and update whenever higher 
                # confidence occurs 

            # for computing pr curve values, the candidate with the greatest overlap is considered
            # if the greatest overlap is an ignored detection (min_height), the overlapping detection is used
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[
                      j] == 1):  # ignored_dt = 1 (intended for height difficulty), det of another class may show up 
                # here 
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        # nothing was assigned to the valid gt
        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1

        # assign detections as true for labels ignored based on difficulty - this will result in meaningful fps
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True

        # valid true positive
        elif valid_detection != NO_DETECTION:
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1

            # compute error in detected orientation
            if compute_aos:
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1
            assigned_detection[det_idx] = True

    if compute_fp:
        # count fp
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        n_stuff = 0

        # do not consider detections falling under don't care regions as fp
        # in VoD experiments, our annoys are purely within fov and upto 50 mts in front and input pcl
        # is also bounded to that distance. So, dc_boxes are nil for VoD setup
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if assigned_detection[j]:
                        continue
                    if ignored_det[j] == -1 or ignored_det[j] == 1:
                        continue
                    if ignored_threshold[j]:
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        n_stuff += 1
        fp -= n_stuff

        if compute_aos:
            tmp = np.zeros((fp + delta_idx,))
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:gt_num + gt_nums[i]]
            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annotations, dt_annotations, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annotations: dict, must from get_label_annotations() in evaluation_common.py
        dt_annotations: dict, must from get_label_annotations() in evaluation_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annotations) == len(dt_annotations)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annotations], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annotations], 0)
    num_examples = len(gt_annotations)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annotations_part = gt_annotations[example_idx:example_idx + num_part]
        dt_annotations_part = dt_annotations[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([(a["bbox"]) for a in gt_annotations_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annotations_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in gt_annotations_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in gt_annotations_part], 0)
            rots = np.concatenate([(a["rotation_y"]) for a in gt_annotations_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in dt_annotations_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annotations_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annotations_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annotations_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annotations_part], 0)
            rots = np.concatenate([(a["rotation_y"]) for a in gt_annotations_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annotations_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annotations_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annotations_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        # gt_annotations_part = gt_annotations[example_idx:example_idx + num_part]
        # dt_annotations_part = dt_annotations[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annotations, dt_annotations, current_class, difficulty, custom_method=0):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annotations)):
        if custom_method == 0:  # default results
            rets = clean_data(gt_annotations[i], dt_annotations[i], current_class, difficulty)

        if custom_method == 3:  # results in the ROI
            rets = clean_data(gt_annotations[i], dt_annotations[i], current_class, difficulty, roi_clean=True)

        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annotations[i]["bbox"], gt_annotations[i]["alpha"][..., np.newaxis]], 1)
        dt_datas = np.concatenate([
            dt_annotations[i]["bbox"], dt_annotations[i]["alpha"][..., np.newaxis],
            dt_annotations[i]["score"][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def eval_class(gt_annotations,
               dt_annotations,
               current_classes,
               difficulties,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=50,
               custom_method=0):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annotations: dict, must from get_label_annotations() in evaluation_common.py
        dt_annotations: dict, must from get_label_annotations() in evaluation_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficulties: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap, metric, class].
        num_parts: int. a parameter for fast calculate algorithm
        custom_method:0: using normal 1: using distance 2: using moving vs not moving
    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annotations) == len(dt_annotations)
    num_examples = len(gt_annotations)
    split_parts = get_split_parts(num_examples, num_parts)

    # iou is calculated in "parts = cluster of frames"
    rets = calculate_iou_partly(dt_annotations, gt_annotations, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets

    N_SAMPLE_PTS = 41  # what is this number
    num_min_overlap = len(min_overlaps)  # 2
    num_class = len(current_classes)
    num_difficulty = len(difficulties)
    precision = np.zeros(
        [num_class, num_difficulty, num_min_overlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_class, num_difficulty, num_min_overlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_min_overlap, N_SAMPLE_PTS])

    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficulties):
            rets = _prepare_data(gt_annotations, dt_annotations, current_class, difficulty, custom_method=custom_method)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets

            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                new_thresholds = []
                for i in range(len(gt_annotations)):  # per frame

                    rets = compute_statistics_jit(overlaps[i], gt_datas_list[i], dt_datas_list[i], ignored_gts[i],
                                                  ignored_dets[i], dontcares[i], metric, min_overlap=min_overlap,
                                                  thresh=0.0, compute_fp=False)

                    _, _, _, _, thresholds = rets  # only thresholds is used
                    new_thresholds += thresholds.tolist()
                new_thresholds = np.array(new_thresholds)
                thresholds = get_thresholds(new_thresholds, total_num_valid_gt)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
                    idx += num_part
                for i in range(len(thresholds)):
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(
                        precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    ret_dict = {
        "recall": recall,
        "precision": precision,
        "orientation": aos,
    }
    return ret_dict


def get_m_ap(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def get_m_ap_r40(prec):
    sums = 0

    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100


def do_eval(gt_annotations,
            dt_annotations,
            current_classes,
            min_overlaps,
            compute_aos=False,
            pr_detail_dict=None,
            custom_method=0):
    if custom_method == 0:  # normal metric
        difficulties = [0]

    if custom_method == 1:  # range-wise metric
        difficulties = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    if custom_method == 2:
        difficulties = [0, 1]

    if custom_method == 3:
        difficulties = [0]

    ret = eval_class(gt_annotations, dt_annotations, current_classes, difficulties, 0,
                     min_overlaps, compute_aos, custom_method=custom_method)

    print("mAP Image BBox finished")
    mAP_bbox = get_m_ap(ret["precision"])
    mAP_bbox_R40 = get_m_ap_r40(ret["precision"])

    if pr_detail_dict is not None:
        pr_detail_dict['bbox'] = ret['precision']

    mAP_aos = mAP_aos_R40 = None
    if compute_aos:
        mAP_aos = get_m_ap(ret["orientation"])
        mAP_aos_R40 = get_m_ap_r40(ret["orientation"])

        if pr_detail_dict is not None:
            pr_detail_dict['aos'] = ret['orientation']

    ret = eval_class(gt_annotations, dt_annotations, current_classes, difficulties, 1,
                     min_overlaps, custom_method=custom_method)
    print("mAP bev BBox finished")
    mAP_bev = get_m_ap(ret["precision"])
    mAP_bev_R40 = get_m_ap_r40(ret["precision"])

    if pr_detail_dict is not None:
        pr_detail_dict['bev'] = ret['precision']

    ret = eval_class(gt_annotations, dt_annotations, current_classes, difficulties, 2,
                     min_overlaps, custom_method=custom_method)
    print("mAP 3D BBox finished")
    mAP_3d = get_m_ap(ret["precision"])
    mAP_3d_R40 = get_m_ap_r40(ret["precision"])
    if pr_detail_dict is not None:
        pr_detail_dict['3d'] = ret['precision']
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos, mAP_bbox_R40, mAP_bev_R40, mAP_3d_R40, mAP_aos_R40


def get_official_eval_result(gt_annotations, dt_annotations, current_classes, pr_detail_dict=None, custom_method=0):
    if custom_method == 0:
        print("Evaluating kitti by default")
    elif custom_method == 3:
        print("Evaluating kitti by ROI")

    # Original OpenPCDet code
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
    overlap_0_5 = np.array([[0.7, 0.50, 0.50, 0.7, 0.50, 0.5],  # image
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],  # bev
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])  # 3d
    # class:  0,    1,    2,   3,   4,    5
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 6] = [num_overlaps, metrics, classes]

    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'rider',
        4: 'bicycle',
        5: 'bicycle_rack',
        6: 'human_depiction',
        7: 'moped_scooter',
        8: 'motor',
        9: 'ride_other',
        10: 'ride_uncertain',
        11: 'truck',
        12: 'vehicle_other'
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]

    if custom_method == 0:
        result_name = 'kitti'
    elif custom_method == 3:
        result_name = 'kitti_roi'

    # check whether alpha is valid

    compute_aos = True
    for anno in dt_annotations:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break

    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
        gt_annotations, dt_annotations, current_classes, min_overlaps, compute_aos, pr_detail_dict=pr_detail_dict,
        custom_method=custom_method)

    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_min_overlap, metric, class]
        # mAP result: [num_class, num_diff, num_min_overlap]
        for i in range(1, 2):  # min_overlaps.shape[0]):
            if compute_aos:
                if i == 1:
                    ret_dict['%s_aos_all' % class_to_name[curcls]] = mAPaos[j, 0, 1]

            if i == 1:
                ret_dict['%s_3d_all' % class_to_name[curcls]] = mAP3d[
                    j, 0, 1]  # get j class, difficulty, second min_overlap
                ret_dict['%s_bev_all' % class_to_name[curcls]] = mAPbev[
                    j, 0, 1]  # get j class, difficulty, second min_overlap

    if custom_method == 0:
        return {'entire_area': ret_dict}
    elif custom_method == 3:
        return {'roi': ret_dict}
    else:
        raise NotImplementedError
