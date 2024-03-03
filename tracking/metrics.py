import numpy as np


def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    # Write code here
    length_intersection = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
    height_intersection = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    intersection = np.maximum(0, length_intersection) * np.maximum(0, height_intersection)
    union = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - intersection
    iou = intersection / union
    return iou


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        frame_obj = dict([(detection[0], detection[1:]) for detection in frame_obj])
        frame_hyp = dict([(detection[0], detection[1:]) for detection in frame_hyp])

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for idx in matches.keys():
            if (idx in frame_obj) and (matches[idx] in frame_hyp):
                iou = iou_score(frame_obj[idx], frame_hyp[matches[idx]])
                if iou >= threshold:
                    dist_sum += iou
                    frame_obj.pop(idx)
                    frame_hyp.pop(matches[idx])
                    match_count += 1

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        remaining_ious = []
        for obj_idx in frame_obj.keys():
            for hyp_idx in frame_hyp.keys():
                iou = iou_score(frame_obj[obj_idx], frame_hyp[hyp_idx])
                if iou >= threshold:
                    remaining_ious.append((iou, obj_idx, hyp_idx))

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        remaining_ious = sorted(remaining_ious, reverse=True)
        updated_matches = {}
        for iou, obj_idx, hyp_idx in remaining_ious:
            if (obj_idx in frame_obj.keys()) and (hyp_idx in frame_hyp.keys()):
                dist_sum += iou
                updated_matches[obj_idx] = hyp_idx
                frame_obj.pop(obj_idx)
                frame_hyp.pop(hyp_idx)
                match_count += 1

        # Step 5: Update matches with current matched IDs
        for obj_idx, hyp_idx in updated_matches.items():
            matches[obj_idx] = hyp_idx

    # Step 6: Calculate MOTP
    return dist_sum / match_count if match_count != 0 else 0


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys
        frame_obj = dict([(detection[0], detection[1:]) for detection in frame_obj])
        frame_hyp = dict([(detection[0], detection[1:]) for detection in frame_hyp])

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for idx in matches.keys():
            if (idx in frame_obj) and (matches[idx] in frame_hyp):
                iou = iou_score(frame_obj[idx], frame_hyp[matches[idx]])
                if iou >= threshold:
                    dist_sum += iou
                    frame_obj.pop(idx)
                    frame_hyp.pop(matches[idx])
                    match_count += 1

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        remaining_ious = []
        for obj_idx in frame_obj.keys():
            for hyp_idx in frame_hyp.keys():
                iou = iou_score(frame_obj[obj_idx], frame_hyp[hyp_idx])
                if iou >= threshold:
                    remaining_ious.append((iou, obj_idx, hyp_idx))

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        remaining_ious = sorted(remaining_ious, reverse=True)
        new_matches = {}
        for iou, obj_idx, hyp_idx in remaining_ious:
            if (obj_idx in frame_obj.keys()) and (hyp_idx in frame_hyp.keys()):
                dist_sum += iou
                new_matches[obj_idx] = hyp_idx

                if obj_idx in matches and matches[obj_idx] != new_matches[obj_idx]:
                    mismatch_error += 1

                frame_obj.pop(obj_idx)
                frame_hyp.pop(hyp_idx)
                match_count += 1

        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error

        # Step 6: Update matches with current matched IDs
        for obj_idx, hyp_idx in new_matches.items():
            matches[obj_idx] = hyp_idx

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        false_positive += len(frame_hyp)
        missed_count += len(frame_obj)

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count if match_count != 0 else 0
    num_objects = sum(list(map(len, obj)))
    MOTA = (1 - (missed_count + false_positive + mismatch_error) / num_objects) if num_objects != 0 else 0

    return MOTP, MOTA
