from functools import lru_cache

import cv2
import numpy as np
import torch


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims


def save_checkpoint(net, checkpoints_path, epoch=None):
    if epoch is None:
        checkpoint_name = "last_checkpoint.pth"
    else:
        checkpoint_name = f"{epoch:03d}.pth"

    if not checkpoints_path.exists():
        checkpoints_path.mkdir(parents=True)

    checkpoint_path = checkpoints_path / checkpoint_name
    print(f"Save checkpoint to {str(checkpoint_path)}")

    torch.save({"state_dict": net.state_dict()}, str(checkpoint_path))


def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def expand_bbox(bbox, expand_ratio, min_crop_size=None):
    rmin, rmax, cmin, cmax = bbox
    rcenter = 0.5 * (rmin + rmax)
    ccenter = 0.5 * (cmin + cmax)
    height = expand_ratio * (rmax - rmin + 1)
    width = expand_ratio * (cmax - cmin + 1)
    if min_crop_size is not None:
        height = max(height, min_crop_size)
        width = max(width, min_crop_size)

    rmin = int(round(rcenter - 0.5 * height))
    rmax = int(round(rcenter + 0.5 * height))
    cmin = int(round(ccenter - 0.5 * width))
    cmax = int(round(ccenter + 0.5 * width))

    return rmin, rmax, cmin, cmax


def clamp_bbox(bbox, rmin, rmax, cmin, cmax):
    return (
        max(rmin, bbox[0]),
        min(rmax, bbox[1]),
        max(cmin, bbox[2]),
        min(cmax, bbox[3]),
    )


def get_bbox_iou(b1, b2):
    h_iou = get_segments_iou(b1[:2], b2[:2])
    w_iou = get_segments_iou(b1[2:4], b2[2:4])
    return h_iou * w_iou


def get_segments_iou(s1, s2):
    a, b = s1
    c, d = s2
    intersection = max(0, min(b, d) - max(a, c) + 1)
    union = max(1e-6, max(b, d) - min(a, c) + 1)
    return intersection / union


def get_labels_with_sizes(x):
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()
    labels = [x for x in labels if x != 0]
    return labels, obj_sizes[labels].tolist()


@lru_cache(maxsize=16)
def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))


def draw_probmap(x):
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT)


def draw_points(image, points, color, radius=3):
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        if len(p) == 3:
            pradius = {0: 8, 1: 6, 2: 4}[p[2]] if p[2] < 3 else 2
        else:
            pradius = radius
        image = cv2.circle(image, (int(p[1]), int(p[0])), pradius, color, -1)

    return image


def draw_with_blend_and_clicks(
    img,
    mask=None,
    alpha=0.6,
    clicks_list=None,
    pos_color=(0, 255, 0),
    neg_color=(255, 0, 0),
    radius=4,
):
    result = img.copy()

    if mask is not None:
        palette = get_palette(np.max(mask) + 1)
        rgb_mask = palette[mask.astype(np.uint8)]

        mask_region = (mask > 0).astype(np.uint8)
        result = (
            result * (1 - mask_region[:, :, np.newaxis])
            + (1 - alpha) * mask_region[:, :, np.newaxis] * result
            + alpha * rgb_mask
        )
        result = result.astype(np.uint8)

        # result = (result * (1 - alpha) + alpha * rgb_mask).astype(np.uint8)

    if clicks_list is not None and len(clicks_list) > 0:
        pos_points = [click.coords for click in clicks_list if click.is_positive]
        neg_points = [click.coords for click in clicks_list if not click.is_positive]

        result = draw_points(result, pos_points, pos_color, radius=radius)
        result = draw_points(result, neg_points, neg_color, radius=radius)

    return result
