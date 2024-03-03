from typing import Literal

import numpy as np
from skimage.color import rgb2yuv
from scipy.ndimage import minimum_filter1d


def compute_energy(image):
    yuv_image = rgb2yuv(image)
    gy, gx = np.gradient(yuv_image[..., 0])
    return np.sqrt(gx ** 2 + gy ** 2)


def compute_seam_matrix(energy: np.ndarray, mode: Literal["vertical", "horizontal"], mask=None):
    if mask is not None:
        mask = mask.astype(np.float64)
        mask *= energy.shape[0] * energy.shape[1] * 256
        energy += mask

    if mode == 'vertical':
        energy = energy.T

    seam_matrix = energy.copy()
    seam_matrix.astype(np.float64)
    for row in range(1, energy.shape[0]):
        seam_matrix[row, :] += minimum_filter1d(seam_matrix[row - 1, :], size=3)

    if mode == 'vertical':
        seam_matrix = seam_matrix.T

    return seam_matrix


def remove_minimal_seam(image, seam_matrix, mode, mask=None):
    h, w, c = image.shape
    shove = np.zeros((h, w))
    if mode.split(' ')[0] == 'vertical':
        seam_matrix = seam_matrix.T
        shove = shove.T

    rows, cols = shove.shape
    start = np.argmin(seam_matrix[-1:])
    shove[-1, start] = 1
    for row in range(rows - 1, 0, -1):
        if start == 0 or start == cols - 1:
            if start == 0:
                start += np.argmin(seam_matrix[row - 1, start:start + 2])
                shove[row - 1, start] = 1
            else:
                start += np.argmin(seam_matrix[row - 1, start - 1:start + 1]) - 1
                shove[row - 1, start] = 1
        else:
            start += np.argmin(seam_matrix[row - 1, start - 1:start + 2]) - 1
            shove[row - 1, start] = 1

    if not mode.split(' ')[0] == 'horizontal':
        first = np.ma.masked_where(shove, image[..., 0].T).compressed().reshape((w, h - 1)).T
        second = np.ma.masked_where(shove, image[..., 1].T).compressed().reshape((w, h - 1)).T
        third = np.ma.masked_where(shove, image[..., 2].T).compressed().reshape((w, h - 1)).T
        new_image = np.dstack([first, second, third])
        if mask is not None:
            mask = np.ma.masked_where(shove, mask.T).compressed().reshape((w, h - 1)).T

        shove = shove.T
    else:
        first = np.ma.masked_where(shove, image[..., 0]).compressed().reshape((h, w - 1))
        second = np.ma.masked_where(shove, image[..., 1]).compressed().reshape((h, w - 1))
        third = np.ma.masked_where(shove, image[..., 2]).compressed().reshape((h, w - 1))
        new_image = np.dstack([first, second, third])
        if mask is not None:
            mask = np.ma.masked_where(shove, mask).compressed().reshape((h, w - 1))

    return new_image.astype(np.uint8), mask, shove.astype(np.uint8)


def seam_carve(image, mode, mask=None):
    energy = compute_energy(image)
    seam_mask = compute_seam_matrix(energy, mode.split(' ')[0], mask)
    new_img, mask, shove = remove_minimal_seam(image, seam_mask, mode, mask)

    return new_img, mask, shove

