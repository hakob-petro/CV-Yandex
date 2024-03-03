import numpy as np


def get_bayer_masks(n_rows: int, n_cols: int) -> np.ndarray:
    red_block = np.array([[0, 1], [0, 0]])
    green_block = np.array([[1, 0], [0, 1]])
    blue_block = np.array([[0, 0], [1, 0]])

    # One more solution based on https://stackoverflow.com/questions/26374634/numpy-tile-a-non-integer-number-of-times
    # Tested with timeit, this is 8x times slower
    # red_mask = np.pad(
    #     red_block,
    #     tuple((0, i) for i in (np.array((n_rows, n_cols)) - red_block.shape)),
    #     mode='wrap'
    # )

    red_mask = np.tile(red_block, ((n_rows + 1) // 2, (n_cols + 1) // 2))[:n_rows, :n_cols]
    green_mask = np.tile(green_block, ((n_rows + 1) // 2, (n_cols + 1) // 2))[:n_rows, :n_cols]
    blue_mask = np.tile(blue_block, ((n_rows + 1) // 2, (n_cols + 1) // 2))[:n_rows, :n_cols]
    return np.stack([red_mask, green_mask, blue_mask], axis=2).astype(bool)


def get_colored_img(raw_img: np.ndarray) -> np.ndarray:
    return raw_img[..., np.newaxis] * get_bayer_masks(raw_img.shape[0], raw_img.shape[1])


def fill_zeros_with_average(img: np.ndarray) -> np.ndarray:
    img[img == 0] = np.mean(img)
    return img


def bilinear_interpolation(img):
    res = np.zeros_like(img, dtype=np.int64)
    image = img.astype(np.int64)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if i % 2 == 0 and j % 2 == 0:
                res[i, j, 0] = (image[i, j - 1, 0] + image[i, j + 1, 0]) / 2
                res[i, j, 1] = image[i, j, 1]
                res[i, j, 2] = (image[i - 1, j, 2] + image[i + 1, j, 2]) / 2
            elif i % 2 == 1 and j % 2 == 0:
                res[i, j, 0] = (image[i - 1, j - 1, 0] + image[i + 1, j + 1, 0] +
                                image[i - 1, j + 1, 0] + image[i + 1, j - 1, 0]) / 4
                res[i, j, 1] = (image[i - 1, j, 1] + image[i + 1, j, 1] +
                                image[i, j - 1, 1] + image[i, j + 1, 1]) / 4
                res[i, j, 2] = image[i, j, 2]
            elif i % 2 == 0 and j % 2 == 1:
                res[i, j, 0] = image[i, j, 0]
                res[i, j, 1] = (image[i - 1, j, 1] + image[i + 1, j, 1] +
                                image[i, j - 1, 1] + image[i, j + 1, 1]) / 4
                res[i, j, 2] = (image[i - 1, j - 1, 2] + image[i + 1, j + 1, 2] +
                                image[i - 1, j + 1, 2] + image[i + 1, j - 1, 2]) / 4
            else:
                res[i, j, 0] = (image[i - 1, j, 0] + image[i + 1, j, 0]) / 2
                res[i, j, 1] = image[i, j, 1]
                res[i, j, 2] = (image[i, j - 1, 2] + image[i, j + 1, 2]) / 2
    return res.astype(np.uint8)


def compute_psnr(img, img_gt):
    img, img_gt = img.astype(np.float64), img_gt.astype(np.float64)
    mse = np.mean((img - img_gt) ** 2)
    if mse == 0:
        raise ValueError("MSE is 0, can't compute PSNR")
    max_i_gt = img_gt.max()
    return 10 * np.log10(max_i_gt ** 2 / mse)


def improved_interpolation(raw_img):
    raw_img = raw_img.astype(np.float32)

    red = np.zeros_like(raw_img)
    green = np.zeros_like(raw_img)
    blue = np.zeros_like(raw_img)

    # green at red locations and green at blue locations
    filter_1 = np.array([[0, 0, -1, 0, 0],
                        [0, 0, 2, 0, 0],
                        [-1, 2, 4, 2, -1],
                        [0, 0, 2, 0, 0],
                        [0, 0, -1, 0, 0]])
    # red at green in red row, blue column and blue at green in blue row, red column
    filter_2 = np.array([[0, 0, 1 / 2, 0, 0],
                        [0, -1, 0, -1, 0],
                        [-1, 4, 5, 4, -1],
                        [0, -1, 0, -1, 0],
                        [0, 0, 1 / 2, 0, 0]])
    # red at green in blue row, red column and red at blue in blue row, blue column
    filter_3 = filter_2.T
    # blue at green in red row, blue column and blue at red in red row, red column
    filter_4 = np.array([[0, 0, -3 / 2, 0, 0],
                        [0, 2, 0, 2, 0],
                        [-3 / 2, 0, 6, 0, -3 / 2],
                        [0, 2, 0, 2, 0],
                        [0, 0, -3 / 2, 0, 0]])

    sum_filter_1 = np.sum(filter_1)
    sum_filter_2 = np.sum(filter_2)
    sum_filter_3 = np.sum(filter_3)
    sum_filter_4 = np.sum(filter_4)

    for i in range(2, raw_img.shape[0] - 2):
        for j in range(2, raw_img.shape[1] - 2):
            raw_img_fragment = raw_img[i - 2:i + 3, j - 2:j + 3]
            if (i % 2 == 0) and (j % 2 == 0):
                red[i, j] = np.sum(filter_2 / sum_filter_2 * raw_img_fragment)
                green[i, j] = raw_img[i, j]
                blue[i, j] = np.sum(filter_3 / sum_filter_3 * raw_img_fragment)
            elif (i % 2 != 0) and (j % 2 != 0):
                red[i, j] = np.sum(filter_3 / sum_filter_3 * raw_img_fragment)
                green[i, j] = raw_img[i, j]
                blue[i, j] = np.sum(filter_2 / sum_filter_2 * raw_img_fragment)
            elif (i % 2 == 0) and (j % 2 != 0):
                red[i, j] = raw_img[i, j]
                green[i, j] = np.sum(filter_1 / sum_filter_1 * raw_img_fragment)
                blue[i, j] = np.sum(filter_4 / sum_filter_4 * raw_img_fragment)
            else:
                red[i, j] = np.sum(filter_4 / sum_filter_4 * raw_img_fragment)
                green[i, j] = np.sum(filter_1 / sum_filter_1 * raw_img_fragment)
                blue[i, j] = raw_img[i, j]

    res = np.stack([red, green, blue], axis=2)
    np.clip(res, 0, 255, out=res)
    return res.astype(np.uint8)
