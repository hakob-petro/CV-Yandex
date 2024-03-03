import numpy as np


def find_optimal_shift_with_fourier(img_1, img_2):
    shifts_matrix = np.fft.ifft2(np.fft.fft2(img_1) * np.conjugate(np.fft.fft2(img_2)))

    optimal_shift = list(np.unravel_index(np.argmax(shifts_matrix, axis=None), shifts_matrix.shape))
    if optimal_shift[0] > shifts_matrix.shape[0] // 2:
        optimal_shift[0] -= shifts_matrix.shape[0]
    if optimal_shift[1] > shifts_matrix.shape[1] // 2:
        optimal_shift[1] -= shifts_matrix.shape[1]

    return optimal_shift


def align(img, g_coord):
    h, w = img.shape

    r, g, b = img[0:int(h / 3), :], img[int(h / 3):int(2 * h / 3), :], img[int(2 * h / 3):, :]
    min_h = min(r.shape[0], g.shape[0], b.shape[0])
    r = r[:min_h, :]
    g = g[:min_h, :]
    b = b[:min_h, :]

    crop = 0.1
    cropped_r = r[int(r.shape[0] * crop):int(r.shape[0] * (1 - crop)),
                int(r.shape[1] * crop):int(r.shape[1] * (1 - crop))]
    cropped_g = g[int(g.shape[0] * crop):int(g.shape[0] * (1 - crop)),
                int(g.shape[1] * crop):int(g.shape[1] * (1 - crop))]
    cropped_b = b[int(b.shape[0] * crop):int(b.shape[0] * (1 - crop)),
                int(b.shape[1] * crop):int(b.shape[1] * (1 - crop))]

    shift_h_red, shift_w_red = find_optimal_shift_with_fourier(cropped_r, cropped_g)
    shift_h_blue, shift_w_blue = find_optimal_shift_with_fourier(cropped_b, cropped_g)

    shift_h_red += g_coord[0] - int((1 + crop) * h / 3) + int(g.shape[0] * crop)
    shift_w_red += g_coord[1]
    shift_h_blue += g_coord[0] + int((1 + crop) * h / 3) - int(g.shape[0] * crop)
    shift_w_blue += g_coord[1]

    return img, (shift_h_red, shift_w_red), (shift_h_blue, shift_w_blue)
