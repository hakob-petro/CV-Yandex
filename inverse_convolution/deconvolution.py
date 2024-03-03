import numpy as np
from scipy.fft import fft2, ifftshift, ifft2, fftshift


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    x_center = size // 2
    y_center = size // 2
    if size % 2 == 0:
        x_center, y_center = (size - 1) / 2, (size - 1) / 2

    kernel = np.zeros((size, size))
    xv, yv = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    for x in xv:
        for y in yv:
            r_squared = (x - x_center) ** 2 + (y - y_center) ** 2
            kernel[x, y] = np.exp(-r_squared / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return kernel / np.sum(kernel)


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    pad_rows = shape[0] - h.shape[0]
    pad_cols = shape[1] - h.shape[1]
    if pad_rows % 2 != 0:
        pad_width = ((pad_rows // 2 + 1, pad_rows - pad_rows // 2 - 1), (pad_cols // 2 + 1, pad_cols - pad_cols // 2 - 1))
    else:
        pad_width = ((pad_rows // 2, pad_rows - pad_rows // 2), (pad_cols // 2, pad_cols - pad_cols // 2))
    return fft2(ifftshift(np.pad(h, pad_width)))


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    return np.where(np.absolute(H) > threshold, 1 / H, 0)


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    H = fourier_transform(h, blurred_img.shape)
    H_inv = inverse_kernel(H, threshold)
    G = fourier_transform(blurred_img, blurred_img.shape)
    F_tilda = G * H_inv
    f_tilda = ifft2(F_tilda)

    if blurred_img.shape[0] % 2 == 0:
        f_tilda = ifftshift(f_tilda)
    else:
        f_tilda = fftshift(f_tilda)
    return np.absolute(f_tilda)


def wiener_filtering(blurred_img, h, K=0.00005):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    H = fourier_transform(h, blurred_img.shape)
    H_conj = np.conjugate(H)
    H_kvadr_mod = H * H_conj
    G = fourier_transform(blurred_img, blurred_img.shape)
    F_curved = H_conj / (H_kvadr_mod + K) * G
    f_curved = ifft2(F_curved)

    if blurred_img.shape[0] % 2 == 0:
        f_curved = ifftshift(f_curved)
    else:
        f_curved = fftshift(f_curved)
    return np.absolute(f_curved)


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        raise ValueError("MSE is 0, can't compute PSNR")
    return 10 * np.log10(255 ** 2 / mse)
