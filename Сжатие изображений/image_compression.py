import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
from scipy.fftpack import dct as scipy_dct, idct as scipy_idct


# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """

    # Your code here
    # Отцентруем каждую строчку матрицы
    matrix_copy = matrix.copy()
    mean_values = np.mean(matrix_copy, axis=1)
    matrix_copy -= mean_values[:, None]
    # Найдем матрицу ковариации
    cov = np.cov(matrix_copy)
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Посчитаем количество найденных собственных векторов
    num_eigenvalues = eigenvectors.shape[1]
    # Сортируем собственные значения в порядке убывания
    indices = np.argsort(eigenvalues)[::-1]
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    sorted_eigenvectors = eigenvectors[:, indices]
    # Оставляем только p собственных векторов
    sorted_eigenvectors = sorted_eigenvectors[:, :p]
    # Проекция данных на новое пространство
    projection = np.dot(sorted_eigenvectors.T, matrix_copy)
    return sorted_eigenvectors, projection, mean_values


def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """
    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!
        channel = np.dot(comp[0], comp[1]) + comp[2][:, None]
        channel = np.clip(channel, 0, 255)
        result_img.append(channel.astype(np.uint8))
    return np.stack(result_img, axis=2)


def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            compressed.append(pca_compression(img[..., j].astype(np.float64), p))
        img_compressed = pca_decompression(compressed)
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))
        axes[i // 3, i % 3].imshow(img_compressed)

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331364 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack((y, cb, cr), axis=2)


def ycbcr2rgb(img):
    """ Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """
    y = img[..., 0]
    cb = img[..., 1]
    cr = img[..., 2]
    r = y + 1.402 * (cr - 128)
    g = y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return np.stack((r, g, b), axis=2)


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ycbcr = rgb2ycbcr(rgb_img)
    y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    cb = gaussian_filter(cb, sigma=5, radius=10)
    cr = gaussian_filter(cr, sigma=5, radius=10)
    rgb_img = ycbcr2rgb(np.stack((y, cb, cr), axis=2))
    rgb_img = rgb_img.astype(np.uint8)
    plt.imshow(rgb_img)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ycbcr = rgb2ycbcr(rgb_img)
    y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    y = gaussian_filter(y, sigma=5, radius=10)
    rgb_img = ycbcr2rgb(np.stack((y, cb, cr), axis=2))
    rgb_img = rgb_img.astype(np.uint8)
    plt.imshow(rgb_img)
    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """
    return gaussian_filter(component, 10.0)[::2, ::2]


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """
    return scipy_dct(scipy_dct(block.T, norm='ortho').T, norm='ortho')


# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """
    return np.round(block / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    scale_factor = 1
    if 1 <= q < 50:
        scale_factor = 5000.0 / q
    elif 50 <= q <= 99:
        scale_factor = 200 - 2 * q

    custom_quantization_matrix = np.floor((default_quantization_matrix * scale_factor + 50) / 100)
    custom_quantization_matrix[custom_quantization_matrix == 0] = 1
    return custom_quantization_matrix


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    reversed_diagonals = []
    for i in range(-7, 8):
        reversed_diagonals.append(np.diagonal(block[::-1, :], i)[::(2 * (i % 2) - 1)])
    return np.concatenate(reversed_diagonals)


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """

    compressed = []
    zeros_num = 0
    for num in zigzag_list:
        if num == 0:
            if zeros_num == 0:
                compressed.append(num)
            zeros_num += 1
        else:
            if zeros_num != 0:
                compressed.append(zeros_num)
            compressed.append(num)
            zeros_num = 0
    if zeros_num != 0:
        compressed.append(zeros_num)
    return compressed


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """
    # Переходим из RGB в YCbCr
    ycbcr_img = rgb2ycbcr(img).astype(np.float64)

    # Уменьшаем цветовые компоненты
    downsampled_colors = []
    for i in (1, 2):
        downsampled_colors.append(downsampling(ycbcr_img[:, :, i]))

    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    y_blocks = []
    cb_blocks = []
    cr_blocks = []
    block_size = 8

    # For y component
    for j in range(ycbcr_img.shape[0] // block_size):
        for i in range(ycbcr_img.shape[1] // block_size):
            y_blocks.append(
                ycbcr_img[j * block_size:(j + 1) * block_size, i * block_size:(i + 1) * block_size, 0] - 128)

    # For color ccomponents
    for j in range(ycbcr_img.shape[0] // (2 * block_size)):
        for i in range(ycbcr_img.shape[1] // (2 * block_size)):
            cb_blocks.append(
                downsampled_colors[0][j * block_size:(j + 1) * block_size, i * block_size:(i + 1) * block_size] - 128)
            cr_blocks.append(
                downsampled_colors[1][j * block_size:(j + 1) * block_size, i * block_size:(i + 1) * block_size] - 128)

    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    y_blocks = [compression(zigzag(quantization(dct(block), quantization_matrixes[0]))) for block in y_blocks]
    cb_blocks = [compression(zigzag(quantization(dct(block), quantization_matrixes[1]))) for block in cb_blocks]
    cr_blocks = [compression(zigzag(quantization(dct(block), quantization_matrixes[1]))) for block in cr_blocks]
    return [y_blocks, cb_blocks, cr_blocks]


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """
    decompressed = []
    i = 0
    while i < len(compressed_list):
        if compressed_list[i] == 0:
            zeros_num = compressed_list[i + 1]
            decompressed += [0] * zeros_num
            i += 1
        else:
            decompressed.append(compressed_list[i])
        i += 1
    return decompressed


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """
    orig_indexes = np.array([
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63
    ])
    result = np.zeros(len(orig_indexes))
    for i in range(len(orig_indexes)):
        result[i] = input[orig_indexes[i]]
    output = result.reshape((8, 8))
    return output


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """
    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """
    return np.round(scipy_idct(scipy_idct(block.T, norm='ortho').T, norm='ortho'))


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """
    return np.repeat(np.repeat(component, 2, axis=1), 2, axis=0)


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """
    result[0] = [
        inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(block)), quantization_matrixes[0]))
        for block in result[0]]
    for i in (1, 2):
        result[i] = [
            inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(block)), quantization_matrixes[1]))
            for block in result[i]]

    # Перевод блоков из диапазона [-128, 127] назад в [0, 255] и объединение их в компоненты
    y_comp = np.zeros(result_shape[:2], dtype=np.float64)
    h, w = result_shape[:2]
    color_components = np.zeros((h // 2, w // 2, 2), dtype=np.float64)
    block_size = 8

    num = 0
    for j in range(h // block_size):
        for i in range(w // block_size):
            y_comp[j * block_size:(j + 1) * block_size, i * block_size:(i + 1) * block_size] = result[0][num] + 128
            num += 1

    num = 0
    for j in range(h // (2 * block_size)):
        for i in range(w // (2 * block_size)):
            color_components[
                j * block_size:(j + 1) * block_size,
                i * block_size: (i + 1) * block_size,
                0
            ] = result[1][num] + 128
            color_components[
                j * block_size:(j + 1) * block_size,
                i * block_size: (i + 1) * block_size,
                1
            ] = result[2][num] + 128
            num += 1

    # Увеличение цветовых компонент и их объединение в изображение
    upsampled_colors = np.zeros((h, w, 2), dtype=np.float64)
    for k in range(2):
        upsampled_colors[:, :, k] = upsampling(color_components[:, :, k])
    ycbcr = np.stack([y_comp, upsampled_colors[:, :, 0], upsampled_colors[:, :, 1]], axis=2)

    # Переход из YCbCr в RGB
    return ycbcr2rgb(ycbcr).astype(np.uint8)


def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 25, 50, 75, 100]):
        matrixes = [own_quantization_matrix(y_quantization_matrix, p),
                    own_quantization_matrix(color_quantization_matrix, p)]
        compressed = jpeg_compression(img, matrixes)
        decompressed = jpeg_decompression(compressed, img.shape, matrixes)

        axes[i // 3, i % 3].imshow(decompressed)
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))

        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])

    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')

    compressed = np.array(compressed, dtype=np.object_)
    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), compressed)
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))

    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')

    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == "__main__":
    pca_visualize()
    get_gauss_1()
    get_gauss_2()
    jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()
