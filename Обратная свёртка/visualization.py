import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from deconvolution import (
    compute_psnr,
    gaussian_kernel,
    inverse_filtering,
    wiener_filtering,
)


def visualize_images(figsize, images, titles, output_name):
    plt.figure(figsize=figsize)
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image, cmap="gray", vmin=0, vmax=1)
        plt.title(titles[i])
    plt.savefig(output_name, dpi=200, bbox_inches="tight")
    plt.close()


def vis_inverse(original_img, blurred_img, noisy_img):
    kernel = gaussian_kernel(size=15, sigma=5)
    restored_img = inverse_filtering(blurred_img, kernel)
    images = [blurred_img, restored_img, original_img]
    titles = [
        "Размытое изображение",
        "Восстановленное изображение",
        "Оригинальное изображение",
    ]
    visualize_images(
        figsize=(15, 10),
        images=images,
        titles=titles,
        output_name="inverse_filtering_blurred.jpg",
    )

    images = [noisy_img]
    ths = [0.0, 0.0001, 0.001, 0.01, 0.1]
    images += [inverse_filtering(noisy_img, kernel, threshold=th) for th in ths]
    titles = ["Искаженное изображение + шум"] + [f"{threshold=}" for threshold in ths]
    visualize_images(
        figsize=(20, 10),
        images=images,
        titles=titles,
        output_name="inverse_filtering_noisy.jpg",
    )


def vis_wiener(original_img, noisy_img):
    kernel = gaussian_kernel(size=15, sigma=5)
    filtered_img = wiener_filtering(noisy_img, kernel)
    psnr = compute_psnr(filtered_img, original_img) - compute_psnr(
        noisy_img, original_img
    )
    print(f"PSNR difference = {psnr:.3f}")

    images = [noisy_img, filtered_img, original_img]
    titles = [
        "Искаженное изображение + шум",
        "Винеровская фильтрация",
        "Оригинальное изображение",
    ]
    visualize_images(
        figsize=(12, 10),
        images=images,
        titles=titles,
        output_name="wiener_filtering_noisy.jpg",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        metavar="mode",
        choices=["inverse", "wiener"],
        help="Visualize inverse or wiener filtering",
    )
    args = parser.parse_args()

    basedir = "examples"
    original_img = np.load(os.path.join(basedir, "original.npy"))
    noisy_img = np.load(os.path.join(basedir, "noisy.npy"))

    if args.mode == "inverse":
        blurred_img = np.load(os.path.join(basedir, "blurred.npy"))
        vis_inverse(original_img, blurred_img, noisy_img)
    else:
        vis_wiener(original_img, noisy_img)


if __name__ == "__main__":
    main()
