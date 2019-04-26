import numpy as np


def get_psnr(image1, image2):
    MSE = np.mean(np.power(image1 - image2, 2))
    return 20 * np.log10(255 / np.sqrt(MSE))


def gaussianNoise(image, means, sigma):
    noise_image = np.copy(image)
    noise = np.random.normal(means, sigma, image.shape)
    return noise_image + noise


def gaussian_fun(inputs, sigma):
    power = -inputs / (2 * sigma * sigma)
    return np.exp(power)


def get_index(num):
    if num != 0:
        return num
    else:
        return None


def calc_difference(image, window_size, r, c):
    diff_image = image[window_size:-window_size, window_size:-window_size] - image[window_size + r:get_index(
        -window_size + r), window_size + c:get_index(-window_size + c)]
    diff_image = np.power(diff_image, 2)
    diff_image = np.cumsum(np.cumsum(diff_image, 0), 1)
    return diff_image


def denoising(image, patch_size, window_size, sigma):
    padded_image = np.pad(image, patch_size + window_size + 1, 'symmetric')
    padded_v = np.pad(image, window_size, 'symmetric')
    I_s = 0
    wmax = 0
    W_s = 0
    for r in range(-window_size, window_size + 1):
        for c in range(-window_size, window_size + 1):
            if not r and not c:
                continue
            diff_image = calc_difference(padded_image, window_size, r, c)
            diff_image = diff_image[2 * patch_size + 1:-1, 2 * patch_size + 1:-1] \
                         + diff_image[:-2 * patch_size - 2, :-2 * patch_size - 2] \
                         - diff_image[2 * patch_size + 1:-1, :-2 * patch_size - 2] \
                         - diff_image[:-2 * patch_size - 2, 2 * patch_size + 1:-1]
            diff_image = diff_image / np.power(2 * patch_size + 1, 2)
            w = gaussian_fun(diff_image, sigma)
            v = padded_v[window_size + r:get_index(-window_size + r), window_size + c:get_index(-window_size + c)]
            I_s += w * v
            wmax = np.maximum(wmax, w)
            W_s += w
    I_s += wmax * image
    I_s /= (wmax + W_s)
    return I_s
