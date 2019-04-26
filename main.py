import numpy as np
import cv2


class Param:
    means = 0
    sigma = 10
    THETA = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    img_path = 'standard_test_images/lena_gray_512.tif'
    R = 5
    ALPHA = 1.2


def get_psnr(image1, image2):
    MSE = np.mean(np.power(image1 - image2, 2))
    return 20 * np.log10(255 / np.sqrt(MSE))


def gaussianNoise(image):
    noise_image = np.copy(image)
    noise = np.random.normal(Param.means, Param.sigma, image.shape)
    return noise_image + noise


def gaussian_fun(inputs, sigma):
    power = -inputs / (2 * sigma * sigma)
    return np.exp(power)


def get_index(num):
    if num != 0:
        return num
    else:
        return None


def calc_difference_linear(image, theta):
    x, y = theta[0], theta[1]
    diff_image = image[1:-1, 1:-1] - image[1 + y:get_index(-1 + y), 1 + x:get_index(-1 + x)]
    diff_image = np.power(diff_image, 2)
    # diff_image = np.cumsum(np.cumsum(diff_image, 0), 1)
    return diff_image


def calc_mhp(p_theta, x, y):
    mhp = p_theta[Param.R + y:get_index(-Param.R + y), Param.R + x:get_index(-Param.R + x)]
    return mhp


def denoising_mhp(image):
    padded_image = np.pad(image, Param.R + 1, 'symmetric')
    I_s = 0
    W_s = 0
    for theta in Param.THETA:
        # calc_difference p_theta
        p_theta_ = calc_difference_linear(padded_image, theta)
        # low pass filter p_theta
        p_theta = cv2.GaussianBlur(p_theta_, (7, 7), sigmaX=2)
        # distance_mhp
        distance_mhp = 0
        for step in range(1, Param.R + 1):
            x, y = step * theta[0], step * theta[1]
            mhp = calc_mhp(p_theta, x, y)
            # distance_mhp += calc_mhp(p_theta, step, theta)
            distance_mhp = np.maximum(distance_mhp, mhp) * Param.ALPHA
            if distance_mhp.any() > Param.T0:
                break
            w = gaussian_fun(distance_mhp, Param.sigma)
            W_s += w
            I_s += padded_image[Param.R + 1 + y:get_index(-Param.R - 1 + y),
                   Param.R + 1 + x:get_index(-Param.R - 1 + x)] * w
    return I_s / W_s


ori_image = cv2.imread(Param.img_path, 0)
ori_image = ori_image.astype(np.float32)
height, width = ori_image.shape
Param.T0 = 3 * np.std(ori_image)

# noising
noise_image = gaussianNoise(ori_image)

# denoising
denoise_image = denoising_mhp(noise_image)

# psnr
print('psnr ori_image and noise_image: {}'.format(get_psnr(ori_image, noise_image)))
print('psnr ori_image and noise_image: {}'.format(get_psnr(ori_image, denoise_image)))

# show
cv2.namedWindow('ori_image', cv2.WINDOW_NORMAL)
cv2.imshow('ori_image', ori_image.astype(np.uint8))
cv2.namedWindow('noise_image', cv2.WINDOW_NORMAL)
cv2.imshow('noise_image', noise_image.astype(np.uint8))
cv2.namedWindow('denoise_image', cv2.WINDOW_NORMAL)
cv2.imshow('denoise_image', denoise_image.astype(np.uint8))
cv2.waitKey()

'''
# 定义添加椒盐噪声的函数
def PepperandSalt(src, percetage):
    NoiseImg = np.copy(src)
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        if random.randint(0, 1) <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


# 定义添加高斯噪声的函数
def GaussianNoise(src, means, sigma):
    NoiseImg = np.copy(src)
    create_noise = np.random.normal(means, sigma, src.shape)
    return NoiseImg + create_noise

    # NoiseImg = np.copy(src)
    # NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    # for i in range(NoiseNum):
    #     randX = random.randint(0, src.shape[0] - 1)
    #     randY = random.randint(0, src.shape[1] - 1)
    #     NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
    #     if NoiseImg[randX, randY] < 0:
    #         NoiseImg[randX, randY] = 0
    #     elif NoiseImg[randX, randY] > 255:
    #         NoiseImg[randX, randY] = 255
    # return NoiseImg


def get_psnr(image1, image2):
    MSE = np.mean(np.power(image1 - image2, 2))
    return 20 * np.log10(255 / np.sqrt(MSE))


def calc_difference(image, theta):
    diff_image = np.empty(image.shape)
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            if 0 <= x + theta[0] < width and 0 <= y + theta[1] < height:
                diff_image[y, x] = np.power(image[y, x] - image[y + theta[1], x + theta[0]], 2)
            else:
                diff_image[y, x] = 0
    return diff_image


def calc_mhp(p_theta, step, theta):
    diff_image = np.zeros(p_theta.shape)
    height, width = p_theta.shape
    for y in range(height):
        for x in range(width):
            if 0 <= x + step * theta[0] < width and 0 <= y + step * theta[1] < height:
                diff_image[y, x] = p_theta[y + step * theta[1], x + step * theta[0]]
    return diff_image


def gaussian_fun(inputs, sigma):
    power = -inputs / (2 * sigma * sigma)
    return np.exp(power)


def update_Is(image, I_s, weight, step, theta):
    height, width = I_s.shape
    for y in range(height):
        for x in range(width):
            if 0 <= x + step * theta[0] < width and 0 <= y + step * theta[1] < height:
                I_s[y, x] += image[y, x] * weight[y, x]
    return I_s


def update_W_I(W_s, I_s, weight, step, theta):
    # update weight
    W_s += weight
    # update image
    image = np.zeros(I_s.shape)
    height, width = I_s.shape
    for y in range(height):
        for x in range(width):
            image[y, x] = I_s[y, x]
            if 0 <= x + step * theta[0] < width and 0 <= y + step * theta[1] < height:
                image[y, x] += I_s[y + step * theta[1], x + step * theta[0]] * weight[y][x]
    return W_s, image


# if __name__ == '__main__':
#     import cv2
#     import os
#
#     img_path = 'standard_test_images'
#     # initalize
#     ori_I = cv2.imread(os.path.join(img_path, 'lena_gray_256.tif'), 0)
#     # out_I = GaussianNoise(ori_I, 0, 10, 0.8)
#     out_I = PepperandSalt(ori_I, 0.1)
#
#     cv2.namedWindow('ori', cv2.WINDOW_NORMAL)
#     cv2.imshow('ori', ori_I)
#     cv2.namedWindow('noise', cv2.WINDOW_NORMAL)
#     cv2.imshow('noise', out_I.astype(np.uint8))
#     cv2.waitKey()

# parameters
THETA = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
R = 5
SIGMA = 1
ALPHA = 1.2
img_path = 'standard_test_images'

# initialize
image = cv2.imread(os.path.join(img_path, 'lena_gray_256.tif'), 0)
image = image.astype(np.float32)
T0 = 3 * np.std(image)
height, width = image.shape
post_I = np.empty(image.shape)
I_s = np.zeros(image.shape)
W_s = np.zeros(image.shape)

# noising
# ori_I = image
# ori_I = PepperandSalt(image, 0.01)
ori_I = GaussianNoise(image, 0, 2)

# denoising
for theta in THETA:
    # calc_difference p_theta
    p_theta_ = calc_difference(ori_I, theta)
    # low pass filter p_theta
    p_theta = cv2.GaussianBlur(p_theta_, (7, 7), sigmaX=2)
    # distance_mhp
    distance_mhp = np.zeros(ori_I.shape)
    for step in range(1, R + 1):
        # distance_mhp += calc_mhp(p_theta, step, theta)
        distance_mhp = np.maximum(distance_mhp, calc_mhp(p_theta, step, theta)) * ALPHA
        if distance_mhp.any() > T0:
            break
        weight = cv2.GaussianBlur(distance_mhp, (7, 7), sigmaX=2)
        # update W_s and I_s1
        W_s += weight
        update_Is(ori_I, I_s, weight, step, theta)
result = I_s / W_s
print(get_psnr(image, ori_I))
print(get_psnr(image, result))

# show
cv2.namedWindow('ori', cv2.WINDOW_NORMAL)
cv2.imshow('ori', ori_I.astype(np.uint8))
cv2.namedWindow('out', cv2.WINDOW_NORMAL)
cv2.imshow('out', result.astype(np.uint8))
cv2.waitKey()
'''
