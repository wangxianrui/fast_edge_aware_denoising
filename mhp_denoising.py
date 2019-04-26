import numpy as np
import cv2
from utils import *


class Param:
    means = 0
    sigma = 10
    img_path = 'standard_test_images/lena_gray_512.tif'
    patch_size = 1
    window_size = 5


ori_image = cv2.imread(Param.img_path, 0)
ori_image = ori_image.astype(np.float32)
height, width = ori_image.shape

# noising
noise_image = gaussianNoise(ori_image, Param.means, Param.sigma)

# denoising
denoise_image = denoising(noise_image, Param.patch_size, Param.window_size, Param.sigma)

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
