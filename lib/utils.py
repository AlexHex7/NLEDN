import numpy as np
from skimage.measure import compare_ssim as ssim_func
from skimage.measure import compare_psnr as psnr_func


def calc_psnr(img1, img2):
    y1 = np.array(img1.convert('YCbCr'))[:, :, 0]
    y2 = np.array(img2.convert('YCbCr'))[:, :, 0]
    psnr = psnr_func(y2, y1)
    return psnr


def calc_ssim(img1, img2):
    y1 = np.array(img1.convert('YCbCr'))[:, :, 0]
    y2 = np.array(img2.convert('YCbCr'))[:, :, 0]
    ssim = ssim_func(y1, y2)
    return ssim
