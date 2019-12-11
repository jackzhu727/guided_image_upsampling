import numpy as np


def calc_rmse(a, b, minmax, mask):
    a = a[6:-6, 6:-6]
    b = b[6:-6, 6:-6]
    mask = mask[6:-6, 6:-6]
    a = a*(minmax[1]-minmax[0]) + minmax[1]
    b = b*(minmax[1]-minmax[0]) + minmax[1]
    return np.sqrt((np.power(a-b, 2) * mask / mask.sum()).sum())


def modcrop(image, modulo):
    h, w = image.shape[0], image.shape[1]
    h = h - h % modulo
    w = w - w % modulo
    return image[:h,:w]


def merge_img(imgs):
    height, width = imgs[0].shape[:2]
    CROP_H_SIZE = 4
    CROP_W_SIZE = 4
    height, width = (height * CROP_H_SIZE, width * CROP_W_SIZE)
    img = np.zeros((height, width))
    idx = 0
    for ih in range(CROP_H_SIZE):
        for iw in range(CROP_W_SIZE):
            img[ih::CROP_H_SIZE, iw::CROP_W_SIZE] = imgs[idx]
            idx += 1
    return img

def crop_img(img):
    imgs = []
    height, width = img.shape[:2]
    CROP_H_SIZE = 4
    CROP_W_SIZE = 4
    height //= CROP_H_SIZE
    width //= CROP_W_SIZE
    for ih in range(CROP_H_SIZE):
        for iw in range(CROP_W_SIZE):
            if len(img.shape) == 3:
                imgs.append(img[ih::CROP_H_SIZE, iw::CROP_W_SIZE, :])
            else:
                imgs.append(img[ih::CROP_H_SIZE, iw::CROP_W_SIZE])
    return imgs


def rmse_two_pic(gt, inference, masks):
    return calc_rmse(gt, inference, [gt.min(), gt.max()], merge_img([x[0].cpu().numpy() for x in masks]))
