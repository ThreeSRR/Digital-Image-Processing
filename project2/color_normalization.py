import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image


def get_object_mask(img_I):
    mask = (img_I > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def get_mean_std(img, mask):
    img = img * np.stack([mask, mask, mask], -1)
    mean = np.mean(np.array(img), axis=(0, 1))
    var = np.std(np.array(img), axis=(0, 1))
    return mean, var


def get_reference(ref_file):
    img = cv2.imread(ref_file)
    img_I = np.mean(img, axis=-1)
    img_I = img_I.astype(np.float32)
    mask = get_object_mask(img_I)
    ref_mean, ref_var = get_mean_std(img, mask)

    return ref_mean, ref_var


def color_transform(img, ref_mean, ref_var):
    img = img.astype(np.float32)
    img_I = np.mean(img, axis=-1)
    mask = get_object_mask(img_I)
    mean, var = get_mean_std(img, mask)
    img_c = (img - mean) / var * ref_var + ref_mean
    img_c = img_c * np.stack([mask, mask, mask], -1)
    return img_c


def color_normalization(img_path, mu_0, sigma_0):
    img = cv2.imread(img_path)

    img_I = np.mean(img, axis=-1)
    img_I = img_I.astype(np.float32)

    img_c = color_transform(img, mu_0, sigma_0)
    img_c = np.clip(img_c, 0, 255).astype(np.uint8)

    return img_c


if __name__ == "__main__":
    # REFERENCE_PATH = './reference.jpg'
    REFERENCE_PATH = './reference.tif'
    ref_mean, ref_var = get_reference(REFERENCE_PATH)

    # imgs = glob('./original_imgs/*.jpg')
    imgs = glob('./images/*.tif')
    export_dir = './color_normalization_results_drive/'
    if not os.path.isdir(export_dir):
        os.makedirs(export_dir)

    for img in tqdm(imgs):
        name = os.path.split(img)[-1].split('.')[0]
        new_img = color_normalization(img, ref_mean, ref_var)
        cv2.imwrite(os.path.join(export_dir, '%s.png' % name), new_img)