import cv2
import os
import numpy as np
import argparse

data_dir = './imgs/noise'


# Sobel算子
def Sobel(img, kernel_size=3):
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    Ix = cv2.convertScaleAbs(Ix)
    Iy = cv2.convertScaleAbs(Iy)
    res_I = cv2.addWeighted(Ix, 0.5, Iy, 0.5, 0)
    return res_I

# Laplacian算子
def Laplacian(img, kernel_size=3):
    grad = cv2.Laplacian(img, -1, ksize = kernel_size)
    return grad.astype(np.uint8)


# 高斯平滑
def guassianfilter(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# LoG算子
def LoG(img, kernel_size=3):
    blur_I = guassianfilter(img, kernel_size)
    return Laplacian(blur_I)

# Canny算子
def Canny(img, t1, t2, kernel_size=3):
    blur_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    edge = cv2.Canny(blur_img, t1, t2)
    return edge

# 中值滤波
def medianfilter(img, kernel_size=11):
    return cv2.medianBlur(img, kernel_size)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--op',
                        default='sobel',
                        type=str)

    args = parser.parse_args()
    operator = args.op.lower()

    export_dir = './result'
    if not os.path.isdir(export_dir):
        os.makedirs(export_dir)

    img = 'salt_pepper_noise.jpg'
    img_path = os.path.join(data_dir, img)
    img_name = img.split('.')[0]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    new_img = medianfilter(img)

    if operator == 'sobel':
        new_img = Sobel(new_img, kernel_size=3)
    elif operator == 'laplacian':
        new_img = Laplacian(new_img)
    elif operator == 'log':
        new_img = LoG(new_img)
    elif operator == 'canny':
        new_img = Canny(new_img, t1=50, t2=150, kernel_size=3)


    thres, edge = cv2.threshold(new_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imshow(img_path+str(thres), edge)
    cv2.waitKey()

    cv2.imwrite(os.path.join(export_dir, '%s_%s_%d.jpg' % (img_name, operator, thres)), edge)
