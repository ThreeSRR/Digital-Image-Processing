import cv2
import os
import numpy as np
import argparse

data_dir = './imgs/gaussian_smoothing'


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
    grad = cv2.convertScaleAbs(grad)
    return grad


# 高斯平滑
def guassianfilter(img, kernel_size):
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    img = cv2.convertScaleAbs(img)
    return img

# LoG算子
def LoG(img, kernel_size=3):
    blur_I = guassianfilter(img, kernel_size)
    return Laplacian(blur_I)

# Canny算子
def Canny(img, t1, t2, kernel_size=3):
    blur_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    edge = cv2.Canny(blur_img, t1, t2)
    return edge


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--op',
                        default='sobel',
                        type=str)
    parser.add_argument('--t1',
                        default=50,
                        type=int)
    parser.add_argument('--t2',
                        default=150,
                        type=int)
    parser.add_argument('--T',
                        default=40,
                        type=int)

    args = parser.parse_args()
    operator = args.op.lower()
    t1 = args.t1
    t2 = args.t2
    T = args.T

    export_dir = './result'
    if not os.path.isdir(export_dir):
        os.mkdir(export_dir)

    for img in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img)
        img_name = img.split('.')[0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if operator == 'sobel':
            new_img = Sobel(img, kernel_size=3)
        elif operator == 'laplacian':
            new_img = Laplacian(img)
        elif operator == 'log':
            new_img = LoG(img)
        elif operator == 'canny':
            new_img = Canny(img, t1, t2)

        _, edge = cv2.threshold(new_img, T, 255, cv2.THRESH_BINARY)

        cv2.imshow(img_path, edge)
        cv2.waitKey()

        cv2.imwrite(os.path.join(export_dir, '%s_%s_%d.jpg' % (img_name, operator, T)), edge)

