import os
import cv2
import numpy as np
import argparse
from skimage import filters
from skimage.filters import prewitt
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import copy

def Robert(img):

    Gx = np.array([[-1, 0], [0, 1]])
    Gy = np.array([[0, -1], [1, 0]])
    Ix = cv2.filter2D(img, -1, Gx)
    Iy = cv2.filter2D(img, -1, Gy)
    grad = (np.abs(Ix) + np.abs(Iy)).astype(np.float32)

    return grad


def Sobel(img):
    
    grad_X=cv2.Sobel(img, -1, 1, 0)
    grad_Y=cv2.Sobel(img, -1, 0, 1)
    grad=cv2.addWeighted(grad_X, 0.5, grad_Y, 0.5, 0)

    return grad


def Prewitt(img):
    Gx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    Gy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    Ix = cv2.filter2D(img, -1, Gx)
    Iy = cv2.filter2D(img, -1, Gy)
    grad = (np.abs(Ix) + np.abs(Iy)).astype(np.float32)

    return grad


def Lapacian(img, kernel_size=3):
    res_I = cv2.Laplacian(img.astype(np.float32), -1, ksize = kernel_size)
    return res_I


def zero_cross(lap):

    lap = np.pad(lap, ((0, 1), (0, 1)), 'constant', constant_values = (0, 0))
    diff_x = lap[:-1, :-1] - lap[:-1, 1:] < 0
    diff_y = lap[:-1, :-1] - lap[1:, :-1] < 0

    edges =  np.logical_or(diff_x, diff_y).astype(np.float32)

    return edges


def interpolate(img, lap, border_mask):
    boundary_value = []
    for i in range(border_mask.shape[0]):
        for j in range(border_mask.shape[1]):
            if border_mask[i][j] == 1:
                if j + 1 < border_mask.shape[1] and lap[i][j] * lap[i][j+1] < 0:
                    a = lap[i][j]
                    b = lap[i][j+1]
                    boundary_value.append(img[i][j] * abs(b/(b-a)) + img[i][j+1] * abs(a/(b-a)))
                if i + 1 < border_mask.shape[0] and lap[i][j] * lap[i+1][j] < 0:
                    a = lap[i][j]
                    b = lap[i+1][j]
                    boundary_value.append(img[i][j] * abs(b/(b-a)) + img[i+1][j] * abs(a/(b-a)))

    return boundary_value



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold',
                        default=8,
                        type=int)
    parser.add_argument('--operator',
                        default='sobel',
                        type=str)

    args = parser.parse_args()
    GRAD_THRE = args.threshold
    grad_operator = args.operator.lower()

    mapping = {'robert': Robert, 'sobel': Sobel, 'prewitt': Prewitt}

    dirpath = "./multi-thres"
    file = '23.bmp'
    file_path = os.path.join(dirpath, file)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    grad = mapping[grad_operator](img)
    grad_mask = grad > 2 * GRAD_THRE

    lap = Lapacian(img)
    zero_cross_mask = zero_cross(lap)

    # zero_cross and gradient threshold
    border_mask = zero_cross_mask * grad_mask

    boundary_values = interpolate(img, lap, border_mask)

    # 查看边缘采样点灰度直方图
    boundary = (img * border_mask).ravel()
    boundary = boundary[boundary!=0]

    plt.hist(boundary, bins=255, range=(0,255))
    plt.title('histogram of discrete sampling points of boundaries')
    plt.show()
    
    # calculate the mean of gray values of each cluster
    boundary_values = np.array(boundary)
    peak1 = boundary_values[boundary_values < 10]
    peak2 = boundary_values[boundary_values >= 10]
    peak2 = peak2[peak2 < 90]
    peak3 = boundary_values[boundary_values >= 90]
    threshold1 = sum(peak1) / len(peak1)
    threshold2 = sum(peak2) / len(peak2)
    threshold3 = sum(peak3) / len(peak3)

    print('threshold1 of %s: %d' % (file, threshold1))
    print('threshold2 of %s: %d' % (file, threshold2))
    print('threshold3 of %s: %d' % (file, threshold3))
    
    # 分割背景
    _, binary_seg = cv2.threshold(img, threshold1, 255, cv2.THRESH_BINARY)
    cv2.imwrite('res_%s_1.jpg' % (file.split('.')[0]), binary_seg)
    
    # 分割结缔组织
    img2 = copy.deepcopy(img)
    img2[img2>threshold2] = 0
    _, binary_seg = cv2.threshold(img2, threshold1, 255, cv2.THRESH_BINARY)
    cv2.imwrite('res_%s_2.jpg' % (file.split('.')[0]), binary_seg)

    # 分割肌肉
    img3 = copy.deepcopy(img)
    img3[img3 > threshold3] = 0
    _, binary_seg = cv2.threshold(img3, threshold2, 255, cv2.THRESH_BINARY)
    cv2.imwrite('res_%s_3.jpg' % (file.split('.')[0]), binary_seg)
    
    # 分割骨头
    _, binary_seg = cv2.threshold(img, threshold3, 255, cv2.THRESH_BINARY)
    cv2.imwrite('res_%s_4.jpg' % (file.split('.')[0]), binary_seg)


