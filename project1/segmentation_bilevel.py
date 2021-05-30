import os
import cv2
import numpy as np
import argparse
from skimage import filters
from skimage.filters import prewitt
from matplotlib import pyplot as plt 

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
                        default=20,
                        type=int)
    parser.add_argument('--operator',
                        default='robert',
                        type=str)

    args = parser.parse_args()
    GRAD_THRE = args.threshold
    grad_operator = args.operator.lower()

    mapping = {'robert': Robert, 'sobel': Sobel, 'prewitt': Prewitt}

    dirpath = "./bilevel-thres"
    export_dir = './bilevel_res'
    if not os.path.isdir(export_dir):
        os.mkdir(export_dir)

    for file in os.listdir(dirpath):
        file_path = os.path.join(dirpath, file)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        grad = mapping[grad_operator](img)
        grad_mask = grad > 2 * GRAD_THRE

        lap = Lapacian(img)
        zero_cross_mask = zero_cross(lap)

         # zero_cross and gradient threshold
        border_mask = zero_cross_mask * grad_mask

        boundary_values = interpolate(img, lap, border_mask)

        # calculate the mean of gray values of points lying on the boundary
        threshold = sum(boundary_values) / len(boundary_values)

        print('threshold of %s: %d' % (file, threshold))

        _, binary_seg = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        cv2.imshow(file, binary_seg)
        cv2.waitKey()

        export_path = os.path.join(export_dir, 'res_%s_%s_grad_%d_t_%d.jpg' % (file.split('.')[0], grad_operator, GRAD_THRE, threshold))
        cv2.imwrite(export_path, binary_seg)



