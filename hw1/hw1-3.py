import cv2
import os

GaussianSmoothing_path = './img/Gaussian_smoothing.jpg'
GaussianNoise_path = './img/Gaussian_Noise.jpg'


def get_GaussianKernel(kernel_size):
    k = cv2.getGaussianKernel(kernel_size, 0)
    return k * k.transpose()


# 高斯平滑
def guassianfilter(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def img_process(img_path, export_path, kernel_size):

    img_name = os.path.split(img_path)[-1].split('.')[0]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_img = guassianfilter(img, kernel_size)
    cv2.imwrite(os.path.join(export_path, '%s_gaussian_kernelsize_%s.jpg' % (img_name, kernel_size)), new_img)


if __name__ == '__main__':

    export_path = './hw1-3'
    if not os.path.isdir(export_path):
        os.mkdir(export_path)

    kernel_sizes = [5, 9, 13, 15]
    for kernel_size in kernel_sizes:
        print(get_GaussianKernel(kernel_size))
        img_process(GaussianNoise_path, export_path, kernel_size)
        img_process(GaussianSmoothing_path, export_path, kernel_size)