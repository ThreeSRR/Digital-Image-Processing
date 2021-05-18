import cv2
import os

SaltPepperNoise_path = './img/SaltPepper_Noise.jpg'


# 中值滤波
def medianfilter(img, kernel_size):
    return cv2.medianBlur(img, kernel_size)


def img_process(img_path, export_path, kernel_size):
    img_name = os.path.split(img_path)[-1].split('.')[0]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_img = medianfilter(img, kernel_size)
    cv2.imwrite(os.path.join(export_path, '%s_median_kernelsize_%s.jpg' % (img_name, kernel_size)), new_img)


if __name__ == '__main__':

    export_path = './hw1-4'
    if not os.path.isdir(export_path):
        os.mkdir(export_path)

    kernel_sizes = [3, 5, 7, 9]
    for kernel_size in kernel_sizes:
        img_process(SaltPepperNoise_path, export_path, kernel_size)
        