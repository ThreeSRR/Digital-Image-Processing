import cv2
import os

GaussianNoise_path = './img/Gaussian_Noise.jpg'
SaltPepperNoise_path = './img/SaltPepper_Noise.jpg'

# 均值滤波
def meanfilter(img, kernel_size=5):
    return cv2.blur(img, (kernel_size, kernel_size))


# 高斯平滑
def guassianfilter(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


# 中值滤波
def medianfilter(img, kernel_size=5):
    return cv2.medianBlur(img, kernel_size)


def img_process(img_path, export_path, filter, kernel_size):

    filterlabel = {
        'mean': meanfilter,
        'guassian': guassianfilter,
        'median': medianfilter
    }

    assert filter in filterlabel, 'filter name ERROR'

    img_name = os.path.split(img_path)[-1].split('.')[0]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_img = filterlabel[filter](img, kernel_size)
    cv2.imwrite(os.path.join(export_path, '%s_%s.jpg' % (img_name, filter)), new_img)



if __name__ == '__main__':

    export_path = './hw1-1'
    if not os.path.isdir(export_path):
        os.mkdir(export_path)

    filters = ['mean', 'guassian', 'median']
    for filter in filters:
        img_process(GaussianNoise_path, export_path, filter, kernel_size=5)
        img_process(SaltPepperNoise_path, export_path, filter, kernel_size=5)
