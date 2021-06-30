import cv2
import numpy as np
import math
import os
from glob import glob
from tqdm import tqdm


def pre_process(image):
    kernel = np.ones((65, 65), np.uint8)
    result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return result


def RGB_count(img):
    # The number of '255's in the green channel, for example, is int(g[255]))
    (B, G, R) = cv2.split(img)
    b = cv2.calcHist([B], [0], None, [256], [0, 256])
    g = cv2.calcHist([G], [0], None, [256], [0, 256])
    r = cv2.calcHist([R], [0], None, [256], [0, 256])
    return np.reshape(r, (256,)), np.reshape(g, (256,)), np.reshape(b, (256,))


def maxEntropyChannel(img):
    (B, G, R) = cv2.split(img)
    b = np.reshape(cv2.calcHist([B], [0], None, [256], [0, 256]), (256,))
    g = np.reshape(cv2.calcHist([G], [0], None, [256], [0, 256]), (256,))
    r = np.reshape(cv2.calcHist([R], [0], None, [256], [0, 256]), (256,))
    total = img.shape[0] * img.shape[1]
    b /= total
    g /= total
    r /= total

    Hb, Hg, Hr = 0.0, 0.0, 0.0
    for i in range(256):
        if b[i] > 0:
            Hb -= b[i] * math.log(b[i], 2)
        if g[i] > 0:
            Hg -= g[i] * math.log(g[i], 2)
        if r[i] > 0:
            Hr -= r[i] * math.log(r[i], 2)
    # print(Hb, Hg, Hr)
    if Hr >= Hg and Hr >= Hb:
        return R
    if Hg >= Hb and Hg > Hr:
        return G
    return B


def vote(image, threshold, radius):
    shape = image.shape
    kernel = np.zeros((2 * radius, 2 * radius), dtype=np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1)
    _, image = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)
    return cv2.filter2D(image, -1, kernel)


def positionWithMostVotes(votes):
    shape = votes.shape
    most_votes = -1
    positions = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            if votes[i][j] > most_votes:
                most_votes = votes[i][j]
                positions = [np.array((i, j))]
            if votes[i][j] == most_votes:
                positions.append(np.array((i, j)))
    return np.average(positions, axis=0)


def find_centers(img):
    """

    :param img: file name of 4288*2848 image
    :return:OD Center and Fovea Center, in two arrays, like [1218.21865533  795.40400832] [1663.22873953 1954.15065074]

    """
    image = cv2.imread(img)
    shape = image.shape
    maxE_image = maxEntropyChannel(pre_process(image))
    ODCenter = positionWithMostVotes(vote(maxE_image, 254, 250))

    if ODCenter[1] > shape[1] / 2:
        FSR = maxE_image[max((int(ODCenter[0]) - 225), 0):min((int(ODCenter[0]) + 675), shape[0]),
              (int(ODCenter[1]) - 1237):(int(ODCenter[1]) - 675)]
        distribution = np.reshape(cv2.calcHist([FSR], [0], None, [256], [0, 256]), (256,))
        darkest_num = FSR.shape[0] * FSR.shape[1] / 100
        s = 0
        threshold = 0
        while s < darkest_num:
            s += distribution[threshold]
            threshold += 1
        positions = []
        for i in range(FSR.shape[0]):
            for j in range(FSR.shape[1]):
                if FSR[i][j] < threshold:
                    positions.append(np.array((i, j)))
        center = np.average(positions, axis=0)
        FCenter = np.array((max((int(ODCenter[0]) - 225), 0) + center[0], int(ODCenter[1]) - 1237 + center[1]))
    else:
        FSR = maxE_image[max((int(ODCenter[0]) - 225), 0):min((int(ODCenter[0]) + 675), shape[0]),
              (int(ODCenter[1]) + 675):(int(ODCenter[1]) + 1237)]
        distribution = np.reshape(cv2.calcHist([FSR], [0], None, [256], [0, 256]), (256,))
        darkest_num = FSR.shape[0] * FSR.shape[1] / 100
        s = 0
        threshold = 0
        while s < darkest_num:
            s += distribution[threshold]
            threshold += 1
        positions = []
        for i in range(FSR.shape[0]):
            for j in range(FSR.shape[1]):
                if FSR[i][j] < threshold:
                    positions.append(np.array((i, j)))
        center = np.average(positions, axis=0)
        FCenter = np.array((max((int(ODCenter[0]) - 225), 0) + center[0], int(ODCenter[1]) + 675 + center[1]))
    return ODCenter, FCenter


def transform(img, origin1, origin2, target1, target2):
    """

    :param img: image parameter of 4288*2848 image
    :param origin1: OD Center of the image being processed, 2d array like [1260.60227367  826.91868854]
    :param origin2: Fovea Center of the image being processed, 2d array
    :param target1: OD Center of the reference image, 2d array
    :param target2: Fovea Center of the reference image, 2d array
    :return:image parameter of the result of transformation process
    """
    result = img
    new_origin1, new_origin2 = origin1, origin2
    if (origin1[1] - origin2[1]) * (target1[1] - target2[1]) < 0:
        result = cv2.flip(result, 1)
        new_origin1[1] = 4287 - origin1[1]
        new_origin2[1] = 4287 - origin2[1]
    mat_translation = np.float32([[1, 0, int((target1[1] + target2[1] - new_origin1[1] - new_origin2[1]) / 2)],
                                  [0, 1, int((target1[0] + target2[0] - new_origin1[0] - new_origin2[0]) / 2)]])
    result = cv2.warpAffine(result, mat_translation, (4288, 2848))
    mat_rotation = cv2.getRotationMatrix2D(((target1[0] + target2[0]) * 0.5, (target1[1] + target2[1]) * 0.5),
                                           math.atan((new_origin1[1] - new_origin2[1]) / (
                                                   new_origin1[0] - new_origin2[0])) - math.atan(
                                               (target1[1] - target2[1]) / (target1[0] - target2[0])), math.sqrt(
            (target1[0] - target2[0]) * (target1[0] - target2[0]) + (target1[1] - target2[1]) * (
                    target1[1] - target2[1])) / math.sqrt(
            (new_origin1[0] - new_origin2[0]) * (new_origin1[0] - new_origin2[0]) + (
                    new_origin1[1] - new_origin2[1]) * (new_origin1[1] - new_origin2[1])))
    result = cv2.warpAffine(result, mat_rotation, (4288, 2848))
    return result


if __name__ == '__main__':
    imgs = glob('./color_normalization_results/*.png')
    export_dir = './dot_detection_and_align_results/'
    if not os.path.isdir(export_dir):
        os.makedirs(export_dir)

    O2, F2 = find_centers('reference.jpg')
    print('reference:', O2, F2)
    for img in tqdm(imgs):
        O1, F1 = find_centers(img)
        print(O1, F1)
        name = os.path.split(img)[-1].split('.')[0]
        new_img = transform(cv2.imread(img), O1, F1, O2, F2)
        cv2.imwrite(os.path.join(export_dir, '%s.png' % name), new_img)

        original_img = cv2.imread(img)
        cv2.circle(original_img, tuple(int(i) for i in O1[::-1]), 256, (255, 0, 0), 32)
        cv2.circle(original_img, tuple(int(i) for i in F1[::-1]), 256, (0, 0, 255), 32)
        cv2.imwrite(os.path.join(export_dir, '%s_detection.png' % name), original_img)
