import cv2
import os
import numpy as np

export_dir = '../../results/gray'
if not os.path.isdir(export_dir):
    os.makedirs(export_dir)
img = cv2.imread("../../images/gray/2.png", cv2.IMREAD_GRAYSCALE)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow("open", open_img)
cv2.waitKey()


_, bin_res = cv2.threshold(open_img, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
result_img = bin_res * open_img

cv2.imshow("result", result_img)
cv2.waitKey()

cv2.imwrite(os.path.join(export_dir, "2.png"), img)
cv2.imwrite(os.path.join(export_dir, "2_open.png"), open_img)
cv2.imwrite(os.path.join(export_dir, "2_result.png"), result_img)
