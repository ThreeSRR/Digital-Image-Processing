import cv2
import os

export_dir = '../../results/binary'
if not os.path.isdir(export_dir):
    os.makedirs(export_dir)
img = cv2.imread("../../images/binary/4.bmp")

# 定义尺寸小于圆点直径但大于杆的宽度的圆形结构元
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

cv2.imshow("result", open_img)
cv2.waitKey()

cv2.imwrite(os.path.join(export_dir, "4.png"), img)
cv2.imwrite(os.path.join(export_dir, "4_result.png"), open_img)