import cv2
import os

export_dir = '../../results/binary'
if not os.path.isdir(export_dir):
    os.makedirs(export_dir)
img = cv2.imread("../../images/binary/9.bmp")

# 定义圆形结构元
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

close_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow("result", close_img)
cv2.waitKey()

cv2.imwrite(os.path.join(export_dir, "9.png"), img)
cv2.imwrite(os.path.join(export_dir, "9_result.png"), close_img)