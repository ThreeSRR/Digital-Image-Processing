import cv2
import os


export_dir = '../../results/binary'
if not os.path.isdir(export_dir):
    os.makedirs(export_dir)
img = cv2.imread("../../images/binary/6.bmp")

# 定义3x15竖直结构元
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
# 定义15x3水平结构元
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))

vertical = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1)
horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)

cv2.imshow("vertical", vertical)
cv2.waitKey()

cv2.imshow("horizontal", horizontal)
cv2.waitKey()

cv2.imwrite(os.path.join(export_dir, "6.png"), img)
cv2.imwrite(os.path.join(export_dir, "6_vertical.png"), vertical)
cv2.imwrite(os.path.join(export_dir, "6_horizontal.png"), horizontal)

