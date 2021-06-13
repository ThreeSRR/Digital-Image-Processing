import cv2
import os

export_dir = '../../results/binary'
if not os.path.isdir(export_dir):
    os.makedirs(export_dir)
img = cv2.imread("../../images/binary/1.bmp")

# 指定结构元
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))

# 膨胀
dilate_img = cv2.dilate(img, kernel)
# 腐蚀
erode_img = cv2.erode(img, kernel)
# 开操作
open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# 闭操作
close_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow("dilate", dilate_img)
cv2.waitKey()

cv2.imshow("erode", erode_img)
cv2.waitKey()

cv2.imshow("open", open_img)
cv2.waitKey()

cv2.imshow("close", close_img)
cv2.waitKey()

cv2.imwrite(os.path.join(export_dir, "1.png"), img)
cv2.imwrite(os.path.join(export_dir, "1_dilate.png"), dilate_img)
cv2.imwrite(os.path.join(export_dir, "1_erode.png"), erode_img)
cv2.imwrite(os.path.join(export_dir, "1_open.png"), open_img)
cv2.imwrite(os.path.join(export_dir, "1_close.png"), close_img)