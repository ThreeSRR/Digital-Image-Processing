import cv2
import os

export_dir = '../../results/binary'
if not os.path.isdir(export_dir):
    os.makedirs(export_dir)
img = cv2.imread("../../images/binary/8.jpg")

# 定义矩形结构元
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))

open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
close_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

top_hat = img - open_img
black_hat = close_img - img

cv2.imshow("top_hat", top_hat)
cv2.waitKey()

cv2.imshow("black_hat", black_hat)
cv2.waitKey()

cv2.imwrite(os.path.join(export_dir, "8.png"), img)
cv2.imwrite(os.path.join(export_dir, "8_res1.png"), top_hat)
cv2.imwrite(os.path.join(export_dir, "8_res2.png"), black_hat)