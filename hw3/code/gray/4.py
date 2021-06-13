import cv2
import os

export_dir = '../../results/gray'
if not os.path.isdir(export_dir):
    os.makedirs(export_dir)
img = cv2.imread("../../images/gray/4.jpg", cv2.IMREAD_GRAYSCALE)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
close_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

res1 = img - open_img
res2 = close_img - img

_, bin_res1 = cv2.threshold(res1, 6, 1, cv2.THRESH_BINARY)
_, bin_res2 = cv2.threshold(res2, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

res1_img = bin_res1 * img
res2_img = bin_res2 * 255

cv2.imshow("res1", res1_img)
cv2.waitKey()

cv2.imshow("res2", res2_img)
cv2.waitKey()

cv2.imwrite(os.path.join(export_dir, "4.png"), img)
cv2.imwrite(os.path.join(export_dir, "4_result1.png"), res1_img)
cv2.imwrite(os.path.join(export_dir, "4_result2.png"), res2_img)