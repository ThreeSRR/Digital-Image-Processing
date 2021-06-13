import cv2
import os


export_dir = '../../results/gray'
if not os.path.isdir(export_dir):
    os.makedirs(export_dir)
img = cv2.imread("../../images/gray/3.jpg", cv2.IMREAD_GRAYSCALE)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow("img", open_img)
cv2.waitKey()

close_img = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, kernel)
cv2.imshow("img", close_img)
cv2.waitKey()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilate_img = cv2.dilate(close_img, kernel)
erode_img = cv2.erode(close_img, kernel)

gradient_img = dilate_img - erode_img
cv2.imshow("gradient_img", gradient_img)
cv2.waitKey()

cv2.imwrite(os.path.join(export_dir, "3.png"), img)
cv2.imwrite(os.path.join(export_dir, "3_result1.png"), close_img)
cv2.imwrite(os.path.join(export_dir, "3_result2.png"), gradient_img)