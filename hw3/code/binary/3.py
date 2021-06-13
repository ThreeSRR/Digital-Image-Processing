import cv2
import os


export_dir = '../../results/binary'
if not os.path.isdir(export_dir):
    os.makedirs(export_dir)
img = cv2.imread("../../images/binary/3.bmp")



kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 15))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))

dilate_img = cv2.dilate(img, kernel1)
cv2.imshow("step1", dilate_img)
cv2.waitKey()
erode_img = cv2.erode(dilate_img, kernel2)
cv2.imshow("step2", erode_img)
cv2.waitKey()
cv2.imwrite(os.path.join(export_dir, "3_result1.png"), erode_img)

dilate_img = cv2.dilate(erode_img, kernel3)
cv2.imshow("step3", dilate_img)
cv2.waitKey()
erode_img = cv2.erode(dilate_img, kernel2)
cv2.imshow("step4", erode_img)
cv2.waitKey()

cv2.imwrite(os.path.join(export_dir, "3.png"), img)
cv2.imwrite(os.path.join(export_dir, "3_result2.png"), erode_img)
