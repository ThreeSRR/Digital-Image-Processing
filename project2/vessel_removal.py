from re import I
import cv2
from PIL import Image
import numpy as np
import imageio

def repairimg(imgpath,maskpath):
    gifmask = imageio.mimread(maskpath)
    imgmask = np.array(gifmask[0],dtype=np.uint8)
    img = cv2.imread(imgpath)
    fina_ok = cv2.inpaint(img, imgmask,15,cv2.INPAINT_TELEA)
    return fina_ok



maskpath='./1st_manual/21_manual1.gif'
imgpath='./images/21_training.tif'
fian_ok=repairimg(imgpath,maskpath)
cv2.imshow('',fina_ok)
cv2.waitKey(0)

# img[np.where(imgmask==255)]=[0,0,0]


# img_copy=img
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
# fina_ok = cv2.morphologyEx(img_copy, cv2.MORPH_CLOSE, kernel)

#fina_ok[np.where(imgmask==0)]=img[np.where(imgmask==0)]