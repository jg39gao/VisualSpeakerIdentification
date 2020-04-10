# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:39:16 2020

@author: George Tsoumis
"""
import cv2

##cv2 implementation
def cropAndSave(img,x,y,w,h):
    img = cv2.imread(imF)
    crop_img = img[y:y+h, x:x+w] #crop_img is of type numpy.ndarray
    print(crop_img.T.shape)
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)
    cv2.imwrite("starry_night.png", crop_img)


imF = 'frame6.png'
x,y,w,h = 155,32,113,193
cropAndSave(imF,x,y,w,h)

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from PIL import Image


# image = Image.open(imF)
# print(image)
# The crop method from the Image module takes four coordinates as input.
# The right can also be represented as (left+width)
# and lower can be represented as (upper+height).

# im_crop = image.crop((x,y,x+w,y+h))
# im_crop.show()
#im_crop.save('frame6area.png')