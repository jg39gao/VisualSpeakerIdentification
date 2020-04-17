#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:32:39 2020

@author: gaojiejun
"""

from matplotlib import pyplot
#import keras_vggface
#print(keras_vggface.__version__)
from mtcnn.mtcnn import MTCNN
#from keras_vggface.vggface import VGGFace 
#from keras_vggface.utils import preprocess_input
#from keras_vggface.utils import decode_predictions
import os, sys
import numpy as np
import cv2



from frameROI import FrameROI 

#import cv2

class VGGface:
    def __init__(self, rawimg, saveto_directory):
        self.rawimg= rawimg
        if not os.path.exists(self.rawimg): 
            print("*** ERROR 0: image not found  ***", file=sys.stderr)
            return
        self.pixels= pyplot.imread(self.rawimg)
        self.filename= self.rawimg.split('/')[-1].split('.')[0]
        
        self.savedir= saveto_directory
        if not os.path.exists(self.savedir): 
            os.mkdir(self.savedir)
            print("**WARNING :  create directory :{}**".format(self.savedir), file=sys.stderr)
        self.image= cv2.imread(rawimg)
#        print('shape:',self.pixels.shape) #:height, width, channels  , (576,768, 3)
#        print('shape cv:',self.image.shape) #:height, width, channels
        return
    def detect_roi_fromRawImg(rawimg):
        '''
        detect_face method. !!!!! BE CAREFUL, rawimg should be jpg only
        '''
        detector = MTCNN()

        results = detector.detect_faces(pyplot.imread(rawimg))
        n= len(results)
#        print("{} detected".format(n))
        annotations=[]
        for i in range(n):
            x1, y1, w, h = results[i]['box']
            annotations.append((x1, y1, w, h))
            
        return np.array(annotations) 
    
    def detect_roi_fromarrayimg(arrayimage):
        '''
        detect_face method.  !!!!! BE CAREFUL, THE ARRAYIMAGE should be obtained by pyplot.imread(rawimage)
        '''
        detector = MTCNN()

        results = detector.detect_faces(arrayimage)
        n= len(results)
#        print("{} detected".format(n))
        annotations=[]
        labels=[]
        for i in range(n):
            x1, y1, w, h = results[i]['box']
            annotations.append((x1, y1, w, h))
            labels.append('vgg')
        return np.array(annotations), np.array(labels)
    
    def detect_roi_fromarrayimg_returnnolabels(arrayimage):
        '''
        detect_face method. !!!!! BE CAREFUL, THE ARRAYIMAGE should be obtained by pyplot.imread(rawimage)
        '''
        return VGGface.detect_roi_fromarrayimg(arrayimage)[0]
#        return np.array(annotations), np.array(labels)
    
    def detect_roi(self):
    
        return VGGface.detect_roi_fromarrayimg(self.pixels)
    
    def detect(self, crop=0):
        a,l=self.detect_roi()
        ROI=FrameROI(self.rawimg, a, l , self.savedir )
        ROI.createROIs(crop=crop)
        
        
if __name__ == '__main__':
    
    rawimg=  '../data/videoframe2/5695231002474224804_veg300_6.jpg'#'../Data/test8.jpg'
    saveto_directory= '../data/facevgg1'
    vggf= VGGface(rawimg, saveto_directory)
    vggf.detect( crop= 1)
    
    model_ann = VGGface.detect_roi_fromRawImg(rawimg)
    print('{} faces detected \n'.format(len(model_ann)),model_ann)
    
    print('vggf.pixels.shape:',vggf.pixels[0][0])
    print('vggf.image.shape:',vggf.image[0][0])
    
    
    cv2.imwrite(os.path.join(saveto_directory, 'test_plt_pixels_{}.png').format(vggf.filename), vggf.pixels)
    cv2.imwrite(os.path.join(saveto_directory, 'test_cv2_images_{}.png').format(vggf.filename), vggf.image)

        
        
    