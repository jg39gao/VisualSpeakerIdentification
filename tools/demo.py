#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:25:24 2020

@author: gaojiejun

This demo is helping to test a face detection model's work. 

"""

import numpy as np
from frameROI import FrameROI as  fr
from vgg_face import VGGface 
from face_detector1 import FaceDetector 
from facedetector_evaluation import videoDetectorEvaluation , Video2Frame
import cv2 

rawimg= '../data/test2.jpg'#videoframe/5695231002474224804_veg300_20.jpg'
bgr=cv2.imread(rawimg)
#rawimg= '../data/test2_noface.png'
saveto= '../data/model_evaluation4'


#model= VGGface.detect_roi_fromRawImg#, 
model= FaceDetector.detect_cv2dnn_fromRawImg#_conf7 
#------------------------------------------------------------------------------
#annotations = model(rawimg)
#print(annotations)
#
##create a relative labels for the annotation
#labels= np.zeros(annotations.shape[0])
#print(labels)

#------------------------------------------------------------------------------
# annotate face boundingbox and save
#ROI=fr(rawimg, annotations=annotations, labels=labels , saveto_directory=saveto )
#ROI=fr(bgr, annotations=annotations, labels=labels , saveto_directory=saveto ,  rawimg_is_array='BGR', filename='unnamed')
#n,faces= ROI.createROIs(crop=1, 
#               save=1,boundingbox_color=(200,255,0),
#               return_RoiArray= True,
#               flatten=1,
#               return_RoiArray_2RGB=True,
#               resize_wh=(224,224)
#               )
#print('faces:{},\n {}'.format((faces.dtype), faces[0]))

#------------------------------------------------------------------------------
#
video='../data/5695231002474224804_veg350.wmv'
annotation_txt= '../data/5695231002474224804_veg350_gt.txt'
savedir= '../data/model_evaluation2/createfaces2'

frame= '../data/5695231002474224804_veg300_0.jpg'
frame2= '../data/5695231002474224804_veg350_0.jpg'
test2= '../data/test2.jpg'
f= cv2.imread(frame2)

annotations = model(frame2, is_rawimg= 1#, inner_blob_size=(500,500)
)
print('here:',len(annotations),'founded. \n',annotations)

for i, ann in enumerate(annotations):
    x,y,w,h= ann
    print(fr.xywh2Points(f.shape, x,y,w,h))

##--to test single image-
#ROI= fr(f,annotations, saveto_directory= savedir, rawimg_is_array='BGR')
#n, rois= ROI.createROIs(crop=0, save=1, boundingbox_color=(36,255,12), 
#                        return_RoiArray= 1, 
#                        flatten=0, 
#                        return_RoiArray_2RGB=False,  
#                        resize_wh=(224,224))
#print(n)
###--end of #to test single image-

vd= Video2Frame(video, saveframe=0, savedir= '../data/model_evaluation2/createfaces')
rsl= vd.createfaces( model=model,
                    
               crop=0, 
               save=1,boundingbox_color=(200,255,0),
               return_RoiArray= True,
               flatten=0,
               return_RoiArray_2RGB=True,
               resize_wh=(224,225)#(224,224)
                   )
print(rsl[0].shape)

#------------------------------------------------------------------------------
### 
#video='../data/5695231002474224804_veg300.wmv'
#annotation_txt= '../data/5695231002474224804_veg300_gt.txt'
#
#vd= Video2Frame(video, saveframe=0, savedir= '../data/model_evaluation2/eva1')
#

#
#
##------------------
# # evaluation 
#vdE= videoDetectorEvaluation([VGGface.detect_roi_fromRawImg, 
#                              FaceDetector.detect_cv2dnn_fromRawImg,
#                              #FaceDetector.detect_cv2dnn_fromRawImg_conf7,
#                              FaceDetector.detect_cv2dnn_fromRawImg_conf3
#                              ], video,annotation_txt,
#                             save= 1, savedir= vd.savedir)
#
#print(vdE.eva)

#r_p= vdE.recall_precision
