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

rawimg= '../data/test2.jpg'#videoframe/5695231002474224804_veg300_20.jpg'
saveto= '../data/model_evaluation2'


#model= VGGface.detect_roi_fromRawImg#, 
model= FaceDetector.detect_cv2dnn_fromRawImg_conf7 
#------------------------------------------------------------------------------
annotations = model(rawimg)
print(annotations)

#create a relative labels for the annotation
labels= np.zeros(annotations.shape[0])
print(labels)

#------------------------------------------------------------------------------
# annotate face boundingbox and save
ROI=fr(rawimg, annotations=annotations, labels=labels , saveto_directory=saveto )
ROI.createROIs(crop=0, 
               save=1,
               boundingbox_color=(255,255,0))
#------------------------------------------------------------------------------
# #evaluation 
video='../data/5695231002474224804_veg301.wmv'
annotation_txt= '../data/5695231002474224804_veg301_gt.txt'

vd= Video2Frame(video, saveframe=0, savedir= '../data/model_evaluation2/eva')
vdE= videoDetectorEvaluation([#VGGface.detect_roi_fromRawImg, 
                              FaceDetector.detect_cv2dnn_fromRawImg,
                              FaceDetector.detect_cv2dnn_fromRawImg_conf7,
                              FaceDetector.detect_cv2dnn_fromRawImg_conf3
                              ], video,annotation_txt,
                             save= 1, savedir= vd.savedir)

print(vdE.eva)