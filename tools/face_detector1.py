#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 02:48:37 2020

@author: gaojiejun
"""


### reference: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/ 

# import the necessary packages
import os, sys
import numpy as np
import cv2

from frameROI import FrameROI  as fr

# construct the argument parse and parse the arguments


## ------------------------------------------------------------------------------------
# load our serialized model from disk



class FaceDetector:
    def __init__(self, rawimg, prototxt, model, confidence, saveto_directory):
        self.rawimg= rawimg
        if not os.path.exists(self.rawimg): 
            print("*** ERROR 0: image not found  ***", file=sys.stderr)
            return
        print("[INFO] loading model...")
        self.image= cv2.imread(self.rawimg)
        self.filename= self.rawimg.split('/')[-1].split('.')[0]
        
        self.prototxt= prototxt
        if not os.path.exists(self.prototxt): 
            print("*** ERROR 1: prototxt not found  ***", file=sys.stderr)
            return
        self.model= model
        if not os.path.exists(self.model): 
            print("*** ERROR 2: model not found  ***", file=sys.stderr)
            return
        self.confidence= confidence 
        self.savedir= saveto_directory
        if not os.path.exists(self.savedir): 
            os.mkdir(self.savedir)
            print("**WARNING :  create directory :{}**".format(self.savedir), file=sys.stderr)

    def detect_cv2dnn_fromRawImg_conf7(rawimg):
        return FaceDetector.detect_cv2dnn_fromRawImg(rawimg, conf=0.7)
    def detect_cv2dnn_fromRawImg_conf3(rawimg):
        return FaceDetector.detect_cv2dnn_fromRawImg(rawimg, conf=0.3)
    
    def detect_cv2dnn_fromRawImg(rawimg, 
                              conf=0.5 ,
                              prototxt='deploy.prototxt.txt',
                              model='res10_300x300_ssd_iter_140000.caffemodel'
                              ):
        '''
        model1: cv2.dnn.readNetFromCaffe
        
        '''
#        prototxt='deploy.prototxt.txt'
#        model= 'res10_300x300_ssd_iter_140000.caffemodel'
#        conf= 0.5 
        image= cv2.imread(rawimg)
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
#        imagecopy= image.copy()
        (h, w) = image.shape[:2]
        resize= (w,h)#(300, 300)
        blob = cv2.dnn.blobFromImage(cv2.resize(image, resize), 1.0,
            resize, (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward() #  detections.shape: (1, 1, 200, 7)
        cnt=0
        
        annotations=[]
        labels=[]
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > conf:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                cnt+=1 
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                w1= endX- startX
                h1= endY- startY
                
                annotations.append((startX, startY, w1, h1))
                labels.append(confidence)
        return np.array(annotations)#, np.array(labels)


    def detect(self, crop= 0):
        
        net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        image = self.image
        imagecopy= image.copy()
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
    
        # pass the blob through the network and obtain the detections and
        # predictions
        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward() #  detections.shape: (1, 1, 200, 7)
    
        # loop over the detections
        cnt=0
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                cnt+=1 
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                imagecopy= image.copy()
                # crop and save
                if crop!=0: fr.cropImg(imagecopy, startX, startY, endX, endY, label='face_{}'.format(i), 
                                  savedir=self.savedir, filename= self.filename, resize_wh=0)
    
                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
        
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                
            
        cv2.imwrite(os.path.join(self.savedir, '{}_detection.png').format(self.filename), image)
        print('summary: {} /{} faces filtered/detected.'.format(cnt,detections.shape[2] ))
        return
    
    def detect1(self, crop= 0):
        annotations, labels= FaceDetector.detect_cv2dnn_fromRawImg(self.rawimg)
        
        print('anns.shape11:', annotations )
        print('labels.shape:',labels)
        ROI=fr(rawimg= self.rawimg, annotations= annotations, labels=labels, saveto_directory= self.savedir)
                        #self, rawimg, annotations, labels, saveto_directory):
        ROI.createROIs(crop=crop)
        
        cnt= len(labels)
        print('summary: {} / faces filtered/detected.'.format(cnt))#,detections.shape[2] ))
        

        ## show the output image
        #cv2.imshow("Output", image)
        #cv2.waitKey(0)