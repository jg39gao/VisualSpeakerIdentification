#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:47:41 2020
@author: Jiejun Gao
------------------------------------------------------------
This class is going to evaluate different face detector models. 

USAGE: # in the terminal: 
    
    ------------------------------------------------------------\
"""

from frameROI import FrameROI as  fr
from vgg_face import VGGface 
from face_detector1 import FaceDetector 
import cv2
import time 
import os, sys,re
import numpy as np


video='../data/5695231002474224804_veg301.wmv'
annotation_txt= '../data/5695231002474224804_veg301_gt.txt'

class Video2Frame:
    def __init__(self, videofile, saveframe=1, savedir= '../data'):
        ''' input video file and transfer to frames. 
            return:
                frames  ,a dictionary of image arrays {frameno: (height, width, colorchannel). \n
                frame_num, the number of frames extracted. \n
                if saveframe= 1 , then all the frames will be saved to <savedir>.
        '''
        # Opens the Video file
        if not os.path.exists(videofile): 
            print("***ERROR: video({}) doesn't exist".format(videofile))
            return
        if(saveframe):
            if not os.path.exists(savedir):
                os.mkdir(savedir)
        self.savedir= savedir
        
        self.filename= videofile.split('/')[-1].split('.')[0]
        cap= cv2.VideoCapture(video)
        i=0
        frames_d={}
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
#            cv2.imwrite('kang'+str(i)+'.jpg',frame)
            if saveframe: 
                cv2.imwrite(os.path.join(savedir, '{}_{}.jpg').format(self.filename,i), frame)
            
            frames_d[i]= frame
            
            i+=1
         
        cap.release()
        cv2.destroyAllWindows()
        self.frame_num= i 
        self.frames= frames_d
        self.frameshape=frames_d[0].shape
        self.height, self.width, self.channels= frames_d[0].shape
        



class videoDetectorEvaluation:
    
    def __init__(self, models, video, gt_annotation_txt, save=1, savedir= '../data/evaluation' ):
        '''
        model: should be a function that take single frame as input and output 2-d array of annotations(n* 4), 
        representing n ROIs(xywh) detected. \n
        video:  raw video. \n
        gt_annotation_txt : ground truth of annotated face/head. \n
        '''
        self.savedir= savedir
        if not os.path.exists(self.savedir):
                os.mkdir(self.savedir)
        self.save= save 
        self.vf= Video2Frame(video, saveframe=0, savedir= savedir)
        
        self.gt_ann_dict= fr.paradigm_annotation_fromTxt(gt_annotation_txt)
        
        self.gt_ann_head_dict={}
#        overlap_pct1,overlap_pct2= fr.overlapping_area_analyse()
        
#        ------gt_processing------
        for i in range(len(self.gt_ann_dict)): 
            gt_anns, gt_labels= self.gt_ann_dict[i]
            # select only head
            head= re.compile(r'^head')
            h= [  bool(head.match(labl)) for labl in gt_labels]
            gt_anns= gt_anns[h]
            gt_labels= gt_labels[h]
#            print(i,gt_anns.shape ) #(4,4)
            self.gt_ann_head_dict[i]=(gt_anns, gt_labels)

#        ------------

        
        eva= []
        
        for k, model in enumerate(models):
            
            print('model:',k,'..')
            t_start= time.time()
            # -------------model test----------------------------------------------------------- 
            self.run_model(model, model_name=k )

            # ------------------------------------------------------------------------ 
            t_end= time.time()
            time_cost= t_end- t_start  # time costumed.
            print('cost time {}s'.format(time_cost))
            
            
            recall=[]
            precision=[]
            for i in range(len(self.gt_ann_head_dict)): # for each frame
            # -- recall  
                head_Num= len(self.gt_ann_head_dict[i])
                head_detected= len(self.model_ann_dict[i])
                
                if head_Num==0 and head_detected==0: 
                    recall.append(None)
                    precision.append(None)
                    continue
                elif head_detected==0 : 
                    recall.append(0)
                    precision.append(None)
                    continue
                elif head_Num==0:
                    recall.append(None)
                    precision.append(0)
                    continue
                
                recall.append( head_detected/head_Num if head_Num>0 else None )  # it could be bigger than head_Num, so not always feasible 

            # -- precision
                single_ann_prec=[]
                for m_xywh in self.model_ann_dict[i]: # 
                    overlap=[]
                    #compare with the ground truth
                    for gt_xywh in self.gt_ann_head_dict[i][0]:
#                        print('m_xywh',m_xywh.shape)
#                        print('gt_xywh',gt_xywh.shape)
                        overlap_pct1,overlap_pct2= fr.overlapping_area_analyse( self.vf.frameshape, m_xywh, gt_xywh )
                        overlap.append(overlap_pct1)
                        
                        if overlap_pct1>= 1:
                            break # when model detected face are 90% included in the ground truth, assume it is right detected.
                    single_ann_prec.append(np.max(overlap))
                precision.append(np.mean(single_ann_prec))
                
            recall= [ r for r in recall if r !=None]
            precision= [p for p in precision if p!=None]
        
            eva.append((k, model, time_cost, np.mean(recall), np.mean(precision)) )
            
        self.eva= eva
         

    def paradigm_model(model, *args, **kw):
        annotations=  model(*args, **kw)
        return annotations

    def run_model(self, model, model_name=''):
        self.model_ann_dict={}

        for i in range(len(self.gt_ann_dict)): #  ground truth annotated frames 
            if(i%20==0 or i==len(self.gt_ann_dict)-1):print(' '*4,'frame',i,'..')
            # for each frame 
            
            image= self.vf.frames[i]
            copy= image.copy()
            temp_rawimg_path= os.path.join(self.savedir, 'temp_rawimg.jpg')
            img_w=cv2.imwrite(temp_rawimg_path, copy)
            if img_w: 
#                print('save temp image')
                detect_anns = videoDetectorEvaluation.paradigm_model(model, temp_rawimg_path)
#            print(detect_anns) #[450 284 129 200]
            self.model_ann_dict[i]= detect_anns
#            print( 'xxxx {} faces found by model'.format(len(detect_anns)) )
            fr.annotates(copy, self.gt_ann_head_dict[i][0], color=(0,255,0)) #green
            fr.annotates(copy, self.model_ann_dict[i], color=(0,0,255)) # red #blue (255,0,0)
#            print('save image to {}'.format(self.savedir))
            if self.save: 
                cv2.imwrite(os.path.join(self.savedir, '{}_frame{}_compare_{}.png').format(self.vf.filename,i,model_name), copy)

                
            
#==============================================================================
# MAIN

if __name__ == '__main__':
    vd= Video2Frame(video, saveframe=0, savedir= '../data/videoframe3')
#    print('{} frames extracted'.format(vd.frame_num) )
#    print(vd.frames[0].shape)
    vdE= videoDetectorEvaluation([ VGGface.detect_roi_fromRawImg, 
                                  FaceDetector.detect_cv2dnn_fromRawImg 
                                  ], video,annotation_txt,
                                 save= 1, savedir= vd.savedir)
    
#    print('gt,','*'*10,' \n',vdE.gt_ann_dict[4])#96
#    print('gt_head,','*'*10,' \n',vdE.gt_ann_head_dict[4])
#    print('model,','*'*10,' \n',vdE.model_ann_dict[4])
    print(vdE.eva)
#    print(vdE.gt_ann_dict.items())
    
#    print(fr.paradigm_annotation_fromTxt(annotation_txt))
    
    
        
        