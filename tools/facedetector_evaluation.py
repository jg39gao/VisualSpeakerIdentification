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

#
#video='../data/5695231002474224804_veg3011.wmv'
#annotation_txt= '../data/5695231002474224804_veg3011_gt.txt'

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
        
        self.savedir= savedir
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)
        
        
        self.filename= videofile.split('/')[-1].split('.')[0]
        cap= cv2.VideoCapture(videofile)
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
        

    def createfaces(self,model, *argv, **kw):
        '''
        set a face detector model for it,
        then the left parameters should be same as frameROI.createROIS() 
        
        
        return a dictionary of {frameNo: matrix of faces} 
        '''
        
        faces={}
        
        #---- setup toolbar-----------
        toolbar, toolbar_width= 0, 40 
        sys.stdout.write("[%s]" % ("" * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b " * (toolbar_width)) # return to start of line, after '['
        
              
        #---------create ROIs----------------------------
        
        for i, f in self.frames.items():
            annotations = model(f, is_rawimg= 0)
#            print('frame{} shape{},annotations:\n{}'.format(i,f.shape,annotations)) #@jjg
            ROI= fr(f,annotations, saveto_directory= self.savedir, rawimg_is_array='BGR',filename='{}_frame{}'.format(self.filename, i) )
            n, rois= ROI.createROIs(*argv, **kw)
            faces[i]= rois
        
        
        #---update the bar----------------------------------
            progress= toolbar_width*((i+1)/self.frame_num )
            if np.floor(progress)>toolbar : 
                sys.stdout.write("#")
#                sys.stdout.write("\r%d/%d " % (i+1, self.frame_num ))
                toolbar= np.floor(progress)
            sys.stdout.flush()
        sys.stdout.write(f"\b] {self.frame_num} frames completed!\n") # this ends the progress bar
        
        
        return faces

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
        for i in self.gt_ann_dict.keys(): 
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
        self.recall_precision={}
        
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
            r_p=[] 
            for i in self.gt_ann_head_dict.keys(): 
                r, p= videoDetectorEvaluation.recall_precision(self.vf.frameshape, 
                                                                        gt_anns= self.gt_ann_head_dict[i][0],
                                                                        model_anns= self.model_ann_dict[i],
                                                                        details= False)
                recall.append(r)
                precision.append(p)
                r_p.append((i, r,p))# frame, recall, precision
            self.recall_precision[k]=(r_p) # K: (frame, recall, precision)
                
#            ##### module r-p ------------------------------------------------------------------------
#            for i in self.gt_ann_head_dict.keys(): # for each frame
#            # -- recall  
#                head_Num= len(self.gt_ann_head_dict[i])
#                head_detected= len(self.model_ann_dict[i])
#                
#                if head_Num==0 and head_detected==0: 
#                    recall.append(None)
#                    precision.append(None)
#                    continue
#                elif head_detected==0 : 
#                    recall.append(0)
#                    precision.append(None)
#                    continue
#                elif head_Num==0:
#                    recall.append(None)
#                    precision.append(0)
#                    continue
#                
#                ## recall: calculate the head number
##                recall.append( head_detected/head_Num if head_Num>0 else None )  # it could be bigger than head_Num, so not always feasible 
#                
#                ## recall: calculate the heads that has been detected by model
#                single_gt_recall=[]
#                for gt_xywh in  self.gt_ann_head_dict[i][0]: # gt_ann_head_dict:(anns,labels)
#                    overlap=[]
#                    for m_xywh in self.model_ann_dict[i]:
#                        overlap_pct1,overlap_pct2= fr.overlapping_area_analyse( self.vf.frameshape, m_xywh, gt_xywh )
#                        overlap.append((overlap_pct2, overlap_pct1))
#                        if overlap_pct2>= 0.4:
#                            break
#                    overlap= sorted(overlap, key= lambda x:(x[0], x[1]), reverse= 1)
#                    x= 1 if (overlap[0][0]>=0.3 and overlap[0][1]>=0.9) else overlap[0][0]
#                    single_gt_recall.append(x)
#                recall.append(np.mean(single_gt_recall))
#                
#            # -- precision
#                single_ann_prec=[]
#                for m_xywh in self.model_ann_dict[i]: # 
#                    overlap=[]
#                    #compare with the ground truth
#                    for gt_xywh in self.gt_ann_head_dict[i][0]:
##                        print('m_xywh',m_xywh.shape)
##                        print('gt_xywh',gt_xywh.shape)
#                        overlap_pct1,overlap_pct2= fr.overlapping_area_analyse( self.vf.frameshape, m_xywh, gt_xywh )
#                        overlap.append(overlap_pct1)
#                        
#                        if overlap_pct1>= 1:
#                            break # when model detected face are 90% included in the ground truth, assume it is right detected.
#                    x= 1 if np.max(overlap)>0.9 else np.max(overlap)
#                    single_ann_prec.append(np.max(overlap))
#                precision.append(np.mean(single_ann_prec))
#            ##### module r-p ------------------------------------------------------------------------
            recall= [ r for r in recall if r !=None]
            precision= [p for p in precision if p!=None]
        
            eva.append((k, model, time_cost, np.mean(recall), np.mean(precision)) )
            
        self.eva= eva
         
    def recall_precision(frameshape, gt_anns, model_anns, details= False):
        '''
        to calculate the recall and precision for a face detection model
        
        frameshape: Height, width, channels
        gt_anns : ground truth annotations [xywh]
        model_anns:  model annotations [xywh]
        
        details: if true, return list of every anns, otherwise return means of them respectively
        
        be careful, it might return nan if len(gt_anns)=0 or len(model_anns)=0 
        '''
        
        single_gt_recall=[]
        if len(gt_anns)==0: single_gt_recall.append(1)
        else: #len(gt_anns)>0:
            for gt_xywh in  gt_anns: # gt_ann_head_dict:(anns,labels)
                overlap=[]
                x=0 
                if len(model_anns)>0:
                    for m_xywh in model_anns:
                        overlap_pct1,overlap_pct2= fr.overlapping_area_analyse( frameshape, m_xywh, gt_xywh )
                        overlap.append((overlap_pct2, overlap_pct1))
                        if overlap_pct2>= 0.4 and overlap_pct1>=0.9:
                            break
                    overlap= sorted(overlap, key= lambda x:(x[0], x[1]), reverse= 1)
                    x= 1 if (overlap[0][0]>=0.3 and overlap[0][1]>=0.9) else overlap[0][0]
    #               x= 1 if (overlap[0][0]>=0.3 and overlap[0][1]>=0.9) else 0
    
                single_gt_recall.append(x)
#        recall.append(np.mean(single_gt_recall))
        
    # -- precision
        single_ann_prec=[]
        if len(model_anns)==0: single_ann_prec.append(1)
        else: #len(model_anns)>0
            for m_xywh in model_anns: # 
                overlap=[]
                x=0
                #compare with the ground truth
                if len(gt_anns)>0:
                    for gt_xywh in gt_anns:
        #                        print('m_xywh',m_xywh.shape)
        #                        print('gt_xywh',gt_xywh.shape)
                        overlap_pct1,overlap_pct2= fr.overlapping_area_analyse( frameshape, m_xywh, gt_xywh )
                        overlap.append(overlap_pct1)
                        
                        if overlap_pct1>= 1:
                            break # when model detected face are 90% included in the ground truth, assume it is right detected.
                    x= 1 if np.max(overlap)>0.9 else np.max(overlap)
                single_ann_prec.append(x)
#        precision.append(np.mean(single_ann_prec))
        if details:
            return single_gt_recall, single_ann_prec
        else: 
            return np.mean(single_gt_recall), np.mean(single_ann_prec)
        

    def paradigm_model(model, *args, **kw):
        annotations=  model(*args, **kw)
        return annotations

    def run_model(self, model, model_name=''):
        self.model_ann_dict={}
        
        for i in self.gt_ann_dict.keys(): #  ground truth annotated frames 
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
    
    
        
        