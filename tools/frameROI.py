#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:17:22 2020
@author: Jiejun Gao
------------------------------------------------------------
This tool helps to make single image with annotation.txt(should desinate frameNo specifically) to seperate region(s) of interest(ROI).
OPTIONS:
    -h : print this help message
    -i : input rawimage
    -t : annotation txt, can be a txt file or a single annotation string like'frame#,n[x,y,w,h,label]'
    -f : frameNo, if -t is a .txt file, then need to desinate a frameNo
    -o : output results saved to desinated dirtory
RETURN:
    annotated full image
    ROI(s)

USAGE: # in the terminal: 
    python frameROI.py -i <rawImage> -t <annotation_txt> -o <output_directory> [-f <frameNo>]
    e.g.: python frameROI.py -i ../data/out111.jpg -t '111,3,-4,188,210,313,head_speaker' -o ../data/roi
    python frameROI.py -i ../data/out112.jpg -t ../data/5695231002474224804_veg350_gt.txt -f 112 -o ../data/roi
------------------------------------------------------------\
"""

# #usage of ffmpeg:
## to count the total frames
#time ffmpeg -y -i 5695231002474224804_veg350.wmv -vcodec copy -acodec copy -f null /dev/null 2>&1 | grep 'frame=' | awk '{print $2}'


## Output a single frame from the video into an image file:
#ffmpeg -i input.mov -ss 00:00:14.435 -vframes 1 out.png
#
## Output one image every second, named out1.png, out2.png, out3.png, etc.
## The %01d dictates that the ordinal number of each output image will be formatted using 1 digits.
#ffmpeg -i input.mov -vf fps=1 out%d.png
#
## Output one image every minute, named out001.jpg, out002.jpg, out003.jpg, etc. 
## The %02d dictates that the ordinal number of each output image will be formatted using 2 digits.
#ffmpeg -i input.mov -vf fps=1/60 out%02d.jpg
#
## Extract all frames from a 24 fps movie using ffmpeg
## The %03d dictates that the ordinal number of each output image will be formatted using 3 digits.
#ffmpeg -i input.mov -r 24/1 out%03d.jpg
#
## Output one image every ten minutes:
#ffmpeg -i input.mov -vf fps=1/600 out%04d.jpg
#
#---------------------------------------------------------------------------------------------------

##--------------------------------------------------------------------------------------------------
import os,re, sys
import numpy as np
import cv2
import getopt
#==============================================================================
# Command line processing



class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'i:t:o:f:h')
        opts = dict(opts)
        
        self.exit = True
#        if (len(opts) < 3):
#            self.printHelp()
#            print("*** ERROR 0: 3 options(i,t,o) needed ***", file=sys.stderr)
#            return

        if '-i' in opts:
            self.rawImg= opts['-i']
            if not os.path.exists(self.rawImg):
                print("*** ERROR 1: Image does not exist ***", file=sys.stderr)
                self.printHelp()
                return
        
        if '-f' in opts:
            self.frameno= opts['-f']
#            print(self.frameno)
            
        if '-t' in opts:
            self.annotatetxt= opts['-t']
            self.pattern_ann= re.compile('\d+,\d+(,(-?\d+,-?\d+,-?\d+,-?\d+,[^,0-9]+[^,]+))+')
            findframe= False
            if os.path.exists(self.annotatetxt):
                if '-f' not in opts:
                    print("*** ERROR 2: You have to desinate the frameNo in the annotation txt file ***", file=sys.stderr)
                    self.printHelp()
                    return
                else:
                    with open(opts['-t']) as f:
                        line = f.readline()
                        while line:
                            if int(line.split(',')[0]) == int(self.frameno): 
                                self.annotatetxt= line 
                                print('annotations:\n', line)
                                findframe= True
                                break 
                            line= f.readline()
                            
                        if not findframe:
                            print('*** ERROR4: cannot find frameno in annotation txt file:***\n')
                            return
                        

            else: # check if match the annotation text
                if not self.pattern_ann.match(self.annotatetxt):
                    print("*** ERROR 3: annotation file does not exist ***", file=sys.stderr)
                    self.printHelp()
                    return
            

                    
        if '-o' in opts:
            self.saveto_directory= opts['-o']
            if not os.path.exists(self.saveto_directory):
                os.mkdir(self.saveto_directory)
        
        if '-h' in opts:
            self.printHelp()
            

        self.exit = False
        
    def printHelp(self):
        progname = sys.argv[0]
        progname = progname.split('/')[-1] # strip off extended path
        help = __doc__.replace('<PROGNAME>', progname, 1)
        print(help, file=sys.stderr)

#==============================================================================
# Body parts
        
class FrameROI:
    
    
    def __init__(self, rawimg, annotations, labels=None, saveto_directory='./', rawimg_is_array='RAW', filename='img'):
        '''annotatetxt.shape: (n,4) . annotatetxt[i]=(x,y,w,h)
            label.shape:(n,), if labels=0, labels will be automatically none for every annotations
            
            rawimg_is_array: 'RAW'(default) or 'BGR'(from cv2.imread())
            '''
        if rawimg_is_array=='RAW':
            self.rawimg= rawimg
            self.filename= rawimg.split('/')[-1].split('.')[0]
            self.image= cv2.imread(rawimg)
        elif rawimg_is_array=='BGR':#  from cv2.imread(rawimg)
            self.rawimg=None
            self.filename= filename
            self.image= rawimg
        else: 
            print('###ERROR of image input####')
            return
        
        self.annotations= annotations
        self.labels= labels
        if self.labels is None:
            self.labels= np.zeros(len(annotations)).astype(np.uint32)
            
        self.saveto_directory= saveto_directory
        if not os.path.exists(saveto_directory):
            os.mkdir(saveto_directory)
        self.img_height, self.img_width, self.img_channels= self.image.shape
        #image shape:  :height, width, channels
        
    def annotate(image, xywh, color=(36,255,12)):
        x,y,w,h= xywh
        #Syntax: cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.rectangle(image,(x,y),(x+w,y+h),color,2) # annotation
        return image
    
    def annotateWithLabel(image, xywh, label, color=(36,255,12)):
        x,y,w,h= xywh
        x1,y1,x2,y2= FrameROI.xywh2Points(image.shape, x,y,w,h)
        y_label = y1 - 10 if y1 - 10 > 10 else y1 + 10
        #Syntax: cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.rectangle(image,(x,y),(x+w,y+h),color,2) # annotation
        cv2.putText(image, str(label), (x1, y_label),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        return image

    
    def annotates(image, annotations, color):
        '''annotations.shape should be (n,4),  
           annotation paradigm: frame#,n[x,y,w,h,label]
        '''
        if len(annotations)==0: 
            return image

        if np.array(annotations).ndim!=2 or annotations.shape[1]!=4  :
            print('***ERROR: annotations.shape should be (n,4)')
            return 
        img= image
        for xywh in annotations:
            FrameROI.annotate(img, xywh, color=color)
        return img
    
    def xywh2Points(imageshape_hwc, x,y,w,h):
        x1,x2= (x+w, x) if w<0 else (x, x+w)
        y1,y2= (y+h, y) if h<0 else (y, y+h)
        if x1< 0: x1= 0
        if x2> imageshape_hwc[1]: x2= imageshape_hwc[1] #image shape:  :height, width, channels
        if y1< 0: y1= 0
        if y2> imageshape_hwc[0]: y2= imageshape_hwc[0] # 
        return (x1,y1,x2,y2)

    
    def overlapping_line( line_seg1, line_seg2):
        '''
          input: line_seg1= (x1,x2), line_seg2= (x3, x4)
          return: overlap(start, end)
        '''
        
        a1, a2= sorted((line_seg1[0], line_seg1[1]))
        b1, b2= sorted((line_seg2[0], line_seg2[1]))
        rsl=(0,0)
        if (b2< a1) or (b1>a2): # no overlap
            rsl= (0,0)
        else:
            rsl=(max(a1,b1), min(a2,b2))
        return rsl
    
    def overlapping_area(imageshape_hwc, ann1_xywh, ann2_xywh ):
        '''
        return( overlap_area, overlap(xywh))
        '''
        a_x1,a_y1,a_x2,a_y2= FrameROI.xywh2Points(imageshape_hwc, *ann1_xywh)
        b_x1,b_y1,b_x2,b_y2= FrameROI.xywh2Points(imageshape_hwc, *ann2_xywh)
        
        x_ol= FrameROI.overlapping_line((a_x1, a_x2), (b_x1, b_x2))
        y_ol= FrameROI.overlapping_line( (a_y1, a_y2), (b_y1, b_y2))
        
        w,h = x_ol[1]- x_ol[0], y_ol[1]- y_ol[0]
        area= w* h 
        
        return (area, (x_ol[0], y_ol[0], w,h ))
        
    def overlapping_area_analyse(imageshape_hwc, ann1_xywh, ann2_xywh ):
        '''
        return( overlap_area/ areaOfAnnotation1, overlap_area/ areaOfAnnotation2)
        '''
        a_x1,a_y1,a_x2,a_y2= FrameROI.xywh2Points(imageshape_hwc, *ann1_xywh)
        b_x1,b_y1,b_x2,b_y2= FrameROI.xywh2Points(imageshape_hwc, *ann2_xywh)
        x_ol= FrameROI.overlapping_line((a_x1, a_x2), (b_x1, b_x2))
        y_ol= FrameROI.overlapping_line( (a_y1, a_y2), (b_y1, b_y2))
        
        w,h = x_ol[1]- x_ol[0], y_ol[1]- y_ol[0]
        area= w* h 
        
        area_a= (a_x2- a_x1)* (a_y2- a_y1)
        area_b= (b_x2- b_x1)* (b_y2- b_y1)
        return (area/area_a, area/area_b)
        
    
    def cropImg(image, x1, y1, x2, y2, label, save, savedir='./', filename='', resize_wh= 0):
        '''crop image by 2 points(x1,y1, x2,y2) and save(if save= true)
           if resize_wh assigned ,it should be a tuple(width, height)
           if save= true, croped image will be saved to <savedir>. 
           
           return roi numpyarray
           
           '''
        ROI = image[y1:y2, x1:x2]
        
        if resize_wh!=0: ROI= cv2.resize(ROI, resize_wh, interpolation = cv2.INTER_AREA)
        
        if save: 
            if not os.path.exists(savedir):
                    os.mkdir(savedir)
            cv2.imwrite(os.path.join(savedir, '{}_crop_{}.png').format(filename,label), ROI)
        return ROI
    
    def cropImgXywh(image, x, y, w, h, label, save= 1, savedir='./', filename='', resize_wh=0):
        '''crop image by xywh and save ,  
           if resize_wh assigned ,it should be a tuple(width, height)
           if save= true, croped image will be saved to <savedir>. 
           
           return roi numpyarray
           '''
        x1, y1, x2, y2= FrameROI.xywh2Points(image.shape, x,y,w,h)
        return FrameROI.cropImg(image, x1, y1, x2, y2, label, save, savedir, filename, resize_wh)
        
    # #---------------------------------------------------------------------------------------------------
    def paradigm_annotation_fromTxt(txtfile):
        ''' input : annotation txtfile ( frame#,n[x,y,w,h,label])
            return: dictionary {frameno:([xywh],[label])}
        '''
        if not os.path.exists(txtfile): 
            print("***ERROR: file({}) doesn't exist".format(txtfile))
            return
        else:
            rsl= dict()
            with open(txtfile) as f:
                line = f.readline().strip()
                while line:
                    frame= int(line.split(',')[0])
                    annotations, labels= FrameROI.paradigm_annotation(line)
                    rsl[frame]= (annotations, labels)
                    line= f.readline()
                
            return rsl
    
    def paradigm_annotation(singleline_txt):
        ''' input : frame#,n[x,y,w,h,label]
            return: [xywh],[label]
        '''
        anns= re.compile('-?\d+,-?\d+,-?\d+,-?\d+,[^,0-9]+[^,]+').findall(singleline_txt)
        annotations= []
        labels=[]
        for i in  anns:
            xywh= np.array(i.split(',')[:4], np.int)
            label= i.split(',')[4]
            annotations.append(xywh)
            labels.append(label)
        return np.array(annotations), np.array(labels)
    
    def createROIs(self, crop=0, save=1, boundingbox_color=(36,255,12), return_RoiArray= False, flatten=0, return_RoiArray_2RGB=False,  resize_wh=(224,224)):
        '''
         annotate all the annotaions and save to desinated directory
         if crop==true: crop and save all the ROIs
         save: whether save annotated image. means annotate rois on the original image.
         boundingbox_color: annotation color
         
         return_RoiArray: if true, return (n, detected ROIs (numpyarray)).if false ,return n. default FALSE
         
         flatten:  if true, return flattened numpyarrays of the ROIs. default false
         
         return_RoiArray_2RGB: if true, return RGB numpyarray, otherwise default BGR
         
         resize_wh: if not 0, then the roi will be resized to desinated shape (width, height) , default 0 
         
        '''
        annotations= self.annotations
        labels=self.labels
        copy = self.image.copy()
        
        rois=[]
        for k,i in enumerate(annotations):
            x,y,w,h= annotations[k]
            label= labels[k]

            
            roi=  FrameROI.cropImgXywh(self.image, x,y,w,h, label='{}_{}'.format(label,k), 
                                             save= crop, savedir=self.saveto_directory, 
                                             filename=self.filename, resize_wh=resize_wh )
            if return_RoiArray_2RGB: roi= cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            if flatten: roi= roi.flatten()
            rois.append(roi)
            
            FrameROI.annotateWithLabel(copy,(x,y,w,h), '{}_{}'.format(label,k), color= boundingbox_color)

        if save: 
            cv2.imwrite(os.path.join(self.saveto_directory, '{}_gt.png').format(self.filename), copy)
            #cv2.imshow('copy', copy)
            #cv2.waitKey()
            cv2.destroyAllWindows()
        n = len(annotations)
#        print('{} rois have been annotated'.format(n))
        
        
        return n if not return_RoiArray else (n, np.array(rois))

#==============================================================================
# MAIN

if __name__ == '__main__':

    config = CommandLine()
    if config.exit:
        sys.exit(0) 
    rawImg='../Data/out111.jpg'
    annotatetxt0= '111,3,-4,188,210,313,head_speaker,382,33,182,309,head,434,270,71,46,lips'
    annotatetxt= '111,3,-4,188,210,313,head_speaker'
    
    a,l=FrameROI.paradigm_annotation(annotatetxt0)
    a1,l1= FrameROI.paradigm_annotation(config.annotatetxt)
#    print('a.shape:',a.shape,a)
#    print('l.shape:',l.shape,l)
    saveto_directory= r'../Data/roi'
    
#    frameroi= FrameROI(rawImg, a,l, saveto_directory)
    frameroi= FrameROI(config.rawImg, a1, l1, config.saveto_directory)
    frameroi.createROIs(crop= 1)

    