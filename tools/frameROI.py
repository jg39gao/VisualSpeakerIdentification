#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:17:22 2020
@author: Jiejun Gao
------------------------------------------------------------
OPTIONS:
    -h : print this help message
    -i : input directory
    -t : annotation txt 
    -f : frameNo, if -t is a .txt file, then need to desinate a frameNo
    -o : output results saved to desinated dirtory
Usage: # in the terminal: 
     python frameROI.py -i rawImage -t annotation_txt -o output_directory [-f frameNo]
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
        if (len(opts) < 3):
            self.printHelp()
            print("*** ERROR 0: 3 options(i,t,o) needed ***", file=sys.stderr)
            return

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
        
class frameROI:
    
    def __init__(self, rawimg, annotatetxt, saveto_directory):
        self.rawimg= rawimg
        self.annotatetxt= annotatetxt
        self.saveto_directory= saveto_directory
        self.filename= rawimg.split('/')[-1].split('.')[0]
        self.image= cv2.imread(rawimg)
        self.img_height, self.img_width, self.img_channels= self.image.shape
        #image shape:  :height, width, channels


        self.pattern_single= re.compile('-?\d+,-?\d+,-?\d+,-?\d+,[^,0-9]+[^,]+')
        
    def annotate(self,img, xywh):
        x,y,w,h= xywh
        #Syntax: cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.rectangle(img,(x,y),(x+w,y+h),(36,255,12),2) # annotation
        return cv2
    
    def annotates(self,img, annotations):
        '''annotations.shape should be (n,4),  
           annotation paradigm: frame#,n[x,y,w,h,label]
        '''
           
        if np.array(annotations).ndim!=2 or annotations.shape[1]!=4  :
            print('***ERROR: annotations.shape should be (n,4)')
            return 
        for xywh in annotations:
            self.annotate(img, xywh)
        return cv2
    
    # crop and save the ROIs 
    def crop(self,image, annotations,labels, savedir, filename):
        for i in range(len(annotations)):
            x,y,w,h = annotations[i]
            label= labels[i].strip()
            
            x1,x2= (x+w, x) if w<0 else (x, x+w)
            y1,y2= (y+h, y) if h<0 else (y, y+h)
            if x1< 0: x1= 0
            if x2> self.img_width: x2= self.img_width
            if y1< 0: y1= 0
            if y2> self.img_height: y2= self.img_height 
            
            ROI = image[y1:y2, x1:x2]# when the h or w be negatie, image[] could lead to error: libpng error: Invalid IHDR data
            cv2.imwrite(os.path.join(savedir, '{}_ROI_{}_{}.png').format(filename,label,i), ROI)
        return 
    # #---------------------------------------------------------------------------------------------------
    
    def createROIs(self):
#        _img= _rawImg
#        filename= _img.split('/')[-1].split('.')[0]
        
        # dealing with the annotaion txt on each frame: annotation should like' frame#,n[x,y,w,h,label]'
        anns= self.pattern_single.findall(self.annotatetxt)
        annotations= []
        labels=[]
        for i in anns:
            annotations.append(np.array(i.split(',')[:4], np.int))
            labels.append(i.split(',')[4])
        annotations= np.array(annotations)

        # crop and save 
        self.crop(self.image, annotations, labels, savedir=self.saveto_directory, filename=self.filename)
        
        # make annotations 
        copy = self.image.copy()
        self.annotates(copy, annotations)
    
        
        cv2.imwrite(os.path.join(self.saveto_directory, '{}_gt.png').format(self.filename), copy)
        
        #cv2.imshow('copy', copy)
        #cv2.waitKey()
        cv2.destroyAllWindows()
        print('{} rois have been found'.format(len(annotations)))

#==============================================================================
# MAIN

if __name__ == '__main__':

    config = CommandLine()
    if config.exit:
        sys.exit(0) 
    rawImg='../Data/out111.jpg'
    annotatetxt= '111,3,-4,188,210,313,head_speaker,382,33,182,309,head,434,270,71,46,lips'
    saveto_directory= r'../Data/roi'
    
#    frameroi= frameROI(rawImg, annotatetxt, saveto_directory)
    frameroi= frameROI(config.rawImg, config.annotatetxt, config.saveto_directory)
    frameroi.createROIs()

    