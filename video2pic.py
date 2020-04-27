# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:39:16 2020

@author: George Tsoumis
"""
import glob
import os
import re
import cv2
import ntpath
import numpy as np


def dims(l):
    '''
    Parameters
    ----------
    l : list
        5tuple containing x,y,w,h and label of annotation.
    Returns
    -------
    x : int
    y : int
    w : int
    h : int
        x,y,w,h starting point and dimensions of bounding box
    label : string
        DESCRIPTION.
    -------
    If x or y are outside of the frame (i.e. negative value) will set x or y to 0 accordingly
    and adjust w or h
    '''
    
    (x,y,w,h),label = map(int, l[:-1]), l[-1]
    if x < 0:
        w = w+x
        x = 0
    if y < 0:
        h = h+x
        y = 0        
    return x,y,w,h,label

def splitAnnotations(numLabels, line):
    '''
    Parameters
    ----------
    numLabels : int
        number of labels in current annotation line.
    line : list
        csv line of annotations.

    Returns 
    -------
    boxList : list
        list of labels as lists
    
    -------
    Example:
    in: [x,y,w,h,label,x,y,w,h,label...]
    out: [[x,y,w,h,label],[x,y,w,h,label]...]
    '''
    
    boxList =[]
    for i in range(numLabels):
        box = line[i*5:i*5+5]
        boxList.append(box)
    return boxList

def findFace(annotationList):
    '''
    Given a list of lists, where each list is an annotation quintuple,
    will calculate if there are lips in the head by calculating if the
    center of the area annotated as lips is within the dimensions of
    the area annotated as lips.
    
    Parameters
    ----------
    annotationList : list
        list of annotation quintuples.

    Returns
    -------
    heads : list
        list of labels as lists.
    '''
    
    heads = []
    while len(annotationList)>1: #while there are more than 1 elements
        annotation = annotationList.pop(0)
        
        if(len(annotation)==0):
            continue
        
        if annotation[-1] == 'head' or annotation[-1] == 'head-speaker': #if it's a head
            xh,yh,wh,hh,labelh = dims(annotation)
            for ann in annotationList: #go through remaining labels
                if ann[-1] == 'lips' or ann[-1] == 'lips-speaker': #if it's lips
                    #calculate the lips center
                    xl,yl,wl,hl,labell = dims(ann)
                    centerX = xl+wl/2
                    centerY = yl+hl/2
                    centerXisIn = centerX in np.arange(start = xh, stop = xh+wh, step = 0.5)
                    centerYisIn = centerY in np.arange(start = yh, stop = yh+hh, step = 0.5)
                    if centerXisIn and centerYisIn: #if center is within head area
                        heads.append(annotation) #add head to detected faces
                        annotationList.remove(ann) #remove lips as one set of lips can't belong to more heads
                        break
    return heads

def cropSave(frameNum,vidname,frame,faces):
    '''
    Will extract annotated areas from a frame and store them .

    Parameters
    ----------
    frameNum : int
        current frame number.
    vidname : string
        clip name.
    frame : nd.array
        frame image converted to nd.array by open cv.
    faces : list
        list of annotations.

    Returns
    -------
    None.
    '''
    
    if(frame is None):
        return
    
    for face in faces:
        if(face is None):
            continue

        x,y,w,h,label = dims(face)
        #print(frame)
        cropped = frame[y:y+h, x:x+w]
        filename = vidname + '_frame_' + frameNum + '_' +label + '_' + str(faces.index(face)) + '.png'
        cv2.imwrite(filename,cropped)
    return

def smart_extract_areas(clip,annotation):
    '''
    Given a clip name and an annotation, will detect areas of the image that
    are marked as head or head-speaker and contain areas marked as lips or
    lips-speaker within them, extract them as noted by the annotation and
    save them as video file name(without extension)_framenumber_label_counter.png
    
    example: 5695231002474224804_veg120_frame_0_head_speaker_0.png
    
    Parameters
    ----------
    clip : string
        filepath of clip.
    annotation : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    '''

    vidcap = cv2.VideoCapture(clip)
    vidname = clip.split('.')[0]
    lines = annotation.splitlines()
    
    for line in lines:
        s,frame = vidcap.read()
        ln = line.split(",")
        frameNum = ln.pop(0)
        
        # Make sure you don't pop empty list
        if(len(ln) != 0):
            numLabels = int(ln.pop(0))
        
        boxList = splitAnnotations(numLabels,ln)
        faces = findFace(boxList)
        cropSave(frameNum,vidname,frame,faces)

def extract_areas(clip,annotation):
    '''
    Given a clip name and an annotation, will extract all areas of the image
    as noted by the annotation and save them as video filename(without extension)_framenumber_label_counter.png
    
    example: 5695231002474224804_veg120_frame_0_head_speaker_0.png
    
    Parameters
    ----------
    clip : string
        filepath of clip.
    annotation : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    '''
    vidcap = cv2.VideoCapture(clip)
    vidname = clip.split('.')[0]
    lines = annotation.splitlines()
    
    for line in lines:
        s,frame = vidcap.read()
        ln = line.split(",")
        frameNum = ln.pop(0)
        numLabels = int(ln.pop(0))
        for i in range(numLabels):
            labelData = ln[:5]
            del ln[:5]
            x,y,w,h,label = dims(labelData)

            cropped = frame[y:y+h, x:x+w]
            filename = vidname + '_frame_' + frameNum + '_' +label + '_' + str(i) + '.png'
            cv2.imwrite(filename,cropped)

def dataset2faces(path=""):
    
    file_list = glob.glob(os.path.join(os.getcwd() + path, "*_gt.txt"))
    corpus = []
    
    print(str(len(file_list)) + " files found")
    
    for file_path in file_list:
        with open(file_path) as f_input:
            corpus.append(f_input.read())
            
    for file in range (len(corpus)):
        filename = ntpath.basename(file_list[file]) #get filename without path
        vid = re.sub(r'_gt.txt','.wmv',filename)
        smart_extract_areas(vid,corpus[file]) 

dataset2faces() 
