# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:44:22 2020
@author: Ben Mouncer



def video2array(path_video, path_txt):
    
Description -   Takes a video and a annotation txt file and gives an np array
                of the images in the shape [224 * 224 * 3 , No. pictures], 
                and another array with the label of if the the picture is 
                of a speaker or not. 
    

            
def dataset2array(path="/data", exclude = ["lips", "offscreen"])

Description -   Takes the path to the folder with all the videos and txts in,
                then uses the previous function to extract all the arrays of 
                data and labels. It then gets rid of any data entry that contains
                something from "exclude" and changes the labels into boolean.
                speaker = 1
                non-speaker = 0
"""
import cv2
import csv
import numpy as np
import glob
import os
import sys

def dataset2array(path="\data", exclude = {"offscreen-speaker", "lips", "lips-speaker"}):
    
    # Get a list of the files in the path
    file_list = glob.glob(os.path.join(os.getcwd() + path, "*_gt.txt"))
    video_list = glob.glob(os.path.join(os.getcwd() + path, "*.wmv"))
    
    if(len(file_list) != len(video_list)):
        print("ERROR: Found %d videos but only %d annotations," % len(video_list), (len(file_list)))
        return 1
    
    # arrays for the data
    X_list = []
    Y_list = []
    
    # scrape the data from the first 300 files 
    # Not enough mem for len(file_list) files
    for i in range(150):
        tmp_X, tmp_Y = video2array(video_list[i],file_list[i],exclude)
        
        for j in range(len(tmp_X)):
            
            if(len(tmp_X[j]) != 150528):
                print(len(tmp_X[j]))
                print(i)
                del tmp_X[j]
                del tmp_Y[j]
        
        if(len(tmp_X) != len(tmp_Y)):
            print("Error")
            print(len(tmp_X))
            print(len(tmp_Y))
            print(i)
            

        X_list.append(tmp_X)
        Y_list.append(tmp_Y)
    
    # make np arrays to return
    y = np.concatenate(Y_list)
    x = np.concatenate(X_list)


    print("X Array Takes Up: %4.3f GB" % (sys.getsizeof(x)/ 1073741824))
    print("Y Array Takes Up: %4.3f MB" % (sys.getsizeof(y)/ 1048576))

    return x,y

def video2array(path_video, path_txt, exclude):
    
    # Taken from: https://stackoverflow.com/questions/18954889/how-to-process-images-of-a-video-frame-by-frame-in-video-streaming-using-openc
    # Accessed: April 2020
    
    frm_indx = 0
    return_data = []
    return_labels = []
    accepted_labels = {"head", "lips", "lips-speaker","head-speaker","offscreen-speaker"}
    
    capture = cv2.VideoCapture(path_video)
    
    # open the txt file
    with open(path_txt,'r') as dest_f:
        data_iter = csv.reader(dest_f,
                               delimiter = ",",
                               quotechar = '"')
        
        data = [data for data in data_iter]
    
    data_array = np.asarray(data)
    
    
    while capture.isOpened():
        
        # Extract the frame
        ret,frame = capture.read()
        
        if (ret is not True):
            break

        if(len(data_array[frm_indx]) == 0):
            continue

        # for that frame extract all the images
        for i in range(int(data_array[frm_indx][1])):
            
            # get the coordinates of the annotation
            x = int(data_array[frm_indx][2 + 5 * i])
            y = int(data_array[frm_indx][3 + 5 * i])
            w = int(data_array[frm_indx][4 + 5 * i])
            h = int(data_array[frm_indx][5 + 5 * i])
            label = data_array[frm_indx][6 + 5 * i] 
            
            # Check no bad labels have crept through
            if (label not in accepted_labels):
                print("Bad label: ", label)
                print(path_txt)
                continue
            
            if label in exclude:
                continue
                
            # Crop the image to get just the head
            crop_img = frame[y:y+h, x:x+w]
        
            # Skip annotations with no size
            if(0 in crop_img.shape):
                continue
            
            # Resize the image into the right size
            resized_img = cv2.resize(crop_img, (224, 224))
            
            # Flatten image into a single dimension array
            flattened = resized_img.flatten()
            
            return_data.append(np.array(flattened))
            
            if("speaker" in label):
                return_labels.append(1)
            else:
                return_labels.append(0)
            
            # If you want to see the images you are turning into arrays
            #image_filename = "frame" + str(frm_indx) +"_" + str(i) + "_" + label + ".jpg"
            #cv2.imwrite(image_filename, resized_img)

        frm_indx += 1
        
        if(frm_indx == len(data_array)):
            break
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
    
    return return_data, return_labels

#x,y = dataset2array()

    
    
