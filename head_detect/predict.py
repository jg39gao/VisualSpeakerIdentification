import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json


class HeadDetector:
    
    def __init__(self, config_path='config.json', weights_path='model.h5'):
        
        self.config_path  = config_path
        self.weights_path = weights_path
    
    def detect_head(self, filepath, exportImg=False, outputpath = "./output/"):
        
        filenames = os.listdir(filepath)
        
        filelist = []
        for name in filenames:
            if (name.endswith('.jpg') or name.endswith('.png')):
                filelist.append(filepath + name)
                

        with open(self.config_path) as config_buffer:    
            config = json.load(config_buffer)


        #   Make the model 
        yolo = YOLO(backend             = config['model']['backend'],
                    input_size          = config['model']['input_size'], 
                    labels              = config['model']['labels'], 
                    max_box_per_image   = config['model']['max_box_per_image'],
                    anchors             = config['model']['anchors'])


        #   Load trained weights
        yolo.load_weights(self.weights_path)
    
        listOfArrays = []
        for imageName in filelist:
            image = cv2.imread(imageName)
            boxes = yolo.predict(image)
            image, box_array = draw_boxes(image, boxes, config['model']['labels'])
            listOfArrays.append(box_array)
            # print(len(boxes), 'boxes are found')
            if exportImg == True:
                if os.path.exists(outputpath) == False:
                    os.mkdir(outputpath)
                cv2.imwrite(outputpath + imageName[-7:-4] + '_detected' + imageName[-4:], image)   
                
        return listOfArrays
