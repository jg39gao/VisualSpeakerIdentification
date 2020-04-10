# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:32:17 2020

@author: George Tsoumis
"""

import glob
import os
import ntpath


def label(row):
    #given a list of data will create xml entry for a particular label
    # label()
    # <clip name = "1233">
    #     <framenumber name="1">
    #         <label name = "head">
    #             <x>111</x>
    #             <y>111</y>
    #             <h>111</h>
    #             <w>111</w>
    #         </label>
    #         <label name = "lips">
    #             <x>222</x>
    #             <y>222</y>
    #             <h>222</h>
    #             <w>2</w>
    #         </label>
    #     </framenumber>
    # </clip>
    label = '''        <label name="%s">
            <x>%s</x>
            <y>%s</y>
            <h>%s</h>
            <w>%s</w>
        </label>\n''' % (row[4], row[0], row[1], row[2], row[3])
    return(label)

def label2(row):
    #given a list of data will create xml entry for a particular label
    # label2() output
    # <clip name = "1234>
    #     <framenumber name="1">
    #         <boundingBox>
    #             <x>111</x>
    #             <y>111</y>
    #             <h>111</h>
    #             <w>111</w>
    #             <label>head</label>
    #         </boundingBox>
    #         <boundingBox>
    #             <x>222</x>
    #             <y>222</y>
    #             <h>222</h>
    #             <w>222</w>
    #             <label>lips</label>
    #         </boundingBox>
    #     </framenumber>
    # </clip>
    label = '''        <boundingBox>
            <x>%s</x>
            <y>%s</y>
            <h>%s</h>
            <w>%s</w>
            <label>%s</label>
        </boundingBox>\n''' % (row[0], row[1], row[2], row[3], row[4])
    return(label)

def label3(row):
    # label3()
    # <clip name = "1234>
    #     <framenumber name="1">
    #         <boundingBox>
    #             <x>111</x>
    #             <y>111</y>
    #             <h>111</h>
    #             <w>111</w>
    #             <label>head</label>
    #             <speaker>True</speaker>
    #         </boundingBox>
    #         <boundingBox>
    #             <x>222</x>
    #             <y>222</y>
    #             <h>222</h>
    #             <w>222</w>
    #             <label>lips</label>
    #             <speaker>False</speaker>
    #         </boundingBox>
    #         .
    #         .
    #         .
    #     </framenumber>
    # </clip>
    delim = "-"
    if delim in row[4]:
        lbl = row[4].split(delim)[0]
        spkrBool = "True"
    else:
        lbl = row[4]
        spkrBool = "False"
        
    label = '''        <boundingBox>
            <x>%s</x>
            <y>%s</y>
            <h>%s</h>
            <w>%s</w>
            <label>%s</label>
            <isSpeaker>%s</isSpeaker>
        </boundingBox>\n''' % (row[0], row[1], row[2], row[3], lbl, spkrBool)
    return(label)


def csv2xml(file,filename):
    
    lines = file.splitlines()
    frame = ''
    for line in lines:
        ln = line.split(",")
        frameNum = ln.pop(0)
        numLabels = int(ln.pop(0))
        subchildren = ''
        for i in range(numLabels):
            labelData = ln[:5]
            del ln[:5]
            subchildren += label(labelData)
        frame += '    <frameNumber name="%s">\n%s    </framenumber>\n' % (frameNum,subchildren)
    clip =  '''<?xml version="1.0" encoding="UTF-8"?>\n<clip name="%s">\n%s</clip>''' % (filename,frame)
    return(clip)

def store2xml(xmlstring,filename):
    #given a string and a filename, will create 'filename.xml' and store string in it    
    filename = filename.split('.')[0]+'.xml'
    with open(filename, 'w+') as file:
        file.write(xmlstring)

file_list = glob.glob(os.path.join(os.getcwd(), "*_gt.txt"))
corpus = []
for file_path in file_list:
    with open(file_path) as f_input:
        corpus.append(f_input.read())

for file in range (len(corpus)):
    filename = ntpath.basename(file_list[file]) #get filename without extension
    xml = csv2xml(corpus[file],filename) #create xml string
    print(xml)
    store2xml(xml,filename) #store




