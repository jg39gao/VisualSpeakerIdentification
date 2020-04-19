import os      
import subprocess


class VideotoImg:

    def __init__(self):
        
        pass
        
    def VoI(self, inputpath, outputpath):
        
        filenames = os.listdir(inputpath)
        
        filelist = []
        for name in filenames:
            if (name.endswith('.wmv') or name.endswith('.mp4')):
                filelist.append(inputpath + name)
                
        for clipname in filelist:
            command = "ffmpeg -i " + clipname + " -r 24/1 " + outputpath + clipname[-7:-4] + "_%03d.jpg"
            subprocess.call(command, shell=True)
