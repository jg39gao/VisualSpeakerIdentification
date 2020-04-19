from predict import HeadDetector


hd = HeadDetector()

inputpath = "./avengers_img/"
outputpath = "./avengers_img/output/"
print(hd.detect_head(inputpath, exportImg=True, outputpath = outputpath))
