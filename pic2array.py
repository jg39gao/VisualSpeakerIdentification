# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27

@author: Ben Mouncer

def faces2array(path):
    
Description -   Takes the path to a folder of images to be turned into one single array
                and worked out the class by the file name of the picture. It the returns an
                a matrix of uint8 values, each row being an flattened image. Also array of 
                bool values indicating the class the image was.
"""

def faces2array(path="\data_test_train_dev\\train"):
    
    # Get a list of the files in the path
    file_list = glob.glob(os.path.join(os.getcwd() + path, "*.png"))
    
    X_data = np.zeros((len(file_list), 150528), dtype = np.ubyte)
    Y_data = np.zeros(len(file_list))
    
    for i in range(len(file_list)):
        
        # read the image
        x = cv2.imread(file_list[i])

        # Resize the image into the right size
        resized_img = cv2.resize(x, (224, 224))
        
        # Flatten image into a single dimension array
        flattened = resized_img.flatten()
        
        X_data[i] = flattened
        
        if("speaker" in file_list[i]):
            Y_data[i] = 1
    
    
    Y_data = Y_data.astype(np.bool_)
    print("X Array Takes Up: %4.3f GB" % (sys.getsizeof(X_data)/ 1073741824))
    print("Y Array Takes Up: %4.3f MB" % (sys.getsizeof(Y_data)/ 1048576))

    return X_data, Y_data
