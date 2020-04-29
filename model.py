"""
Created on Tue Apr 21 11:21:24 2020

@author: Xiaozhou Huang

Description - Aggregation of all ML models
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
from keras.preprocessing.image import array_to_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn.naive_bayes import GaussianNB


def basic(X_train, X_test, Y_train, Y_test, seed = 42):
    

    #### Logistical Regression
    reg_model = LogisticRegression(solver = 'liblinear', random_state=seed).fit(X_train, Y_train)
    accuracy = reg_model.score(X_test, Y_test)
    print("Logistical Regression:\t\t", accuracy)
    
    #### Decision Tree
    clf = DecisionTreeClassifier(random_state=seed)
    clf = clf.fit(X_train,Y_train)
    score_c = clf.score(X_test,Y_test)
    print("Decision Tree:\t\t\t", score_c)
    
    #### Random Forest
    rfc = RandomForestClassifier(n_estimators=150,max_depth=4, random_state=seed)
    rfc = rfc.fit(X_train,Y_train)
    score_r = rfc.score(X_test,Y_test)
    print("Random Forest:\t\t\t", score_r)
    
    #### Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    score_g = gnb.score(X_test,Y_test)
    print("Gaussian Naive Bayes:\t\t", score_g)

def pca(X_train, X_test, Y_train, Y_test, seed = 42):
    
    no_components = len(X_train)
    pca_face = PCA(n_components=no_components, svd_solver='randomized', whiten=True).fit(X_train)
    
    #### Decide the components to be kept
    perc = []
    evs = 0
    sum_evs = sum(pca_face.explained_variance_)
    
    for i in range(no_components):
        evs += pca_face.explained_variance_[i]
        perc.append(evs/sum_evs)
    
    plt.xlim(0, 500)
    plt.plot(range(1, no_components+1), perc)
    plt.show()
    no_components = int(input("Enter the number of components to keep: "))
    #no_components = 10
    pca_face = PCA(n_components=no_components, svd_solver='randomized', whiten=True).fit(X_train)
    results = pca_face.transform(X_train)
    
    #### Show PCA image
    face_approx = pca_face.inverse_transform(results)
    pca_image = np.reshape(face_approx[0], (224,224,3))
    pca_image = array_to_img(pca_image.astype(np.uint8), scale=False, dtype=np.uint8)
    plt.imshow(pca_image)
    plt.show()
    
    #### Process the Test Data
    pca_x_test = PCA(n_components=no_components, svd_solver='randomized', whiten=True).fit(X_test)
    results_test = pca_face.transform(X_test)
    face_approx_test = pca_face.inverse_transform(results_test)
    
    # Reshape image array
    print("pca result:",face_approx.shape)
    pca_x_train = np.reshape(face_approx, (face_approx.shape[0], 224,224,3))
    print('X_train shape:', pca_x_train.shape)
    X_test = np.reshape(face_approx_test, (face_approx_test.shape[0], 224,224,3))
    print('X_test shape:', X_test.shape)
    
    # Train Logistic regression
    LR = LogisticRegression(solver='sag')
    LR.fit(results, Y_train)
    pre_lr = LR.predict(results_test)
    LR_accuracy = accuracy_score(pre_lr,Y_test)
    print('LogisticRegression Accuracy:', LR_accuracy)

def pca2(X_train, X_test, Y_train, Y_test, seed = 42, auto_pic = True, threshold_percent = 99):
    
    # Discover the best number of components to use 
    initial_pca = PCA(n_components=len(X_train))
    initial_pca.fit(X_train)
    
    percent_var = (initial_pca.explained_variance_ratio_ * 100)
    
    plt.xlabel('Component')
    plt.ylabel('Percentage Variance Contribution')
    plt.title("Variance Contribution from each Component")
    plt.xlim(0, 100)
    plt.plot(percent_var)
    plt.show()
    
    perc = []
    evs = 0
    sum_evs = sum(initial_pca.explained_variance_)
    
    for i in range(len(X_train)):
        evs += initial_pca.explained_variance_[i]
        perc.append(evs/sum_evs*100)
    
    plt.xlim(0, 100)
    plt.xlabel('Component')
    plt.ylabel('Cumulative Percentage Variance Contribution')
    plt.title("Cumulative Variance Contribution from each Component")
    plt.plot(perc)
    plt.show()
    
    if(auto_pic is True):
        
        for i in range(len(perc)):
            if(perc[i] > threshold_percent):
                print(i, "components will  be kept as these contribute over",threshold_percent,"% of variance.")
                break
        
        no_components = i
        
    else:    
        no_components = int(input("Enter the number of components to keep: "))

    # Transform the data
    transform_pca = PCA(n_components=no_components)
    X_train = transform_pca.fit_transform(X_train)
    X_test = transform_pca.transform(X_test)
    
    #### Test out the reduced data 
    basic(X_train, X_test, Y_train, Y_test)
          
def cnn(X_train, X_test, X_dev, Y_train, Y_test, Y_dev, seed = 42):
    
    CNN_X_train = np.reshape(X_train, (X_train.shape[0], 224,224,3))
    CNN_X_test = np.reshape(X_test, (X_test.shape[0], 224,224,3))
    CNN_X_dev = np.reshape(X_dev, (X_dev.shape[0], 224,224,3))
    
    epochs = 60
    batch_size = 16
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_CNN_model.h5'
    img_width, img_height = 224, 224
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    
    # Let's train the model using RMSprop
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    x_train = CNN_X_train.astype('float32')
    x_test = CNN_X_test.astype('float32')
    x_dev = CNN_X_dev.astype('float32')
    x_train /= 255
    x_test /= 255
    x_dev /= 225
    
    model.fit(x_train, Y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_dev, Y_dev),
                  shuffle=True)
    
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    
    # Score trained model.
    scores = model.evaluate(x_test, Y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
