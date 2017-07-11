# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 18:48:04 2017

@author: Yugal
"""

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
from matplotlib.pyplot import imshow
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import backend as K



histarray={'PEACE':0, 'PUNCH':0, 'STOP': 0, 'Thumbs Up':0}


def load_model():
    try:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("weights.hdf5")
        print("Model successfully loaded from disk.")
        
        #compile again
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model
    except:
        print("""Model not found. Please train the CNN by running the script 
cnn_train.py. Note that the training and test samples should be properly 
set up in the dataset directory.""")
        return None
    
    
def visualize( img, layer_index=0, filter_index=0 ,all_filters=False ):
    
    act_fun = K.function([model.layers[0].input, K.learning_phase()], 
                                  [model.layers[layer_index].output,])
    
    #img = load_img('Dataset/test_set/punch/punch70.jpg',target_size=(200,200))
    x=img_to_array(img)
    img = cv2.cvtColor( x, cv2.COLOR_RGB2GRAY )
    img=img.reshape(img.shape+(1,))
    img=img.reshape((1,)+img.shape)
    img = act_fun([img,0])[0]
    
    if all_filters:
        fig=plt.figure(figsize=(7,7))
        filters = len(img[0,0,0,:])
        for i in range(filters):
                plot = fig.add_subplot(6, 6, i+1)
                plot.imshow(img[0,:,:,i],'gray')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
        plt.tight_layout()
    else:
        img = np.rollaxis(img, 3, 1)
        img=img[0][filter_index]
        print(img.shape)
        imshow(img)


def update(histarray2):
    global histarray
    histarray=histarray2


#realtime:
def realtime():
    #initialize preview
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    
    if vc.isOpened(): #get the first frame
        rval, frame = vc.read()
        
    else:
        rval = False
    
    classes=["peace","punch","stop","thumbs_up"]
    
    while rval:
        frame=cv2.flip(frame,1)
        cv2.rectangle(frame,(300,200),(500,400),(0,255,0),1)
        cv2.putText(frame,"Place your hand in the green box.", (50,50), cv2.FONT_HERSHEY_PLAIN , 1, 255)
        cv2.putText(frame,"Press esc to exit.", (50,100), cv2.FONT_HERSHEY_PLAIN , 1, 255)
        
        cv2.imshow("preview", frame)
        frame=frame[200:400,300:500]
        #frame = cv2.resize(frame, (200,200))
        frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY)
        frame=frame.reshape((1,)+frame.shape)
        frame=frame.reshape(frame.shape+(1,))
        test_datagen = ImageDataGenerator(rescale=1./255)
        m=test_datagen.flow(frame,batch_size=1)
        y_pred=model.predict_generator(m,1)
        histarray2={'PEACE': y_pred[0][0], 'PUNCH': y_pred[0][1], 'STOP': y_pred[0][2], 'Thumbs Up': y_pred[0][3]}
        update(histarray2)
        print(classes[list(y_pred[0]).index(y_pred[0].max())])
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")
    vc=None
    

#loading the model

model=load_model()
#visualize(load_img('Dataset/test_set/stop/stop1.jpg',target_size=(200,200)),filter_index=0,all_filters=True)


if model is not None:
    
    ans=str(input("Do you want to plot a realtime histogram as well? (slower) y/n\n"))
    
    if ans.lower()=='y':
        #the code for histogram 
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        
        def animate(i): 
            xar= [1, 2, 3, 4]
            yar = []
            xtitles = ['']
            for items in histarray:
                yar.append(histarray[items])
                xtitles.append(items)
            
            ax1.clear()        
            plt.bar(xar,yar, align='center')
            plt.xticks(np.arange(5), xtitles)
            
        ani = animation.FuncAnimation(fig, animate, interval=500)
        fig.show()

    #threading.Thread(target=realtime).start()
    realtime()
    
