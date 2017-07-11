# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:36:21 2017

@author: Yugal
"""

from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img
from numpy import array
from keras import regularizers
import cv2


#init the model
model= Sequential()

#add conv layers and pooling layers 
model.add(Convolution2D(32,3,3, input_shape=(200,200,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,3,3, input_shape=(200,200,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5)) #to reduce overfitting

model.add(Flatten())

#Now two hidden(dense) layers:
model.add(Dense(output_dim = 150, activation = 'relu',
                kernel_regularizer=regularizers.l2(0.01)))

model.add(Dropout(0.5))#again for regularization

model.add(Dense(output_dim = 150, activation = 'relu',
                kernel_regularizer=regularizers.l2(0.01)))


model.add(Dropout(0.5))#last one lol

model.add(Dense(output_dim = 150, activation = 'relu',
                kernel_regularizer=regularizers.l2(0.01)))

#output layer
model.add(Dense(output_dim = 4, activation = 'sigmoid'))


#Now copile it
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#Now generate training and test sets from folders

train_datagen=ImageDataGenerator(
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.,
                                   horizontal_flip = False
                                 )

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory("Dataset/training_set",
                                               target_size = (200,200),
                                               color_mode='grayscale',
                                               batch_size=10,
                                               class_mode='categorical')

test_set=test_datagen.flow_from_directory("Dataset/test_set",
                                               target_size = (200,200),
                                               color_mode='grayscale',
                                               batch_size=10,
                                               class_mode='categorical')






#finally, start training
model.fit_generator(training_set,
                         samples_per_epoch = 1956,
                         nb_epoch = 10,
                         validation_data = test_set,
                         nb_val_samples = 320)


#after 10 epochs:
    #training accuracy: 0.9005
    #training loss:     0.4212
    #test set accuracy: 0.8813
    #test set loss:     0.5387

#saving the weights
model.save_weights("weights.hdf5",overwrite=True)

#saving the model itself in json format:
model_json = model.to_json()
with open("model.json", "w") as model_file:
    model_file.write(model_json)
print("Model has been saved.")


#testing it to a random image from the test set
img = load_img('Dataset/test_set/stop/stop26.jpg',target_size=(200,200))
x=array(img)
img = cv2.cvtColor( x, cv2.COLOR_RGB2GRAY )
img=img.reshape((1,)+img.shape)
img=img.reshape(img.shape+(1,))

test_datagen = ImageDataGenerator(rescale=1./255)
m=test_datagen.flow(img,batch_size=1)
y_pred=model.predict_generator(m,1)


#save the model schema in a pic
plot_model(model, to_file='model.png', show_shapes = True)






