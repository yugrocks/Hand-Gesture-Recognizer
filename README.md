# Hand-Gesture-Recognizer  
A python implementation of a Convolutional Neural Net that recognizes four hand gestures, namely- peace, punch, stop and thumbs_up.    
More custom gestures can be added easily by tinkering little bit with the code.  

Requirements: Python 3.5 ,Keras 2.0.2 , Tensorflow 1.2.1 , OpenCV 3.2, numpy 1.11.0   

## Demo:


### Usage:   

$ python main.py  

## Contents /Scripts:  

### - cnn_train.py: 
    This script contains code which is used to create and train the CNN. The code can be modified a little in case            to    define new custom gestures.    
### - main.py: 
    The main file which launches the gesture recognizer. It contains code to launch the camera using OpenCV and to load the pre trained model from a json file named model.json.
### - generate_data.py: 
    This script helps in creating the data for training and cross validation. Once you set the names and number of gestures as required, it automatically guides you to the process of creating the data required. For every category, it opens the camera to record your responses for every gestures, and automatically saves the pictures in corresponding directories.
### - model.json:   
    The pre trained model.
### -weights.mdf5:   
    Contains the weights of the trained model.
### -stop_conv1.png:    
    This picture visualizes the first convolutional layer, given an input image of the cartegory 'stop'.    
### The dataset: 
    The dataset currently contains a training and a test set, in which only four classes are there namely- punch, peace, thumbs_up, stop. More folders can be added as per new custom classes, and images should be placed inside the corresponding folders.   


# About the Model    

## Convolutional Neural Net schema:
![alt tag](https://raw.githubusercontent.com/yugrocks/Hand-Gesture-Recognizer/master/model.png)    

### After training for 10 epochs:    
    training accuracy: 0.9005    
    test set accuracy: 0.8813    
    training loss    : 0.4212    
    test set loss    : 0.5387    
 
