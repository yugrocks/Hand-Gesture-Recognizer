import cv2
from time import sleep
import os


print("""
INSTRUCTIONS:

This script helps in generating the Dataset for training and cross validation.
The directory structure for the dataset looks like the following, when completed (automatically created):

/Dataset
   /test_set
       /thumbs_up
       /peace
       /stop
       /punch
   /training_set
       /thumbs_up
       /peace
       /stop
       /punch

It starts with one of the Training set or Test set.
For creation of each sets, the four directories - thumbs_up/thumbs_up, peace/peace, stop/stop and punch/punch
are filled one by one. User is prompted to place their hand in the green box in front of their webcam, as soon
as it starts recording. It takes 500 images for the training set and 80 images for test_set.

""")



if not os.path.exists("Dataset"):os.mkdir("Dataset")
if not os.path.exists("Dataset/training_set"):os.mkdir("Dataset/training_set")
if not os.path.exists("Dataset/test_set"):os.mkdir("Dataset/test_set")

dirs=['thumbs_up/thumbs_up','peace/peace','stop/stop','punch/punch']
sets={'training_set':500,'test_set':80}

for set_name in sets:
    print("Taking images for the {}. Press enter when ready. ".format(set_name.upper()))
    input()
    if not os.path.exists("Dataset"):os.mkdir("Dataset/{}".format(set_name))
    for dir_name in dirs:
        print("""\nTaking images for the {} dataset. Press enter whenever ready. Note: Place the gesture to be recorded inside the green rectangle shown in the preview until it automatically disappears.""".format(dir_name))
        input()
        for _ in range(5):
            print(5-_)
            sleep(1)
        print("GO!")
        if not os.path.exists("Dataset/{}/{}".format(set_name,os.path.basename(dir_name))):os.mkdir("Dataset/{}/{}".format(set_name,os.path.basename(dir_name)))
        vc=cv2.VideoCapture(0)
        if vc.isOpened():
            rval,frame= vc.read()
        else:
            rval=False
        index=0
        
        while rval:
            index+=1
            rval, frame = vc.read()
            frame=cv2.flip(frame,1)
            cv2.putText(frame,"Keep your hand gesture in the green box.", (20,50), cv2.FONT_HERSHEY_PLAIN , 1, 255)
            cv2.putText(frame,"Taking images for {} dataset".format(dir_name), (20,80), cv2.FONT_HERSHEY_PLAIN , 1, 255)
            cv2.rectangle(frame,(300,200),(500,400),(0,255,0),1)
            cv2.imshow("Recording", frame)
            cv2.imwrite("Dataset/{}/".format(set_name)+str(dir_name)+"{}.jpg".format(index),frame[200:400,300:500]) #save image
            print("images taken: {}".format(index))
            key = cv2.waitKey(20)
            if key == 27 or index==sets[set_name]: # exit on ESC or when enough images are taken
                break

        cv2.destroyWindow("Recording")
        vc=None
