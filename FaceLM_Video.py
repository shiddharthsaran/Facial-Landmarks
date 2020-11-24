import cv2
import numpy as np
import dlib
import os
import copy

def plot(img,landmarks,r1,r2):
    '''
    plot() is used to plot facial landmarks in the given frame.
    Parameters:
    img(numpy.ndarray): img is where landmarks are plotted
    r1(int): r1 is the staring range of the 68 facial landmarks
    r2(int): r2 is the ending range of the 68 facial landmarks
    Output:
    Returns frame with facial landmarks plotted
    '''
    for n in range(r1,r2+1): #==> iterating through the facial landmark points which range from 0 to 67
            x = landmarks.part(n).x #==> getting the x co-ordinate of that particular landmark point
            y = landmarks.part(n).y #==> getting the y co-ordinate of that particular landmark point
            cv2.circle(img, (x, y), 2, (255, 255, 255), -1) #==> plotting the point using the (x,y) co-ordinates
            
    return img
    

detector=dlib.get_frontal_face_detector() #==> used to detect frontal faces
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #==> used to identify the 68 facial points

facial_landmark_range={"Jawline":(0,16),"Eyebrows":(17,26),"Nose":(27,35),"Eyes":(36,47),"Mouth":(48,67),"All":(0,67)} # the points range corresponding to each facial landmark

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

        
    img_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #==> converting to GRAY for usign less computational power
        
    faces=detector(img_gray) #==> using the builtin dlib face detector to get the co-ordinates of the faces detected
        
    if(len(faces)>0): #==> if faces are detected using those co-ordinates to find the landmarks
        for face in faces:
            landmarks=predictor(img_gray,face) #==> with the face co-ordinates predicting the facial landmarks
   
        img_gray=plot(img_gray,landmarks,facial_landmark_range["All"][0],facial_landmark_range["All"][1]) # plotting different facial landmarks   
    cv2.imshow('frame',img_gray) # Display the resulting frame

    if cv2.waitKey(1) & 0xFF == ord('q'): # Quiting the program when 'q' is pressed
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
  
       