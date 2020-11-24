import cv2
import numpy as np
import dlib
import os
import copy

def plot(img,landmarks,file_name,r1,r2,directory,sub_folder_name):
    '''
    plot() is used to plot facial landmarks in the given image.
    Parameters:
    img(numpy.ndarray): img is where landmarks are plotted
    file_name(str): file_name is the name of the file to be saved with file format
    r1(int): r1 is the staring range of the 68 facial landmarks
    r2(int): r2 is the ending range of the 68 facial landmarks
    directory(str): directory is where the plotted image is to be saved
    sub_folder_name(str): sub_folder_name is the name of the sub folder in which you need to save the images
    Output:
    Saves facial landmark plotted images
    '''
    for n in range(r1,r2+1): #==> iterating through the facial landmark points which range from 0 to 67
            x = landmarks.part(n).x #==> getting the x co-ordinate of that particular landmark point
            y = landmarks.part(n).y #==> getting the y co-ordinate of that particular landmark point
            cv2.circle(img, (x, y), 5, (0, 0, 0), -1) #==> plotting the point using the (x,y) co-ordinates
            
    path=directory+"\\"+sub_folder_name #==> path to save the plotted image
    
    if(os.path.exists(path)): #==> checking if the path exists
        pass
    else:
        os.makedirs(path) #==> if doesn't exists creating the path
    cv2.imwrite(path+"\\"+file_name,img) #==> saving the image in the specified directory

detector=dlib.get_frontal_face_detector() #==> used to detect frontal faces
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #==> used to identify the 68 facial points

facial_landmarks=["Jawline","Eyebrows","Nose","Eyes","Mouth","All"] #==> all the facial landmarks

facial_landmark_range={"Jawline":(0,16),"Eyebrows":(17,26),"Nose":(27,35),"Eyes":(36,47),"Mouth":(48,67),"All":(0,67)} # the points range corresponding to each facial landmark

for file_name in os.listdir("Images\\"): #==> iterating through every images in the T3-Candidate folder
    img=cv2.imread("Images\\"+file_name) #==> reading the image using cv2
    temp_img=copy.copy(img) #==> making a copy of img to return the values back once points are plotted
    
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #==> converting to GRAY for usign less computational power
    
    faces=detector(img_gray) #==> using the builtin dlib face detector to get the co-ordinates of the faces detected
    
    if(len(faces)>0): #==> if faces are detected using those co-ordinates to find the landmarks
        for face in faces:
            landmarks=predictor(img_gray,face) #==> with the face co-ordinates predicting the facial landmarks
    else:
        face=dlib.rectangle(left=0, top=0, right=img.shape[1], bottom=img.shape[0]) #==> As T1 were mostly cropped face images i took the whole image as face-coordinates
        landmarks=predictor(img_gray,face) #==> with the face co-ordinates predicting the facial landmarks
    
    for fl in facial_landmarks:
        plot(img,landmarks,file_name,facial_landmark_range[fl][0],facial_landmark_range[fl][1],"Face_Landmarks",fl) # plotting and saving different facial landmarks
        img=copy.copy(temp_img) # reversing the img state to original    