import cv2
import numpy as np


#initialize camera
cap=cv2.VideoCapture(0)# 0 is id for webcam

#face detection
face_detect=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

face_data=[]
skip=0
data_path='./data/'
file_name=input('Enter the Name of the person:')

while True:
    ret,frame= cap.read()

    if not ret :
        continue

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=face_detect.detectMultiScale(frame, 1.3, 5)
    faces=sorted(faces,key=lambda f:f[2]*f[3])

    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        skip+=1
        if skip%10==0:
            cv2.imshow('face', face_section)
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow('photo', frame)

    key=cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

#convert our face data from list into numpy array
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)


#save data into file system
np.save(data_path+file_name+".npy",face_data)



cap.release()
cv2.destroyAllWindows()