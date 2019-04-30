import cv2
import numpy as np
import os

def distance(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())

def knn(x_train,y_train,x_test,k=5):
    dist_val=[]
    m=x_train.shape[0]
    for ix in range(m):
        d=distance(x_test,x_train[ix])
        dist_val.append([d,y_train[ix]])

    dist_val=sorted(dist_val)
    dist_val=dist_val[:k]

    y=np.array(dist_val)
    ans=np.unique(y[:,1],return_counts=True)
    index=ans[1].argmax()
    prediction=ans[0][index]
    return int(prediction)


data_path='./data/'
face_data=[]
labels=[]

class_id=0
names={} #map id-name

#data preparation
for fx in os.listdir(data_path):
    if fx.endswith('.npy'):
        #Mapping class_id and name of person(file)
        names[class_id]=fx[:-4]
        data_item=np.load(data_path+fx)
        face_data.append(data_item)
        # print(fx)
        # print(data_item.shape)
        target=class_id*np.ones((data_item.shape[0]))
        labels.append(target)
        class_id += 1

# print(len(face_data))
face_set=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0) ##.reshape(-1,1)

# print(face_set.shape)
# print(face_labels.shape)

# I don't need that 2 line of code bacause i used separate x_test and y_test in knn
# train_set=np.concatenate((face_set,face_labels),axis=1)
# print(train_set.shape)




#testing
#initialize camera
cap=cv2.VideoCapture(0)# 0 is id for webcam

#face detection
face_detect=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
    ret,frame=cap.read()

    if not ret:
        continue

    faces=face_detect.detectMultiScale(frame,1.3,5)

    for face in faces:
        x,y,w,h=face
        #get the face section
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))

        prediction=knn(face_set,face_labels,face_section.flatten())
        predicted_name=names[prediction]
        print(predicted_name)
        cv2.putText(frame,predicted_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    cv2.imshow('photo',frame)

    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
