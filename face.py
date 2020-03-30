import numpy as np
import cv2 
import matplotlib.pyplot as plt
import urllib

face_data = "haarcascade_frontalface_default.xml"
classifier =  cv2.CascadeClassifier(face_data)


URL = "http://10.23.129.192:8080/shot.jpg"

data = []

ret = True
while True:
    img = urllib.request.urlopen(URL)
    image = np.array(bytearray(img.read()),np.uint8)
    image = cv2.imdecode(image,-1)
    faces = classifier.detectMultiScale(image,1.2,5)
    for x,y,w,h in faces:
        face_img = image[y:y+h,x:x+w].copy()
       # face_img =cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)    
        
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),5)
        
        if len(data)<200:
            data.append(face_img)
        else:
            cv2.putText(image,"done",(100,100),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
    cv2.imshow('live video',image)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()



name = input("Enter the name ")
c=0
for i in data:
    cv2.imwrite('images/'+name+'_'+str(c)+'.jpg',i)
    c+=1



















