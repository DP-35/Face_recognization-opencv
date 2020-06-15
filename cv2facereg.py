import cv2
import numpy as np
import os

clf=cv2.face.LBPHFaceRecognizer_create()
clf.read('trainer.yml')
cascadepath="haarcascade_frontalface_default.xml"
faceCascade=cv2.CascadeClassifier(cascadepath)
font=cv2.FONT_HERSHEY_SIMPLEX

id=0
name=['none','Divya','tony']


cap=cv2.VideoCapture(0)
#minW=0.1*cap.get(3)
#minH=0.1*cap.get(4)
while True:
	ret,img=cap.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray,1.5,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		id,confidence=clf.predict(gray[y:y+h,x:x+w])
		if confidence<100:
			id=name[id]
			confidence=" {0}%".format(round(100-confidence))
		else:
			id="unknown"
			confidence=" {0}%".format(round(100-confidence))
		cv2.putText(img,str(id),(x+5,y-5),font,1,(255,255,255),2)
		cv2.putText(img,str(confidence),(x+5,y+h-5),font,1,(255,255,0),1)
	cv2.imshow('image',img)
	k=cv2.waitKey(10) & 0xFF
	if k==27:
		break

cap.release()
cv2.destroyAllWindows()


    
